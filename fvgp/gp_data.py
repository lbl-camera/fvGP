import numpy as np
import warnings


MULTI_TASK_MODES = ("independent", "cartesian", "kronecker")


class GPdata:
    def __init__(self, x_data, y_data,
                 args=None,
                 noise_variances=None,
                 gradient_per_direction=False,
                 gp2Scale=False,
                 compute_device="cpu",
                 dask_client=None,
                 multi_task_mode="independent"):

        if multi_task_mode not in MULTI_TASK_MODES:
            raise ValueError(f"multi_task_mode must be 'independent', 'cartesian' or 'kronecker', "
                             f"got {multi_task_mode!r}.")
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list), \
            "x_data must be np.ndarray or list"
        assert isinstance(y_data, np.ndarray) and (np.ndim(y_data) == 1 or np.ndim(y_data) == 2), \
            "y_data must be a 1-d or 2-d np.ndarray"
        assert len(x_data) == len(y_data), "x_data and y_data have different lengths."
        if np.ndim(y_data) == 1: y_data = y_data.reshape(len(y_data), 1)

        # The shape of y_data decides: one column is a single-task GP, several are a multi-task
        # GP in multi_task_mode -- "independent", several right-hand sides sharing one K;
        # "cartesian", a single-task GP over (point, task) pairs; "kronecker", one (n_x, n_out)
        # tensor on a full grid. None for a single-task GP.
        self.output_num = y_data.shape[1]
        self.multi_task = multi_task = multi_task_mode if self.output_num > 1 else None
        self.kronecker = multi_task == "kronecker"
        self._check_noise_variances(noise_variances, y_data)

        # analyze data
        if isinstance(x_data, np.ndarray):
            assert np.ndim(x_data) == 2, "Euclidean x_data must be 2-d (n_points × input_dim)"
            self.input_set_dim = len(x_data[0])
            self.Euclidean = True
        if isinstance(x_data, list):
            self.input_set_dim = 1
            self.Euclidean = False
        # the output index is one more index dimension: a coordinate of the Cartesian points,
        # and the flat output index of the Kronecker default kernel
        self.index_set_dim = self.input_set_dim
        if self.Euclidean and multi_task in ("cartesian", "kronecker"): self.index_set_dim += 1
        # the Cartesian posterior predicts at (x_pred, task) pairs; the other modes select columns
        self.x_out = np.arange(self.output_num) if multi_task == "cartesian" else None
        if multi_task == "cartesian":
            x_data, y_data, noise_variances = self._cartesian_transform(x_data, y_data, noise_variances)
            y_data = y_data.reshape(len(y_data), 1)

        self.x_data = x_data
        self.y_data = y_data
        self.noise_variances = noise_variances
        self.x_new = None
        self.y_new = None
        self.noise_variances_new = None
        self.x_old = None
        self.y_old = None
        self.noise_variances_old = None
        self.point_number = len(self.x_data)
        self._check_for_nan()
        self.args = args
        self.gradient_per_direction = gradient_per_direction
        self.gp2Scale = gp2Scale
        self.compute_device = compute_device
        self.dask_client = dask_client
        self.compute_workers = []
        if gp2Scale and dask_client is not None:
            self.compute_workers = list(dask_client.scheduler_info()["workers"].keys())

    def _check_noise_variances(self, noise_variances, y_data):
        """Measured noise is one variance per observation: (N, Ni), like y_data, in the Cartesian
        mode (np.nan where y_data is), else one per input point, (N), shared by all outputs."""
        if noise_variances is None: return
        shape = y_data.shape if self.multi_task == "cartesian" else (len(y_data),)
        if not isinstance(noise_variances, np.ndarray) or noise_variances.shape != shape:
            got = noise_variances.shape if isinstance(noise_variances, np.ndarray) else type(noise_variances)
            raise ValueError(f"noise_variances must be a np.ndarray of shape {shape} "
                             f"(multi_task_mode={self.multi_task!r}), got {got}.")

    def update(self, x_data_new, y_data_new, noise_variances_new=None, append=True):
        """New data arrive in the format of the constructor's (x, (V, Ni) outputs, noise per
        mode); a Cartesian chunk is transformed to (point, task) pairs before it is stored."""
        assert isinstance(x_data_new, np.ndarray) or isinstance(x_data_new, list), \
            "x_data_new must be np.ndarray or list"
        assert isinstance(y_data_new, np.ndarray) and (np.ndim(y_data_new) == 1 or np.ndim(y_data_new) == 2), \
            "y_data_new must be a 1-d or 2-d np.ndarray"
        if np.ndim(y_data_new) == 1: y_data_new = y_data_new.reshape(len(y_data_new), 1)
        if y_data_new.shape[1] != self.output_num:
            raise ValueError(f"y_data_new must have {self.output_num} column(s), one per output as in y_data, "
                             f"got {y_data_new.shape[1]}.")
        self._check_noise_variances(noise_variances_new, y_data_new)
        if self.Euclidean: assert isinstance(x_data_new, np.ndarray) and np.ndim(x_data_new) == 2, \
            "Euclidean x_data_new must be a 2-d np.ndarray"
        # A non-Euclidean point is an arbitrary object, so there is no dimensionality to
        # check: __init__ accepts any list here and sets index_set_dim to 1 regardless.
        else: assert isinstance(x_data_new, list), "non-Euclidean x_data_new must be a list"

        if self.noise_variances is not None and noise_variances_new is None:
            raise Exception("Please provide noise_variances in the data update because you did at initialization "
                            "or during a previous update.")
        if self.noise_variances is None and noise_variances_new is not None:
            raise Exception("You did not initialize noise and but included noise in the update."
                            "Please reinitialize in this case.")
        if self.multi_task == "cartesian":
            x_data_new, y_data_new, noise_variances_new = \
                self._cartesian_transform(x_data_new, y_data_new, noise_variances_new)
            y_data_new = y_data_new.reshape(len(y_data_new), 1)

        if append is False:
            self.x_data = x_data_new
            self.y_data = y_data_new
            self.noise_variances = noise_variances_new
            self.x_old = None
            self.y_old = None
            self.noise_variances_old = None
            self.x_new = None
            self.y_new = None
            self.noise_variances_new = None
        else:
            self.x_old = self.x_data
            self.y_old = self.y_data
            self.noise_variances_old = self.noise_variances
            self.x_new = x_data_new
            self.y_new = y_data_new
            self.noise_variances_new = noise_variances_new
            if self.Euclidean: self.x_data = np.vstack([self.x_data, x_data_new])
            else: self.x_data = self.x_data + x_data_new
            self.y_data = np.vstack([self.y_data, y_data_new])
            if isinstance(noise_variances_new, np.ndarray):
                self.noise_variances = np.append(self.noise_variances, noise_variances_new)
        self.point_number = len(self.x_data)
        self._check_for_nan()

    def _cartesian_transform(self, x_data, y_data, noise_variances):
        """Every observed (point, task) pair becomes one point [x, task] (a Euclidean x gets the
        task index appended, a non-Euclidean point is paired with it), task-major; NaN outputs
        are skipped."""
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list), \
            "x_data must be np.ndarray or list"
        assert isinstance(y_data, np.ndarray), "y_data must be np.ndarray"
        assert len(x_data) == len(y_data), "x_data and y_data have different lengths"
        if noise_variances is not None: assert len(noise_variances) == len(y_data), \
            "noise_variances and y_data have different lengths"
        new_x_data = []
        new_y_data = []
        output_indices = np.arange(0, self.output_num)
        if noise_variances is not None: new_variances = []
        else: new_variances = None
        for i in range(self.output_num):
            for j in range(len(x_data)):
                if noise_variances is not None: assert len(noise_variances[j]) == self.output_num, \
                    f"noise_variances row {j} length must equal output_num={self.output_num}"
                assert len(y_data[j]) == self.output_num, \
                    f"y_data row {j} length must equal output_num={self.output_num}"
                if np.isnan(y_data[j, i]): continue
                if isinstance(x_data, np.ndarray): new_x_data.append(np.append(x_data[j], output_indices[i]))
                else: new_x_data.append([x_data[j], output_indices[i]])
                new_y_data.append(y_data[j, i])
                if new_variances is not None:
                    new_variances.append(noise_variances[j, i])
        if isinstance(x_data, np.ndarray): new_x_data = np.asarray(new_x_data)
        new_y_data = np.asarray(new_y_data)
        if new_variances is not None: new_variances = np.asarray(new_variances)

        assert isinstance(new_x_data, list) or isinstance(new_x_data, np.ndarray), \
            "transformed x_data must be list or np.ndarray"
        assert isinstance(new_y_data, np.ndarray) and np.ndim(new_y_data) == 1, \
            "transformed y_data must be a 1-d np.ndarray"
        if new_variances is not None: assert isinstance(new_variances, np.ndarray) and np.ndim(new_variances) == 1, \
            "transformed noise_variances must be a 1-d np.ndarray"
        return new_x_data, new_y_data, new_variances

    def _check_for_nan(self):
        if self.multi_task in ("kronecker", "independent") and np.isnan(np.sum(self.y_data)):
            raise ValueError(f"multi_task_mode={self.multi_task!r} needs every output at every input (a full "
                             f"grid), but y_data contains NaN. Use multi_task_mode='cartesian' for missing outputs.")
        if self.Euclidean:
            if np.isnan(np.sum(self.x_data) + np.sum(self.y_data)): raise Exception("NaNs encountered in dataset.")

    def __getstate__(self):
        state = dict(
            x_data=self.x_data,
            y_data=self.y_data,
            noise_variances=self.noise_variances,
            Euclidean=self.Euclidean,
            index_set_dim=self.index_set_dim,
            x_new=self.x_new,
            y_new=self.y_new,
            noise_variances_new=self.noise_variances_new,
            x_old=self.x_old,
            y_old=self.y_old,
            noise_variances_old=self.noise_variances_old,
            point_number=self.point_number,
            output_num=self.output_num,
            x_out=self.x_out,
            input_set_dim=self.input_set_dim,
            args=self.args,
            gradient_per_direction=self.gradient_per_direction,
            gp2Scale=self.gp2Scale,
            compute_device=self.compute_device,
            dask_client=None,
            compute_workers=self.compute_workers,
            kronecker=self.kronecker,
            multi_task=self.multi_task,
            )
        return state

    def __setstate__(self, state):
        # pickled before 4.11: the old name of gradient_per_direction, and fvGP's copies of the
        # data before its transform, which nothing reads any more
        if "ram_economy" in state: state["gradient_per_direction"] = state.pop("ram_economy")
        for key in ("fvgp_x_data", "fvgp_y_data", "fvgp_noise_variances"): state.pop(key, None)
        self.__dict__.update(state)
        if "kronecker" not in self.__dict__: self.kronecker = False
        if "multi_task" not in self.__dict__: self.multi_task = "kronecker" if self.kronecker else None
        if "output_num" not in self.__dict__:
            self.output_num = len(self.x_out) if self.x_out is not None else self.y_data.shape[1]


