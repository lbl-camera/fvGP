import numpy as np
import warnings


class GPdata:
    def __init__(self, x_data, y_data, noise_variances=None):
        # make sure the inputs are in the right format
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list)
        assert isinstance(y_data, np.ndarray) and (np.ndim(y_data) == 1 or np.ndim(y_data) == 2)
        assert ((isinstance(noise_variances, np.ndarray) and np.ndim(noise_variances) == 1)
                or noise_variances is None)
        assert len(x_data) == len(y_data), "x_data and y_data have different lengths."

        # analyse data
        if isinstance(x_data, np.ndarray):
            assert np.ndim(x_data) == 2
            self.index_set_dim = len(x_data[0])
            self.input_set_dim = len(x_data[0])
            self.Euclidean = True
        if isinstance(x_data, list):
            self.index_set_dim = 1  #in non-Euclidean spaces there is no notion of dimensionality
            self.input_set_dim = 1
            self.Euclidean = False

        self.x_data = x_data
        self.y_data = y_data
        self.noise_variances = noise_variances
        self.point_number = len(self.x_data)
        self._check_for_nan()
        self.fvgp_x_data = None
        self.fvgp_y_data = None
        self.fvgp_noise_variances = None
        self.x_out = None

    def set_fvgp_data(self, fvgp_x_data, fvgp_y_data, fvgp_noise_variances, x_out):
        self.fvgp_x_data = fvgp_x_data
        self.fvgp_y_data = fvgp_y_data
        self.fvgp_noise_variances = fvgp_noise_variances
        self.x_out = x_out
        if self.Euclidean: self.input_set_dim = self.index_set_dim - 1

    def update(self, x_data_new, y_data_new, noise_variances_new=None, append=True):
        assert isinstance(x_data_new, np.ndarray) or isinstance(x_data_new, list)
        assert isinstance(y_data_new, np.ndarray) and np.ndim(y_data_new) == 1
        assert ((isinstance(noise_variances_new, np.ndarray) and np.ndim(noise_variances_new) == 1)
                or noise_variances_new is None)
        if self.Euclidean: assert isinstance(x_data_new, np.ndarray) and np.ndim(x_data_new) == 2
        else: assert (isinstance(x_data_new, list) and
                      np.ndim(x_data_new) == 2 and
                      self.index_set_dim == x_data_new.shape[1])

        if self.noise_variances is not None and noise_variances_new is None:
            raise Exception("Please provide noise_variances in the data update because you did at initialization "
                            "or during a previous update.")
        if self.noise_variances is None and noise_variances_new is not None:
            raise Exception("You did not initialize noise and but included noise in the update."
                            "Please reinitialize in this case.")
        if callable(noise_variances_new): raise Exception("The update noise_variances cannot be a callable.")
        if noise_variances_new is not None:
            assert isinstance(noise_variances_new, np.ndarray) and np.ndim(noise_variances_new) == 1

        if append is False:
            self.x_data = x_data_new
            self.y_data = y_data_new
            self.noise_variances = noise_variances_new
        else:
            if self.Euclidean: self.x_data = np.vstack([self.x_data, x_data_new])
            else: self.x_data = self.x_data + x_data_new
            self.y_data = np.append(self.y_data, y_data_new)
            if isinstance(noise_variances_new, np.ndarray):
                self.noise_variances = np.append(self.noise_variances, noise_variances_new)
        self.point_number = len(self.x_data)
        self._check_for_nan()

    def _check_for_nan(self):
        if self.Euclidean:
            if np.isnan(np.sum(self.x_data) + np.sum(self.y_data)): raise Exception("NaNs encountered in dataset.")

    def __getstate__(self):
        state = dict(
            x_data=self.x_data,
            y_data=self.y_data,
            Euclidean=self.Euclidean,
            index_set_dim=self.index_set_dim,
            noise_variances=self.noise_variances,
            point_number=self.point_number,
            fvgp_x_data=self.fvgp_x_data,
            fvgp_y_data=self.fvgp_y_data,
            fvgp_noise_variances=self.fvgp_noise_variances,
            x_out=self.x_out,
            input_set_dim=self.input_set_dim
            )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


