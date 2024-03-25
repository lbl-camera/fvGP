import numpy as np


class GPdata:
    def __init__(self, x_data, y_data, noise_variances=None):
        # make sure the inputs are in the right format
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list)
        assert isinstance(y_data, np.ndarray) and np.ndim(y_data) == 1
        assert ((isinstance(noise_variances, np.ndarray) and np.ndim(noise_variances) == 1)
                or noise_variances is None)
        assert len(x_data) == len(y_data)
        if noise_variances is not None: assert len(x_data) == len(noise_variances)

        # analyse data
        if isinstance(x_data, np.ndarray):
            assert np.ndim(x_data) == 2
            self.input_space_dim = len(x_data[0])
            self.Euclidean = True
        if isinstance(x_data, list):
            self.input_space_dim = 1
            self.Euclidean = False

        self.x_data = x_data
        self.y_data = y_data
        self.noise_variances = noise_variances
        self.point_number = len(self.x_data)
        self._check_for_nan()

    def update(self, x_data_new, y_data_new, noise_variances_new=None, append=True):
        assert isinstance(x_data_new, np.ndarray) or isinstance(x_data_new, list)
        assert isinstance(y_data_new, np.ndarray) and np.ndim(y_data_new) == 1
        assert ((isinstance(noise_variances_new, np.ndarray) and np.ndim(noise_variances_new) == 1)
                or noise_variances_new is None)
        if self.Euclidean: assert isinstance(x_data_new, np.ndarray) and np.ndim(x_data_new) == 2
        else: assert isinstance(x_data_new, list)
        if self.noise_variances and noise_variances_new is None:
            raise Exception("Please provide noise_variances in the data update.")

        if append is False:
            self.x_data = x_data_new
            self.y_data = y_data_new
        else:
            if self.Euclidean:
                np.row_stack([self.x_data, x_data_new])
            else:
                self.x_data = self.x_data + x_data_new
            self.y_data = np.append(self.y_data, y_data_new)
            if noise_variances_new is not None:
                self.noise_variances = np.append(self.noise_variances, noise_variances_new)
        self.point_number = len(self.x_data)
        self._check_for_nan()

    def _check_for_nan(self):
        if self.Euclidean:
            if np.isnan(np.sum(self.x_data) + np.sum(self.y_data)): raise Exception("NaNs encountered in dataset.")