import numpy as np
import warnings


class GPlikelihood:  # pragma: no cover
    def __init__(self, data_obj,
                 hyperparameters=None,
                 gp_noise_function=None,
                 gp_noise_function_grad=None,
                 ram_economy=False,
                 gp2Scale=False,
                 online=False):

        assert isinstance(hyperparameters, np.ndarray) and np.ndim(hyperparameters) == 1
        self.data_obj = data_obj
        self.x_data = self.data_obj.x_data
        self.y_data = self.data_obj.y_data
        self.noise_variances = self.data_obj.noise_variances
        assert self.noise_variances is None or isinstance(self.noise_variances, np.ndarray)

        if isinstance(self.noise_variances, np.ndarray):
            assert np.ndim(self.noise_variances) == 1
            assert all(self.noise_variances > 0.0)

        self.gp2Scale = gp2Scale

        if self.noise_variances is not None and callable(gp_noise_function):
            raise Exception("Noise function and measurement noise provided. Decide which one to use.", stacklevel=2)
        if callable(gp_noise_function):
            self.noise_function = gp_noise_function
        elif self.noise_variances is not None:
            self.noise_function = self._measured_noise_function
        else:
            warnings.warn(
                "No noise function or measurement noise provided. Noise variances will be set to 1% of mean(y_data).",
                stacklevel=2)
            self.noise_function = self._default_noise_function

        if callable(gp_noise_function_grad):
            self.noise_function_grad = gp_noise_function_grad
        elif callable(gp_noise_function):
            if ram_economy is True:
                self.noise_function_grad = self._finitediff_dnoise_dh_econ
            else:
                self.noise_function_grad = self._finitediff_dnoise_dh
        else:
            if ram_economy is True:
                self.noise_function_grad = self._default_dnoise_dh_econ
            else:
                self.noise_function_grad = self._default_dnoise_dh

        self.V = self.noise_function(self.x_data, hyperparameters, self)
        self.online = online

    ##################################################################################
    def update(self, hyperparameters=None):
        self.x_data = self.data_obj.x_data
        self.y_data = self.data_obj.y_data
        self.noise_variances = self.data_obj.noise_variances
        self.V = self.noise_function(self.x_data, hyperparameters, self)

    def _default_noise_function(self, x, hyperparameters, gp_obj):
        noise = np.ones((len(x))) * (np.mean(abs(self.y_data)) / 100.0)
        if self.gp2Scale: return self.gp2Scale_obj.calculate_sparse_noise_covariance(noise)
        else: return np.diag(noise)

    def _measured_noise_function(self, x, hyperparameters, gp_obj):
        if self.gp2Scale: return self.gp2Scale_obj.calculate_sparse_default_noise_covariance(noise) #should be here
        else: return np.diag(self.noise_variances)

    def _default_dnoise_dh(self, x, hps, gp_obj):
        gr = np.zeros((len(hps), len(x), len(x)))
        return gr

    def _default_dnoise_dh_econ(self, x, i, hps, gp_obj):
        gr = np.zeros((len(x), len(x)))
        return gr

    ##########################
    def _finitediff_dnoise_dh(self, x, hps, gp_obj):
        gr = np.zeros((len(hps), len(x), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.noise_function(x, temp_hps1, self)
            b = self.noise_function(x, temp_hps2, self)
            gr[i] = (a - b) / 2e-6
        return gr

    ##########################
    def _finitediff_dnoise_dh_econ(self, x, i, hps, gp_obj):
        temp_hps1 = np.array(hps)
        temp_hps1[i] = temp_hps1[i] + 1e-6
        temp_hps2 = np.array(hps)
        temp_hps2[i] = temp_hps2[i] - 1e-6
        a = self.noise_function(x, temp_hps1, self)
        b = self.noise_function(x, temp_hps2, self)
        gr = (a - b) / 2e-6
        return gr
