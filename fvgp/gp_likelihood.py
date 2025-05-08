import numpy as np
import warnings
from loguru import logger

class GPlikelihood:
    def __init__(self,
                 data,
                 hyperparameters=None,
                 noise_function=None,
                 noise_function_grad=None,
                 ram_economy=False,
                 gp2Scale=False,
                 args=None
                 ):

        assert isinstance(hyperparameters, np.ndarray) and np.ndim(hyperparameters) == 1
        self.data = data
        assert self.data.noise_variances is None or isinstance(self.data.noise_variances, np.ndarray)

        if isinstance(self.data.noise_variances, np.ndarray):
            assert np.ndim(self.data.noise_variances) == 1
            assert all(self.data.noise_variances > 0.0)

        self.gp2Scale = gp2Scale
        self.args = args

        if self.data.noise_variances is not None and callable(noise_function):
            raise Exception("Noise function and measurement noise provided. Decide which one to use.")
        if callable(noise_function):
            self.noise_function = noise_function
        elif self.data.noise_variances is not None:
            self.noise_function = self._measured_noise_function
        else:
            warnings.warn(
                "No noise function or measurement noise provided. "
                "Noise variances will be set to (0.01 * mean(|y_data|))^2.",
                stacklevel=2)
            self.noise_function = self._default_noise_function

        if callable(noise_function_grad):
            self.noise_function_grad = noise_function_grad
        elif callable(noise_function):
            if ram_economy is True:
                self.noise_function_grad = self._finitediff_dnoise_dh_econ
            else:
                self.noise_function_grad = self._finitediff_dnoise_dh
        else:
            if ram_economy is True:
                self.noise_function_grad = self._default_dnoise_dh_econ
            else:
                self.noise_function_grad = self._default_dnoise_dh
        self.V = self.noise_function(self.data.x_data, hyperparameters)

    ##################################################################################
    def update(self, hyperparameters):
        logger.debug("Updating noise matrix V after new hyperparameters communicated.")
        self.V = self.calculate_V(hyperparameters)

    #def augment(self, x_old, x_new): #for later to augment V given new data instead of recompute
    #    self.V =

    def calculate_V(self, hyperparameters):
        logger.debug("Calculating V.")
        noise = self.noise_function(self.data.x_data, hyperparameters)
        return noise

    def _default_noise_function(self, x, hyperparameters):
        noise = np.ones((len(x))) * (np.mean(abs(self.data.y_data)) / 100.0)**2
        return noise

    def _measured_noise_function(self, x, hyperparameters):
        if len(x) == len(self.data.noise_variances): return self.data.noise_variances
        else: return np.zeros((len(x))) + 0.0000001

    @staticmethod
    def _default_dnoise_dh(x, hps):
        gr = np.zeros((len(hps), len(x)))
        return gr

    @staticmethod
    def _default_dnoise_dh_econ(x, i, hps):
        gr = np.zeros((len(x)))
        return gr

    ##########################
    def _finitediff_dnoise_dh(self, x, hps):
        #gr = np.zeros((len(hps), len(x)))
        gr = np.array([np.zeros(self.V.shape)] * len(hps))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.noise_function(x, temp_hps1)
            b = self.noise_function(x, temp_hps2)
            gr[i] = (a - b) / 2e-6
        return gr

    ##########################
    def _finitediff_dnoise_dh_econ(self, x, i, hps):
        temp_hps1 = np.array(hps)
        temp_hps1[i] = temp_hps1[i] + 1e-6
        temp_hps2 = np.array(hps)
        temp_hps2[i] = temp_hps2[i] - 1e-6
        a = self.noise_function(x, temp_hps1)
        b = self.noise_function(x, temp_hps2)
        gr = (a - b) / 2e-6
        return gr
