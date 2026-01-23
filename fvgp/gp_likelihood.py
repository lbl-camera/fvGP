import numpy as np
import warnings
from loguru import logger
import inspect


class GPlikelihood:
    def __init__(self,
                 data,
                 trainer,
                 noise_function=None,
                 noise_function_grad=None,
                 ):

        self.data = data
        self.trainer = trainer
        assert self.noise_variances is None or isinstance(self.noise_variances, np.ndarray)

        if isinstance(self.noise_variances, np.ndarray):
            assert np.ndim(self.noise_variances) == 1
            assert np.all(self.noise_variances > 0.0)

        if self.noise_variances is not None and callable(noise_function):
            raise Exception("Noise function and measurement noise provided. Decide which one to use.")

        self.v_n_params = 2
        if callable(noise_function):
            self.noise_function = noise_function
            self.v_n_params = len(inspect.signature(noise_function).parameters)
        elif self.noise_variances is not None:
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
            if self.ram_economy is True:
                self.noise_function_grad = self._finitediff_dnoise_dh_econ
            else:
                self.noise_function_grad = self._finitediff_dnoise_dh
        else:
            if self.ram_economy is True:
                self.noise_function_grad = self._default_dnoise_dh_econ
            else:
                self.noise_function_grad = self._default_dnoise_dh
        self.V = self.noise_function(self.x_data, self.hyperparameters)

    ##################################################################################
    @property
    def args(self):
        return self.data.args

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters

    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def noise_variances(self):
        return self.data.noise_variances

    @property
    def ram_economy(self):
        return self.data.ram_economy

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    ##################################################################################
    #functions available from outside the class
    def update_state(self):
        logger.debug("Updating noise matrix V after new hyperparameters communicated.")
        self.V = self.calculate_V(self.x_data, self.hyperparameters)

    def calculate_V(self, x_data, hyperparameters):
        logger.debug("Calculating V.")
        if self.v_n_params == 2: noise = self.noise_function(x_data, hyperparameters)
        elif self.v_n_params == 3: noise = self.noise_function(x_data, hyperparameters, self.args)
        else: raise Exception("No valid noise function signature.")
        return noise

    def calculate_V_grad(self, x, hyperparameters, direction=None):
        logger.debug("calculating noise gradient")
        if self.ram_economy is True: return self.noise_function_grad(x, hyperparameters, direction)
        else: return self.noise_function_grad(x, hyperparameters)

    ##################################################################################
    def _default_noise_function(self, x, hyperparameters):
        noise = np.ones((len(x))) * (np.mean(abs(self.y_data)) / 100.0)**2
        return noise

    def _measured_noise_function(self, x, hyperparameters):
        if len(x) == len(self.noise_variances):
            return self.noise_variances
        else:
            return np.zeros((len(x))) + np.mean(self.noise_variances)

    @staticmethod
    def _default_dnoise_dh(x, hps):
        gr = np.zeros((len(hps), len(x)))
        return gr

    @staticmethod
    def _default_dnoise_dh_econ(x, hps, i):
        gr = np.zeros((len(x)))
        return gr

    ##########################
    def _finitediff_dnoise_dh(self, x, hps):
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
    def _finitediff_dnoise_dh_econ(self, x, hps, i):
        temp_hps1 = np.array(hps)
        temp_hps1[i] = temp_hps1[i] + 1e-6
        temp_hps2 = np.array(hps)
        temp_hps2[i] = temp_hps2[i] - 1e-6
        a = self.noise_function(x, temp_hps1)
        b = self.noise_function(x, temp_hps2)
        gr = (a - b) / 2e-6
        return gr

    def __getstate__(self):
        state = dict(
            data=self.data,
            trainer=self.trainer,
            V=self.V,
            noise_function=self.noise_function,
            noise_function_grad=self.noise_function_grad,
            v_n_params=self.v_n_params
            )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
