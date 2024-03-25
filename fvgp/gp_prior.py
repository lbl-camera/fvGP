import numpy as np
from .gp_kernels import *


class GPrior:  # pragma: no cover
    def __init__(self,
                 input_space_dim,
                 x_data,
                 Euclidean,
                 gp_kernel_function=None,
                 gp_kernel_function_grad=None,
                 gp_mean_function=None,
                 gp_mean_function_grad=None,
                 hyperparameters=None,
                 ram_economy=False,
                 cov_comp_mode="trad",
                 compute_device='cpu',
                 constant_mean=0.0
                 ):

        assert callable(gp_kernel_function) or gp_kernel_function is None
        assert callable(gp_mean_function) or gp_mean_function is None
        assert isinstance(hyperparameters, np.ndarray) or hyperparameters is None
        if isinstance(hyperparameters, np.ndarray): assert np.ndim(hyperparameters) == 1
        assert isinstance(cov_comp_mode, str)
        assert isinstance(constant_mean, float)

        self.input_space_dim = input_space_dim
        self.Euclidean = Euclidean
        self.gp_kernel_function = gp_kernel_function
        self.gp_mean_function = gp_mean_function
        self.hyperparameters = hyperparameters
        self.ram_economy = ram_economy
        self.constant_mean = constant_mean

        if not self.Euclidean and not callable(gp_kernel_function):
            raise Exception(
                "For GPs on non-Euclidean input spaces you need a user-defined kernel and initial hyperparameters.")
        if not self.Euclidean and self.hyperparameters is None:
            raise Exception(
                "You are running fvGP on non-Euclidean inputs. Please provide initial hyperparameters.")
        if compute_device == 'gpu':
            try:
                import torch
            except:
                raise Exception(
                    "You have specified the 'gpu' as your compute device. You need to install pytorch\
                     manually for this to work.")

        if (callable(gp_kernel_function) or callable(gp_mean_function)) and self.hyperparameters is None:
            warnings.warn(
                "You have provided callables for kernel, mean, or noise functions but no initial \n \
                hyperparameters. It is likely they have to be defined for a success initialization",
                stacklevel=2)

        # kernel
        if callable(gp_kernel_function):
            self.kernel = gp_kernel_function
        elif gp_kernel_function is None:
            self.kernel = self._default_kernel
        else:
            raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self._d_gp_kernel_dx
        if callable(gp_kernel_function_grad):
            self.dk_dh = gp_kernel_function_grad
        else:
            if self.ram_economy is True:
                self.dk_dh = self._gp_kernel_derivative
            else:
                self.dk_dh = self._gp_kernel_gradient

        # prior mean
        if callable(gp_mean_function):
            self.mean_function = gp_mean_function
        else:
            self.mean_function = self._default_mean_function

        if callable(gp_mean_function_grad):
            self.dm_dh = gp_mean_function_grad
        elif callable(gp_mean_function):
            self.dm_dh = self._finitediff_dm_dh
        else:
            self.dm_dh = self._default_dm_dh

        self.prior_mean_vector, self.K = self._compute_prior(x_data)

    def augment_data(self, x_data, x_new):
        self.prior_mean_vector, self.K = self._update_prior(x_data, x_new)

    def update_data(self, x_data, constant_mean=0.0):
        self.constant_mean = constant_mean
        self.prior_mean_vector, self.K = self._compute_prior(x_data)

    def update_hyperparameters(self, x_data, hyperparameters):
        self.hyperparameters = hyperparameters
        self.prior_mean_vector, self.K = self._compute_prior(x_data)

    def _compute_prior(self, x_data):
        m = self.compute_mean(x_data)
        K = self.compute_K(x_data, x_data)
        assert np.ndim(m) == 1
        assert np.ndim(K) == 2
        return m, K

    def _update_prior(self, x_data, x_new):
        m = self.update_mean(x_new)
        K = self.update_K(x_data, x_new)
        assert np.ndim(prior_mean_vec) == 1
        assert np.ndim(K) == 2
        return m, K

    def _compute_K(self, x1, x2):
        """computes the covariance matrix from the kernel"""
        # if gp2Scale:
        # else:
        K = self.kernel(x1, x2, self.hyperparameters, self)
        return K

    def _update_K(self, x_data, x_new):
        # if gp2Scale: ...
        # else:
        k = self._compute_K(x_data, x_new)
        kk = self._compute_K(x_new, x_new)
        K = np.block([
            [self.K, k],
            [k.T, kk]
        ])
        return K

    def _compute_mean(self, x_data):
        """computes the covariance matrix from the kernel"""
        # if gp2Scale:
        # else:
        m = self.mean_function(x_data,self. hyperparameters, self)
        return m

    def _update_mean(self, x_new):
        # if gp2Scale: ...
        # else:
        m = np.append(self.prior_mean_vec, self.mean_function(x_new, self.hyperparameters, self))
        return m

    ####################################################
    ####################################################
    ####################################################
    ####################################################

    def _default_kernel(self, x1, x2, hyperparameters, obj):
        """
        Function for the default kernel, a Matern kernel of first-order differentiability.

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D).
        x2 : np.ndarray
            Numpy array of shape (V x D).
        hyperparameters : np.ndarray
            Array of hyperparameters. For this kernel we need D + 1 hyperparameters.
        obj : object instance
            GP object instance.

        Return
        ------
        Covariance matrix : np.ndarray
        """
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])):
            distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        distance_matrix = np.sqrt(distance_matrix)
        return hps[0] * matern_kernel_diff1(distance_matrix, 1)

    def _d_gp_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-8
        new_points[:, direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters, self)
        b = self.kernel(points1, points2, hyperparameters, self)
        derivative = (a - b) / epsilon
        return derivative

    def _gp_kernel_gradient(self, points1, points2, hyperparameters, obj):
        gradient = np.empty((len(hyperparameters), len(points1), len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def _gp_kernel_derivative(self, points1, points2, direction, hyperparameters, obj):
        # gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        derivative = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return derivative

    def _dkernel_dh(self, points1, points2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-8
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyperparameters1, self)
        b = self.kernel(points1, points2, new_hyperparameters2, self)
        derivative = (a - b) / (2.0 * epsilon)
        return derivative

    def _default_mean_function(self, x, hyperparameters, gp_obj):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)))
        mean[:] = self.constant_mean
        return mean

    def _finitediff_dm_dh(self, x, hps, gp_obj):
        gr = np.empty((len(hps), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.mean_function(x, temp_hps1, self)
            b = self.mean_function(x, temp_hps2, self)
            gr[i] = (a - b) / 2e-6
        return gr

    def _default_dm_dh(self, x, hps, gp_obj):
        gr = np.zeros((len(hps), len(x)))
        return gr