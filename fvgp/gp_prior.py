import numpy as np
from .gp_kernels import *


class GPrior:  # pragma: no cover
    def __init__(self, data,
                 gp_kernel_function=None,
                 gp_kernel_function_grad=None,
                 gp_mean_function=None,
                 gp_mean_function_grad=None,
                 init_hyperparameters=None,
                 ram_economy=False,
                 online=False):

        assert callable(gp_kernel_function) or gp_kernel_function is None
        assert callable(gp_mean_function) or gp_mean_function is None
        assert isinstance(init_hyperparameters, np.ndarray)
        assert np.ndim(init_hyperparameters) == 1
        assert isinstance(cov_comp_mode, str)
        assert isinstance(online, bool)

        self.data = data
        self.gp_kernel_function = gp_kernel_function
        self.gp_mean_function = gp_mean_function
        self.init_hyperparameters = init_hyperparameters
        self.online = online
        self.ram_economy = ram_economy

        if not data.Euclidean and not callable(gp_kernel_function):
            raise Exception(
                "For GPs on non-Euclidean input spaces you need a user-defined kernel and initial hyperparameters.")
        if not data.Euclidean and init_hyperparameters is None:
            raise Exception(
                "You are running fvGP on non-Euclidean inputs. Please provide initial hyperparameters.")
        if compute_device == 'gpu':
            try:
                import torch
            except:
                raise Exception(
                    "You have specified the 'gpu' as your compute device. You need to install pytorch\
                     manually for this to work.")

        if (callable(gp_kernel_function) or callable(gp_mean_function)) and init_hyperparameters is None:
            warnings.warn(
                "You have provided callables for kernel, mean, or noise functions but no initial \n \
                hyperparameters. It is likely they have to be defined for a success initialization",
                stacklevel=2)

        if init_hyperparameters is None: init_hyperparameters = np.ones((data.input_space_dim + 1))
        self.hyperparameters = init_hyperparameters

        # kernel
        if callable(gp_kernel_function):
            self.kernel = gp_kernel_function
        elif gp_kernel_function is None:
            self.kernel = self.default_kernel
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

        self.prior_mean_vector, self.K = self.compute_prior(x_data, hyperparameters)

    def compute_prior(self, x_data, hyperparameters):
        m = self.mean_function(x_data, hyperparameters, self)
        K = self.cov_function(x_data, hyperparameters)
        assert np.ndim(prior_mean_vec) == 1
        assert np.ndim(K) == 2
        return m, K

    def _compute_GPpriorV(self, x_data, y_data, hyperparameters, calc_inv=False):
        # get the prior mean
        prior_mean_vec = self.mean_function(x_data, hyperparameters, self)
        assert np.ndim(prior_mean_vec) == 1
        # get the latest noise
        V = self.noise_function(x_data, hyperparameters, self)
        assert np.ndim(V) == 2
        K = self._compute_K(x_data, x_data, hyperparameters)
        # check if shapes are correct
        if K.shape != V.shape: raise Exception("Noise covariance and prior covariance not of the same shape.")
        # get K + V
        KV = K + V

        # get Kinv/KVinvY, LU, Chol, logdet(KV)
        KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(y_data-prior_mean_vec, KV,
                                                calc_inv=calc_inv)
        return K, KV, KVinvY, KVlogdet, factorization_obj, KVinv, prior_mean_vec, V

    ##################################################################################

    def _update_GPpriorV(self, x_data_old, x_new, y_data, hyperparameters, calc_inv=False):
        # get the prior mean
        prior_mean_vec = np.append(self.prior_mean_vec, self.mean_function(x_new, hyperparameters, self))
        assert np.ndim(prior_mean_vec) == 1
        # get the latest noise
        V = self.noise_function(self.data.x_data, hyperparameters, self) #can be avoided by update
        assert np.ndim(V) == 2
        # get K
        K = self.update_K(x_data_old, x_new, hyperparameters)

        # check if shapes are correct
        if K.shape != V.shape: raise Exception("Noise covariance and prior covariance not of the same shape.")

        # get K + V
        KV = K + V
        # get Kinv/KVinvY, LU, Chol, logdet(KV)

        if self.online_mode is True: KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(
            y_data - prior_mean_vec, k, kk)
        else: KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(y_data-prior_mean_vec, KV,
                                                                             calc_inv=calc_inv)
        return K, KV, KVinvY, KVlogdet, factorization_obj, KVinv, prior_mean_vec, V

    ##################################################################################
    def _compute_gp_linalg(self, vec, KV, calc_inv=False):
        if calc_inv:
            KVinv = self._inv(KV)
            factorization_obj = ("Inv", None)
            KVinvY = KVinv @ vec
            KVlogdet = self._logdet(KV)
        else:
            KVinv = None
            KVinvY, KVlogdet, factorization_obj = self._Chol(KV, vec)
        return KVinvY, KVlogdet, factorization_obj, KVinv

    def _update_gp_linalg(self, vec, k, kk):
        X = self._inv(kk - C @ self.KVinv @ B)
        F = -self.KVinv @ k @ X
        KVinv = np.block([[self.KVinv + self.KVinv @ B @ X @ C @ self.KVinv, F],
                          [F.T,                                              X]])
        factorization_obj = ("Inv", None)
        KVinvY = KVinv @ vec
        KVlogdet = self.KVlogdet + self._logdet(kk - k.T @ self.KVinv @ k)
        return KVinvY, KVlogdet, factorization_obj, KVinv

    def _LU(self, KV, vec):
        st = time.time()
        if self.info: print("LU in progress ...")
        LU = splu(KV.tocsc())
        factorization_obj = ("LU", LU)
        KVinvY = LU.solve(vec)
        upper_diag = abs(LU.U.diagonal())
        KVlogdet = np.sum(np.log(upper_diag))
        if self.info: print("LU compute time: ", time.time() - st, "seconds.")
        return KVinvY, KVlogdet, factorization_obj

    def _Chol(self, KV, vec):
        c, l = cho_factor(KV)
        factorization_obj = ("Chol", c, l)
        KVinvY = cho_solve((c, l), vec)
        upper_diag = abs(c.diagonal())
        KVlogdet = 2.0 * np.sum(np.log(upper_diag))
        return KVinvY, KVlogdet, factorization_obj

    ##################################################################################
    def compute_K(self, x, hyperparameters):
        """computes the covariance matrix from the kernel"""
        K = self.kernel(x, hyperparameters, self)
        return K

    def update_k(self, x_data_old, x_new, hyperparameters):
        k = self._compute_K(x_data_old, x_new, hyperparameters)
        kk = self._compute_K(x_new, x_new, hyperparameters)
        K = np.block([
                         [self.K,          k],
                         [k.T,            kk]
                         ])
        return K
