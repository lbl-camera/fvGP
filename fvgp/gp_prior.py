import numpy as np
from .gp_kernels import *


class GPrior: # pragma: no cover
    def __init__(self, x_data,
                 gp_kernel_function=None,
                 gp_kernel_function_grad=None,
                 gp_mean_function=None,
                 gp_mean_function_grad=None,
                 init_hyperparameters=None,
                 gp2Scale_dask_client=None,
                 cov_comp_mode='default', #default, gp2SCale, ...
                 online=False):
        assert callable(gp_kernel_function) or gp_kernel_function is None
        assert callable(gp_mean_function) or gp_mean_function is None
        assert isinstance(init_hyperparameters, np.ndarray)
        assert np.ndim(init_hyperparameters) == 1
        assert isinstance(cov_comp_mode, str)
        assert isinstance(online, bool)

        self.gp_kernel_function = gp_kernel_function
        self.gp_mean_function = gp_mean_function
        self.init_hyperparameters = init_hyperparameters
        self.cov_comp_mode = comp_mode
        self.online = online

        if cov_comp_mode == 'gp2Scale':
            try:
                import imate
            except:
                raise Exception(
                    "You have activated `gp2Scale`. You need to install imate manually for this to work.")
            if gp2Scale_dask_client is None:
                gp2Scale_dask_client = Client()
                warnings.warn("gp2Scale needs a 'gp2Scale_dask_client'. \
                Set to distributed.Client().", stacklevel=2)
            self.gp2Scale_dask_client = gp2Scale_dask_client

            if not callable(gp_kernel_function):
                warnings.warn("You have chosen to activate gp2Scale. A powerful tool! \n \
                        But you have not supplied a kernel that is compactly supported. \n \
                        I will use an anisotropic Wendland kernel for now.",
                              stacklevel=2)
                if compute_device == "cpu":
                    gp_kernel_function = wendland_anisotropic_gp2Scale_cpu
                elif compute_device == "gpu":
                    gp_kernel_function = wendland_anisotropic_gp2Scale_gpu

            self.gp2Scale_obj = gp2S(batch_size=gp2Scale_batch_size,
                                     gp_kernel_function=gp_kernel_function,
                                     covariance_dask_client=gp2Scale_dask_client,
                                     info=info)
            self.cov_function = self.gp2Scale_obj.compute_covariance
            self.store_inv = False
            warnings.warn("WARNING: gp2Scale activated. Only training via MCMC will be performed. \
                    Only noise variances (no noise covariances can be considered). \
                    A customized sparse kernel should be used, otherwise an anisotropic Wendland kernel is used.",
                          stacklevel=2)
        elif cov_comp_mode == "default":
            self.cov_function = self.compute_K

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
        if callable(self.noise_function): V = self.noise_function(x_data, hyperparameters, self)
        else: V = self.V
        assert np.ndim(V) == 2

        # get K
        try_sparse_LU = False
        if self.gp2Scale:
            st = time.time()
            K = self.gp2Scale_obj.compute_covariance(x_data, hyperparameters, self.gp2Scale_dask_client)
            Ksparsity = float(K.nnz) / float(len(x_data) ** 2)
            if isinstance(V, np.ndarray): raise Exception("You are running gp2Scale. \
            Your noise model has to return a `scipy.sparse.coo_matrix`.")
            if len(x_data) < 50000 and Ksparsity < 0.0001: try_sparse_LU = True
            if self.info: print("Computing and transferring the covariance matrix took ", time.time() - st,
                                " seconds | sparsity = ", Ksparsity, flush=True)
        else:
            K = self._compute_K(x_data, x_data, hyperparameters)

        # check if shapes are correct
        if K.shape != V.shape: raise Exception("Noise covariance and prior covariance not of the same shape.")

        # get K + V
        KV = K + V

        # get Kinv/KVinvY, LU, Chol, logdet(KV)
        KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(y_data-prior_mean_vec, KV,
                                                calc_inv=calc_inv, try_sparse_LU=try_sparse_LU)

        return K, KV, KVinvY, KVlogdet, factorization_obj, KVinv, prior_mean_vec, V

    ##################################################################################

    def _update_GPpriorV(self, x_data_old, x_new, y_data, hyperparameters, calc_inv=False):
        #where is the new variance?
        #do I need the x_data here? Or can it be self.x_data

        # get the prior mean
        prior_mean_vec = np.append(self.prior_mean_vec, self.mean_function(x_new, hyperparameters, self))
        assert np.ndim(prior_mean_vec) == 1
        # get the latest noise
        if callable(self.noise_function): V = self.noise_function(self.x_data, hyperparameters, self)
        else: V = self.V
        assert np.ndim(V) == 2
        # get K
        try_sparse_LU = False
        if self.gp2Scale:
            st = time.time()
            K = self.gp2Scale_obj.update_covariance(x_new, hyperparameters, self.gp2Scale_dask_client, self.K)
            Ksparsity = float(K.nnz) / float(len(self.x_data) ** 2)
            if len(self.x_data) < 50000 and Ksparsity < 0.0001: try_sparse_LU = True
            if self.info: print("Computing and transferring the covariance matrix took ", time.time() - st,
                                " seconds | sparsity = ", Ksparsity, flush=True)
        else:
            off_diag = self._compute_K(x_data_old, x_new, hyperparameters)
            K = np.block([
                         [self.K,          off_diag],
                         [off_diag.T,      self._compute_K(x_new, x_new, hyperparameters)]
                         ])
        # check if shapes are correct
        if K.shape != V.shape: raise Exception("Noise covariance and prior covariance not of the same shape.")

        # get K + V
        KV = K + V
        #y_data = np.append(y_data, y_new)
        # get Kinv/KVinvY, LU, Chol, logdet(KV)

        KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(y_data-prior_mean_vec, KV,
                                                     calc_inv=calc_inv, try_sparse_LU=try_sparse_LU)

        return K, KV, KVinvY, KVlogdet, factorization_obj, KVinv, prior_mean_vec, V

    ##################################################################################
    def _compute_gp_linalg(self, vec, KV, calc_inv=False, try_sparse_LU=False):
        if self.gp2Scale:
            st = time.time()
            from imate import logdet
            # try fast but RAM intensive SuperLU first
            if try_sparse_LU:
                try:
                    LU = splu(KV.tocsc())
                    factorization_obj = ("LU", LU)
                    KVinvY = LU.solve(vec)
                    upper_diag = abs(LU.U.diagonal())
                    KVlogdet = np.sum(np.log(upper_diag))
                    if self.info: print("LU compute time: ", time.time() - st, "seconds.")
                # if that did not work, do random lin algebra magic
                except:
                    KVinvY, exit_code = minres(KV.tocsc(), vec)
                    factorization_obj = ("gp2Scale", None)
                    if self.compute_device == "gpu": gpu = True
                    else: gpu = False
                    if self.info: print("logdet() in progress ... ", time.time() - st, "seconds.")
                    KVlogdet, info_slq = logdet(KV, method='slq', min_num_samples=10, max_num_samples=100,
                                                lanczos_degree=20, error_rtol=0.1, gpu=gpu,
                                                return_info=True, plot=False, verbose=False)
                    if self.info: print("logdet/LU compute time: ", time.time() - st, "seconds.")
            # if the problem is large go with rand. lin. algebra straight away
            else:
                if self.info: print("MINRES solve in progress ...", time.time() - st, "seconds.")
                factorization_obj = ("gp2Scale", None)
                KVinvY, exit_code = minres(KV.tocsc(), vec)
                if self.info: print("MINRES solve compute time: ", time.time() - st, "seconds.")
                if self.compute_device == "gpu": gpu = True
                else: gpu = False
                if self.info: print("logdet() in progress ... ", time.time() - st, "seconds.")
                KVlogdet, info_slq = logdet(KV, method='slq', min_num_samples=10, max_num_samples=100,
                                            lanczos_degree=20, error_rtol=0.1, orthogonalize=0, gpu=gpu,
                                            return_info=True, plot=False, verbose=False)
                if self.info: print("logdet/LU compute time: ", time.time() - st, "seconds.")
            KVinv = None
        else:
            if calc_inv:
                KVinv = self._inv(KV)
                factorization_obj = ("Inv", None)
                KVinvY = KVinv @ vec
                KVlogdet = self._logdet(KV)
            else:
                KVinv = None
                c, l = cho_factor(KV)
                factorization_obj = ("Chol", c, l)
                KVinvY = cho_solve((c, l), vec)
                upper_diag = abs(c.diagonal())
                KVlogdet = 2.0 * np.sum(np.log(upper_diag))
        return KVinvY, KVlogdet, factorization_obj, KVinv

    ##################################################################################
    def compute_K(self, x, hyperparameters):
        """computes the covariance matrix from the kernel"""
        K = self.kernel(x, hyperparameters, self)
        return K









