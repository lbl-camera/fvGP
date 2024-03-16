


class GPriorV: # pragma: no cover
    def __init__(self,
                 gp_kernel_function=None,
                 gp_mean_function=None,
                 noise_variances=None,
                 V=None,
                 init_hyperparameters=None,
                 comp_mode='default',
                 online=False):
        assert callable(gp_kernel_function) or gp_kernel_function is None
        assert callable(gp_mean_function)
        assert callable(V) or isinstance(V, np.ndarray)
        assert isinstance(init_hyperparameters, np.ndarray)
        assert isinstance(comp_mode, str)
        assert isinstance(online, bool)

        self.gp_kernel_function = gp_kernel_function
        self.gp_mean_function = gp_mean_function
        self.V = V
        self.init_hyperparameters = init_hyperparameters
        self.comp_mode = comp_mode
        self.online = online

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
    def _compute_K(self, x1, x2, hyperparameters):
        """computes the covariance matrix from the kernel"""
        K = self.kernel(x1, x2, hyperparameters, self)
        return K

    ##################################################################################
    def _KVsolve(self, b):
        if self.factorization_obj[0] == "LU":
            LU = self.factorization_obj[1]
            return LU.solve(b)
        elif self.factorization_obj[0] == "Chol":
            c, l = self.factorization_obj[1], self.factorization_obj[2]
            return cho_solve((c, l), b)
        else:
            res = np.empty((len(b), b.shape[1]))
            if b.shape[1] > 100: warnings.warn(
                "You want to predict at >100 points. \n When using gp2Scale, this takes a while. \n \
                Better predict at only a handful of points.")
            for i in range(b.shape[1]):
                res[:, i], exit_status = minres(self.KV, b[:, i])
                # if exit_status != 0: res[:,i], exit_status = minres(self.KV,b[:,i])
            return res

    def log_likelihood(self, hyperparameters=None):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Return
        ------
        log marginal likelihood of the data : float
        """
        if hyperparameters is None:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                (self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj,
                 self.KVinv, self.prior_mean_vec, self.V)
        else:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv=False)
        n = len(self.y_data)
        return -(0.5 * ((self.y_data - mean).T @ KVinvY)) - (0.5 * KVlogdet) - (0.5 * n * np.log(2.0 * np.pi))

    ##################################################################################
    def neg_log_likelihood(self, hyperparameters=None):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N)
            If not provided, the covariance will not be recomputed.

        Return
        ------
        negative log marginal likelihood of the data : float
        """
        return -self.log_likelihood(hyperparameters=hyperparameters)

    ##################################################################################
    def neg_log_likelihood_gradient(self, hyperparameters=None):
        """
        Function that computes the gradient of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Return
        ------
        Gradient of the negative log marginal likelihood : np.ndarray
        """
        logger.debug("log-likelihood gradient is being evaluated...")
        if hyperparameters is None:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, \
                    self.KVinv, self.prior_mean_vec, self.V
        else:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv=False)

        b = KVinvY
        y = self.y_data - mean
        if self.ram_economy is False:
            try:
                dK_dH = self.dk_dh(self.x_data, self.x_data, hyperparameters, self) + self.noise_function_grad(
                    self.x_data, hyperparameters, self)
            except Exception as e:
                raise Exception(
                    "The gradient evaluation dK/dh + dNoise/dh was not successful. \n \
                    That normally means the combination of ram_economy and definition \
                    of the gradient function is wrong. ",
                    str(e))
            KV = np.array([KV, ] * len(hyperparameters))
            a = self._solve(KV, dK_dH)
        bbT = np.outer(b, b.T)
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.dm_dh(self.x_data, hyperparameters, self)

        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if self.ram_economy is False:
                matr = a[i]
            else:
                try:
                    dK_dH = self.dk_dh(self.x_data, self.x_data, i, hyperparameters, self) + self.noise_function_grad(
                        self.x_data, i, hyperparameters, self)
                except:
                    raise Exception(
                        "The gradient evaluation dK/dh + dNoise/dh was not successful. \n \
                        That normally means the combination of ram_economy and definition of \
                        the gradient function is wrong.")
                matr = np.linalg.solve(KV, dK_dH)
            if dL_dHm[i] == 0.0:
                if self.ram_economy is False:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH[i])
                else:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH)
                dL_dH[i] = - 0.5 * (mtrace - np.trace(matr))
            else:
                dL_dH[i] = 0.0
        logger.debug("gradient norm: {}", np.linalg.norm(dL_dH + dL_dHm))
        return dL_dH + dL_dHm

    ##################################################################################
    def neg_log_likelihood_hessian(self, hyperparameters=None):
        """
        Function that computes the Hessian of the marginal log-likelihood.
        It does so by a first-order approximation of the exact gradient.

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Return
        ------
        Hessian of the negative log marginal likelihood : np.ndarray
        """
        ##implemented as first-order approximation
        len_hyperparameters = len(hyperparameters)
        d2L_dmdh = np.zeros((len_hyperparameters, len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.neg_log_likelihood_gradient(hyperparameters=hyperparameters)
        for i in range(len_hyperparameters):
            hps_temp = np.array(hyperparameters)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i, i:] = ((self.neg_log_likelihood_gradient(hyperparameters=hps_temp) - grad_at_hps) / epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))






