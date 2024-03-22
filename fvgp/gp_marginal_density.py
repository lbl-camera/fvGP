import numpy as np


class GPMarginalDensity:
    def __init__(self,
                 data_obj,
                 prior_obj,
                 likelihood_obj,
                 store_inv=False):
        self.data_obj = data_obj
        self.prior_obj = prior_obj
        self.likelihood_obj = likelihood_obj
        self.store_inv = store_inv
        self.K = prior_obj.K
        self.k = None
        self.kk = None
        self.V = likelihood_obj.V
        self.y_data = data_obj.y_data
        self.y_mean = data_obj.y_data - prior_obj.prior_mean_vector

        self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv = self._compute_GPpriorV()

    def update(self):
        self.K = prior_obj.K
        self.k = prior_obj.k
        self.kk = prior_obj.kk
        self.V = likelihood_obj.V
        self.y_data = data_obj.y_data
        self.y_mean = data_obj.y_data - prior_obj.prior_mean_vector
        self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv = self._update_GPpriorV()

    def _compute_GPpriorV(self):

        # get K + V
        K = self.prior_obj.K
        V = self.likelihood_obj.V
        # check if shapes are correct
        assert K.shape == V.shape
        KV = K + V
        # get Kinv/KVinvY, LU, Chol, logdet(KV)
        KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(self.y_mean, KV,
                                                                             calc_inv=calc_inv)
        return KV, KVinvY, KVlogdet, factorization_obj, KVinv

    ##################################################################################

    def _update_GPpriorV(self):
        # get K
        K = self.prior_obj.K
        k = self.prior_obj.k #this should be kk + the right part of V
        kk = self.prior_obj.kk #this should be kk + the right part of V
        V = self.likelihood_obj.V
        # check if shapes are correct
        assert K.shape == V.shape

        # get K + V
        KV = K + V
        # get Kinv/KVinvY, LU, Chol, logdet(KV)

        if self.online_mode is True:
            KVinvY, KVlogdet, factorization_obj, KVinv = self._update_gp_linalg(
                self.y_mean, k, kk)
        else:
            KVinvY, KVlogdet, factorization_obj, KVinv = self._compute_gp_linalg(self.y_mean, KV)
        return KV, KVinvY, KVlogdet, factorization_obj, KVinv

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
                          [F.T, X]])
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
        if self.info: print("Dense Cholesky in progress ...")
        c, l = cho_factor(KV)
        factorization_obj = ("Chol", c, l)
        KVinvY = cho_solve((c, l), vec)
        upper_diag = abs(c.diagonal())
        KVlogdet = 2.0 * np.sum(np.log(upper_diag))
        if self.info: print("Dense Cholesky compute time: ", time.time() - st, "seconds.")
        return KVinvY, KVlogdet, factorization_obj

    ##################################################################################

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

    def test_log_likelihood_gradient(self, hyperparameters):
        thps = np.array(hyperparameters)
        grad = np.empty((len(thps)))
        eps = 1e-6
        for i in range(len(thps)):
            thps_aux = np.array(thps)
            thps_aux[i] = thps_aux[i] + eps
            grad[i] = (self.log_likelihood(hyperparameters=thps_aux) - self.log_likelihood(hyperparameters=thps)) / eps
        analytical = -self.neg_log_likelihood_gradient(hyperparameters=thps)
        if np.linalg.norm(grad - analytical) > np.linalg.norm(grad) / 100.0:
            print("Gradient possibly wrong")
            print("finite diff appr: ", grad)
            print("analytical      : ", analytical)
        else:
            print("Gradient correct")
            print("finite diff appr: ", grad)
            print("analytical      : ", analytical)
        assert np.linalg.norm(grad - analytical) < np.linalg.norm(grad) / 100.0

        return grad, analytical
