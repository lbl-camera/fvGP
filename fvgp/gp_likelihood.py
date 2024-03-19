import numpy


class GPlikelihood:  # pragma: no cover
    def __init__(self, K, noise_variances, gp_noise_function):
        assert isinstance(K, np.ndarray) and np.ndim(K) == 2


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

    def _default_noise_function(self, x, hyperparameters, gp_obj):
        noise = np.ones((len(x))) * (np.mean(abs(self.y_data)) / 100.0)
        if self.gp2Scale:
            return self.gp2Scale_obj.calculate_sparse_noise_covariance(noise)
        else:
            return np.diag(noise)
