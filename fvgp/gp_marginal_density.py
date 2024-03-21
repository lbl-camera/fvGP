import numpy as np


class GPMarginalDensity:
    def __init__(self, data_obj, prior_obj, likelihood_obj):
        self.data_obj = data_obj
        self.prior_obj = prior_obj
        self.likelihood_obj = likelihood_obj

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
        if self.info: print("Dense Cholesky in progress ...")
        c, l = cho_factor(KV)
        factorization_obj = ("Chol", c, l)
        KVinvY = cho_solve((c, l), vec)
        upper_diag = abs(c.diagonal())
        KVlogdet = 2.0 * np.sum(np.log(upper_diag))
        if self.info: print("Dense Cholesky compute time: ", time.time() - st, "seconds.")
        return KVinvY, KVlogdet, factorization_obj

    ##################################################################################


