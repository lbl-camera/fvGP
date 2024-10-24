import numpy as np
from loguru import logger
from .gp_lin_alg import *
import warnings
from scipy.sparse import issparse
import scipy.sparse as sparse


class GPMarginalDensity:
    def __init__(self,
                 data_obj,
                 prior_obj,
                 likelihood_obj,
                 calc_inv=False,
                 gp2Scale=False,
                 gp2Scale_linalg_mode=None,
                 compute_device='cpu'):

        self.data_obj = data_obj
        self.prior_obj = prior_obj
        self.likelihood_obj = likelihood_obj
        self.calc_inv = calc_inv
        self.y_data = data_obj.y_data
        self.gp2Scale = gp2Scale
        self.compute_device = compute_device
        self.gp2Scale_linalg_mode = gp2Scale_linalg_mode
        if self.gp2Scale:
            self.calc_inv = False
            warnings.warn("gp2Scale use forbids calc_inv=True; it has been set to False")
        self.KVlinalg = KVlinalg(compute_device)
        K, V, m = self._get_KVm()
        if self.gp2Scale:
            mode = self._set_gp2Scale_mode(K)
        elif self.calc_inv:
            mode = "Inv"
        else:
            mode = "Chol"

        self.KVinvY = self._set_KVinvY(K, V, m, mode)

    ##################################################################
    def update_data(self, append):
        """Update the marginal PDF when the data has changed in data likelihood or prior objects"""
        self.y_data = self.data_obj.y_data
        K, V, m = self._get_KVm()
        if append:
            self.KVinvY = self._update_KVinvY(K, V, m)
        else:
            self.KVinvY = self._set_KVinvY(K, V, m, self.KVlinalg.mode)

    def update_hyperparameters(self):
        """Update the marginal PDF when if hyperparameters have changed"""
        K, V, m = self._get_KVm()
        self.KVinvY = self._set_KVinvY(K, V, m, self.KVlinalg.mode)

    ##################################################################
    def _update_KVinvY(self, K, V, m):
        """This updates KVinvY after new data was communicated"""
        y_mean = self.y_data - m
        KV = self._addKV(K, V)
        self.KVlinalg.update_KV(KV)
        KVinvY = self.KVlinalg.solve(y_mean, x0=self.KVinvY)
        return KVinvY.reshape(len(y_mean))

    def _set_KVinvY(self, K, V, m, mode):
        """Set or reset KVinvY for new hyperparameters"""
        y_mean = self.data_obj.y_data - m
        #update lin alg obj
        KV = self._addKV(K, V)
        self.KVlinalg.set_KV(KV, mode)
        KVinvY = self.KVlinalg.solve(y_mean)
        return KVinvY.reshape(len(y_mean))

    ##################################################################
    def compute_new_KVinvY(self, KV, m):
        """
        Recompute KVinvY for new hyperparameters (e.g. during training, for instance)
        This is only used by some posterior functions and in the log likelihood functions.
        This does not change the KV obj
        """
        y_mean = self.data_obj.y_data - m
        if self.gp2Scale:
            mode = self._set_gp2Scale_mode(KV)
            if mode == "sparseLU":
                LU_factor = calculate_LU_factor(KV)
                KVinvY = calculate_LU_solve(LU_factor, y_mean)
            elif mode == "Chol":
                if issparse(KV): KV = KV.toarray()
                Chol_factor = calculate_Chol_factor(KV)
                KVinvY = calculate_Chol_solve(Chol_factor, y_mean)
            elif mode == "sparseCG":
                KVinvY = calculate_sparse_conj_grad(KV, y_mean)
            elif mode == "sparseMINRES":
                KVinvY = calculate_sparse_minres(KV, y_mean)
            elif mode == "sparseMINRESpre":
                B = sparse.linalg.spilu(KV, drop_tol=1e-8)
                KVinvY = calculate_sparse_minres(KV, y_mean, M=B.L.T @ B.L)
            elif mode == "sparseCGpre":
                B = sparse.linalg.spilu(KV, drop_tol=1e-8)
                KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=B.L.T @ B.L)
            elif mode == "sparseSolve":
                KVinvY = calculate_sparse_solve(KV, y_mean)
            else:
                raise Exception("No mode in gp2Scale", mode)
        else:
            Chol_factor = calculate_Chol_factor(KV)
            KVinvY = calculate_Chol_solve(Chol_factor, y_mean)
        return KVinvY.reshape(len(y_mean))

    def compute_new_KVlogdet_KVinvY(self, K, V, m):
        """
        Recomputing KVinvY and logdet(KV) for new hyperparameters.
        This is only used by the training (the log likelihood)
        """
        KV = self._addKV(K, V)
        y_mean = self.data_obj.y_data - m
        if self.gp2Scale:
            mode = self._set_gp2Scale_mode(KV)
            if mode == "sparseLU":
                LU_factor = calculate_LU_factor(KV)
                KVinvY = calculate_LU_solve(LU_factor, y_mean)
                KVlogdet = calculate_LU_logdet(LU_factor)
            elif mode == "Chol":
                if issparse(KV): KV = KV.toarray()
                Chol_factor = calculate_Chol_factor(KV)
                KVinvY = calculate_Chol_solve(Chol_factor, y_mean)
                KVlogdet = calculate_Chol_logdet(Chol_factor)
            elif mode == "sparseCG":
                KVinvY = calculate_sparse_conj_grad(KV, y_mean)
                KVlogdet = calculate_random_logdet(KV, self.compute_device)
            elif mode == "sparseMINRES":
                KVinvY = calculate_sparse_minres(KV, y_mean)
                KVlogdet = calculate_random_logdet(KV, self.compute_device)
            elif mode == "sparseMINRESpre":
                B = sparse.linalg.spilu(KV, drop_tol=1e-8)
                KVinvY = calculate_sparse_minres(KV, y_mean, M=B.L.T @ B.L)
                KVlogdet = calculate_random_logdet(KV, self.compute_device)
            elif mode == "sparseCGpre":
                B = sparse.linalg.spilu(KV, drop_tol=1e-8)
                KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=B.L.T @ B.L)
                KVlogdet = calculate_random_logdet(KV, self.compute_device)
            elif mode == "sparseSolve":
                KVinvY = calculate_sparse_solve(KV, y_mean)
                KVlogdet = calculate_random_logdet(KV, self.compute_device)
            else:
                raise Exception("No mode in gp2Scale", mode)
        else:
            Chol_factor = calculate_Chol_factor(KV)
            KVinvY = calculate_Chol_solve(Chol_factor, y_mean).reshape(len(y_mean))
            KVlogdet = calculate_Chol_logdet(Chol_factor)
        return KVinvY, KVlogdet

    def _get_KVm(self):
        K = self.prior_obj.K
        m = self.prior_obj.m
        V = self.likelihood_obj.V
        return K, V, m

    @staticmethod
    def _addKV(K, V):
        assert np.ndim(K) == 2
        assert K.shape[0] == K.shape[1]

        if issparse(K):
            if issparse(V):
                KV = K + V
                return KV
            else:
                assert np.ndim(V) == 1, "K is sparse, but V is a dense matrix"
                assert len(V) == K.shape[0]
                KV = K.copy()
                KV.setdiag(KV.diagonal() + V)
                return KV.tocsr()
        elif isinstance(K, np.ndarray):
            assert isinstance(V, np.ndarray), "K is np.ndarray, V is not"
            assert np.ndim(V) == 1 or np.ndim(V) == 2, "V has strange dimensionality"
            if np.ndim(V) == 2:
                KV = K + V
                return KV
            else:
                KV = K.copy()
                np.fill_diagonal(KV, np.diag(K) + V)
                return KV
        else:
            raise Exception("K+V not possible with the given formats")

    ##################################################################################
    def _set_gp2Scale_mode(self, KV):
        if self.gp2Scale_linalg_mode is not None: return self.gp2Scale_linalg_mode
        Ksparsity = float(KV.nnz) / float(len(self.data_obj.x_data) ** 2)
        if len(self.data_obj.x_data) < 50001 and Ksparsity < 0.0001:
            mode = "sparseLU"
        elif len(self.data_obj.x_data) < 2001 and Ksparsity >= 0.0001:
            mode = "Chol"
        else:
            mode = "sparseMINRES"
        return mode

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
        logger.debug("log marginal likelihood is being evaluated")
        if hyperparameters is None:
            K, V, m = self._get_KVm()
            KVinvY = self.KVinvY
            KVlogdet = self.KVlinalg.logdet()
        else:
            st = time.time()
            K = self.prior_obj.compute_prior_covariance_matrix(self.data_obj.x_data, hyperparameters=hyperparameters)
            logger.debug("   Prior covariance matrix computed after {} seconds.", time.time() - st)
            V = self.likelihood_obj.calculate_V(hyperparameters)
            logger.debug("   V computed after {} seconds.", time.time() - st)
            m = self.prior_obj.compute_mean(self.data_obj.x_data, hyperparameters=hyperparameters)
            logger.debug("   Prior mean computed after {} seconds.", time.time() - st)
            KVinvY, KVlogdet = self.compute_new_KVlogdet_KVinvY(K, V, m)
            logger.debug("   KVinvY and logdet computed after {} seconds.", time.time() - st)

        n = len(self.y_data)
        return -(0.5 * ((self.y_data - m).T @ KVinvY)) - (0.5 * KVlogdet) - (0.5 * n * np.log(2.0 * np.pi))

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

        if hyperparameters is None:
            KVinvY = self.KVinvY
            K = self.prior_obj.K
            V = self.likelihood_obj.V
            KV = self._addKV(K, V)
        else:
            K = self.prior_obj.compute_prior_covariance_matrix(self.data_obj.x_data, hyperparameters=hyperparameters)
            V = self.likelihood_obj.calculate_V(hyperparameters)
            m = self.prior_obj.compute_mean(self.data_obj.x_data, hyperparameters=hyperparameters)
            KV = self._addKV(K, V)
            KVinvY = self.compute_new_KVinvY(KV, m)

        b = KVinvY
        if self.prior_obj.ram_economy is False:
            try:
                noise_der = self.likelihood_obj.noise_function_grad(self.data_obj.x_data, hyperparameters)
                assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 3
                if np.ndim(noise_der) == 2:
                    noise_der_V = np.zeros((len(hyperparameters), len(self.data_obj.x_data), len(self.data_obj.x_data)))
                    for i in range(len(hyperparameters)): np.fill_diagonal(noise_der_V[i], noise_der[i])
                else: noise_der_V = noise_der
                dK_dH = self.prior_obj.dk_dh(self.data_obj.x_data, self.data_obj.x_data, hyperparameters) + noise_der_V
            except Exception as e:
                raise Exception(
                    "The gradient evaluation dK/dh + dNoise/dh was not successful. "
                    "That normally means the combination of ram_economy and definition "
                    "of the gradient function is wrong. ",
                    str(e))
            KV = np.array([KV, ] * len(hyperparameters))
            a = solve(KV, dK_dH, compute_device=self.compute_device)
        bbT = np.outer(b, b.T)
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.prior_obj.dm_dh(self.data_obj.x_data, hyperparameters)

        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if self.prior_obj.ram_economy is False:
                matr = a[i]
            else:
                try:
                    noise_der = self.likelihood_obj.noise_function_grad(self.data_obj.x_data, i, hyperparameters)
                    assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 1
                    if np.ndim(noise_der) == 1:
                        dK_dH = self.prior_obj.dk_dh(
                            self.data_obj.x_data, self.data_obj.x_data, i, hyperparameters) + np.diag(noise_der)
                    else:
                        dK_dH = self.prior_obj.dk_dh(
                            self.data_obj.x_data, self.data_obj.x_data, i, hyperparameters) + noise_der
                except:
                    raise Exception(
                        "The gradient evaluation dK/dh + dNoise/dh was not successful. "
                        "That normally means the combination of ram_economy and definition of "
                        "the gradient function is wrong.")
                matr = solve(KV, dK_dH, compute_device=self.compute_device)
            if dL_dHm[i] == 0.0:
                if self.prior_obj.ram_economy is False:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH[i])
                else:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH)
                dL_dH[i] = - 0.5 * (mtrace - np.trace(matr))
            else:
                dL_dH[i] = 0.0
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
        return grad, analytical


class KVlinalg:
    def __init__(self, compute_device):
        self.mode = None
        self.compute_device = compute_device
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None

    def set_KV(self, KV, mode):
        self.mode = mode
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None
        assert self.mode is not None
        if self.mode == "Chol":
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV)
        elif self.mode == "Inv":
            self.KV = KV
            self.KVinv = calculate_inv(KV, compute_device=self.compute_device)
        elif self.mode == "sparseMINRES":
            self.KV = KV
        elif self.mode == "sparseCG":
            self.KV = KV
        elif self.mode == "sparseLU":
            self.LU_factor = calculate_LU_factor(KV)
        elif self.mode == "sparseMINRESpre":
            self.KV = KV
        elif self.mode == "sparseCGpre":
            self.KV = KV
        elif self.mode == "sparseSolve":
            self.KV = KV
        else:
            raise Exception("No Mode")

    def update_KV(self, KV):
        if self.mode == "Chol":
            if issparse(KV): KV = KV.toarray()
            if len(KV) <= len(self.Chol_factor):
                res = calculate_Chol_factor(KV)
            else:
                res = update_Chol_factor(self.Chol_factor, KV)
            self.Chol_factor = res
        elif self.mode == "Inv":
            self.KV = KV
            if len(KV) <= len(self.KVinv):
                self.KVinv = calculate_inv(KV, compute_device=self.compute_device)
            else:
                self.KVinv = update_inv(self.KVinv, KV, self.compute_device)
        elif self.mode == "sparseMINRES":
            self.KV = KV
        elif self.mode == "sparseCG":
            self.KV = KV
        elif self.mode == "sparseLU":
            self.LU_factor = calculate_LU_factor(KV)
        elif self.mode == "sparseMINRESpre":
            self.KV = KV
        elif self.mode == "sparseCGpre":
            self.KV = KV
        elif self.mode == "sparseSolve":
            self.KV = KV
        else:
            raise Exception("No Mode")

    def solve(self, b, x0=None):
        if self.mode == "Chol":
            return calculate_Chol_solve(self.Chol_factor, b)
        elif self.mode == "Inv":
            return self.KVinv @ b
        elif self.mode == "sparseCG":
            return calculate_sparse_conj_grad(self.KV, b, x0=x0)
        elif self.mode == "sparseMINRES":
            return calculate_sparse_minres(self.KV, b, x0=x0)
        elif self.mode == "sparseLU":
            return calculate_LU_solve(self.LU_factor, b)
        elif self.mode == "sparseMINRESpre":
            B = sparse.linalg.spilu(self.KV, drop_tol=1e-8)
            return calculate_sparse_minres(self.KV, b, M=B.L.T @ B.L, x0=x0)
        elif self.mode == "sparseCGpre":
            B = sparse.linalg.spilu(self.KV, drop_tol=1e-8)
            return calculate_sparse_conj_grad(self.KV, b, M=B.L.T @ B.L, x0=x0)
        elif self.mode == "sparseSolve":
            return calculate_sparse_solve(self.KV, b)
        else:
            raise Exception("No Mode")

    def logdet(self):
        if self.mode == "Chol": return calculate_Chol_logdet(self.Chol_factor)
        elif self.mode == "sparseLU": return calculate_LU_logdet(self.LU_factor)
        elif self.mode == "Inv": return calculate_logdet(self.KV)
        elif self.mode == "sparseCG": return calculate_random_logdet(self.KV, self.compute_device)
        elif self.mode == "sparseMINRES": return calculate_random_logdet(self.KV, self.compute_device)
        elif self.mode == "sparseMINRESpre": return calculate_random_logdet(self.KV, self.compute_device)
        elif self.mode == "sparseCGpre": return calculate_random_logdet(self.KV, self.compute_device)
        elif self.mode == "sparseSolve": return calculate_random_logdet(self.KV, self.compute_device)


