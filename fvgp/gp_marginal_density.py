import numpy as np
from .gp_lin_alg import *
from scipy.sparse import issparse
import scipy.sparse as sparse
from loguru import logger


class GPMarginalDensity:
    def __init__(self,
                 data,
                 prior,
                 likelihood,
                 trainer,
                 gp2Scale_linalg_mode=None):

        self.data = data
        self.prior = prior
        self.likelihood = likelihood
        self.gp2Scale_linalg_mode = gp2Scale_linalg_mode
        self.trainer = trainer

        self.KVlinalg = KVlinalg(self.compute_device, data)
        K, V, m = self._get_KVm()
        if self.gp2Scale:
            mode = self._set_gp2Scale_mode(K)
        elif self.calc_inv: mode = "CholInv"
        else: mode = "Chol"

        self.KVinvY = self._set_state_KVinvY(K, V, m, mode)

    ##################################################################
    @property
    def args(self):
        return self.data.args

    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def K(self):
        return self.prior.K

    @property
    def m(self):
        return self.prior.m

    @property
    def V(self):
        return self.likelihood.V

    @property
    def ram_economy(self):
        return self.data.ram_economy

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    @property
    def compute_device(self):
        return self.data.compute_device

    @property
    def calc_inv(self):
        return self.data.calc_inv

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters

    ##################################################################
    def compute_prior_covariance_matrix(self, x_data, hyperparameters):
        return self.prior.compute_prior_covariance_matrix(x_data, hyperparameters)

    def compute_mean(self, x_data, hyperparameters):
        return self.prior.compute_mean(x_data, hyperparameters)

    def dk_dh(self, x1, x2, hyperparameters, direction=None):
        return self.prior.dk_dh(x1, x2, hyperparameters, direction=direction)

    def dm_dh(self, x, hyperparameters):
        return self.prior.dm_dh(x, hyperparameters)

    def calculate_V(self, x, hyperparameters):
        return self.likelihood.calculate_V(x, hyperparameters)

    def calculate_V_grad(self, x, hyperparameters, direction=None):
        return self.likelihood.calculate_V_grad(x, hyperparameters, direction=direction)

    ##################################################################
    def update_state_data(self, append):
        """Update the marginal PDF when the data has changed in data likelihood or prior objects"""
        logger.debug("Updating marginal density after new data was appended.")
        K, V, m = self._get_KVm()
        if append: self.KVinvY = self._update_state_KVinvY(K, V, m)
        else: self.KVinvY = self._set_state_KVinvY(K, V, m, self.KVlinalg.mode)

    def update_state_hyperparameters(self):
        """Update the marginal PDF when if hyperparameters have changed"""
        logger.debug("Updating marginal density after new hyperparameters were appended.")
        K, V, m = self._get_KVm()
        self.KVinvY = self._set_state_KVinvY(K, V, m, self.KVlinalg.mode)

    def compute_new_KVinvY(self, KV, m):
        """
        Recompute KVinvY for new hyperparameters (e.g. during training, for instance)
        This is only used by some posterior functions and in the gradient of the log likelihood function.
        This does not change the KV obj
        """
        y_mean = self.y_data - m[:, None]
        if self.gp2Scale:   # pragma: no cover
            raise Exception("Can't compute a new KVinvY for gp2Scale")
        else:
            Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            KVinvY = calculate_Chol_solve(Chol_factor, y_mean, compute_device=self.compute_device, args=self.args)
        return KVinvY.reshape(y_mean.shape)
    ##################################################################
    def _update_state_KVinvY(self, K, V, m):
        """This updates KVinvY after new data was communicated"""
        #assert self.y_data.shape == m.shape
        y_mean = self.y_data - m[:, None]
        KV = self.addKV(K, V)
        self.KVlinalg.update_KV(KV)
        KVinvY = self.KVlinalg.solve(y_mean, x0=self.KVinvY).reshape(y_mean.shape)
        return KVinvY

    def _set_state_KVinvY(self, K, V, m, mode):
        """Set or reset KVinvY for new hyperparameters"""
        #assert self.y_data.shape == m.shape
        y_mean = self.y_data - m[:, None]
        KV = self.addKV(K, V)
        logger.debug("K+V computed")
        self.KVlinalg.set_KV(KV, mode)
        logger.debug("KVlinalg obj set")
        logger.debug("Solve in progress")
        KVinvY = self.KVlinalg.solve(y_mean).reshape(y_mean.shape)
        return KVinvY

    ##################################################################
    def _compute_new_KVlogdet_KVinvY(self, K, V, m):
        """
        Recomputing KVinvY and logdet(KV) for new hyperparameters.
        This is only used by the training (the log likelihood)
        """
        KV = self.addKV(K, V)
        #assert self.y_data.shape == m.shape
        y_mean = self.y_data - m[:, None]
        if self.gp2Scale:
            mode = self._set_gp2Scale_mode(KV)
            if mode == "sparseLU":
                LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
                KVinvY = calculate_LU_solve(LU_factor, y_mean, args=self.args)
                KVlogdet = calculate_LU_logdet(LU_factor, args=self.args)
            elif mode == "Chol":
                if issparse(KV): KV = KV.toarray()
                Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
                KVinvY = calculate_Chol_solve(Chol_factor, y_mean, compute_device=self.compute_device, args=self.args)
                KVlogdet = calculate_Chol_logdet(Chol_factor, compute_device=self.compute_device, args=self.args)
            elif mode == "sparseCG":
                KVinvY = calculate_sparse_conj_grad(KV, y_mean, args=self.args)
                KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
            elif mode == "sparseMINRES":
                KVinvY = calculate_sparse_minres(KV, y_mean, args=self.args)
                KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
            elif mode == "sparseMINRESpre":
                B = sparse.linalg.spilu(KV, drop_tol=1e-8)
                KVinvY = calculate_sparse_minres(KV, y_mean, M=B.L.T @ B.L, args=self.args)
                KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
            elif mode == "sparseCGpre":
                B = sparse.linalg.spilu(KV, drop_tol=1e-8)
                KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=B.L.T @ B.L, args=self.args)
                KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
            elif mode == "sparseSolve":
                KVinvY = calculate_sparse_solve(KV, y_mean, args=self.args)
                KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
            elif callable(mode[0]) and callable(mode[1]) and callable(mode[2]):
                factor = mode[0](KV)
                KVinvY = mode[1](factor, y_mean)
                KVlogdet = mode[2](factor)
            else:
                raise Exception("No mode in gp2Scale", mode)
        else:
            Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            KVinvY = calculate_Chol_solve(Chol_factor, y_mean, compute_device=self.compute_device, args=self.args)
            KVlogdet = calculate_Chol_logdet(Chol_factor, compute_device=self.compute_device, args=self.args)
        return KVinvY.reshape(y_mean.shape), KVlogdet

    def _get_KVm(self):
        return self.K, self.V, self.m

    @staticmethod
    def addKV(K, V):
        assert np.ndim(K) == 2
        assert K.shape[0] == K.shape[1]

        if issparse(K):
            if issparse(V):
                KV = K + V
                return KV
            else:
                assert np.ndim(V) == 1, "K is sparse, but V is a dense matrix"
                assert len(V) == K.shape[0]
                logger.debug("Evaluating K+V in gp2Scale")
                KV = K.copy()
                K_diag = K.diagonal()
                KV.setdiag(K_diag + V)
                logger.debug("K+V in gp2Scale Computed")
                return KV
        elif isinstance(K, np.ndarray):
            if issparse(V): V = V.toarray()
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
        Ksparsity = float(KV.nnz) / float(len(self.x_data) ** 2)
        if self.gp2Scale_linalg_mode is not None:
            return self.gp2Scale_linalg_mode
        elif len(self.x_data) < 50001 and Ksparsity < 0.0001:
            mode = "sparseLU"
        elif len(self.x_data) < 2001 and Ksparsity >= 0.0001:
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
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            logger.debug("   Prior covariance matrix computed after {} seconds.", time.time() - st)
            V = self.calculate_V(self.x_data, hyperparameters)
            logger.debug("   V computed after {} seconds.", time.time() - st)
            m = self.compute_mean(self.x_data, hyperparameters)
            logger.debug("   Prior mean computed after {} seconds.", time.time() - st)
            KVinvY, KVlogdet = self._compute_new_KVlogdet_KVinvY(K, V, m)
            logger.debug("   KVinvY and logdet computed after {} seconds.", time.time() - st)

        n = len(self.y_data)
        y_mean = self.y_data - m[:, None]
        assert np.ndim(y_mean) == 2
        assert y_mean.shape == KVinvY.shape, "(y-m).shape != KVinvY.shape in log_likelihood"+str(y_mean.shape)+" ,"+str(KVinvY.shape)
        if np.ndim(y_mean) == 2: l1 = np.sum(y_mean * KVinvY)/y_mean.shape[1]
        else: l1 = np.sum(y_mean * KVinvY)
        L = -0.5 * (l1 + KVlogdet + n * np.log(2.0 * np.pi))
        return L

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
    def neg_log_likelihood_gradient(self, hyperparameters=None, component=0):
        """
        Function that computes the gradient of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.
        component : int, optional
            In case many GPs are computed in parallel, this specifies which one is considered.

        Return
        ------
        Gradient of the negative log marginal likelihood : np.ndarray
        """
        if self.gp2Scale: raise Exception("Can't compute neg_log_likelihood_gradient for gp2Scale")

        if hyperparameters is None:
            KVinvY = self.KVinvY
            K = self.K
            V = self.V
            KV = self.addKV(K, V)
            hyperparameters = self.hyperparameters
        else:
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            V = self.calculate_V(self.x_data, hyperparameters)
            m = self.compute_mean(self.x_data, hyperparameters)
            KV = self.addKV(K, V)
            KVinvY = self.compute_new_KVinvY(KV, m)

        b = KVinvY[:, component]
        #dK_dH = None
        a = None
        if self.ram_economy is False:
            try:
                noise_der = self.calculate_V_grad(self.x_data, hyperparameters)
                assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 3
                if np.ndim(noise_der) == 2:
                    noise_der_V = np.zeros((len(hyperparameters), len(self.x_data), len(self.x_data)))
                    for i in range(len(hyperparameters)): np.fill_diagonal(noise_der_V[i], noise_der[i])
                else: noise_der_V = noise_der
                dK_dH = self.dk_dh(self.x_data, self.x_data, hyperparameters) + noise_der_V
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
        dm_dh = self.dm_dh(self.x_data, hyperparameters)

        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if self.ram_economy is False:
                matr = a[i]
            else:
                try:
                    noise_der = self.calculate_V_grad(self.x_data, hyperparameters, direction=i)
                    assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 1
                    if np.ndim(noise_der) == 1:
                        dK_dH = self.dk_dh(
                            self.x_data, self.x_data, hyperparameters, direction=i) + np.diag(noise_der)
                    else:
                        dK_dH = self.dk_dh(
                            self.x_data, self.x_data, hyperparameters, direction=i) + noise_der
                except Exception as e:
                    raise Exception(
                        "The gradient evaluation dK/dh + dNoise/dh was not successful. "
                        "That normally means the combination of ram_economy and definition of "
                        "the gradient function is wrong.",e)
                matr = solve(KV, dK_dH, compute_device=self.compute_device)
            if dL_dHm[i] == 0.0:
                if self.ram_economy is False:
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

    def test_log_likelihood_gradient(self, hyperparameters, epsilon=1e-6):
        thps = np.array(hyperparameters)
        grad = np.empty((len(thps)))
        eps = epsilon
        for i in range(len(thps)):
            thps_aux = np.array(thps)
            thps_aux[i] = thps_aux[i] + eps
            grad[i] = (self.log_likelihood(hyperparameters=thps_aux) - self.log_likelihood(hyperparameters=thps)) / eps
        analytical = -self.neg_log_likelihood_gradient(hyperparameters=thps)
        return grad, analytical

    def __getstate__(self):
        state = dict(
            data=self.data,
            prior=self.prior,
            likelihood=self.likelihood,
            gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
            trainer=self.trainer,
            KVlinalg=self.KVlinalg,
            KVinvY=self.KVinvY
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class KVlinalg:
    def __init__(self, compute_device, data):
        self.mode = None
        self.data = data
        self.compute_device = compute_device
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None
        self.custom_obj = None
        self.allowed_modes = ["Chol", "CholInv", "Inv", "sparseMINRES", "sparseCG",
                              "sparseLU", "sparseMINRESpre", "sparseCGpre", "sparseSolve", "a set of callables"]

    @property
    def args(self):
        return self.data.args

    def set_KV(self, KV, mode):
        self.mode = mode
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None
        assert self.mode is not None
        if self.mode == "Chol":
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv":
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            self.KVinv = calculate_inv_from_chol(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "Inv":
            self.KV = KV
            self.KVinv = calculate_inv(KV, compute_device=self.compute_device, args=self.args)
        elif self.mode == "sparseMINRES":
            self.KV = KV
        elif self.mode == "sparseCG":
            self.KV = KV
        elif self.mode == "sparseLU":
            self.LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
        elif self.mode == "sparseMINRESpre":
            self.KV = KV
        elif self.mode == "sparseCGpre":
            self.KV = KV
        elif self.mode == "sparseSolve":
            self.KV = KV
        elif callable(self.mode[0]):
            self.custom_obj = self.mode[0](KV)
        else:
            raise Exception("No Mode. Choose from: ", self.allowed_modes)

    def update_KV(self, KV):
        if self.mode == "Chol":
            if issparse(KV): KV = KV.toarray()
            if len(KV) <= len(self.Chol_factor):
                res = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            else:
                res = update_Chol_factor(self.Chol_factor, KV, compute_device="cpu", args=self.args)
            self.Chol_factor = res
        elif self.mode == "CholInv":
            if issparse(KV): KV = KV.toarray()
            if len(KV) <= len(self.Chol_factor):
                res = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            else:
                res = update_Chol_factor(self.Chol_factor, KV, compute_device="cpu", args=self.args)
            self.Chol_factor = res
            self.KVinv = calculate_inv_from_chol(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "Inv":
            self.KV = KV
            if len(KV) <= len(self.KVinv):
                self.KVinv = calculate_inv(KV, compute_device=self.compute_device, args=self.args)
            else:
                self.KVinv = update_inv(self.KVinv, KV, self.compute_device, args=self.args)
        elif self.mode == "sparseMINRES":
            self.KV = KV
        elif self.mode == "sparseCG":
            self.KV = KV
        elif self.mode == "sparseLU":
            self.LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
        elif self.mode == "sparseMINRESpre":
            self.KV = KV
        elif self.mode == "sparseCGpre":
            self.KV = KV
        elif self.mode == "sparseSolve":
            self.KV = KV
        elif callable(self.mode[0]):
            self.custom_obj = self.mode[0](KV)
        else:
            raise Exception("No Mode. Choose from: ", self.allowed_modes)

    def solve(self, b, x0=None):
        if self.mode == "Chol":
            return calculate_Chol_solve(self.Chol_factor, b, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv":
            return calculate_Chol_solve(self.Chol_factor, b, compute_device=self.compute_device, args=self.args)
        elif self.mode == "Inv":
            #return matmul(self.KVinv, b, compute_device=self.compute_device) #is this really faster?
            return self.KVinv @ b
        elif self.mode == "sparseCG":
            return calculate_sparse_conj_grad(self.KV, b, x0=x0, args=self.args)
        elif self.mode == "sparseMINRES":
            return calculate_sparse_minres(self.KV, b, x0=x0, args=self.args)
        elif self.mode == "sparseLU":
            return calculate_LU_solve(self.LU_factor, b, args=self.args)
        elif self.mode == "sparseMINRESpre":
            B = sparse.linalg.spilu(self.KV, drop_tol=1e-8)
            return calculate_sparse_minres(self.KV, b, M=B.L.T @ B.L, x0=x0, args=self.args)
        elif self.mode == "sparseCGpre":
            B = sparse.linalg.spilu(self.KV, drop_tol=1e-8)
            return calculate_sparse_conj_grad(self.KV, b, M=B.L.T @ B.L, x0=x0, args=self.args)
        elif self.mode == "sparseSolve":
            return calculate_sparse_solve(self.KV, b, args=self.args)
        elif callable(self.mode[1]):
            return self.mode[1](self.custom_obj, b)
        else:
            raise Exception("No Mode. Choose from: ", self.allowed_modes)

    def logdet(self):
        if self.mode == "Chol": return calculate_Chol_logdet(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv": return calculate_Chol_logdet(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "sparseLU": return calculate_LU_logdet(self.LU_factor, args=self.args)
        elif self.mode == "Inv": return calculate_logdet(self.KV, args=self.args)
        elif self.mode == "sparseCG": return calculate_random_logdet(self.KV, self.compute_device, args=self.args)
        elif self.mode == "sparseMINRES": return calculate_random_logdet(self.KV, self.compute_device, args=self.args)
        elif self.mode == "sparseMINRESpre": return calculate_random_logdet(self.KV, self.compute_device, args=self.args)
        elif self.mode == "sparseCGpre": return calculate_random_logdet(self.KV, self.compute_device, args=self.args)
        elif self.mode == "sparseSolve": return calculate_random_logdet(self.KV, self.compute_device, args=self.args)
        elif callable(self.mode[2]): return self.mode[2](self.custom_obj)
        else: raise Exception("No Mode. Choose from: ", self.allowed_modes)

    def __getstate__(self):
        state = dict(
            mode=self.mode,
            data=self.data,
            compute_device=self.compute_device,
            KVinv=self.KVinv,
            KV=self.KV,
            Chol_factor=self.Chol_factor,
            LU_factor=self.LU_factor,
            custom_obj=self.custom_obj,
            allowed_modes=self.allowed_modes
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


