import numpy as np
from .gp_lin_alg import *
from scipy.sparse import issparse
import scipy.sparse as sparse
from loguru import logger


class GPkv:
    def __init__(self, 
                 data, 
                 prior, 
                 likelihood, 
                 linalg_mode=None):
        """
        Initialize the GPkv object. This class keeps track of the current state of the K+V matrix 
        and its inverse, as well as the current mode of linear algebra being used. It also provides 
        methods for updating the state when new data is added or when hyperparameters are changed. 
        The KVlinalg object is used to compute and update the K+V matrix and its inverse efficiently, depending on the chosen mode.
        """

        self.data = data
        self.prior = prior
        self.likelihood = likelihood
        self.linalg_mode = linalg_mode ###there should only be one mode
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None
        self.custom_obj = None
        self.cached_solve = None
        self.cached_precond = None
        self.allowed_modes = ["Chol", "CholInv", "Inv", "sparseMINRES", "sparseCG",
                              "sparseLU", "sparseMINRESpre", "sparseCGpre", "sparseSolve", "a set of callables"]
        K, V, m = self._get_KVm()

        if self.gp2Scale: self.mode = self._set_gp2Scale_mode(K)
        elif linalg_mode is not None: self.mode = linalg_mode
        else: self.mode = "Chol"
        self._refresh(rank_n_update=False)

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
    def compute_device(self):
        return self.data.compute_device
    
    @property
    def gp2Scale(self):
        return self.data.gp2Scale
    
    ##################################################################
    def _set_gp2Scale_mode(self, KV):
        Ksparsity = float(KV.nnz) / float(len(self.x_data) ** 2)
        if self.linalg_mode is not None: mode = self.linalg_mode
        elif len(self.x_data) < 50001 and Ksparsity < 0.0001: mode = "sparseLU"
        elif len(self.x_data) < 2001 and Ksparsity >= 0.0001: mode = "Chol"
        else: mode = "sparseMINRES"
        return mode
    
    ##################################################################
    #####################UPDATE THE OBJ STATE#########################
    ##################################################################
    def update_state_hyperparameters(self):
        """Hyperparameters changed: full KV recompute, then KVinvY."""
        logger.debug("Updating marginal density after hyperparameters were updated.")
        self._refresh(rank_n_update=False)

    def update_state_data(self, append):
        """Data changed: rank-n KV update if appending, full recompute otherwise, then KVinvY."""
        logger.debug("Updating marginal density after new data was %s.",
                     "appended" if append else "overwritten")
        self._refresh(rank_n_update=append)

    def _refresh(self, rank_n_update):
        """Refresh both the KV factorization (Chol_factor / KVinv / LU_factor / ...) and KVinvY.

        rank_n_update=True   reuse the current factorization (rank-n update via update_KV)
                             and warm-start the solve from the previous KVinvY.  Used after
                             appending data.
        rank_n_update=False  full recompute via set_KV with no warm-start.  Used after
                             hyperparameter changes or data overwrite.
        """
        K, V, m = self._get_KVm()
        KV = self.addKV(K, V)
        logger.debug("K+V computed")
        if rank_n_update: self.update_KV(KV)
        else: self.set_KV(KV)
        logger.debug("KV factorization set")
        logger.debug("Solve in progress")
        y_mean = self.y_data - m[:, None]
        x0 = self.KVinvY if rank_n_update else None
        self.KVinvY = self.solve(y_mean, x0=x0).reshape(y_mean.shape)

    def set_KV(self, KV):
        if self.mode == "Chol":
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv":
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            self.KVinv = calculate_inv_from_chol(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "Inv":
            if issparse(KV): KV = KV.toarray()
            self.KV = KV
            self.KVinv = calculate_inv(KV, compute_device=self.compute_device, args=self.args)
        elif self.mode == "sparseMINRES":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseCG":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseLU":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
        elif self.mode == "sparseMINRESpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif callable(self.mode[0]):
            self.custom_obj = self.mode[0](KV)
        else:
            raise Exception(f"No Mode. Choose from: {self.allowed_modes}")

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
            if issparse(KV): KV = KV.toarray()
            self.KV = KV
            if len(KV) <= len(self.KVinv):
                self.KVinv = calculate_inv(KV, compute_device=self.compute_device, args=self.args)
            else:
                self.KVinv = update_inv(self.KVinv, KV, self.compute_device, args=self.args)
        elif self.mode == "sparseMINRES":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseCG":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseLU":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
        elif self.mode == "sparseMINRESpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif self.mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif callable(self.mode[0]):
            self.custom_obj = self.mode[0](KV)
        else:
            raise Exception(f"No Mode. Choose from: {self.allowed_modes}")

    def compute_new_KVinvY(self, KV, m):
        """Recompute KVinvY for a given KV and m without updating state (used during training)."""
        y_mean = self.y_data - m[:, None]
        if self.gp2Scale:   # pragma: no cover
            mode = self._set_gp2Scale_mode(KV)
        else:
            mode = self.mode
        if mode in ("Chol", "CholInv"):
            if issparse(KV): KV = KV.toarray()
            Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            KVinvY = calculate_Chol_solve(Chol_factor, y_mean, compute_device=self.compute_device, args=self.args)
        elif mode == "Inv":
            if issparse(KV): KV = KV.toarray()
            KVinvY = calculate_inv(KV, compute_device=self.compute_device, args=self.args) @ y_mean
        elif mode == "sparseLU":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
            KVinvY = calculate_LU_solve(LU_factor, y_mean, args=self.args)
        elif mode == "sparseCG":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, args=self.args)
        elif mode == "sparseMINRES":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_minres(KV, y_mean, args=self.args)
        elif mode == "sparseMINRESpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            B = sparse.linalg.spilu(KV, drop_tol=1e-8)
            M = sparse.linalg.LinearOperator(KV.shape, matvec=B.solve)
            KVinvY = calculate_sparse_minres(KV, y_mean, M=M, args=self.args)
        elif mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            B = sparse.linalg.spilu(KV, drop_tol=1e-8)
            M = sparse.linalg.LinearOperator(KV.shape, matvec=B.solve)
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=M, args=self.args)
        elif mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_solve(KV, y_mean, args=self.args)
        elif callable(mode[0]) and callable(mode[1]):
            factor = mode[0](KV)
            KVinvY = mode[1](factor, y_mean)
        else:
            raise Exception(f"No mode: {mode}")
        return KVinvY.reshape(y_mean.shape)

    def compute_new_KVlogdet_KVinvY(self, K, V, m):
        """Compute KVinvY and log|KV| jointly in one factorization pass (used during training)."""
        KV = self.addKV(K, V)
        y_mean = self.y_data - m[:, None]
        if self.gp2Scale:   # pragma: no cover
            mode = self._set_gp2Scale_mode(KV)
        else:
            mode = self.mode
        if mode in ("Chol", "CholInv"):
            if issparse(KV): KV = KV.toarray()
            Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            KVinvY = calculate_Chol_solve(Chol_factor, y_mean, compute_device=self.compute_device, args=self.args)
            KVlogdet = calculate_Chol_logdet(Chol_factor, compute_device=self.compute_device, args=self.args)
        elif mode == "Inv":
            if issparse(KV): KV = KV.toarray()
            KVinvY = calculate_inv(KV, compute_device=self.compute_device, args=self.args) @ y_mean
            KVlogdet = calculate_logdet(KV, compute_device=self.compute_device, args=self.args)
        elif mode == "sparseLU":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
            KVinvY = calculate_LU_solve(LU_factor, y_mean, args=self.args)
            KVlogdet = calculate_LU_logdet(LU_factor, args=self.args)
        elif mode == "sparseCG":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, args=self.args)
            KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
        elif mode == "sparseMINRES":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_minres(KV, y_mean, args=self.args)
            KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
        elif mode == "sparseMINRESpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            B = sparse.linalg.spilu(KV, drop_tol=1e-8)
            M = sparse.linalg.LinearOperator(KV.shape, matvec=B.solve)
            KVinvY = calculate_sparse_minres(KV, y_mean, M=M, args=self.args)
            KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
        elif mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            B = sparse.linalg.spilu(KV, drop_tol=1e-8)
            M = sparse.linalg.LinearOperator(KV.shape, matvec=B.solve)
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=M, args=self.args)
            KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
        elif mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_solve(KV, y_mean, args=self.args)
            KVlogdet = calculate_random_logdet(KV, self.compute_device, args=self.args)
        elif callable(mode[0]) and callable(mode[1]) and callable(mode[2]):
            factor = mode[0](KV)
            KVinvY = mode[1](factor, y_mean)
            KVlogdet = mode[2](factor)
        else:
            raise Exception(f"No mode: {mode}")
        return KVinvY.reshape(y_mean.shape), KVlogdet

    ##################################################################
    ##################################################################
    ##################################################################
    def _get_KVm(self):
        return self.K, self.V, self.m

    @staticmethod
    def addKV(K, V):
        assert np.ndim(K) == 2, "K must be a 2-d matrix"
        assert K.shape[0] == K.shape[1], "K must be square"

        if issparse(K):
            if issparse(V):
                KV = K + V
                return KV
            else:
                assert np.ndim(V) == 1, "K is sparse but V is a dense matrix; expected 1-d diagonal"
                assert len(V) == K.shape[0], "diagonal noise V length must match K dimension"
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

    def solve(self, b, x0=None):
        if self.mode == "Chol":
            return calculate_Chol_solve(self.Chol_factor, b, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv":
            # CholInv mode pre-computes and caches the explicit inverse in set_KV/update_KV;
            # using it here turns every downstream solve (posterior mean, covariance, gradients,
            # state-update KVinvY) into a single GEMM/GEMV instead of two triangular solves.
            return self.KVinv @ b
        elif self.mode == "Inv":
            return self.KVinv @ b
        elif self.mode == "sparseCG":
            return calculate_sparse_conj_grad(self.KV, b, x0=x0, args=self.args)
        elif self.mode == "sparseMINRES":
            return calculate_sparse_minres(self.KV, b, x0=x0, args=self.args)
        elif self.mode == "sparseLU":
            return calculate_LU_solve(self.LU_factor, b, args=self.args)
        elif self.mode == "sparseMINRESpre":
            B = sparse.linalg.spilu(self.KV, drop_tol=1e-8)
            M = sparse.linalg.LinearOperator(self.KV.shape, matvec=B.solve)
            return calculate_sparse_minres(self.KV, b, M=M, x0=x0, args=self.args)
        elif self.mode == "sparseCGpre":
            B = sparse.linalg.spilu(self.KV, drop_tol=1e-8)
            M = sparse.linalg.LinearOperator(self.KV.shape, matvec=B.solve)
            return calculate_sparse_conj_grad(self.KV, b, M=M, x0=x0, args=self.args)
        elif self.mode == "sparseSolve":
            return calculate_sparse_solve(self.KV, b, args=self.args)
        elif callable(self.mode[1]):
            return self.mode[1](self.custom_obj, b)
        else:
            raise Exception(f"No Mode. Choose from: {self.allowed_modes}")

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
        else: raise Exception(f"No Mode. Choose from: {self.allowed_modes}")

    def __getstate__(self):
        state = dict(
            mode=self.mode,
            linalg_mode=self.linalg_mode,
            data=self.data,
            prior=self.prior,
            likelihood=self.likelihood,
            KVinv=self.KVinv,
            KV=self.KV,
            Chol_factor=self.Chol_factor,
            LU_factor=self.LU_factor,
            KVinvY=self.KVinvY,
            cached_solve=self.cached_solve,
            cached_precond=self.cached_precond,
            custom_obj=self.custom_obj,
            allowed_modes=self.allowed_modes,
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
