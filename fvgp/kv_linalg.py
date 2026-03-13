import numpy as np
from .gp_lin_alg import *
from scipy.sparse import issparse
import scipy.sparse as sparse
from loguru import logger


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
