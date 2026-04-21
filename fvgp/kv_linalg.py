import numpy as np
import warnings
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
        self.Preconditioner_factor = None
        self.Preconditioner_operator = None
        self.Preconditioner_signature = None
        self.Preconditioner_reuse_counter = 0
        self.Last_preconditioner_error = None
        self.Last_iterative_solution = None
        self.custom_obj = None
        self.allowed_modes = ["Chol", "CholInv", "Inv", "sparseMINRES", "sparseCG",
                              "sparseLU", "sparseMINRESpre", "sparseCGpre",
                              "sparseMINRESpre_<type>", "sparseCGpre_<type>",
                              "sparseSolve", "a set of callables"]

    @property
    def args(self):
        return self.data.args

    def _warm_start_enabled(self):
        return bool(self.args.get("sparse_krylov_warm_start", False))

    def _preconditioner_refresh_interval(self):
        return max(1, int(self.args.get("sparse_preconditioner_refresh_interval", 1)))

    def _preconditioner_signature(self):
        relevant = {
            key: value
            for key, value in self.args.items()
            if key.startswith("sparse_preconditioner_")
        }
        return tuple(sorted(relevant.items()))

    def _reset_sparse_preconditioner(self):
        self.Preconditioner_factor = None
        self.Preconditioner_operator = None
        self.Preconditioner_signature = None
        self.Preconditioner_reuse_counter = 0
        self.Last_preconditioner_error = None

    def _can_reuse_sparse_preconditioner(self, KV, mode):
        if mode not in {"sparseMINRESpre", "sparseCGpre"}:
            return False
        if self.Preconditioner_operator is None:
            return False
        if self.mode != mode:
            return False
        if self.KV is None:
            return False
        if self.KV.shape != KV.shape:
            return False
        if self.Preconditioner_signature != self._preconditioner_signature():
            return False
        if self.Preconditioner_reuse_counter >= self._preconditioner_refresh_interval() - 1:
            return False
        return True

    def _refresh_sparse_preconditioner(self):
        if self.mode not in {"sparseMINRESpre", "sparseCGpre"} or self.KV is None:
            self._reset_sparse_preconditioner()
            return
        try:
            factor, operator = calculate_sparse_preconditioner(self.KV, args=self.args)
        except Exception as exc:
            self.Last_preconditioner_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"Failed to build sparse preconditioner for mode {self.mode}; "
                f"falling back to the unpreconditioned iterative solve. "
                f"Reason: {self.Last_preconditioner_error}. "
                f"{sparse_preconditioner_failure_guidance(self.args)}"
            )
            logger.warning("Sparse preconditioner construction failed for {}: {}", self.mode, exc)
            self._reset_sparse_preconditioner()
            return
        self.Preconditioner_factor = factor
        self.Preconditioner_operator = operator
        self.Preconditioner_signature = self._preconditioner_signature()
        self.Preconditioner_reuse_counter = 0
        self.Last_preconditioner_error = None

    def _get_sparse_preconditioner(self):
        if self.mode in {"sparseMINRESpre", "sparseCGpre"} and self.Preconditioner_operator is None:
            self._refresh_sparse_preconditioner()
        return self.Preconditioner_operator

    def _iterative_modes(self):
        return {"sparseMINRES", "sparseCG", "sparseMINRESpre", "sparseCGpre"}

    def _store_iterative_solution(self, solution, training=False):
        if training and self._warm_start_enabled():
            self.Last_iterative_solution = np.array(solution, copy=True)

    def _consume_warm_start(self, b, x0, training=False):
        if x0 is not None:
            return x0
        if not training:
            return None
        if not self._warm_start_enabled():
            return None
        if self.Last_iterative_solution is None:
            return None
        guess = np.asarray(self.Last_iterative_solution)
        rhs = np.asarray(b)
        if rhs.ndim == 1:
            rhs = rhs.reshape(-1, 1)
        if guess.ndim == 1:
            guess = guess.reshape(-1, 1)
        if guess.shape[0] != rhs.shape[0]:
            return None
        if guess.shape[1] == rhs.shape[1]:
            return guess
        if guess.shape[1] == 1 and rhs.shape[1] > 1:
            return np.repeat(guess, rhs.shape[1], axis=1)
        return None

    def _prepare_iterative_state(self, KV, mode):
        if mode in {"sparseMINRESpre", "sparseCGpre"}:
            if self._can_reuse_sparse_preconditioner(KV, mode):
                self.KV = KV
                self.mode = mode
                self.Preconditioner_reuse_counter += 1
            else:
                self.KV = KV
                self.mode = mode
                self._reset_sparse_preconditioner()
                self._refresh_sparse_preconditioner()
        else:
            if mode not in {"sparseMINRES", "sparseCG"}:
                self.Last_iterative_solution = None
            self._reset_sparse_preconditioner()
            self.KV = KV
            self.mode = mode

    def set_KV(self, KV, mode):
        previous_solution = self.Last_iterative_solution
        previous_mode = self.mode
        previous_custom_obj = self.custom_obj
        mode, resolved_args = resolve_gp2scale_linalg_mode(mode, self.args)
        self.data.args = resolved_args
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None
        self.custom_obj = None
        if mode not in self._iterative_modes():
            self.Last_iterative_solution = None
        assert mode is not None
        self.mode = mode
        if self.mode == "Chol":
            self._reset_sparse_preconditioner()
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv":
            self._reset_sparse_preconditioner()
            if issparse(KV): KV = KV.toarray()
            self.Chol_factor = calculate_Chol_factor(KV, compute_device=self.compute_device, args=self.args)
            self.KVinv = calculate_inv_from_chol(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "Inv":
            self._reset_sparse_preconditioner()
            self.KV = KV
            self.KVinv = calculate_inv(KV, compute_device=self.compute_device, args=self.args)
        elif self.mode == "sparseMINRES":
            self._prepare_iterative_state(KV, self.mode)
        elif self.mode == "sparseCG":
            self._prepare_iterative_state(KV, self.mode)
        elif self.mode == "sparseLU":
            self._reset_sparse_preconditioner()
            self.LU_factor = calculate_sparse_LU_factor(KV, args=self.args)
        elif self.mode == "sparseMINRESpre":
            self._prepare_iterative_state(KV, self.mode)
        elif self.mode == "sparseCGpre":
            self._prepare_iterative_state(KV, self.mode)
        elif self.mode == "sparseSolve":
            self._reset_sparse_preconditioner()
            self.KV = KV
        elif callable(self.mode[0]):
            self._reset_sparse_preconditioner()
            self.custom_obj = self.mode[0](KV)
        else:
            self.custom_obj = previous_custom_obj
            raise Exception("No Mode. Choose from: ", self.allowed_modes)
        if self._warm_start_enabled() and previous_solution is not None and mode in self._iterative_modes():
            previous = np.asarray(previous_solution)
            if previous.ndim == 1:
                previous = previous.reshape(-1, 1)
            if previous.shape[0] == KV.shape[0]:
                self.Last_iterative_solution = previous
            elif mode != previous_mode:
                self.Last_iterative_solution = None

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
            if self._can_reuse_sparse_preconditioner(KV, self.mode):
                self.KV = KV
                self.Preconditioner_reuse_counter += 1
            else:
                self.KV = KV
                self._reset_sparse_preconditioner()
                self._refresh_sparse_preconditioner()
        elif self.mode == "sparseCGpre":
            if self._can_reuse_sparse_preconditioner(KV, self.mode):
                self.KV = KV
                self.Preconditioner_reuse_counter += 1
            else:
                self.KV = KV
                self._reset_sparse_preconditioner()
                self._refresh_sparse_preconditioner()
        elif self.mode == "sparseSolve":
            self.KV = KV
        elif callable(self.mode[0]):
            self.custom_obj = self.mode[0](KV)
        else:
            raise Exception("No Mode. Choose from: ", self.allowed_modes)
        if self.mode not in self._iterative_modes():
            self.Last_iterative_solution = None

    def solve(self, b, x0=None, training=False):
        if self.mode == "Chol":
            return calculate_Chol_solve(self.Chol_factor, b, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv":
            return calculate_Chol_solve(self.Chol_factor, b, compute_device=self.compute_device, args=self.args)
        elif self.mode == "Inv":
            #return matmul(self.KVinv, b, compute_device=self.compute_device) #is this really faster?
            return self.KVinv @ b
        elif self.mode == "sparseCG":
            res = calculate_sparse_conj_grad(self.KV, b, x0=self._consume_warm_start(b, x0, training=training), args=self.args)
            self._store_iterative_solution(res, training=training)
            return res
        elif self.mode == "sparseMINRES":
            res = calculate_sparse_minres(self.KV, b, x0=self._consume_warm_start(b, x0, training=training), args=self.args)
            self._store_iterative_solution(res, training=training)
            return res
        elif self.mode == "sparseLU":
            return calculate_LU_solve(self.LU_factor, b, args=self.args)
        elif self.mode == "sparseMINRESpre":
            res = calculate_sparse_minres(
                self.KV,
                b,
                M=self._get_sparse_preconditioner(),
                x0=self._consume_warm_start(b, x0, training=training),
                args=self.args,
            )
            self._store_iterative_solution(res, training=training)
            return res
        elif self.mode == "sparseCGpre":
            res = calculate_sparse_conj_grad(
                self.KV,
                b,
                M=self._get_sparse_preconditioner(),
                x0=self._consume_warm_start(b, x0, training=training),
                args=self.args,
            )
            self._store_iterative_solution(res, training=training)
            return res
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
            Preconditioner_factor=None,
            Preconditioner_operator=None,
            Preconditioner_signature=self.Preconditioner_signature,
            Preconditioner_reuse_counter=self.Preconditioner_reuse_counter,
            Last_preconditioner_error=self.Last_preconditioner_error,
            Last_iterative_solution=self.Last_iterative_solution,
            custom_obj=self.custom_obj,
            allowed_modes=self.allowed_modes
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "Preconditioner_factor" not in self.__dict__:
            self.Preconditioner_factor = None
        if "Preconditioner_operator" not in self.__dict__:
            self.Preconditioner_operator = None
        if "Preconditioner_signature" not in self.__dict__:
            self.Preconditioner_signature = None
        if "Preconditioner_reuse_counter" not in self.__dict__:
            self.Preconditioner_reuse_counter = 0
        if "Last_preconditioner_error" not in self.__dict__:
            self.Last_preconditioner_error = None
        if "Last_iterative_solution" not in self.__dict__:
            self.Last_iterative_solution = None
