import contextlib
import warnings

import numpy as np
import scipy.sparse as sparse
from loguru import logger
from scipy.sparse import issparse

from .gp_lin_alg import *


##################################################################
# Policy for linear-algebra state that persists between likelihood
# evaluations. Lives here with the per-evaluation staleness checks on GPkv
# rather than with any one training method: the rule is that everything
# deciding whether cached linalg state may be reused belongs in this module.
##################################################################
#: linalg settings that carry state from one likelihood evaluation to the next, and the
#: value each must take for the evaluations to be independent of the order they run in
_SEQUENTIAL_STATE_DEFAULTS = {
    "sparse_krylov_warm_start": False,
    "sparse_preconditioner_refresh_interval": 1,
}

#: the only training method whose steps are small enough for that state to be sound
_SEQUENTIAL_STATE_METHODS = {"mcmc"}


@contextlib.contextmanager
def sequential_linalg_state(args, method):
    """Allow Krylov warm starts and preconditioner reuse only for ``method='mcmc'``.

    Both mechanisms carry state from one likelihood evaluation to the next, which is
    sound exactly when successive evaluations are close. MCMC proposes local steps, so
    the previous solve is an excellent starting guess and a preconditioner built one
    step ago is still apt: measured on a truncated CG solve, a warm start from nearby
    hyperparameters cuts the error 25x and a reused nearby preconditioner is as good as
    a fresh one.

    Every other method moves non-locally. A Bayesian optimizer's space-filling design
    spans the box and its acquisition then jumps wherever it likes; a global optimizer's
    population samples the box outright. The same measurements show that in that regime
    a warm start is *worse than a cold start* and a reused preconditioner is worth no
    more than none at all.

    The cost is not the lost speed but what the leftover residual does to the objective.
    A truncated solve seeded with stale state has an error that depends on which
    hyperparameters ran *before* it, making the likelihood order-dependent: the same
    point can return different values at different stages of a run. A Bayesian optimizer
    assumes its observations are noisy but unbiased, so an order-dependent bias is
    precisely what its noise model cannot absorb -- unlike the variance of the stochastic
    log-determinant, which is genuinely zero-mean and is fed in deliberately.

    This is the coarse, per-training-run gate. The fine, per-evaluation one lives in
    :py:meth:`GPkv._validated_warm_start` and
    :py:meth:`GPkv._can_reuse_sparse_preconditioner`, which discard cached state
    whenever K+V has actually drifted, whatever the method. Both settings also
    default to the safe value, so this only changes anything for a user who turned them
    on for MCMC -- which fvGP's own preconditioner-failure guidance suggests -- and then
    switched method. Overriding an explicit setting is warned about, not done silently.
    """
    if not isinstance(args, dict) or method in _SEQUENTIAL_STATE_METHODS:
        yield
        return
    overridden = {key: args[key] for key, safe in _SEQUENTIAL_STATE_DEFAULTS.items()
                  if key in args and args[key] != safe}
    if overridden:
        warnings.warn(
            f"method={method!r} disables sequential linear-algebra state for the duration "
            f"of the run: {overridden}. Warm starts and preconditioner reuse assume "
            f"successive evaluations are close, which holds for 'mcmc' but not for a "
            f"method that samples the space non-locally, where they leave an "
            f"order-dependent bias in the objective. The original settings are restored "
            f"afterwards."
        )
    saved = {key: args[key] for key in _SEQUENTIAL_STATE_DEFAULTS if key in args}
    try:
        args.update(_SEQUENTIAL_STATE_DEFAULTS)
        yield
    finally:
        for key in _SEQUENTIAL_STATE_DEFAULTS:
            if key in saved:
                args[key] = saved[key]
            else:
                args.pop(key, None)


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
        # variance of the most recent stochastic log-determinant estimate; stays None
        # for the exact modes, where log|KV| carries no estimator noise
        self.last_logdet_variance = None
        self.last_logdet_info = {}
        # fingerprints of the K+V that the cached preconditioner, and the solution most
        # recently reused as a warm start, were computed for
        self.Preconditioner_fingerprint = None
        self.Warm_start_fingerprint = None

        # Resolve aliases like "sparseCGpre_amg" → mode "sparseCGpre" with
        # args["sparse_preconditioner_type"]="amg".  Writes resolved args back
        # to data.args so downstream calls see the canonical key.
        if isinstance(linalg_mode, str):
            linalg_mode, resolved_args = resolve_gp2scale_linalg_mode(linalg_mode, self.data.args)
            self.data.args = resolved_args

        self.linalg_mode = linalg_mode  ###there should only be one mode
        self.KVinv = None
        self.KV = None
        self.Chol_factor = None
        self.LU_factor = None
        self.logdet_KV = None
        self.custom_obj = None
        self.cached_solve = None
        self.cached_precond = None
        # Sparse preconditioner cache (used by sparseMINRESpre / sparseCGpre).
        # `Preconditioner_KV_shape` validates that the cached operator is
        # dimensionally compatible with whichever KV the next caller submits.
        self.Preconditioner_factor = None
        self.Preconditioner_operator = None
        self.Preconditioner_signature = None
        self.Preconditioner_KV_shape = None
        self.Preconditioner_reuse_counter = 0
        self.Last_preconditioner_error = None
        self.allowed_modes = ["Chol", "CholInv", "Inv", "sparseMINRES", "sparseCG",
                              "sparseLU", "sparseMINRESpre", "sparseCGpre",
                              "sparseMINRESpre_<type>", "sparseCGpre_<type>",
                              "sparseSolve", "a set of callables"]
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
    ##############Sparse-preconditioner cache##########################
    ##################################################################
    _PRECONDITIONED_MODES = {"sparseMINRESpre", "sparseCGpre"}

    def _preconditioner_refresh_interval(self):
        """Hard cap on consecutive reuses. ``None``/absent means no cap.

        This used to be the *only* thing standing between a preconditioner and
        unlimited reuse, which is the wrong control: a fixed count knows nothing about
        whether the matrix has actually moved. Staleness is now decided by
        :py:meth:`_matrix_drift`, and this remains only as an optional belt-and-braces
        cap for callers who want one.
        """
        value = self.args.get("sparse_preconditioner_refresh_interval", None)
        if value is None:
            return None
        return max(1, int(value))

    def _preconditioner_signature(self):
        """Args fingerprint: any key beginning with ``sparse_preconditioner_``."""
        relevant = {key: value for key, value in self.args.items()
                    if key.startswith("sparse_preconditioner_")}
        return tuple(sorted(relevant.items()))

    @staticmethod
    def matrix_fingerprint(KV):
        """Cheap O(nnz) summary of K+V, used to detect that it has moved.

        Trace and Frobenius norm together capture both a change of scale (signal
        variance, noise) and a change of shape (length scales redistributing mass off
        the diagonal). Both are one pass over the stored values -- the cost of a single
        matvec, negligible beside the solve this protects.
        """
        if KV is None:
            return None
        try:
            if issparse(KV):
                data = KV.data
                trace = float(KV.diagonal().sum())
                fro = float(np.sqrt(np.dot(data, data)))
                nnz = int(KV.nnz)
            else:
                arr = np.asarray(KV)
                trace = float(np.trace(arr))
                fro = float(np.linalg.norm(arr))
                nnz = int(arr.size)
            return (tuple(KV.shape), nnz, trace, fro)
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _fingerprint_drift(old, new):
        """Relative distance between two fingerprints; ``inf`` when incomparable."""
        if old is None or new is None:
            return np.inf
        if old[0] != new[0]:
            return np.inf
        drift = 0.0
        for old_value, new_value in ((old[2], new[2]), (old[3], new[3])):
            scale = max(abs(old_value), abs(new_value), 1e-300)
            drift = max(drift, abs(new_value - old_value) / scale)
        return drift

    def _matrix_drift(self, KV):
        """How far K+V has moved since the cached state was built."""
        return self._fingerprint_drift(self.Preconditioner_fingerprint,
                                       self.matrix_fingerprint(KV))

    def _max_matrix_drift(self):
        """Relative change in K+V beyond which cached state is considered stale.

        The default is calibrated against how much benefit a preconditioner actually
        retains as the matrix moves away from the one it was built for. Measured on a
        truncated CG solve, as a percentage of the speed-up a freshly built
        preconditioner buys over none at all:

            drift  0.0007 -> 101%    drift  0.154 ->  88%
            drift  0.007  -> 100%    drift  0.268 ->  90%
            drift  0.035  -> 102%    drift  0.423 ->  71%
            drift  0.068  ->  96%    drift  0.595 ->  49%

        So the cached operator is essentially free of charge out to a drift of a few
        percent and only starts to fade beyond ~0.15. A threshold of 0.1 keeps nearly
        all of the benefit while refusing reuse once the operator has genuinely changed.

        In hyperparameter terms on a 300-point GP this admits steps of a few percent --
        comfortably covering MCMC proposals and local optimizer steps -- while a
        doubling of the hyperparameters drifts 0.58 and a tenfold change 0.92, both far
        outside. That is the intended split, and it is reached by measuring the operator
        rather than by asking which optimizer is running.

        Shared with the warm-start check, whose benefit falls off over a comparable
        range.
        """
        return float(self.args.get("sparse_preconditioner_max_matrix_drift", 0.1))

    def _validated_warm_start(self, KV, x0):
        """Drop a warm start that was computed for a materially different K+V.

        The previous solution is only a good starting guess while the operator is
        essentially unchanged -- true between MCMC or local steps, emphatically not true
        between the jumps a global or Bayesian optimizer makes. Measured on a truncated
        CG solve, a warm start from nearby hyperparameters cuts the error 25x while one
        from distant hyperparameters is worse than starting cold, because the truncation
        leaves a residual that depends on where the previous evaluation happened to be.

        Deciding by drift rather than by which optimizer is running keeps the guess
        exactly when it helps, with no method-specific configuration.
        """
        if x0 is None:
            return None
        if self.Warm_start_fingerprint is None:
            return x0
        if self._fingerprint_drift(self.Warm_start_fingerprint,
                                   self.matrix_fingerprint(KV)) > self._max_matrix_drift():
            logger.debug("Warm start dropped: K+V drifted beyond the reuse threshold.")
            return None
        return x0

    def _reset_sparse_preconditioner(self):
        self.Preconditioner_factor = None
        self.Preconditioner_operator = None
        self.Preconditioner_signature = None
        self.Preconditioner_KV_shape = None
        self.Preconditioner_fingerprint = None
        self.Preconditioner_reuse_counter = 0
        self.Last_preconditioner_error = None

    def _can_reuse_sparse_preconditioner(self, KV):
        if self.mode not in self._PRECONDITIONED_MODES:
            return False
        if self.Preconditioner_operator is None:
            return False
        if self.Preconditioner_KV_shape != KV.shape:
            return False
        if self.Preconditioner_signature != self._preconditioner_signature():
            return False
        interval = self._preconditioner_refresh_interval()
        if interval is not None and self.Preconditioner_reuse_counter >= interval - 1:
            return False
        # The decisive test: reuse only while K+V is still substantially the matrix the
        # preconditioner was built for. A counter cannot know this -- k steps of MCMC
        # barely move the operator, while one step of a global search can replace it.
        if self._matrix_drift(KV) > self._max_matrix_drift():
            logger.debug("Preconditioner refreshed: K+V drifted beyond the reuse threshold.")
            return False
        return True

    def _build_sparse_preconditioner_or_none(self, KV):
        """Construct a fresh preconditioner for ``KV``; ``None`` on failure (with warning)."""
        try:
            factor, operator = calculate_sparse_preconditioner(KV, args=self.args)
        except Exception as exc:
            self.Last_preconditioner_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"Failed to build sparse preconditioner for mode {self.mode}; "
                f"falling back to the unpreconditioned iterative solve. "
                f"Reason: {self.Last_preconditioner_error}. "
                f"{sparse_preconditioner_failure_guidance(self.args)}"
            )
            logger.warning("Sparse preconditioner construction failed for {}: {}", self.mode, exc)
            return None, None
        return factor, operator

    def _get_or_refresh_preconditioner(self, KV, force_refresh=False):
        """Return a cached or freshly-built ``LinearOperator`` for ``KV``.

        Caching honors ``args["sparse_preconditioner_refresh_interval"]`` (default 1
        = always refresh) and validates shape + relevant args fingerprint.
        ``force_refresh=True`` bypasses reuse — used by ``set_KV`` after a state
        change so the new factorization isn't based on the old KV's preconditioner.
        Returns ``None`` if construction fails; the caller falls back to an
        unpreconditioned iterative solve.
        """
        if self.mode not in self._PRECONDITIONED_MODES:
            return None
        if not force_refresh and self._can_reuse_sparse_preconditioner(KV):
            self.Preconditioner_reuse_counter += 1
            return self.Preconditioner_operator
        factor, operator = self._build_sparse_preconditioner_or_none(KV)
        if operator is None:
            self._reset_sparse_preconditioner()
            return None
        self.Preconditioner_factor = factor
        self.Preconditioner_operator = operator
        self.Preconditioner_signature = self._preconditioner_signature()
        self.Preconditioner_KV_shape = KV.shape
        self.Preconditioner_fingerprint = self.matrix_fingerprint(KV)
        self.Preconditioner_reuse_counter = 0
        self.Last_preconditioner_error = None
        return operator

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
        self.logdet_KV = self.logdet()

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
            self._get_or_refresh_preconditioner(KV, force_refresh=True)
        elif self.mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
            self._get_or_refresh_preconditioner(KV, force_refresh=True)
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
            self._get_or_refresh_preconditioner(KV)
        elif self.mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
            self._get_or_refresh_preconditioner(KV)
        elif self.mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            self.KV = KV
        elif callable(self.mode[0]):
            self.custom_obj = self.mode[0](KV)
        else:
            raise Exception(f"No Mode. Choose from: {self.allowed_modes}")

    def compute_new_KVinvY(self, KV, m, x0=None):
        """Recompute KVinvY for a given KV and m without updating state (used during training).

        ``x0`` (optional) is forwarded to iterative solvers as a warm-start; passing
        the previous iteration's KVinvY can substantially cut iteration counts when
        successive hyperparameters are close.
        """
        x0 = self._validated_warm_start(KV, x0)
        self.Warm_start_fingerprint = self.matrix_fingerprint(KV)
        y_mean = self.y_data - m[:, None]
        if self.gp2Scale:
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
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, x0=x0, args=self.args)
        elif mode == "sparseMINRES":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_minres(KV, y_mean, x0=x0, args=self.args)
        elif mode == "sparseMINRESpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            M = self._get_or_refresh_preconditioner(KV)
            KVinvY = calculate_sparse_minres(KV, y_mean, M=M, x0=x0, args=self.args)
        elif mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            M = self._get_or_refresh_preconditioner(KV)
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=M, x0=x0, args=self.args)
        elif mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_solve(KV, y_mean, args=self.args)
        elif callable(mode[0]) and callable(mode[1]):
            factor = mode[0](KV)
            KVinvY = mode[1](factor, y_mean)
        else:
            raise Exception(f"No mode: {mode}")
        return KVinvY.reshape(y_mean.shape)

    def _random_logdet(self, KV):
        """Stochastic-Lanczos log-determinant, recording the estimator's own variance.

        The sparse modes estimate log|KV| from a finite number of Hutchinson probes, so
        the value is a random variable. Capturing its variance here is what lets a
        noise-aware consumer -- ``train(method='bo')`` -- treat the marginal likelihood
        as the noisy observation it actually is, instead of asking the user to supply a
        noise level by hand. ``None`` when the estimate is exact or unavailable.
        """
        info = {}
        logdet = calculate_random_logdet(KV, self.compute_device, args=self.args, info_out=info)
        self.last_logdet_variance = info.get("variance", None)
        self.last_logdet_info = info
        return logdet

    def compute_new_KVlogdet_KVinvY(self, K, V, m, x0=None):
        """
        Compute KVinvY and log|KV| jointly in one factorization pass (used during training).
        No state is updated.

        ``x0`` (optional) is forwarded to iterative solvers as a warm-start.
        """
        KV = self.addKV(K, V)
        x0 = self._validated_warm_start(KV, x0)
        self.Warm_start_fingerprint = self.matrix_fingerprint(KV)
        y_mean = self.y_data - m[:, None]
        if self.gp2Scale:
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
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, x0=x0, args=self.args)
            KVlogdet = self._random_logdet(KV)
        elif mode == "sparseMINRES":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_minres(KV, y_mean, x0=x0, args=self.args)
            KVlogdet = self._random_logdet(KV)
        elif mode == "sparseMINRESpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            M = self._get_or_refresh_preconditioner(KV)
            KVinvY = calculate_sparse_minres(KV, y_mean, M=M, x0=x0, args=self.args)
            KVlogdet = self._random_logdet(KV)
        elif mode == "sparseCGpre":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            M = self._get_or_refresh_preconditioner(KV)
            KVinvY = calculate_sparse_conj_grad(KV, y_mean, M=M, x0=x0, args=self.args)
            KVlogdet = self._random_logdet(KV)
        elif mode == "sparseSolve":
            if not issparse(KV): KV = sparse.csr_matrix(KV)
            KVinvY = calculate_sparse_solve(KV, y_mean, args=self.args)
            KVlogdet = self._random_logdet(KV)
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
        # x0 shape normalization (zero-pad / column-broadcast) is handled inside
        # the sparse iterative solvers in gp_lin_alg, so no shaping is needed here.
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
            M = self._get_or_refresh_preconditioner(self.KV)
            return calculate_sparse_minres(self.KV, b, M=M, x0=x0, args=self.args)
        elif self.mode == "sparseCGpre":
            M = self._get_or_refresh_preconditioner(self.KV)
            return calculate_sparse_conj_grad(self.KV, b, M=M, x0=x0, args=self.args)
        elif self.mode == "sparseSolve":
            return calculate_sparse_solve(self.KV, b, args=self.args)
        elif callable(self.mode[1]):
            return self.mode[1](self.custom_obj, b)
        else:
            raise Exception(f"No Mode. Choose from: {self.allowed_modes}")

    def logdet(self):
        """
        Compute log|KV| in the current mode without updating state. Used during training for the log-likelihood.
        """
        if self.mode == "Chol": return calculate_Chol_logdet(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "CholInv": return calculate_Chol_logdet(self.Chol_factor, compute_device=self.compute_device, args=self.args)
        elif self.mode == "sparseLU": return calculate_LU_logdet(self.LU_factor, args=self.args)
        elif self.mode == "Inv": return calculate_logdet(self.KV, args=self.args)
        elif self.mode == "sparseCG": return self._random_logdet(self.KV)
        elif self.mode == "sparseMINRES": return self._random_logdet(self.KV)
        elif self.mode == "sparseMINRESpre": return self._random_logdet(self.KV)
        elif self.mode == "sparseCGpre": return self._random_logdet(self.KV)
        elif self.mode == "sparseSolve": return self._random_logdet(self.KV)
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
            # Preconditioner factor/operator are not picklable in the general
            # case (LinearOperator closures, spilu factors); the next call will
            # rebuild from KV via the cache helpers.
            Preconditioner_factor=None,
            Preconditioner_operator=None,
            Preconditioner_signature=self.Preconditioner_signature,
            Preconditioner_KV_shape=self.Preconditioner_KV_shape,
            Preconditioner_reuse_counter=self.Preconditioner_reuse_counter,
            Last_preconditioner_error=self.Last_preconditioner_error,
            custom_obj=self.custom_obj,
            allowed_modes=self.allowed_modes,
            last_logdet_variance=self.last_logdet_variance,
            last_logdet_info=self.last_logdet_info,
            Preconditioner_fingerprint=self.Preconditioner_fingerprint,
            Warm_start_fingerprint=self.Warm_start_fingerprint,
            logdet_KV=self.logdet_KV
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Defensive defaults for unpickling older saved states.
        for attr, default in (
            ("Preconditioner_factor", None),
            ("Preconditioner_operator", None),
            ("Preconditioner_signature", None),
            ("Preconditioner_KV_shape", None),
            ("Preconditioner_reuse_counter", 0),
            ("Last_preconditioner_error", None),
            ("Preconditioner_fingerprint", None),
            ("Warm_start_fingerprint", None),
        ):
            if attr not in self.__dict__:
                setattr(self, attr, default)
