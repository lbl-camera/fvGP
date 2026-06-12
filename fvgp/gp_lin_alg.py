"""CPU/GPU linear algebra backend for fvGP.

Provides Cholesky, LU, and sparse solvers with optional dispatch to PyTorch
(CUDA / MPS) or CuPy GPU backends.  All public functions accept a
``compute_device`` argument (``"cpu"`` or ``"gpu"``) and an ``args`` dict for
fine-grained options such as ``"GPU_engine"`` and ``"GPU_device"``.

Also exposes a sparse preconditioner framework (``calculate_sparse_preconditioner``)
with ILU, incomplete Cholesky, block Jacobi, additive Schwarz, and AMG
backends, along with a block conjugate-gradient solver for multi-RHS systems.
"""
import importlib
import time
import warnings
from collections import deque

import numpy as np
from loguru import logger
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse import identity
from scipy.sparse.linalg import cg, minres, onenormest, splu, spsolve, spsolve_triangular

warnings.simplefilter("once", UserWarning)


class NonPositiveDefiniteError(np.linalg.LinAlgError):
    """Covariance matrix is not positive definite."""
    pass


def _non_pd_message(M, original_error):
    n = M.shape[0]
    diag_min = float(np.min(np.diag(M)))
    sym_err = float(np.max(np.abs(M - M.T)))
    return (
        f"Cholesky factorization failed: the {n}x{n} prior covariance matrix "
        f"is not positive definite.\n"
        f"Most common causes in fvGP:\n"
        f"  1. A user-defined kernel that is not positive definite for all inputs.\n"
        f"     (The kernel must produce a symmetric PD matrix for every set of points.)\n"
        f"  2. Duplicate or near-duplicate rows in x_data causing a rank-deficient K.\n"
        f"  3. Noise/jitter on the diagonal is too small for the conditioning of K.\n"
        f"Diagnostics: min(diag(M)) = {diag_min:.3e}, "
        f"max|M - M.T| = {sym_err:.3e} (should be ~0).\n"
        f"Try: (a) verify the kernel is PD, (b) add jitter to the diagonal, "
        f"(c) deduplicate x_data.\n"
        f"Original linear-algebra error: {original_error}"
    )


def _rank1_update_non_pd_message(disc):
    return (
        f"Cholesky rank-1 update failed: Schur complement {float(disc):.3e} <= 0, "
        f"the augmented matrix is not positive definite. "
        f"This usually indicates the new data row is linearly dependent on old rows "
        f"or the kernel is not PD on the augmented set."
    )


# --------------------------------------------------------------------------
# GPU engine / device detection
# --------------------------------------------------------------------------

def _normalize_args(args):
    return {} if args is None else args


def _torch_gpu_device(args=None):
    """Return a usable torch device (CUDA or MPS) or ``None``.

    Honors ``args["GPU_device"]`` (e.g. ``"cuda:1"``, ``"mps"``) and
    ``args["GPU_device_index"]`` when picking the default CUDA device.
    """
    args = _normalize_args(args)
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if "GPU_device" in args:
        try:
            requested_device = torch.device(str(args["GPU_device"]))
        except Exception:
            return None
        if requested_device.type == "cuda":
            return requested_device if torch.cuda.is_available() else None
        if requested_device.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            return requested_device if mps_backend is not None and torch.backends.mps.is_available() else None
        return requested_device

    if torch.cuda.is_available():
        device_index = int(args.get("GPU_device_index",
                                    torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0))
        return torch.device(f"cuda:{device_index}")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return None


def _cupy_gpu_available():
    """Return ``True`` if cupy is importable and a CUDA device is visible."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_gpu_engine(args):
    """Return the active GPU engine name (``"torch"``, ``"cupy"``, or ``None``).

    Unlike a simple package-presence check, this also verifies that the
    selected backend has a usable GPU available; otherwise returns ``None``.
    """
    args = _normalize_args(args)
    if "GPU_engine" in args:
        preferred_engine = str(args["GPU_engine"]).lower()
        if preferred_engine == "torch":
            return "torch" if _torch_gpu_device(args) is not None else None
        if preferred_engine == "cupy":
            return "cupy" if _cupy_gpu_available() else None
        return None
    if _torch_gpu_device(args) is not None:
        return "torch"
    if _cupy_gpu_available():
        return "cupy"
    return None


def _imate_gpu_enabled(args=None):
    """imate's GPU backend works through cupy (or torch CUDA); gate accordingly."""
    args = _normalize_args(args)
    if _cupy_gpu_available():
        return True
    device = _torch_gpu_device(args)
    return device is not None and device.type == "cuda"


# --------------------------------------------------------------------------
# LU
# --------------------------------------------------------------------------

def calculate_sparse_LU_factor(M, args=None):
    """Compute the sparse LU factorization of ``M`` via SuperLU."""
    assert sparse.issparse(M), "M must be a sparse matrix for LU factorization"
    logger.debug("calculate_sparse_LU_factor")
    LU = splu(M.tocsc())
    return LU


def calculate_LU_solve(LU, vec, args=None):
    """Solve ``M x = vec`` given a pre-computed SuperLU factorization."""
    assert isinstance(vec, np.ndarray), "vec must be np.ndarray for LU solve"
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if vec.dtype != LU.L.dtype:
        vec = vec.astype(LU.L.dtype, copy=False)
    logger.debug("calculate_LU_solve")
    res = LU.solve(vec)
    if np.ndim(res) == 1: res = res.reshape(len(res), 1)
    assert np.ndim(res) == 2, "LU solve result must be 2-d"
    return res


def calculate_LU_logdet(LU, args=None):
    """Return log|det M| from a SuperLU factorization via the U diagonal."""
    logger.debug("calculate_LU_logdet")
    upper_diag = abs(LU.U.diagonal())
    logdet = np.sum(np.log(upper_diag))
    assert np.isscalar(logdet), "LU logdet result must be scalar"
    return logdet


# --------------------------------------------------------------------------
# Dense Cholesky
# --------------------------------------------------------------------------

def calculate_Chol_factor(M, compute_device="cpu", args=None):
    """Return the lower-triangular Cholesky factor of the symmetric PD matrix ``M``."""
    assert isinstance(M, np.ndarray), "M must be np.ndarray for Cholesky factorization"
    args = _normalize_args(args)
    if "Chol_factor_compute_device" in args: compute_device = args["Chol_factor_compute_device"]
    logger.debug(f"calculate_Chol_factor on {compute_device}")
    try:
        if compute_device == "cpu":
            c, _ = cho_factor(M, lower=True)
        elif compute_device == "gpu":
            engine = get_gpu_engine(args)
            if engine == "torch":  # pragma: no cover
                import torch
                device = _torch_gpu_device(args)
                A = torch.as_tensor(M, device=device)
                L = torch.linalg.cholesky(A)
                c = L.detach().cpu().numpy()
            elif engine == "cupy":  # pragma: no cover
                import cupy as cp
                A = cp.asarray(M)
                L = cp.linalg.cholesky(A)
                c = cp.asnumpy(L)
            else:
                warnings.warn(
                    "No usable GPU backend found for Cholesky factorization. Falling back to CPU.",
                    stacklevel=2,
                )
                c, _ = cho_factor(M, lower=True)
        else:
            raise Exception("No valid compute device found. ")
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise NonPositiveDefiniteError(_non_pd_message(M, e)) from e
    return c


def update_Chol_factor(old_chol_factor, new_matrix, compute_device="cpu", args=None):
    """Extend an existing Cholesky factor to cover ``new_matrix`` via rank-n update.

    Note: ``compute_device`` is currently fixed to ``"cpu"`` regardless of the
    argument; GPU support is not yet wired up for this wrapper.
    """
    assert isinstance(new_matrix, np.ndarray), "new_matrix must be np.ndarray for Cholesky update"
    compute_device = "cpu"
    #if "update_Chol_factor_compute_device" in args: compute_device = args["update_Chol_factor_compute_device"]
    logger.debug(f"update_Chol_factor on {compute_device}")
    size = len(old_chol_factor)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    return cholesky_update_rank_n(old_chol_factor, k.T, kk, compute_device=compute_device, args=args)


def calculate_Chol_solve(factor, vec, compute_device="cpu", args=None):
    """Solve ``A x = vec`` given the lower-triangular Cholesky factor of ``A``."""
    assert isinstance(vec, np.ndarray), "vec must be np.ndarray for Cholesky solve"
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if vec.dtype != factor.dtype:
        vec = vec.astype(factor.dtype, copy=False)
    args = _normalize_args(args)
    if "Chol_solve_compute_device" in args: compute_device = args["Chol_solve_compute_device"]
    logger.debug(f"calculate_Chol_solve on {compute_device}")
    if compute_device == "cpu":
        res = cho_solve((factor, True), vec)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            L = torch.as_tensor(factor, device=device)
            b = torch.as_tensor(vec, device=device)
            y = torch.linalg.solve_triangular(L, b, upper=False)
            x = torch.linalg.solve_triangular(L.T, y, upper=True)
            res = x.detach().cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
            L = cp.asarray(factor)
            b = cp.asarray(vec)
            y = cp_solve_triangular(L, b, lower=True)
            x = cp_solve_triangular(L.T, y, lower=False)
            res = cp.asnumpy(x)
        else:
            warnings.warn(
                "No usable GPU backend found for Cholesky solve. Falling back to CPU.",
                stacklevel=2,
            )
            res = cho_solve((factor, True), vec)
    else:
        raise Exception("NO valid compute device found. ")

    if np.ndim(res) == 1: res = res.reshape(len(res), 1)
    return res


def calculate_Chol_logdet(factor, compute_device="cpu", args=None):
    """Return log|det A| = 2 * sum(log(diag(L))) from the Cholesky factor ``L``."""
    args = _normalize_args(args)
    if "Chol_logdet_compute_device" in args: compute_device = args["Chol_logdet_compute_device"]
    logger.debug(f"calculate_Chol_logdet on {compute_device}")
    if compute_device == "cpu":
        upper_diag = abs(factor.diagonal())
        logdet = 2.0 * np.sum(np.log(upper_diag))
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            L = torch.as_tensor(factor, device=device)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L))).cpu().item()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            L = cp.asarray(factor)
            logdet = 2.0 * cp.sum(cp.log(cp.diag(L))).get()
        else:
            warnings.warn(
                "No usable GPU backend found for Cholesky logdet. Falling back to CPU.",
                stacklevel=2,
            )
            upper_diag = abs(factor.diagonal())
            logdet = 2.0 * np.sum(np.log(upper_diag))
    else:
        raise Exception("No valid compute device found. ")
    assert np.isscalar(logdet), "Cholesky logdet result must be scalar"
    return logdet


def spai(A, m, args=None):
    """Sparse Approximate Inverse preconditioner via m-step SPAI iteration."""
    assert sparse.issparse(A), "A must be sparse for SPAI"
    assert isinstance(m, int), "m must be int (SPAI iteration count)"
    logger.debug("spai preconditioning")
    n = A.shape[0]

    ident = identity(n, format='csr')
    alpha = 2 / onenormest(A @ A.T)
    M = alpha * A

    for index in range(m):
        C = A @ M
        G = ident - C
        AG = A @ G
        trace = (G.T @ AG).diagonal().sum()
        alpha = trace / np.linalg.norm(AG.data) ** 2
        M = M + alpha * G

    assert sparse.issparse(M), "SPAI result must remain sparse"
    return M


# --------------------------------------------------------------------------
# Sparse preconditioner framework
#
# `calculate_sparse_preconditioner(KV, args)` dispatches on
# args["sparse_preconditioner_type"] (default "ilu") and returns
# (factor, operator). The factor is implementation-specific; the operator
# is a scipy.sparse.linalg.LinearOperator suitable for passing as M to
# cg/minres. `resolve_gp2scale_linalg_mode` lets a mode string like
# "sparseCGpre_amg" auto-populate args["sparse_preconditioner_type"].
# --------------------------------------------------------------------------

def normalize_sparse_preconditioner_type(preconditioner_type):
    """Resolve user-facing preconditioner aliases to canonical names."""
    preconditioner_type = str(preconditioner_type).lower()
    aliases = {
        "ilu": "ilu",
        "ic": "ichol",
        "ichol": "ichol",
        "incomplete_cholesky": "ichol",
        "ichol0": "ichol0",
        "native_ic": "native_incomplete_cholesky",
        "native_ichol": "native_incomplete_cholesky",
        "legacy_ic": "native_incomplete_cholesky",
        "legacy_ichol": "native_incomplete_cholesky",
        "native_incomplete_cholesky": "native_incomplete_cholesky",
        "legacy_incomplete_cholesky": "native_incomplete_cholesky",
        "block_jacobi": "block_jacobi",
        "blockjacobi": "block_jacobi",
        "schwarz": "additive_schwarz",
        "additive_schwarz": "additive_schwarz",
        "amg": "amg",
    }
    if preconditioner_type not in aliases:
        raise ValueError(
            "Unknown sparse preconditioner type "
            f"{preconditioner_type!r}. Expected one of "
            "{'ilu', 'ichol', 'ic', 'ichol0', 'incomplete_cholesky', "
            "'native_ic', 'native_ichol', 'legacy_ic', 'legacy_ichol', "
            "'block_jacobi', 'blockjacobi', "
            "'schwarz', 'additive_schwarz', 'amg'}."
        )
    return aliases[preconditioner_type]


def _raise_missing_ilupp(preconditioner_type, exc):
    raise ImportError(
        "The sparse incomplete-Cholesky preconditioners (`ichol`, `ic`, "
        "`incomplete_cholesky`, and `ichol0`) require the optional `ilupp` "
        "package. Install it in the Python environment running fvGP with:\n\n"
        "    pip install ilupp\n\n"
        f"Requested sparse preconditioner resolved to backend={preconditioner_type!r}."
    ) from exc


def sparse_preconditioner_failure_guidance(args=None):
    args = _normalize_args(args)
    preconditioner_type = args.get("sparse_preconditioner_type")
    try:
        preconditioner_type = normalize_sparse_preconditioner_type(preconditioner_type)
    except Exception:
        preconditioner_type = str(preconditioner_type)

    guidance = [
        "Practical guidance: preconditioner failures often mean the covariance graph is too dense or the factor is too expressive for available memory.",
        "First check the compact-support kernel length scale/support radius and keep matrix density low before tuning solver parameters.",
        "Run a small preconditioner build sweep before a full solve run; a buildable preconditioner can still be slow to apply.",
    ]
    if preconditioner_type == "ilu":
        guidance.append(
            "For ILU, sweep `sparse_preconditioner_drop_tol` and `sparse_preconditioner_fill_factor`; looser drop tolerances and smaller fill factors are more likely to fit, while stronger factors may reduce solve time."
        )
    elif preconditioner_type in {"ichol", "ichol0"}:
        guidance.append(
            "For IC/IChol, install the optional backend with `pip install ilupp`; if thresholded IC does not fit, try softer fill/threshold settings or `ichol0`, then verify actual solve time."
        )
    elif preconditioner_type in {"block_jacobi", "additive_schwarz"}:
        guidance.append(
            "For block/local preconditioners, sweep block size and overlap; these may fit easily but can be weaker than ILU on large covariance systems."
        )
    guidance.append(
        "For repeated nearby K+V updates, `sparse_krylov_warm_start=True` and a nontrivial `sparse_preconditioner_refresh_interval` can avoid rebuilding every solve."
    )
    guidance.append(
        "If MINRES returns with a poor raw residual, try a stricter `sparse_minres_tol` before judging the method."
    )
    return " ".join(guidance)


def resolve_gp2scale_linalg_mode(mode, args=None):
    """Split a mode string like ``"sparseCGpre_amg"`` into ``("sparseCGpre", args)``.

    Stores the inferred preconditioner type in ``args["sparse_preconditioner_type"]``.
    Raises ``ValueError`` if the inferred type conflicts with an explicit one.
    """
    args = dict(_normalize_args(args))
    if not isinstance(mode, str):
        return mode, args

    mode_lower = mode.lower()
    alias_prefixes = {
        "sparsecgpre_": "sparseCGpre",
        "sparseminrespre_": "sparseMINRESpre",
    }
    for prefix, canonical_mode in alias_prefixes.items():
        if not mode_lower.startswith(prefix):
            continue
        inferred_type = normalize_sparse_preconditioner_type(mode_lower[len(prefix):])
        explicit_type = args.get("sparse_preconditioner_type")
        if explicit_type is not None:
            explicit_type = normalize_sparse_preconditioner_type(explicit_type)
            if explicit_type != inferred_type:
                raise ValueError(
                    f"Conflicting sparse preconditioner specifications: mode {mode!r} "
                    f"implies {inferred_type!r}, but args['sparse_preconditioner_type'] "
                    f"is {explicit_type!r}."
                )
        args["sparse_preconditioner_type"] = inferred_type
        return canonical_mode, args

    return mode, args


def _as_symmetric_csr(KV):
    assert sparse.issparse(KV)
    A = KV.tocsr().astype(np.float64)
    return ((A + A.T) * 0.5).tocsr()


def _shifted_dense_cholesky(matrix, args=None, shift_key="sparse_preconditioner_shift"):
    args = _normalize_args(args)
    base_shift = float(args.get(shift_key, args.get("sparse_preconditioner_shift", 0.0)))
    growth = float(args.get("sparse_preconditioner_shift_growth", 10.0))
    attempts = int(args.get("sparse_preconditioner_shift_attempts", 5))
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
    last_exc = None
    shift = base_shift

    for _ in range(max(attempts, 1)):
        try:
            local_matrix = matrix if shift == 0.0 else matrix + shift * eye
            return cho_factor(local_matrix, lower=True), shift
        except Exception as exc:
            last_exc = exc
            shift = 1e-10 if shift == 0.0 else shift * growth

    raise np.linalg.LinAlgError(f"Local Cholesky factorization failed after shifted retries: {last_exc}")


def _build_graph_blocks(KV, block_size):
    A = _as_symmetric_csr(KV)
    n = A.shape[0]
    block_size = max(int(block_size), 1)
    assigned = np.zeros(n, dtype=bool)
    blocks = []

    for seed in range(n):
        if assigned[seed]:
            continue
        queue = deque([seed])
        queued = {seed}
        block = []

        while queue and len(block) < block_size:
            node = queue.popleft()
            if assigned[node]:
                continue
            assigned[node] = True
            block.append(node)
            start, end = A.indptr[node], A.indptr[node + 1]
            for neighbor in A.indices[start:end]:
                neighbor = int(neighbor)
                if neighbor == node or assigned[neighbor] or neighbor in queued:
                    continue
                queued.add(neighbor)
                queue.append(neighbor)

        blocks.append(np.array(block, dtype=int))

    return blocks


def _expand_block_overlap(KV, block, overlap):
    A = _as_symmetric_csr(KV)
    overlap = max(int(overlap), 0)
    expanded = set(int(i) for i in block)
    frontier = set(expanded)

    for _ in range(overlap):
        if not frontier:
            break
        next_frontier = set()
        for node in frontier:
            start, end = A.indptr[node], A.indptr[node + 1]
            for neighbor in A.indices[start:end]:
                neighbor = int(neighbor)
                if neighbor not in expanded:
                    expanded.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    return np.array(sorted(expanded), dtype=int)


def _build_blockwise_operator(shape, local_models, dtype):
    def matvec(x):
        x = np.asarray(x, dtype=dtype)
        result = np.zeros(shape[0], dtype=dtype)
        for model in local_models:
            local_rhs = x[model["indices"]]
            local_solution = cho_solve(model["factor"], local_rhs)
            result[model["indices"]] += local_solution
        return result

    return sparse.linalg.LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=dtype)


def _build_block_jacobi_preconditioner(KV, args=None):
    args = _normalize_args(args)
    A = _as_symmetric_csr(KV)
    block_size = int(args.get("sparse_preconditioner_block_size", 64))
    blocks = _build_graph_blocks(A, block_size)
    local_models = []
    for block in blocks:
        local_matrix = A[block][:, block].toarray()
        factor, shift = _shifted_dense_cholesky(local_matrix, args)
        local_models.append({"indices": block, "factor": factor, "shift": shift})

    factor = {
        "type": "block_jacobi",
        "blocks": blocks,
        "local_models": local_models,
    }
    operator = _build_blockwise_operator(A.shape, local_models, A.dtype)
    return factor, operator


def _build_additive_schwarz_preconditioner(KV, args=None):
    args = _normalize_args(args)
    A = _as_symmetric_csr(KV)
    block_size = int(args.get("sparse_preconditioner_block_size", 64))
    overlap = int(args.get("sparse_preconditioner_schwarz_overlap", 1))
    base_blocks = _build_graph_blocks(A, block_size)
    local_models = []

    for base_block in base_blocks:
        local_indices = _expand_block_overlap(A, base_block, overlap)
        local_matrix = A[local_indices][:, local_indices].toarray()
        factor, shift = _shifted_dense_cholesky(local_matrix, args)
        local_models.append(
            {
                "indices": local_indices,
                "core": base_block,
                "factor": factor,
                "shift": shift,
            }
        )

    factor = {
        "type": "additive_schwarz",
        "blocks": base_blocks,
        "overlap": overlap,
        "local_models": local_models,
    }
    operator = _build_blockwise_operator(A.shape, local_models, A.dtype)
    return factor, operator


def _build_ic0_factor(KV, args=None):
    """Pure-Python IC(0) preconditioner.

    Correct but slow; intended only as a legacy/debugging path. Falls back to
    increasing diagonal shifts if a non-positive pivot is encountered.
    """
    args = _normalize_args(args)
    A = _as_symmetric_csr(KV)
    n = A.shape[0]
    base_shift = float(args.get("sparse_preconditioner_ic_shift", args.get("sparse_preconditioner_shift", 0.0)))
    growth = float(args.get("sparse_preconditioner_ic_shift_growth", args.get("sparse_preconditioner_shift_growth", 10.0)))
    attempts = int(args.get("sparse_preconditioner_ic_shift_attempts", args.get("sparse_preconditioner_shift_attempts", 5)))
    last_exc = None
    shift = base_shift

    for _ in range(max(attempts, 1)):
        try:
            A_shifted = A if shift == 0.0 else A + sparse.eye(n, format="csr", dtype=A.dtype) * shift
            lower = sparse.tril(A_shifted, format="csr")
            diag = np.zeros(n, dtype=np.float64)
            rows = []

            for i in range(n):
                start, end = lower.indptr[i], lower.indptr[i + 1]
                row_cols = lower.indices[start:end]
                row_vals = lower.data[start:end]
                diag_val = None
                original_entries = {}

                for col, value in zip(row_cols, row_vals):
                    col = int(col)
                    value = float(value)
                    if col < i:
                        original_entries[col] = value
                    elif col == i:
                        diag_val = value

                if diag_val is None:
                    raise np.linalg.LinAlgError(f"IC(0) encountered a missing diagonal at row {i}")

                computed_entries = {}
                for j in sorted(original_entries):
                    row_j = rows[j]
                    if len(computed_entries) < len(row_j):
                        coupling = sum(
                            value * row_j.get(k, 0.0)
                            for k, value in computed_entries.items()
                        )
                    else:
                        coupling = sum(
                            computed_entries.get(k, 0.0) * value
                            for k, value in row_j.items()
                        )
                    computed_entries[j] = (original_entries[j] - coupling) / diag[j]

                diag_sq = diag_val - sum(value * value for value in computed_entries.values())
                if diag_sq <= 0.0:
                    raise np.linalg.LinAlgError(f"IC(0) produced a non-positive pivot at row {i}")
                diag[i] = np.sqrt(diag_sq)
                rows.append(computed_entries)

            indptr = [0]
            indices = []
            data = []
            for i, row in enumerate(rows):
                for j, value in row.items():
                    indices.append(int(j))
                    data.append(float(value))
                indices.append(i)
                data.append(float(diag[i]))
                indptr.append(len(indices))

            L = sparse.csr_matrix(
                (
                    np.asarray(data, dtype=np.float64),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(indptr, dtype=np.int32),
                ),
                shape=A.shape,
            )
            LT = L.transpose().tocsr()

            def solve(vector):
                vector = np.asarray(vector, dtype=np.float64)
                y = spsolve_triangular(L, vector, lower=True)
                return spsolve_triangular(LT, y, lower=False)

            factor = {
                "type": "native_incomplete_cholesky",
                "diag": diag,
                "rows": rows,
                "L": L,
                "LT": LT,
                "shift": shift,
            }
            operator = _build_dtype_adapted_operator(A.shape, solve, factor_dtype=np.float64)
            return factor, operator
        except Exception as exc:
            last_exc = exc
            shift = 1e-10 if shift == 0.0 else shift * growth

    raise np.linalg.LinAlgError(f"IC(0) preconditioner construction failed after shifted retries: {last_exc}")


def _build_dtype_adapted_operator(shape, solve, factor_dtype, operator_dtype=np.float64):
    factor_dtype = np.dtype(factor_dtype)
    operator_dtype = np.dtype(operator_dtype)

    def _apply(vec):
        arr = np.asarray(vec, dtype=operator_dtype)
        if arr.ndim == 1:
            solved = solve(np.asarray(arr, dtype=factor_dtype))
            return np.asarray(solved, dtype=operator_dtype)
        columns = [
            np.asarray(solve(np.asarray(arr[:, i], dtype=factor_dtype)), dtype=operator_dtype)
            for i in range(arr.shape[1])
        ]
        return np.column_stack(columns)

    return sparse.linalg.LinearOperator(
        shape,
        matvec=_apply,
        rmatvec=_apply,
        matmat=_apply,
        dtype=operator_dtype,
    )


def _build_ilu_preconditioner(KV, args=None):
    args = _normalize_args(args)
    A = KV.tocsc()
    spilu_kwargs = {
        "drop_tol": args.get("sparse_preconditioner_drop_tol", 1e-8),
        "fill_factor": args.get("sparse_preconditioner_fill_factor", 10),
    }
    if "sparse_preconditioner_drop_rule" in args:
        spilu_kwargs["drop_rule"] = args["sparse_preconditioner_drop_rule"]
    if "sparse_preconditioner_permc_spec" in args:
        spilu_kwargs["permc_spec"] = args["sparse_preconditioner_permc_spec"]
    if "sparse_preconditioner_diag_pivot_thresh" in args:
        spilu_kwargs["diag_pivot_thresh"] = args["sparse_preconditioner_diag_pivot_thresh"]

    factor = sparse.linalg.spilu(A, **spilu_kwargs)
    operator = _build_dtype_adapted_operator(A.shape, factor.solve, factor_dtype=A.dtype)
    return factor, operator


def _shift_retry_ilupp_factor(A, build_fn, label, args):
    """Try ``build_fn(A)`` with progressively larger diagonal shifts.

    Mirrors the shift-retry policy of the native IC(0) path
    (:func:`_build_ic0_factor`) using the shared ``sparse_preconditioner_shift``
    / ``sparse_preconditioner_shift_growth`` / ``sparse_preconditioner_shift_attempts``
    args (defaults: 0.0 / 10.0 / 5). Useful when an ``ilupp`` factorization
    fails because of a non-PD or near-singular pivot in K+V.
    """
    args = _normalize_args(args)
    shift = float(args.get("sparse_preconditioner_shift", 0.0))
    growth = float(args.get("sparse_preconditioner_shift_growth", 10.0))
    attempts = int(args.get("sparse_preconditioner_shift_attempts", 5))
    n = A.shape[0]
    last_exc = None
    for _ in range(max(attempts, 1)):
        A_try = A if shift == 0.0 else (A + shift * sparse.eye(n, format="csr"))
        try:
            return build_fn(A_try)
        except Exception as exc:
            last_exc = exc
            shift = max(shift * growth, 1e-12)
    raise np.linalg.LinAlgError(
        f"{label} preconditioner construction failed after "
        f"{attempts} shifted retries: {last_exc}"
    )


def _build_ichol0_preconditioner(KV, args=None):
    try:
        import ilupp
    except ImportError as exc:
        _raise_missing_ilupp("ichol0", exc)

    A = _as_symmetric_csr(KV).astype(np.float64)
    factor = _shift_retry_ilupp_factor(A, ilupp.IChol0Preconditioner, "IChol0", args)
    operator = _build_dtype_adapted_operator(A.shape, factor.dot, factor_dtype=np.float64)
    return factor, operator


def _build_ichol_preconditioner(KV, args=None):
    args = _normalize_args(args)
    try:
        import ilupp
    except ImportError as exc:
        _raise_missing_ilupp("ichol", exc)

    A = _as_symmetric_csr(KV).astype(np.float64)
    add_fill_in = int(args.get("sparse_preconditioner_ichol_fill_in", 16))
    threshold = float(args.get("sparse_preconditioner_ichol_threshold", 1e-4))

    def _build(A_try):
        return ilupp.ICholTPreconditioner(
            A_try, add_fill_in=add_fill_in, threshold=threshold
        )

    factor = _shift_retry_ilupp_factor(A, _build, "ICholT", args)
    operator = _build_dtype_adapted_operator(A.shape, factor.dot, factor_dtype=np.float64)
    return factor, operator


def _build_amg_preconditioner(KV, args=None):
    args = _normalize_args(args)
    try:
        import pyamg
    except ImportError as exc:
        raise ImportError("pyamg is required for sparse_preconditioner_type='amg'") from exc

    A = _as_symmetric_csr(KV)
    solver_kwargs = {
        "max_levels": int(args.get("sparse_preconditioner_amg_max_levels", 10)),
        "max_coarse": int(args.get("sparse_preconditioner_amg_max_coarse", 500)),
    }
    if "sparse_preconditioner_amg_strength" in args:
        solver_kwargs["strength"] = args["sparse_preconditioner_amg_strength"]
    if "sparse_preconditioner_amg_symmetry" in args:
        solver_kwargs["symmetry"] = args["sparse_preconditioner_amg_symmetry"]
    if "sparse_preconditioner_amg_presmoother" in args:
        solver_kwargs["presmoother"] = args["sparse_preconditioner_amg_presmoother"]
    if "sparse_preconditioner_amg_postsmoother" in args:
        solver_kwargs["postsmoother"] = args["sparse_preconditioner_amg_postsmoother"]

    factor = pyamg.smoothed_aggregation_solver(A, **solver_kwargs)
    cycle = args.get("sparse_preconditioner_amg_cycle", "V")
    operator = factor.aspreconditioner(cycle=cycle)
    return factor, operator


def calculate_sparse_preconditioner(KV, args=None):
    """Return ``(factor, operator)`` for the requested sparse preconditioner.

    The ``operator`` is a ``scipy.sparse.linalg.LinearOperator`` ready to pass
    as the ``M=`` argument to ``cg`` / ``minres``.  The type is selected by
    ``args["sparse_preconditioner_type"]`` (default ``"ilu"``); supported values
    are listed in :func:`normalize_sparse_preconditioner_type`.
    """
    args = _normalize_args(args)
    assert sparse.issparse(KV)
    logger.debug("calculate_sparse_preconditioner")
    preconditioner_type = normalize_sparse_preconditioner_type(
        args.get("sparse_preconditioner_type", "ilu")
    )

    builders = {
        "ilu": _build_ilu_preconditioner,
        "native_incomplete_cholesky": _build_ic0_factor,
        "ichol0": _build_ichol0_preconditioner,
        "ichol": _build_ichol_preconditioner,
        "block_jacobi": _build_block_jacobi_preconditioner,
        "additive_schwarz": _build_additive_schwarz_preconditioner,
        "amg": _build_amg_preconditioner,
    }

    if preconditioner_type not in builders:
        raise ValueError(
            "Unknown sparse preconditioner type "
            f"{preconditioner_type!r}. Expected one of {sorted(builders)}."
        )

    factor, operator = builders[preconditioner_type](KV, args=args)
    if isinstance(factor, dict):
        factor.setdefault("type", preconditioner_type)
    return factor, operator


# --------------------------------------------------------------------------
# Sparse iterative solvers
# --------------------------------------------------------------------------

def _resolve_krylov_mode(args=None):
    args = _normalize_args(args)
    if bool(args.get("sparse_block_krylov", False)):
        return "block"
    mode = str(args.get("sparse_krylov_mode", "single")).lower()
    aliases = {
        "single": "single",
        "columnwise": "single",
        "block": "block",
        "block_cg": "block",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown sparse Krylov mode {mode!r}. Expected one of {{'single', 'block'}}."
        )
    return aliases[mode]


def _resolve_krylov_maxiter(args=None, solver_key=None):
    args = _normalize_args(args)
    if solver_key is not None and solver_key in args:
        value = args[solver_key]
        return None if value is None else int(value)
    if "sparse_krylov_maxiter" in args:
        value = args["sparse_krylov_maxiter"]
        return None if value is None else int(value)
    return None


def _normalize_rhs(vec):
    vec = np.asarray(vec, dtype=np.float64)
    if np.ndim(vec) == 1:
        vec = vec.reshape(len(vec), 1)
    return vec


def _normalize_initial_guess(x0, shape):
    """Reshape/zero-pad ``x0`` so it matches ``(n_rows, n_rhs_columns)``.

    Replaces the historic 1-d-only ``np.append`` zero-pad that silently
    flattened 2-d guesses.
    """
    if x0 is None:
        return None
    guess = np.asarray(x0, dtype=np.float64)
    if guess.ndim == 1:
        guess = guess.reshape(-1, 1)
    if guess.shape[1] == 1 and shape[1] > 1:
        guess = np.repeat(guess, shape[1], axis=1)
    if guess.shape[1] != shape[1]:
        raise ValueError(
            f"Initial guess has {guess.shape[1]} columns but the RHS has {shape[1]} columns."
        )
    if guess.shape[0] < shape[0]:
        padding = np.zeros((shape[0] - guess.shape[0], guess.shape[1]), dtype=guess.dtype)
        guess = np.vstack([guess, padding])
    elif guess.shape[0] > shape[0]:
        guess = guess[:shape[0], :]
    return guess


def _column_initial_guess(x0, column_index):
    if x0 is None:
        return None
    if x0.ndim == 1:
        return x0
    return x0[:, min(column_index, x0.shape[1] - 1)]


def _apply_linear_operator(operator, matrix):
    if matrix.size == 0:
        return np.zeros_like(matrix)
    if hasattr(operator, "matmat"):
        try:
            return np.asarray(operator.matmat(matrix), dtype=np.float64)
        except Exception:
            pass
    return np.column_stack([np.asarray(operator @ matrix[:, i], dtype=np.float64)
                            for i in range(matrix.shape[1])])


def _apply_preconditioner(M, residual):
    if M is None:
        return residual.copy()
    return _apply_linear_operator(M, residual)


def _block_conjugate_gradient(KV, vec, cg_tol, x0=None, M=None, maxiter=None):
    """Block conjugate gradient for symmetric PD systems with multiple RHS.

    Falls back to scipy's single-RHS ``cg`` when there is only one column.
    Returns ``(X, exit_code)`` where ``exit_code == 0`` means converged.
    """
    vec = _normalize_rhs(vec)
    x0 = _normalize_initial_guess(x0, vec.shape)
    n, rhs_count = vec.shape
    if rhs_count == 1:
        rhs = vec[:, 0]
        initial_guess = None if x0 is None else x0[:, 0]
        solution, exit_code = cg(KV, rhs, M=M, rtol=cg_tol, x0=initial_guess, maxiter=maxiter)
        return solution.reshape(n, 1), exit_code

    X = np.zeros((n, rhs_count), dtype=np.float64) if x0 is None else x0.copy()
    R = vec - _apply_linear_operator(KV, X)
    rhs_norm = np.linalg.norm(vec, axis=0)
    rhs_norm[rhs_norm == 0.0] = 1.0
    rel_residual = np.linalg.norm(R, axis=0) / rhs_norm
    if np.max(rel_residual) <= cg_tol:
        return X, 0

    Z = _apply_preconditioner(M, R)
    P = Z.copy()
    G = R.T @ Z
    if maxiter is None:
        maxiter = 10 * n
    last_exit_code = 1

    for _ in range(max(maxiter, 1)):
        AP = _apply_linear_operator(KV, P)
        H = P.T @ AP
        try:
            alpha = np.linalg.solve(H, G)
        except np.linalg.LinAlgError:
            last_exit_code = 2
            break

        X = X + P @ alpha
        R = R - AP @ alpha
        rel_residual = np.linalg.norm(R, axis=0) / rhs_norm
        if np.max(rel_residual) <= cg_tol:
            return X, 0

        Z = _apply_preconditioner(M, R)
        G_new = R.T @ Z
        try:
            beta = np.linalg.solve(G, G_new)
        except np.linalg.LinAlgError:
            last_exit_code = 3
            break
        P = Z + P @ beta
        G = G_new

    return X, last_exit_code


def calculate_random_logdet(KV, compute_device, args=None):
    """Estimate log|det KV| for a sparse matrix via stochastic Lanczos quadrature (imate)."""
    args = _normalize_args(args)
    assert sparse.issparse(KV), "KV must be sparse for stochastic logdet"
    logger.debug("calculate_random_logdet")
    from imate import logdet as imate_logdet
    st = time.time()
    gpu = compute_device == "gpu" and _imate_gpu_enabled(args)

    lanczos_degree = 20
    error_rtol = 0.01
    verbose = False
    print_info = False

    if "random_logdet_lanczos_degree" in args: lanczos_degree = args["random_logdet_lanczos_degree"]
    if "random_logdet_error_rtol" in args: error_rtol = args["random_logdet_error_rtol"]
    if "random_logdet_verbose" in args: verbose = args["random_logdet_verbose"]
    if "random_logdet_print_info" in args: print_info = args["random_logdet_print_info"]

    logdet, info_slq = imate_logdet(KV, method='slq', min_num_samples=10, max_num_samples=5000,
                                    lanczos_degree=lanczos_degree, error_rtol=error_rtol, gpu=gpu,
                                    return_info=True, plot=False, verbose=verbose, orthogonalize=0)
    logger.debug("Stochastic Lanczos logdet() compute time: {} seconds", time.time() - st)
    if print_info: logger.debug(info_slq)
    assert np.isscalar(logdet), "stochastic logdet result must be scalar"
    return logdet


def calculate_sparse_minres(KV, vec, x0=None, M=None, args=None):
    """Solve the sparse symmetric system ``KV x = vec`` with MINRES.

    Honors ``args["sparse_minres_tol"]`` and ``args["sparse_minres_maxiter"]``
    (or the generic ``args["sparse_krylov_maxiter"]``).  ``x0`` is reshaped /
    zero-padded to match ``vec`` before each per-column solve.
    """
    args = _normalize_args(args)
    assert sparse.issparse(KV), "KV must be sparse for MINRES"
    st = time.time()
    logger.debug("MINRES solve in progress ...")
    minres_tol = 1e-5
    if "sparse_minres_tol" in args:
        minres_tol = args["sparse_minres_tol"]
        logger.debug("sparse_minres_tol changed to ", minres_tol)
    maxiter = _resolve_krylov_maxiter(args, solver_key="sparse_minres_maxiter")

    vec = _normalize_rhs(vec)
    x0 = _normalize_initial_guess(x0, vec.shape)
    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        initial_guess = _column_initial_guess(x0, i)
        res[:, i], exit_code = minres(KV, vec[:, i], M=M, rtol=minres_tol, x0=initial_guess, maxiter=maxiter)
        if exit_code != 0: warnings.warn(f"MINRES not successful (exit_code={exit_code})")
    logger.debug("MINRES compute time: {} seconds.", time.time() - st)
    assert np.ndim(res) == 2, "MINRES result must be 2-d"
    return res


def calculate_sparse_conj_grad(KV, vec, x0=None, M=None, args=None):
    """Solve the sparse SPD system ``KV x = vec`` with conjugate gradients.

    Honors ``args["sparse_cg_tol"]`` (or legacy ``cg_minres_tol`` /
    ``sparse_minres_tol``), ``args["sparse_cg_maxiter"]``, and a block-CG mode
    selected by ``args["sparse_block_krylov"]`` or
    ``args["sparse_krylov_mode"]``.  ``x0`` is reshaped / zero-padded to match
    ``vec`` before each per-column solve.
    """
    args = _normalize_args(args)
    assert sparse.issparse(KV), "KV must be sparse for CG"
    st = time.time()
    logger.debug("CG solve in progress ...")
    cg_tol = 1e-5
    # Backward-compatible tolerance resolution:
    # - sparse_cg_tol is the corrected, explicit CG key
    # - cg_minres_tol appeared in published docs
    # - sparse_minres_tol was accidentally read by the published CG path
    if "sparse_cg_tol" in args:
        cg_tol = args["sparse_cg_tol"]
    elif "cg_minres_tol" in args:
        cg_tol = args["cg_minres_tol"]
    elif "sparse_minres_tol" in args:
        cg_tol = args["sparse_minres_tol"]
    krylov_mode = _resolve_krylov_mode(args)
    maxiter = _resolve_krylov_maxiter(args, solver_key="sparse_cg_maxiter")
    vec = _normalize_rhs(vec)
    x0 = _normalize_initial_guess(x0, vec.shape)

    if krylov_mode == "block" and vec.shape[1] > 1:
        block_size = int(args.get("sparse_krylov_block_size", vec.shape[1]))
        block_size = max(1, min(block_size, vec.shape[1]))
        res = np.zeros(vec.shape)
        exit_codes = []
        for start in range(0, vec.shape[1], block_size):
            stop = min(start + block_size, vec.shape[1])
            block_x0 = None if x0 is None else x0[:, start:stop]
            try:
                block_solution, exit_code = _block_conjugate_gradient(
                    KV,
                    vec[:, start:stop],
                    cg_tol,
                    x0=block_x0,
                    M=M,
                    maxiter=maxiter,
                )
            except Exception as exc:
                warnings.warn(
                    "Block CG failed; falling back to columnwise CG for this RHS block: "
                    f"{type(exc).__name__}: {exc}"
                )
                block_solution = np.zeros((KV.shape[0], stop - start))
                exit_code = 4
                for block_index, rhs_index in enumerate(range(start, stop)):
                    initial_guess = _column_initial_guess(x0, rhs_index)
                    block_solution[:, block_index], _ = cg(
                        KV,
                        vec[:, rhs_index],
                        M=M,
                        rtol=cg_tol,
                        x0=initial_guess,
                        maxiter=maxiter,
                    )
            res[:, start:stop] = block_solution
            exit_codes.append(exit_code)
        if any(code != 0 for code in exit_codes):
            warnings.warn(f"Block CG not fully successful (exit_codes={exit_codes})")
        logger.debug("CG compute time: {} seconds.", time.time() - st)
        assert np.ndim(res) == 2, "CG result must be 2-d"
        return res

    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        initial_guess = _column_initial_guess(x0, i)
        res[:, i], exit_code = cg(KV, vec[:, i], M=M, rtol=cg_tol, x0=initial_guess, maxiter=maxiter)
        if exit_code != 0: warnings.warn(f"CG not successful (exit_code={exit_code})")
    logger.debug("CG compute time: {} seconds.", time.time() - st)
    assert np.ndim(res) == 2, "CG result must be 2-d"
    return res


def calculate_sparse_solve(KV, vec, args=None):
    """Solve the sparse system ``KV x = vec`` with a direct sparse solver."""
    assert sparse.issparse(KV), "KV must be sparse for sparse direct solve"
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    st = time.time()
    logger.debug("Sparse solve in progress ...")
    res = sparse.linalg.spsolve(KV.tocsc(), vec)
    logger.debug("Sparse solve compute time: {} seconds.", time.time() - st)
    if np.ndim(res) == 1: res = res.reshape(len(res), 1)
    return res


# --------------------------------------------------------------------------
# Cholesky rank-1 / rank-n updates
# --------------------------------------------------------------------------

def cholesky_update_rank_1(L, b, c, compute_device="cpu", args=None):
    """
    Extend a Cholesky factor by one row/column in O(n²).

    Given the lower-triangular Cholesky factor ``L`` of ``A``, returns the
    Cholesky factor of ``[[A, b], [b.T, c]]``.

    Parameters
    ----------
    L : (n, n) ndarray
        Lower-triangular Cholesky factor.
    b : (n,) ndarray
        Cross-covariance vector between existing and new point.
    c : float
        Variance of the new point.
    compute_device : {"cpu", "gpu"}, optional
    args : dict, optional
        Extra options; ``"GPU_engine"`` selects ``"torch"`` or ``"cupy"``.

    Returns
    -------
    L_prime : (n+1, n+1) ndarray
        Updated lower-triangular Cholesky factor.

    Raises
    ------
    NonPositiveDefiniteError
        If the augmented matrix is not positive definite.
    """
    if compute_device == "cpu": L_prime = cholesky_update_rank_1_numpy(L, b, c)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            L_prime = cholesky_update_rank_1_torch(L, b, c, args=args)
        elif engine == "cupy":  # pragma: no cover
            L_prime = cholesky_update_rank_1_cupy(L, b, c)
        else: L_prime = None
    else: raise Exception("No valid compute device found.")
    return L_prime


def cholesky_update_rank_1_numpy(L, b, c, args=None):
    """
    CPU (NumPy) implementation of :func:`cholesky_update_rank_1`.

    Parameters
    ----------
    L : (n, n) ndarray
        Lower-triangular Cholesky factor.
    b : (n,) ndarray
        Cross-covariance vector.
    c : float
        New-point variance.

    Returns
    -------
    L_prime : (n+1, n+1) ndarray
    """
    # Solve Lv = b for v
    v = solve_triangular(L, b, lower=True, check_finite=False)

    # Compute d
    disc = c - np.dot(v, v)
    if disc <= 0:
        raise NonPositiveDefiniteError(_rank1_update_non_pd_message(disc))
    d = np.sqrt(disc)

    # Form the new L'
    L_prime = np.block([
        [L, np.zeros((len(L), 1), dtype=L.dtype)],
        [v.T, d]
    ])
    return L_prime


def cholesky_update_rank_1_torch(L, b, c, args=None):   # pragma: no cover
    """
    Rank-1 Cholesky update on GPU using PyTorch.

    Parameters
    ----------
    L : (n, n) lower-triangular Cholesky factor on GPU
    b : (n,) or (n,1) vector on GPU
    c : scalar

    Returns
    -------
    L_prime : (n+1, n+1) updated lower-triangular Cholesky factor on GPU
    """
    import torch
    device = _torch_gpu_device(args)
    # Match b/c to L's dtype.
    target = L.dtype if hasattr(L, "dtype") else np.float64
    L_t = torch.as_tensor(L, device=device)
    b_t = torch.as_tensor(np.asarray(b, dtype=target), device=device)
    c_val = np.asarray(c, dtype=target).item() if np.ndim(c) == 0 else np.asarray(c, dtype=target)

    v = torch.linalg.solve_triangular(L_t, b_t.unsqueeze(1), upper=False).squeeze(1)

    # Compute d = sqrt(c - v^T v) with PD check
    disc = c_val - torch.dot(v, v)
    if float(disc) <= 0:
        raise NonPositiveDefiniteError(_rank1_update_non_pd_message(disc))
    d = torch.sqrt(disc)

    # Form new L'
    n = L_t.shape[0]
    L_prime = torch.zeros((n+1, n+1), device=L_t.device, dtype=L_t.dtype)
    L_prime[:n, :n] = L_t
    L_prime[:n, n] = 0.0
    L_prime[n, :n] = v
    L_prime[n, n] = d

    return L_prime.cpu().numpy()


def cholesky_update_rank_1_cupy(L, b, c):   # pragma: no cover
    """
    Rank-1 Cholesky update on GPU using CuPy.

    Parameters
    ----------
    L : (n, n) lower-triangular Cholesky factor (cupy array)
    b : (n,) vector (cupy array)
    c : scalar

    Returns
    -------
    L_prime : (n+1, n+1) updated lower-triangular Cholesky factor
    """
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    target = L.dtype
    L = cp.asarray(L)
    b = cp.asarray(np.asarray(b, dtype=target))
    c = np.asarray(c, dtype=target).item() if np.ndim(c) == 0 else np.asarray(c, dtype=target)
    # Solve L v = b
    v = cp_solve_triangular(L, b[:, None], lower=True).squeeze(1)

    # Compute d
    disc = c - cp.dot(v, v)
    if float(disc) <= 0:
        raise NonPositiveDefiniteError(_rank1_update_non_pd_message(disc))
    d = cp.sqrt(disc)

    # Form new L'
    n = L.shape[0]
    L_prime = cp.zeros((n+1, n+1), dtype=L.dtype)
    L_prime[:n, :n] = L
    L_prime[:n, n] = 0
    L_prime[n, :n] = v
    L_prime[n, n] = d

    return cp.asnumpy(L_prime)


def cholesky_update_rank_n(L, b, c, compute_device="cpu", args=None):
    """Extend ``L`` by ``b.shape[1]`` columns via sequential rank-1 Cholesky updates."""
    L_prime = L.copy()
    for i in range(b.shape[1]):
        L_prime = cholesky_update_rank_1(
            L_prime,
            np.append(b[:, i], c[0:i, i]),
            c[i, i],
            compute_device=compute_device,
            args=args,
        )
    return L_prime


# --------------------------------------------------------------------------
# Dense logdet / inverse / solve
# --------------------------------------------------------------------------

def calculate_logdet(A, compute_device='cpu', args=None):
    """Return log|det A|; GPU path tries torch then cupy, falls back to numpy."""
    logger.debug("calculate_logdet")
    if compute_device == "cpu":
        s, logdet = np.linalg.slogdet(A)
        assert np.isscalar(logdet), "logdet must be scalar"
        return logdet
    elif compute_device == "gpu":
        try:
            engine = get_gpu_engine(args)
            if engine == "torch":  # pragma: no cover
                import torch
                device = _torch_gpu_device(args)
                A_dev = torch.as_tensor(A, device=device)
                sign, logdet = torch.slogdet(A_dev)
                logdet = np.nan_to_num(logdet.detach().cpu().numpy())
                assert np.isscalar(logdet), "logdet must be scalar"
                return logdet
            if engine == "cupy":  # pragma: no cover
                import cupy as cp
                A_dev = cp.asarray(A)
                sign, logdet = cp.linalg.slogdet(A_dev)
                logdet = np.nan_to_num(cp.asnumpy(logdet))
                assert np.isscalar(logdet), "logdet must be scalar"
                return logdet
            raise RuntimeError("No usable GPU backend available")
        except Exception:
            warnings.warn(
                "I encountered a problem using the GPU. Falling back to Numpy and the CPU.")
            s, logdet = np.linalg.slogdet(A)
            assert np.isscalar(logdet), "logdet must be scalar"
            return logdet
    else:
        sign, logdet = np.linalg.slogdet(A)
        assert np.isscalar(logdet), "logdet must be scalar"
        return logdet


def update_logdet(old_logdet, old_inv, new_matrix, compute_device="cpu", args=None):
    """Update log|det| after augmenting the matrix via the Schur complement identity."""
    logger.debug("update_logdet")
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    res = old_logdet + calculate_logdet(kk - k @ old_inv @ k.T, compute_device=compute_device, args=args)
    assert np.isscalar(res), "updated logdet must be scalar"
    return res


def calculate_inv(A, compute_device='cpu', args=None):
    """Return the inverse of square matrix ``A``; GPU path supports torch and cupy."""
    assert isinstance(A, np.ndarray), "A must be np.ndarray for matrix inversion"
    assert np.ndim(A) == 2, "A must be 2-d for matrix inversion"
    logger.debug("calculate_inv")
    if compute_device == "cpu":
        return np.linalg.inv(A)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A_dev = torch.as_tensor(A, device=device)
            B = torch.inverse(A_dev)
            return B.detach().cpu().numpy()
        if engine == "cupy":  # pragma: no cover
            import cupy as cp
            A_dev = cp.asarray(A)
            return cp.asnumpy(cp.linalg.inv(A_dev))
        return np.linalg.inv(A)
    else:
        return np.linalg.inv(A)


def calculate_inv_from_chol(L, compute_device="cpu", args=None):
    """Return A⁻¹ by solving A x = I using the pre-computed Cholesky factor ``L``."""
    logger.debug("calculate_inv_from_chol")
    if compute_device == "cpu": A_inv = cho_solve((L, True), np.eye(L.shape[0]))
    elif compute_device == "gpu": A_inv = calculate_Chol_solve(L, np.eye(L.shape[0]), compute_device="gpu", args=args)
    else: raise Exception("No valid compute device found.")
    return A_inv


def update_inv(old_inv, new_matrix, compute_device="cpu", args=None):
    """Update A⁻¹ after augmenting the matrix via the block-matrix inversion lemma."""
    logger.debug("update_inv")
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    X = calculate_inv(kk - k @ old_inv @ k.T, compute_device=compute_device, args=args)
    F = -old_inv @ k.T @ X
    new_inv = np.block([[old_inv + old_inv @ k.T @ X @ k @ old_inv, F],
                        [F.T, X]])
    return new_inv


def solve(A, b, compute_device='cpu', args=None):
    """Solve ``A x = b``; falls back to least-squares if ``A`` is singular (cpu path)."""
    assert isinstance(A, np.ndarray), "A must be np.ndarray for solve"
    logger.debug("solve")
    if np.ndim(b) == 1: b = b.reshape(len(b), 1)
    if b.dtype != A.dtype:
        b = b.astype(A.dtype, copy=False)
    if compute_device == "cpu":
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if np.ndim(x) == 1: x = x.reshape(len(x), 1)
        assert np.ndim(x) == np.ndim(b), "solve result and rhs have mismatched dimensions"
        return x
    elif compute_device == "gpu":  # pragma: no cover
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            At = torch.as_tensor(A, device=device)
            bt = torch.as_tensor(b, device=device)
            x = torch.linalg.solve(At, bt)
            x = x.detach().cpu().numpy()
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b), "solve result and rhs have mismatched dimensions"
            return x
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            Lt = cp.asarray(A)
            bt = cp.asarray(b)
            x = cp.linalg.solve(Lt, bt)
            x = cp.asnumpy(x)
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b), "solve result and rhs have mismatched dimensions"
            return x
        else:
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b), "solve result and rhs have mismatched dimensions"
            return x
    else:
        raise Exception("No valid solve method specified")


def matmul(A, B, compute_device="cpu", args=None):
    """Return ``A @ B``; sparse inputs are always computed on CPU."""
    if sparse.issparse(A) or sparse.issparse(B): compute_device = "cpu"
    logger.debug("matrix multiplication")
    if compute_device == "cpu":
        res = A @ B
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        target = np.result_type(A, B)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A = torch.as_tensor(A.astype(target, copy=False), device=device)
            B = torch.as_tensor(B.astype(target, copy=False), device=device)
            res = A @ B
            res = res.detach().cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.asarray(A, dtype=target)
            B = cp.asarray(B, dtype=target)
            res = A @ B
            res = cp.asnumpy(res)
        else:
            warnings.warn(
                "No usable GPU backend found for matmul. Falling back to CPU.",
                stacklevel=2,
            )
            res = A @ B
    else:
        raise Exception("NO valid compute device found. ")
    return res


def matmul3(A, B, C, compute_device="cpu", args=None):
    """Return ``A @ B @ C``; sparse inputs are always computed on CPU."""
    if sparse.issparse(A) or sparse.issparse(B) or sparse.issparse(C): compute_device = "cpu"
    assert isinstance(A, np.ndarray), "A must be np.ndarray for matmul3"
    assert isinstance(B, np.ndarray), "B must be np.ndarray for matmul3"
    assert isinstance(C, np.ndarray), "C must be np.ndarray for matmul3"

    logger.debug("matrix multiplication on", compute_device)
    if compute_device == "cpu":
        res = A @ B @ C
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        target = np.result_type(A, B, C)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A = torch.as_tensor(A.astype(target, copy=False), device=device)
            B = torch.as_tensor(B.astype(target, copy=False), device=device)
            C = torch.as_tensor(C.astype(target, copy=False), device=device)
            res = A @ B @ C
            res = res.detach().cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.asarray(A, dtype=target)
            B = cp.asarray(B, dtype=target)
            C = cp.asarray(C, dtype=target)
            res = A @ B @ C
            res = cp.asnumpy(res)
        else:
            warnings.warn(
                "No usable GPU backend found for matmul3. Falling back to CPU.",
                stacklevel=2,
            )
            res = A @ B @ C
    else:
        raise Exception("NO valid compute device found. ")
    return res


##################################################################################
def is_sparse(A):
    """Return ``True`` if fewer than 1 % of elements in ``A`` are non-zero."""
    logger.debug("is_sparse")
    if float(np.count_nonzero(A)) / float(A.shape[0] * A.shape[1]) < 0.01:
        return True
    else:
        return False


def how_sparse_is(A):
    """Return the non-zero fraction of elements in ``A`` (0 = fully sparse, 1 = dense)."""
    logger.debug("how_sparse_is")
    return float(np.count_nonzero(A)) / float(A.shape[0] * A.shape[1])
