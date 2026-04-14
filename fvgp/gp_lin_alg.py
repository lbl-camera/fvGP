import numpy as np
import warnings
warnings.simplefilter("once", UserWarning)
import time
from collections import deque
from loguru import logger
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import minres, cg, spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy import sparse
import importlib


def _normalize_args(args):
    return {} if args is None else args


def normalize_sparse_preconditioner_type(preconditioner_type):
    preconditioner_type = str(preconditioner_type).lower()
    aliases = {
        "ilu": "ilu",
        "ic": "incomplete_cholesky",
        "ichol": "incomplete_cholesky",
        "incomplete_cholesky": "incomplete_cholesky",
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
            "{'ilu', 'ic', 'incomplete_cholesky', 'block_jacobi', 'blockjacobi', "
            "'schwarz', 'additive_schwarz', 'amg'}."
        )
    return aliases[preconditioner_type]


def resolve_gp2scale_linalg_mode(mode, args=None):
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


def _torch_gpu_device(args=None):
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
        device_index = int(args.get("GPU_device_index", torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0))
        return torch.device(f"cuda:{device_index}")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return None


def _cupy_gpu_available():
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_gpu_engine(args):
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
    args = _normalize_args(args)
    if _cupy_gpu_available():
        return True
    device = _torch_gpu_device(args)
    return device is not None and device.type == "cuda"


def calculate_sparse_LU_factor(M, args=None):
    assert sparse.issparse(M)
    logger.debug("calculate_sparse_LU_factor")
    LU = splu(M.tocsc())
    return LU


def calculate_LU_solve(LU, vec, args=None):
    assert isinstance(vec, np.ndarray)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    logger.debug("calculate_LU_solve")
    lu_dtype = getattr(getattr(LU, "L", None), "dtype", None)
    if lu_dtype is None:
        lu_dtype = getattr(getattr(LU, "U", None), "dtype", vec.dtype)
    vec = np.asarray(vec, dtype=lu_dtype)
    res = LU.solve(vec)
    if np.ndim(res) == 1: res = res.reshape(len(res), 1)
    assert np.ndim(res) == 2
    return res


def calculate_LU_logdet(LU, args=None):
    logger.debug("calculate_LU_logdet")
    upper_diag = abs(LU.U.diagonal())
    logdet = np.sum(np.log(upper_diag))
    assert np.isscalar(logdet)
    return logdet


def calculate_Chol_factor(M, compute_device="cpu", args=None):
    args = _normalize_args(args)
    assert isinstance(M, np.ndarray)
    if "Chol_factor_compute_device" in args: compute_device = args["Chol_factor_compute_device"]
    logger.debug(f"calculate_Chol_factor on {compute_device}")
    if compute_device == "cpu":
        c, l = cho_factor(M, lower=True)
        c = np.tril(c)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A = torch.tensor(M, device=device, dtype=torch.float32)
            L = torch.linalg.cholesky(A)
            c = L.cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.array(M, dtype=cp.float32)
            L = cp.linalg.cholesky(A)
            c = cp.asnumpy(L)
        else:
            warnings.warn(
                "No usable GPU backend found for Cholesky factorization. Falling back to CPU.",
                stacklevel=2,
            )
            c, _ = cho_factor(M, lower=True)
            c = np.tril(c)
    else:
        raise Exception("No valid compute device found. ")
    return c


def update_Chol_factor(old_chol_factor, new_matrix, compute_device="cpu", args=None):
    args = _normalize_args(args)
    assert isinstance(new_matrix, np.ndarray)
    compute_device = "cpu"
    #if "update_Chol_factor_compute_device" in args: compute_device = args["update_Chol_factor_compute_device"]
    logger.debug(f"update_Chol_factor on {compute_device}")
    size = len(old_chol_factor)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    return cholesky_update_rank_n(old_chol_factor, k.T, kk, compute_device=compute_device, args=args)


def calculate_Chol_solve(factor, vec, compute_device="cpu", args=None):
    args = _normalize_args(args)
    assert isinstance(vec, np.ndarray)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if "Chol_solve_compute_device" in args: compute_device = args["Chol_solve_compute_device"]
    logger.debug(f"calculate_Chol_solve on {compute_device}")
    if compute_device == "cpu":
        res = cho_solve((factor, True), vec)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            L = torch.tensor(factor, device=device, dtype=torch.float32)
            b = torch.tensor(vec, device=device, dtype=torch.float32)
            y = torch.linalg.solve_triangular(L, b, upper=False)
            x = torch.linalg.solve_triangular(L.T, y, upper=True)
            res = x.cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            L = cp.array(factor, dtype=cp.float32)
            b = cp.array(vec, dtype=cp.float32)
            y = cp.linalg.solve_triangular(L, b, lower=True)
            x = cp.linalg.solve_triangular(L.T, y, lower=False)
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
            L = torch.tensor(factor, device=device, dtype=torch.float32)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L))).cpu().item()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            L = cp.array(factor, dtype=cp.float32)
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
    assert np.isscalar(logdet)
    return logdet


def spai(A, m, args=None):
    assert sparse.issparse(A)
    assert isinstance(m, int)
    logger.debug("spai preconditioning")
    """Perform m step of the SPAI iteration."""
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

    assert sparse.issparse(M)
    return M


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

            columns = [[] for _ in range(n)]
            for i, row in enumerate(rows):
                for j, value in row.items():
                    columns[j].append((i, value))

            def solve(vector):
                vector = np.asarray(vector, dtype=np.float64)
                y = np.zeros_like(vector)
                for i, row in enumerate(rows):
                    total = vector[i]
                    for j, value in row.items():
                        total -= value * y[j]
                    y[i] = total / diag[i]

                z = np.zeros_like(y)
                for i in range(n - 1, -1, -1):
                    total = y[i]
                    for row_index, value in columns[i]:
                        if row_index > i:
                            total -= value * z[row_index]
                    z[i] = total / diag[i]
                return z

            factor = {
                "type": "incomplete_cholesky",
                "diag": diag,
                "rows": rows,
                "columns": columns,
                "solve": solve,
                "shift": shift,
            }
            operator = sparse.linalg.LinearOperator(A.shape, matvec=solve, rmatvec=solve, dtype=A.dtype)
            return factor, operator
        except Exception as exc:
            last_exc = exc
            shift = 1e-10 if shift == 0.0 else shift * growth

    raise np.linalg.LinAlgError(f"IC(0) preconditioner construction failed after shifted retries: {last_exc}")


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
    operator = sparse.linalg.LinearOperator(
        A.shape,
        matvec=factor.solve,
        rmatvec=factor.solve,
        dtype=A.dtype,
    )
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
    args = _normalize_args(args)
    assert sparse.issparse(KV)
    logger.debug("calculate_sparse_preconditioner")
    preconditioner_type = normalize_sparse_preconditioner_type(
        args.get("sparse_preconditioner_type", "ilu")
    )

    builders = {
        "ilu": _build_ilu_preconditioner,
        "incomplete_cholesky": _build_ic0_factor,
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
    return np.column_stack([np.asarray(operator @ matrix[:, i], dtype=np.float64) for i in range(matrix.shape[1])])


def _apply_preconditioner(M, residual):
    if M is None:
        return residual.copy()
    return _apply_linear_operator(M, residual)


def _block_conjugate_gradient(KV, vec, cg_tol, x0=None, M=None, maxiter=None):
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
    args = _normalize_args(args)
    assert sparse.issparse(KV)
    logger.debug("calculate_random_logdet")
    from imate import logdet as imate_logdet
    st = time.time()
    logdet_compute_device = str(args.get("random_logdet_lanczos_compute_device", compute_device)).lower()
    gpu = logdet_compute_device == "gpu" and _imate_gpu_enabled(args)

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
    assert np.isscalar(logdet)
    return logdet


def calculate_sparse_minres(KV, vec, x0=None, M=None, args=None):
    args = _normalize_args(args)
    assert sparse.issparse(KV)
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
    assert np.ndim(res) == 2
    return res


def calculate_sparse_conj_grad(KV, vec, x0=None, M=None, args=None):
    args = _normalize_args(args)
    assert sparse.issparse(KV)
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
        assert np.ndim(res) == 2
        return res

    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        initial_guess = _column_initial_guess(x0, i)
        res[:, i], exit_code = cg(KV, vec[:, i], M=M, rtol=cg_tol, x0=initial_guess, maxiter=maxiter)
        if exit_code != 0: warnings.warn(f"CG not successful (exit_code={exit_code})")
    logger.debug("CG compute time: {} seconds.", time.time() - st)
    assert np.ndim(res) == 2
    return res


def calculate_sparse_solve(KV, vec, args=None):
    assert sparse.issparse(KV)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    st = time.time()
    logger.debug("Sparse solve in progress ...")
    res = sparse.linalg.spsolve(KV, vec)
    logger.debug("Sparse solve compute time: {} seconds.", time.time() - st)
    if np.ndim(res) == 1: res = res.reshape(len(res), 1)
    return res


def cholesky_update_rank_1(L, b, c, compute_device="cpu", args=None):
    """

    Parameters
    ----------
    L matrix
    b vector
    c scalar

    Returns
    -------
    updated Cholesky

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

    Parameters
    ----------
    L matrix
    b vector
    c scalar

    Returns
    -------
    updated Cholesky

    """
    # Solve Lv = b for v
    v = solve_triangular(L, b, lower=True, check_finite=False)

    # Compute d
    d = np.sqrt(c - np.dot(v, v))

    # Form the new L'
    L_prime = np.block([
        [L, np.zeros((len(L), 1))],
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
    # Solve L v = b (forward solve)
    L = torch.tensor(L, device=device, dtype=torch.float32)
    b = torch.tensor(b, device=device, dtype=torch.float32)

    v = torch.linalg.solve_triangular(L, b.unsqueeze(1), upper=False).squeeze(1)

    # Compute d = sqrt(c - v^T v)
    d = torch.sqrt(c - torch.dot(v, v))

    # Form new L'
    n = L.shape[0]
    L_prime = torch.zeros((n+1, n+1), device=L.device, dtype=L.dtype)
    L_prime[:n, :n] = L
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
    L = cp.array(L)
    b = cp.array(b)
    # Solve L v = b
    v = cp.linalg.solve_triangular(L, b[:, None], lower=True).squeeze(1)

    # Compute d
    d = cp.sqrt(c - cp.dot(v, v))

    # Form new L'
    n = L.shape[0]
    L_prime = cp.zeros((n+1, n+1), dtype=L.dtype)
    L_prime[:n, :n] = L
    L_prime[:n, n] = 0
    L_prime[n, :n] = v
    L_prime[n, n] = d

    return cp.asnumpy(L_prime)


def cholesky_update_rank_n(L, b, c, compute_device="cpu", args=None):
    # Solve Lv = b for v
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


def calculate_logdet(A, compute_device='cpu', args=None):
    logger.debug("calculate_logdet")
    if compute_device == "cpu":
        s, logdet = np.linalg.slogdet(A)
        assert np.isscalar(logdet)
        return logdet
    elif compute_device == "gpu":
        try:
            engine = get_gpu_engine(args)
            if engine == "torch":
                import torch
                device = _torch_gpu_device(args)
                A_dev = torch.as_tensor(A, device=device, dtype=torch.float32)
                sign, logdet = torch.slogdet(A_dev)
                logdet = np.nan_to_num(logdet.detach().cpu().numpy())
                assert np.isscalar(logdet)
                return logdet
            if engine == "cupy":
                import cupy as cp
                A_dev = cp.asarray(A, dtype=cp.float32)
                sign, logdet = cp.linalg.slogdet(A_dev)
                logdet = np.nan_to_num(cp.asnumpy(logdet))
                assert np.isscalar(logdet)
                return logdet
            raise RuntimeError("No usable GPU backend available")
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            s, logdet = np.linalg.slogdet(A)
            assert np.isscalar(logdet)
            return logdet
    else:
        sign, logdet = np.linalg.slogdet(A)
        assert np.isscalar(logdet)
        return logdet


def update_logdet(old_logdet, old_inv, new_matrix, compute_device="cpu", args=None):
    logger.debug("update_logdet")
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    res = old_logdet + calculate_logdet(kk - k @ old_inv @ k.T, compute_device=compute_device, args=args)
    assert np.isscalar(res)
    return res


def calculate_inv(A, compute_device='cpu', args=None):
    assert isinstance(A, np.ndarray)
    assert np.ndim(A) == 2
    logger.debug("calculate_inv")
    if compute_device == "cpu":
        return np.linalg.inv(A)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A_dev = torch.as_tensor(A, device=device, dtype=torch.float32)
            B = torch.inverse(A_dev)
            return B.detach().cpu().numpy()
        if engine == "cupy":  # pragma: no cover
            import cupy as cp
            A_dev = cp.asarray(A, dtype=cp.float32)
            return cp.asnumpy(cp.linalg.inv(A_dev))
        return np.linalg.inv(A)
    else:
        return np.linalg.inv(A)


def calculate_inv_from_chol(L, compute_device="cpu", args=None):
    logger.debug("calculate_inv_from_chol")
    if compute_device == "cpu": A_inv = cho_solve((L, True), np.eye(L.shape[0]))
    elif compute_device == "gpu": A_inv = calculate_Chol_solve(L, np.eye(L.shape[0]), compute_device="gpu", args=args)
    else: raise Exception("No valid compute device found.")
    return A_inv


def update_inv(old_inv, new_matrix, compute_device="cpu", args=None):
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
    assert isinstance(A, np.ndarray)
    logger.debug("solve")
    if np.ndim(b) == 1: b = b.reshape(len(b), 1)
    if compute_device == "cpu":
        try:
            x = np.linalg.solve(A, b)
        except:
            x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if np.ndim(x) == 1: x = x.reshape(len(x), 1)
        assert np.ndim(x) == np.ndim(b)
        return x
    elif compute_device == "gpu":  # pragma: no cover
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            At = torch.as_tensor(A, device=device, dtype=torch.float32)
            bt = torch.as_tensor(b, device=device, dtype=torch.float32)
            x = torch.linalg.solve(At, bt)
            x = x.detach().cpu().numpy()
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b)
            return x
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            Lt = cp.array(A, dtype=cp.float32)
            bt = cp.array(b, dtype=cp.float32)
            x = cp.linalg.solve(Lt, bt)
            x = cp.asnumpy(x)
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b)
            return x
        else:
            try:
                x = np.linalg.solve(A, b)
            except:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b)
            return x
    else:
        raise Exception("No valid solve method specified")


def matmul(A, B, compute_device="cpu", args=None):
    if sparse.issparse(A) or sparse.issparse(B): compute_device = "cpu"
    logger.debug("matrix multiplication")
    if compute_device == "cpu":
        res = A @ B
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A = torch.tensor(A, device=device, dtype=torch.float32)
            B = torch.tensor(B, device=device, dtype=torch.float32)
            res = A@B
            res = res.detach().cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.array(A)
            B = cp.array(B)
            res = A @ B
            res = cp.asnumpy(res)
        else: res = A @ B
    else:
        raise Exception("NO valid compute device found. ")
    return res


def matmul3(A, B, C, compute_device="cpu", args=None):
    if sparse.issparse(A) or sparse.issparse(B) or sparse.issparse(C): compute_device = "cpu"
    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert isinstance(C, np.ndarray)

    logger.debug("matrix multiplication on", compute_device)
    if compute_device == "cpu":
        res = A @ B @ C
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            device = _torch_gpu_device(args)
            A = torch.tensor(A, device=device, dtype=torch.float32)
            B = torch.tensor(B, device=device, dtype=torch.float32)
            C = torch.tensor(C, device=device, dtype=torch.float32)
            res = A @ B @ C
            res = res.detach().cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.array(A)
            B = cp.array(B)
            C = cp.array(C)
            res = A @ B @ C
            res = cp.asnumpy(res)
        else: res = A @ B @ C
    else:
        raise Exception("NO valid compute device found. ")
    return res


##################################################################################
def is_sparse(A):
    logger.debug("is_sparse")
    if float(np.count_nonzero(A)) / float(len(A) ** 2) < 0.01:
        return True
    else:
        return False


def how_sparse_is(A):
    logger.debug("how_sparse_is")
    return float(np.count_nonzero(A)) / float(len(A) ** 2)
