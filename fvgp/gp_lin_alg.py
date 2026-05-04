import numpy as np
import warnings
warnings.simplefilter("once", UserWarning)
import time
from loguru import logger
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import minres, cg, spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy import sparse
import importlib


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


def get_gpu_engine(args):
    args = args or {}
    if "GPU_engine" in args: return args["GPU_engine"]
    if importlib.util.find_spec("torch"): return "torch"
    elif importlib.util.find_spec("cupy"): return "cupy"
    else: return None


def calculate_sparse_LU_factor(M, args=None):
    assert sparse.issparse(M)
    logger.debug("calculate_sparse_LU_factor")
    LU = splu(M.tocsc())
    return LU


def calculate_LU_solve(LU, vec, args=None):
    assert isinstance(vec, np.ndarray)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if vec.dtype != LU.L.dtype:
        vec = vec.astype(LU.L.dtype, copy=False)
    logger.debug("calculate_LU_solve")
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
    assert isinstance(M, np.ndarray)
    args = args or {}
    if "Chol_factor_compute_device" in args: compute_device = args["Chol_factor_compute_device"]
    logger.debug(f"calculate_Chol_factor on {compute_device}")
    try:
        if compute_device == "cpu":
            c, l = cho_factor(M, lower=True)
        elif compute_device == "gpu":
            engine = get_gpu_engine(args)
            if engine == "torch":  # pragma: no cover
                import torch
                A = torch.tensor(M, device="cuda:0")
                L = torch.linalg.cholesky(A)
                c = L.cpu().numpy()
            elif engine == "cupy":  # pragma: no cover
                import cupy as cp
                A = cp.array(M)
                L = cp.linalg.cholesky(A)
                c = cp.asnumpy(L)
            else: c = None
        else:
            raise Exception("No valid compute device found. ")
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise NonPositiveDefiniteError(_non_pd_message(M, e)) from e
    return c


def update_Chol_factor(old_chol_factor, new_matrix, compute_device="cpu", args=None):
    assert isinstance(new_matrix, np.ndarray)
    compute_device = "cpu"
    #if "update_Chol_factor_compute_device" in args: compute_device = args["update_Chol_factor_compute_device"]
    logger.debug(f"update_Chol_factor on {compute_device}")
    size = len(old_chol_factor)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    return cholesky_update_rank_n(old_chol_factor, k.T, kk, compute_device=compute_device)


def calculate_Chol_solve(factor, vec, compute_device="cpu", args=None):
    assert isinstance(vec, np.ndarray)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if vec.dtype != factor.dtype:
        vec = vec.astype(factor.dtype, copy=False)
    args = args or {}
    if "Chol_solve_compute_device" in args: compute_device = args["Chol_solve_compute_device"]
    logger.debug(f"calculate_Chol_solve on {compute_device}")
    if compute_device == "cpu":
        res = cho_solve((factor, True), vec)
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            # Move to GPU
            L = torch.tensor(factor, device="cuda")
            b = torch.tensor(vec, device="cuda")
            y = torch.linalg.solve_triangular(L, b, upper=False)
            x = torch.linalg.solve_triangular(L.T, y, upper=True)
            res = x.cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
            L = cp.array(factor)
            b = cp.array(vec)
            y = cp_solve_triangular(L, b, lower=True)
            x = cp_solve_triangular(L.T, y, lower=False)
            res = cp.asnumpy(x)
        else: res = None
    else:
        raise Exception("NO valid compute device found. ")

    if np.ndim(res) == 1: res = res.reshape(len(res), 1)
    return res


def calculate_Chol_logdet(factor, compute_device="cpu", args=None):
    args = args or {}
    if "Chol_logdet_compute_device" in args: compute_device = args["Chol_logdet_compute_device"]
    logger.debug(f"calculate_Chol_logdet on {compute_device}")
    if compute_device == "cpu":
        upper_diag = abs(factor.diagonal())
        logdet = 2.0 * np.sum(np.log(upper_diag))
    elif compute_device == "gpu":
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            L = torch.tensor(factor, device="cuda")
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L))).cpu().item()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            L = cp.array(factor)
            logdet = 2.0 * cp.sum(cp.log(cp.diag(L))).get()
        else: logdet = None
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


def calculate_random_logdet(KV, compute_device, args=None):
    assert sparse.issparse(KV)
    logger.debug("calculate_random_logdet")
    from imate import logdet as imate_logdet
    args = args or {}
    st = time.time()
    if compute_device == "gpu": gpu = True
    else: gpu = False

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
    assert sparse.issparse(KV)
    args = args or {}
    st = time.time()
    logger.debug("MINRES solve in progress ...")
    minres_tol = 1e-5
    if "sparse_minres_tol" in args:
        minres_tol = args["sparse_minres_tol"]
        logger.debug("sparse_minres_tol changed to ", minres_tol)

    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if isinstance(x0, np.ndarray) and len(x0) < KV.shape[0]: x0 = np.append(x0, np.zeros(KV.shape[0] - len(x0)))
    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        res[:, i], exit_code = minres(KV, vec[:, i], M=M, rtol=minres_tol, x0=x0)
        if exit_code == 1: warnings.warn("MINRES not successful")
    logger.debug("MINRES compute time: {} seconds.", time.time() - st)
    assert np.ndim(res) == 2
    return res


def calculate_sparse_conj_grad(KV, vec, x0=None, M=None, args=None):
    assert sparse.issparse(KV)
    args = args or {}
    st = time.time()
    logger.debug("CG solve in progress ...")
    cg_tol = 1e-5
    if "sparse_cg_tol" in args: cg_tol = args["sparse_cg_tol"]
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    if isinstance(x0, np.ndarray) and len(x0) < KV.shape[0]: x0 = np.append(x0, np.zeros(KV.shape[0] - len(x0)))
    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        res[:, i], exit_code = cg(KV, vec[:, i], M=M, rtol=cg_tol, x0=x0)
        if exit_code == 1: warnings.warn("CG not successful")
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
            L_prime = cholesky_update_rank_1_torch(L, b, c)
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


def cholesky_update_rank_1_torch(L, b, c):   # pragma: no cover
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
    # Solve L v = b (forward solve). Match b/c to L's dtype.
    target = L.dtype
    L = torch.tensor(L, device="cuda")
    b = torch.tensor(np.asarray(b, dtype=target), device="cuda")
    c = np.asarray(c, dtype=target).item() if np.ndim(c) == 0 else np.asarray(c, dtype=target)

    v = torch.linalg.solve_triangular(L, b.unsqueeze(1), upper=False).squeeze(1)

    # Compute d = sqrt(c - v^T v)
    disc = c - torch.dot(v, v)
    if float(disc) <= 0:
        raise NonPositiveDefiniteError(_rank1_update_non_pd_message(disc))
    d = torch.sqrt(disc)

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
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    target = L.dtype
    L = cp.array(L)
    b = cp.array(np.asarray(b, dtype=target))
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
    # Solve Lv = b for v
    L_prime = L.copy()
    for i in range(b.shape[1]):
        L_prime = cholesky_update_rank_1(L_prime, np.append(b[:, i], c[0:i, i]), c[i, i], compute_device=compute_device)
    return L_prime


def calculate_logdet(A, compute_device='cpu', args=None):
    logger.debug("calculate_logdet")
    if compute_device == "cpu":
        s, logdet = np.linalg.slogdet(A)
        assert np.isscalar(logdet)
        return logdet
    elif compute_device == "gpu":
        try:
            import torch
            A = torch.from_numpy(A).cuda()
            sign, logdet = torch.slogdet(A)
            logdet = logdet.cpu().numpy()
            logdet = np.nan_to_num(logdet)
            assert np.isscalar(logdet)
            return logdet
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
    res = old_logdet + calculate_logdet(kk - k @ old_inv @ k.T, compute_device=compute_device)
    assert np.isscalar(res)
    return res


def calculate_inv(A, compute_device='cpu', args=None):
    assert isinstance(A, np.ndarray)
    assert np.ndim(A) == 2
    logger.debug("calculate_inv")
    if compute_device == "cpu":
        return np.linalg.inv(A)
    elif compute_device == "gpu":
        import torch
        A = torch.from_numpy(A).cuda()
        B = torch.inverse(A)
        return B.cpu().numpy()
    else:
        return np.linalg.inv(A)


def calculate_inv_from_chol(L, compute_device="cpu", args=None):
    logger.debug("calculate_inv_from_chol")
    if compute_device == "cpu": A_inv = cho_solve((L, True), np.eye(L.shape[0]))
    elif compute_device == "gpu": A_inv = calculate_Chol_solve(L, np.eye(L.shape[0]), compute_device="gpu")
    else: raise Exception("No valid compute device found.")
    return A_inv


def update_inv(old_inv, new_matrix, compute_device="cpu", args=None):
    logger.debug("update_inv")
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    X = calculate_inv(kk - k @ old_inv @ k.T, compute_device=compute_device)
    F = -old_inv @ k.T @ X
    new_inv = np.block([[old_inv + old_inv @ k.T @ X @ k @ old_inv, F],
                        [F.T, X]])
    return new_inv


def solve(A, b, compute_device='cpu', args=None):
    assert isinstance(A, np.ndarray)
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
        assert np.ndim(x) == np.ndim(b)
        return x
    elif compute_device == "gpu":  # pragma: no cover
        engine = get_gpu_engine(args)
        if engine == "torch":  # pragma: no cover
            import torch
            At = torch.from_numpy(A).cuda()
            bt = torch.from_numpy(b).cuda()
            x = torch.linalg.solve(At, bt)
            x = x.cpu().numpy()
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b)
            return x
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            Lt = cp.array(A)
            bt = cp.array(b)
            x = cp.linalg.solve(Lt, bt)
            x = cp.asnumpy(x)
            if np.ndim(x) == 1: x = x.reshape(len(x), 1)
            assert np.ndim(x) == np.ndim(b)
            return x
        else:
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
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
        target = np.result_type(A, B)
        if engine == "torch":  # pragma: no cover
            import torch
            A = torch.tensor(A.astype(target, copy=False), device="cuda")
            B = torch.tensor(B.astype(target, copy=False), device="cuda")
            res = A@B
            res = res.cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.array(A, dtype=target)
            B = cp.array(B, dtype=target)
            res = A @ B
            res = cp.asnumpy(res)
        else: res = None
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
        target = np.result_type(A, B, C)
        if engine == "torch":  # pragma: no cover
            import torch
            A = torch.tensor(A.astype(target, copy=False), device="cuda")
            B = torch.tensor(B.astype(target, copy=False), device="cuda")
            C = torch.tensor(C.astype(target, copy=False), device="cuda")
            res = A @ B @ C
            res = res.cpu().numpy()
        elif engine == "cupy":  # pragma: no cover
            import cupy as cp
            A = cp.array(A, dtype=target)
            B = cp.array(B, dtype=target)
            C = cp.array(C, dtype=target)
            res = A @ B @ C
            res = cp.asnumpy(res)
        else: res = None
    else:
        raise Exception("NO valid compute device found. ")
    return res


##################################################################################
def is_sparse(A):
    logger.debug("is_sparse")
    if float(np.count_nonzero(A)) / float(A.shape[0] * A.shape[1]) < 0.01:
        return True
    else:
        return False


def how_sparse_is(A):
    logger.debug("how_sparse_is")
    return float(np.count_nonzero(A)) / float(A.shape[0] * A.shape[1])
