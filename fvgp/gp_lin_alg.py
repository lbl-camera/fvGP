import numpy as np
import warnings
import time
from loguru import logger
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import minres, cg, spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy import sparse


def calculate_sparse_LU_factor(M, args=None):
    assert sparse.issparse(M)
    logger.debug("calculate_sparse_LU_factor")
    LU = splu(M)
    return LU


def calculate_LU_solve(LU, vec, args=None):
    assert isinstance(vec, np.ndarray)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    logger.debug("calculate_LU_solve")
    res = LU.solve(vec)
    assert np.ndim(res) == 2
    return res


def calculate_LU_logdet(LU, args=None):
    logger.debug("calculate_LU_logdet")
    upper_diag = abs(LU.U.diagonal())
    logdet = np.sum(np.log(upper_diag))
    assert np.isscalar(logdet)

    return logdet


def calculate_Chol_factor(M, args=None):
    assert isinstance(M, np.ndarray)
    logger.debug("calculate_Chol_factor")
    c, l = cho_factor(M, lower=True)
    c = np.tril(c)
    return c


def update_Chol_factor(old_chol_factor, new_matrix, args=None):
    assert isinstance(new_matrix, np.ndarray)
    logger.debug("update_Chol_factor")
    size = len(old_chol_factor)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    return cholesky_update_rank_n(old_chol_factor, k.T, kk)


def calculate_Chol_solve(factor, vec, args=None):
    assert isinstance(vec, np.ndarray)
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    logger.debug("calculate_Chol_solve")
    res = cho_solve((factor, True), vec)
    assert np.ndim(res) == 2
    return res


def calculate_Chol_logdet(factor, args=None):
    logger.debug("calculate_Chol_logdet")
    upper_diag = abs(factor.diagonal())
    logdet = 2.0 * np.sum(np.log(upper_diag))
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
    st = time.time()
    if compute_device == "gpu": gpu = True
    else: gpu = False

    lanczos_degree = 20
    error_rtol = 0.01
    verbose = False
    print_info = False

    if "random_logdet_lanczos_degree" in args: lanczos_degree = args["random_logdet_lanczos_degree"]
    if "random_logdet_error_rtol" in args: lanczos_degree = args["random_logdet_error_rtol"]
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
    st = time.time()
    logger.debug("CG solve in progress ...")
    cg_tol = 1e-5
    if "sparse_cg_tol" in args: cg_tol = args["sparse_minres_tol"]
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
    if vec.shape[1] == 1: res = res.reshape(len(vec), 1)
    assert np.ndim(res) == 2
    return res


def cholesky_update_rank_1(L, b, c, args=None):
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
    #v = np.linalg.solve(L, b)

    # Compute d
    d = np.sqrt(c - np.dot(v, v))

    # Form the new L'
    L_prime = np.block([
        [L, np.zeros((len(L), 1))],
        [v.T, d]
    ])
    return L_prime


def cholesky_update_rank_n(L, b, c, args=None):
    # Solve Lv = b for v
    L_prime = L.copy()
    for i in range(b.shape[1]):
        L_prime = cholesky_update_rank_1(L_prime, np.append(b[:, i], c[0:i, i]), c[i, i])
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
        A = torch.from_numpy(A)
        B = torch.inverse(A)
        return B.numpy()
    else:
        return np.linalg.inv(A)


def calculate_inv_from_chol(L, args=None):
    logger.debug("calculate_inv_from_chol")
    A_inv = cho_solve((L, True), np.eye(L.shape[0]))
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
    if compute_device == "cpu":
        try:
            x = np.linalg.solve(A, b)
        except:
            x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        assert np.ndim(x) == np.ndim(b)
        return x
    elif compute_device == "gpu" or A.ndim < 3:
        try:
            import torch
            A = torch.from_numpy(A).cuda()
            b = torch.from_numpy(b).cuda()
            x = torch.linalg.solve(A, b)
            return x.cpu().numpy()
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            try:
                x = np.linalg.solve(A, b)
            except:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            assert np.ndim(x) == np.ndim(b)
            return x
    elif compute_device == "multi-gpu":
        try:
            import torch
            n = min(len(A), torch.cuda.device_count())
            split_A = np.array_split(A, n)
            split_b = np.array_split(b, n)
            results = []
            for i, (tmp_A, tmp_b) in enumerate(zip(split_A, split_b)):
                cur_device = torch.device("cuda:" + str(i))
                tmp_A = torch.from_numpy(tmp_A).cuda(cur_device)
                tmp_b = torch.from_numpy(tmp_b).cuda(cur_device)
                results.append(torch.linalg.solve(tmp_A, tmp_b)[0])
            total = results[0].cpu().numpy()
            for i in range(1, len(results)):
                total = np.append(total, results[i].cpu().numpy(), 0)
            return total
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            try:
                x = np.linalg.solve(A, b)
            except:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            assert np.ndim(x) == np.ndim(b)
            return x
    else:
        raise Exception("No valid solve method specified")


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
