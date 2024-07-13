import numpy as np
import warnings
import time
from loguru import logger
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import minres, cg, spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def calculate_LU_factor(M):
    logger.info("calculate_LU_factor")
    LU = splu(M.tocsc())
    return LU


def calculate_LU_solve(LU, vec):
    logger.info("calculate_LU_solve")
    return LU.solve(vec)


def calculate_LU_logdet(LU):
    logger.info("calculate_LU_logdet")
    upper_diag = abs(LU.U.diagonal())
    logdet = np.sum(np.log(upper_diag))
    return logdet


def calculate_Chol_factor(M):
    logger.info("calculate_Chol_factor")
    c, l = cho_factor(M, lower=True)
    c = np.tril(c)
    return c


def update_Chol_factor(old_chol_factor, new_matrix):
    logger.info("update_Chol_factor")
    size = len(old_chol_factor)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    return cholesky_update_rank_n(old_chol_factor, k.T, kk)


def calculate_Chol_solve(factor, vec):
    logger.info("calculate_Chol_solve")
    res = cho_solve((factor, True), vec)
    return res


def calculate_Chol_logdet(factor):
    logger.info("calculate_Chol_logdet")
    upper_diag = abs(factor.diagonal())
    logdet = 2.0 * np.sum(np.log(upper_diag))
    return logdet


def spai(A, m):
    logger.info("spai preconditioning")
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

    return M


def calculate_random_logdet(KV, info, compute_device):
    logger.info("calculate_random_logdet")
    from imate import logdet as imate_logdet
    st = time.time()
    if compute_device == "gpu":
        gpu = True
    else:
        gpu = False

    logdet, info_slq = imate_logdet(KV, method='slq', min_num_samples=10, max_num_samples=1000,
                                    lanczos_degree=10, error_rtol=0.01, gpu=gpu,
                                    return_info=True, plot=False, verbose=False, orthogonalize=0)
    if info: logger.info("Stochastic Lanczos logdet() compute time: {} seconds", time.time() - st)
    return logdet


def calculate_sparse_conj_grad(KV, vec, info=False):
    logger.info("calculate_sparse_conj_grad")
    st = time.time()
    if info: logger.info("CG solve in progress ...")
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        res[:, i], exit_code = cg(KV.tocsc(), vec[:, i], rtol=1e-8)
        if exit_code == 1:
            logger.info("CG preconditioning in progress ...")
            M = spai(KV, 20)
            res[:, i], exit_code = cg(KV.tocsc(), vec[:, i], M=M, rtol=1e-8)
        if exit_code == 1: warnings.warn("CG not successful")
    if info: logger.info("CG compute time: {} seconds, exit status {} (0:=successful)", time.time() - st, exit_code)
    return res


def update_sparse_conj_grad(KV, vec, x0, info=False):
    assert np.ndim(vec) == 1
    assert np.ndim(KV) == 2
    assert np.ndim(x0) == 1
    logger.info("update_sparse_conj_grad")
    st = time.time()
    if len(x0) < KV.shape[0]: x0 = np.append(x0, np.zeros(KV.shape[0] - len(x0)))
    if info: logger.info("CG solve in progress ...")
    vec = vec.reshape(len(vec), 1)
    res, exit_code = cg(KV.tocsc(), vec[:, 0], rtol=1e-8, x0=x0)
    if exit_code == 1:
        logger.info("CG preconditioning in progress ...")
        M = spai(KV, 20)
        res, exit_code = cg(KV.tocsc(), vec[:, 0], M=M, x0=x0, rtol=1e-8)
        if exit_code == 1: warnings.warn("CG not successful")
    if info: logger.info("CG compute time: {} seconds, exit status {} (0:=successful)", time.time() - st, exit_code)
    return res


def calculate_sparse_minres(KV, vec, info=False):
    logger.info("calculate_sparse_minres")
    st = time.time()
    if info: logger.info("MINRES solve in progress ...")
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec), 1)
    res = np.zeros(vec.shape)
    for i in range(vec.shape[1]):
        res[:, i], exit_code = minres(KV.tocsc(), vec[:, i], rtol=1e-8)
        if exit_code == 1: warnings.warn("MINRES not successful")
    if info: logger.info("MINRES compute time: {} seconds, exit status {} (0:=successful)",
                         time.time() - st, exit_code)
    return res


def update_sparse_minres(KV, vec, x0, info=False):
    assert np.ndim(vec) == 1
    assert np.ndim(KV) == 2
    assert np.ndim(x0) == 1
    logger.info("update_sparse_minres")
    st = time.time()
    if len(x0) < KV.shape[0]: x0 = np.append(x0, np.zeros(KV.shape[0] - len(x0)))
    if info: logger.info("MINRES solve in progress ...")
    vec = vec.reshape(len(vec), 1)
    res, exit_code = minres(KV.tocsc(), vec[:, 0], rtol=1e-8, x0=x0)
    if exit_code == 1: warnings.warn("MINRES update not successful")
    if info: logger.info("MINRES compute time: {} seconds, exit status {} (0:=successful)",
                         time.time() - st, exit_code)
    return res


def cholesky_update_rank_1(L, b, c):
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


def cholesky_update_rank_n(L, b, c):
    # Solve Lv = b for v
    L_prime = L.copy()
    for i in range(b.shape[1]):
        L_prime = cholesky_update_rank_1(L_prime, np.append(b[:, i], c[0:i, i]), c[i, i])
    return L_prime


def calculate_logdet(A, compute_device='cpu'):
    logger.info("calculate_logdet")
    if compute_device == "cpu":
        s, logdet = np.linalg.slogdet(A)
        return logdet
    elif compute_device == "gpu":
        try:
            import torch
            A = torch.from_numpy(A).cuda()
            sign, logdet = torch.slogdet(A)
            logdet = logdet.cpu().numpy()
            logdet = np.nan_to_num(logdet)
            return logdet
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            s, logdet = np.linalg.slogdet(A)
            return logdet
    else:
        sign, logdet = np.linalg.slogdet(A)
        return logdet


def update_logdet(old_logdet, old_inv, new_matrix, compute_device="cpu"):
    logger.info("update_logdet")
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    res = old_logdet + calculate_logdet(kk - k @ old_inv @ k.T, compute_device=compute_device)
    return res


def calculate_inv(A, compute_device='cpu'):
    logger.info("calculate_inv")
    if compute_device == "cpu":
        return np.linalg.inv(A)
    elif compute_device == "gpu":
        import torch
        A = torch.from_numpy(A)
        B = torch.inverse(A)
        return B.numpy()
    else:
        return np.linalg.inv(A)


def update_inv(old_inv, new_matrix, compute_device="cpu"):
    logger.info("update_inv")
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    X = calculate_inv(kk - k @ old_inv @ k.T, compute_device=compute_device)
    F = -old_inv @ k.T @ X
    new_inv = np.block([[old_inv + old_inv @ k.T @ X @ k @ old_inv, F],
                        [F.T, X]])
    return new_inv


def solve(A, b, compute_device='cpu'):
    logger.info("solve")
    if np.ndim(b) == 1: b = np.expand_dims(b, axis=1)
    if compute_device == "cpu":
        try:
            x = np.linalg.solve(A, b)
        except:
            x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
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
            return x
    else:
        raise Exception("No valid solve method specified")


##################################################################################
def is_sparse(A):
    logger.info("is_sparse")
    if float(np.count_nonzero(A)) / float(len(A) ** 2) < 0.01:
        return True
    else:
        return False


def how_sparse_is(A):
    logger.info("how_sparse_is")
    return float(np.count_nonzero(A)) / float(len(A) ** 2)
