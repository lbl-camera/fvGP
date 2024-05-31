import numpy as np
import warnings
import time
from loguru import logger
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import minres, cg
from scipy.sparse import identity
from scipy.sparse.linalg import onenormest
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def compute_LU_factor(M):
    LU = splu(M.tocsc())
    return LU


def calculate_LU_solve(LU, vec):
    return LU.solve(vec)


def calculate_LU_logdet(LU):
    upper_diag = abs(LU.U.diagonal())
    logdet = np.sum(np.log(upper_diag))
    return logdet


def calculate_Chol_factor(M):
    c, l = cho_factor(M, lower=True)
    c = np.tril(c)
    return c


def update_Chol_factor(old_chol_factor, new_matrix):
    size = len(old_chol_factor)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    return cholesky_update_rank_n(old_chol_factor, k.T, kk)


def calculate_Chol_solve(factor, vec):
    res = cho_solve((factor, True), vec)
    return res


def calculate_Chol_logdet(factor):
    upper_diag = abs(factor.diagonal())
    logdet = 2.0 * np.sum(np.log(upper_diag))
    return logdet


def spai(A, m):
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
    from imate import logdet as imate_logdet
    st = time.time()
    if compute_device == "gpu": gpu = True
    else: gpu = False

    logdet, info_slq = imate_logdet(KV, method='slq', min_num_samples=10, max_num_samples=1000,
                                    lanczos_degree=20, error_rtol=0.001, gpu=gpu,
                                    return_info=True, plot=False, verbose=False)
    if info: logger.info("Random logdet() done after {} seconds", time.time() - st)
    return logdet


def calculate_sparse_conj_grad(KV, vec, info=False):
    st = time.time()
    if info: logger.info("CG solve in progress ...")
    if np.ndim(vec) == 1: vec = vec.reshape(len(vec),-1)
    for i in range(vec.shape[1]):
        res, exit_code = cg(KV.tocsc(), vec[:, i])
        if exit_code == 1:
            M = spai(KV, 20)
            res, exit_code = cg(KV.tocsc(), vec[:,i], M=M)
        if exit_code == 1: warnings.warn("CG not successful")
    if info: logger.info("CG compute time: {} seconds, exit status {} (0:=successful)", time.time() - st, exit_code)
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
    size = len(old_inv)
    KV = new_matrix
    kk = KV[size:, size:]
    k = KV[size:, 0:size]
    res = old_logdet + calculate_logdet(kk - k @ old_inv @ k.T, compute_device=compute_device)
    return res


def calculate_inv(A, compute_device='cpu'):
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
    if float(np.count_nonzero(A)) / float(len(A) ** 2) < 0.01:
        return True
    else:
        return False


def how_sparse_is(A):
    return float(np.count_nonzero(A)) / float(len(A) ** 2)
