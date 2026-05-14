import importlib
import warnings

import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def squared_exponential_kernel(distance, length):
    """
    Function for the squared exponential kernel.
    kernel = np.exp(-(distance ** 2) / (2.0 * (length ** 2)))

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    length : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    kernel = np.exp(-(distance ** 2) / (2.0 * (length ** 2)))
    return kernel


def squared_exponential_kernel_robust(distance, phi):
    """
    Function for the squared exponential kernel (robust version)
    kernel = np.exp(-(distance ** 2) * (phi ** 2))

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    phi : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    kernel = np.exp(-(distance ** 2) * (phi ** 2))
    return kernel


def exponential_kernel(distance, length):
    """
    Function for the exponential kernel.
    kernel = np.exp(-(distance) / (length))

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    length : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    kernel = np.exp(-distance / length)
    return kernel


def exponential_kernel_robust(distance, phi):
    """
    Function for the exponential kernel (robust version)
    kernel = np.exp(-(distance) * (phi**2))

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    phi : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    kernel = np.exp(-(distance) * (phi ** 2))
    return kernel


def matern_kernel_diff1(distance, length):
    """
    Function for the Matern kernel, order of differentiability = 1.
    kernel = (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    length : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    kernel = (1.0 + ((np.sqrt(3.0) * distance) / length)) * np.exp(
        -(np.sqrt(3.0) * distance) / length
    )
    return kernel


def matern_kernel_diff1_grad(distance, dist_der):
    """
    Derivative of the Matern-1 kernel with respect to the hyperparameters.
    kernel_der = -sqrt(3)*d * dist_der * exp(-sqrt(3)*d)

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    dist_der : scalar or np.ndarray
        The derivative of the distance matrix. We assume here that the distance is a function of the hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    a = (np.sqrt(3.0) * distance)
    dadl = np.sqrt(3.0) * dist_der
    ea = np.exp(-a)
    kernel_der = dadl * ea - (1.+a) * dadl * ea
    return kernel_der


def matern_kernel_diff1_robust(distance, phi):
    """
    Function for the Matern kernel, order of differentiability = 1, robust version.
    kernel = (1 + sqrt(3)*d*phi**2) * exp(-sqrt(3)*d*phi**2)

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    phi : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    ##1/l --> phi**2
    kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi ** 2))) * np.exp(
        -(np.sqrt(3.0) * distance) * (phi ** 2))
    return kernel


def matern_kernel_diff2(distance, length):
    """
    Function for the Matern kernel, order of differentiability = 2.
    kernel = (1 + sqrt(5)*d/l + 5*d**2/(3*l**2)) * exp(-sqrt(5)*d/l)

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    length : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    kernel = (
                 1.0
                 + ((np.sqrt(5.0) * distance) / (length))
                 + ((5.0 * distance ** 2) / (3.0 * length ** 2))
             ) * np.exp(-(np.sqrt(5.0) * distance) / length)
    return kernel


def matern_kernel_diff2_robust(distance, phi):
    """
    Function for the Matern kernel, order of differentiability = 2, robust version.
    kernel = (1 + sqrt(5)*d*phi**2 + 15*d**2*phi**4) * exp(-sqrt(5)*d*phi**2)

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    phi : scalar
        The length scale hyperparameters.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    kernel = (
                 1.0
                 + ((np.sqrt(5.0) * distance) * (phi ** 2))
                 + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
             ) * np.exp(-(np.sqrt(5.0) * distance) * (phi ** 2))
    return kernel


def sparse_kernel(distance, radius):
    """
    Function for a compactly supported kernel.

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    radius : scalar
        Radius of support.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    d = np.array(distance)
    d[d == 0.0] = 10e-6
    d[d > radius] = radius
    kernel = (np.sqrt(2.0) / (3.0 * np.sqrt(np.pi))) * \
             ((3.0 * (d / radius) ** 2 * np.log((d / radius) / (1 + np.sqrt(1.0 - (d / radius) ** 2)))) +
              ((2.0 * (d / radius) ** 2 + 1.0) * np.sqrt(1.0 - (d / radius) ** 2)))
    return kernel


def periodic_kernel(distance, length, p):
    """
    Function for a periodic kernel.
    kernel = np.exp(-(2.0/length**2)*(np.sin(np.pi*distance/p)**2))

    Parameters
    ----------
    distance : scalar or np.ndarray
        Distance between a set of points.
    length : scalar
        Length scale.
    p : scalar
        Period.

    Return
    ------
    Kernel output : same as distance parameter.
    """

    kernel = np.exp(-(2.0 / length ** 2) * (np.sin(np.pi * distance / p) ** 2))
    return kernel


def linear_kernel(x1, x2, hp1, hp2, hp3):
    """
    Function for a linear kernel.
    kernel = hp1 + (hp2*(x1-hp3)*(x2-hp3))

    Parameters
    ----------
    x1 : float
        Point 1.
    x2 : float
        Point 2.
    hp1 : float
        Hyperparameter.
    hp2 : float
        Hyperparameter.
    hp3 : float
        Hyperparameter.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    kernel = hp1 + (hp2 * (x1 - hp3) * (x2 - hp3))
    return kernel


def dot_product_kernel(x1, x2, hp, matrix):
    """
    Function for a dot-product kernel.
    kernel = hp + x1.T @ matrix @ x2

    Parameters
    ----------
    x1 : np.ndarray
        Point 1.
    x2 : np.ndarray
        Point 2.
    hp : float
        Offset hyperparameter.
    matrix : np.ndarray
        PSD matrix defining the inner product.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    kernel = hp + x1.T @ matrix @ x2
    return kernel


def polynomial_kernel(x1, x2, p):
    """
    Polynomial kernel: ``(1 + x1ᵀ x2) ** p``.

    Parameters
    ----------
    x1 : np.ndarray
        Point 1, shape (D,).
    x2 : np.ndarray
        Point 2, shape (D,).
    p : float
        Degree hyperparameter.

    Returns
    -------
    kernel : float
        Kernel value.
    """
    kernel = (1.0 + x1.T @ x2) ** p
    return kernel


def wendland_kernel(d):
    """
    Function for the Wendland kernel with a given distance matrix.
    The Wendland kernel is compactly supported, leading to sparse covariance matrices.

    Parameters
    ----------
    d : np.ndarray
        Distance matrix.

    Return
    ------
    Covariance matrix : np.ndarray
    """
    d[d > 1.] = 1.
    kernel = (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return kernel


def wendland_anisotropic(x1, x2, hyperparameters):
    """
    Function for the Wendland kernel.
    The Wendland kernel is compactly supported, leading to sparse covariance matrices.

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).
    hyperparameters : np.ndarray
        Array of hyperparameters. For this kernel we need D + 1 hyperparameters.

    Return
    ------
    Covariance matrix : np.ndarray
    """
    hps = hyperparameters
    distance_matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0])): distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
    d = np.sqrt(distance_matrix)
    d[d > 1.] = 1.
    kernel = (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return hps[0] * kernel


def non_stat_kernel(x1, x2, x0, w, l):
    """
    Non-stationary kernel.
    kernel = g(x1) g(x2)

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).
    x0 : np.ndarray
        Numpy array of the basis function locations.
    w : np.ndarray
        1d np.ndarray of weights. len(w) = len(x0).
    l : float
        Width measure of the basis functions.

    Return
    ------
    Covariance matrix : np.ndarray
    """
    non_stat = np.outer(_g(x1, x0, w, l), _g(x2, x0, w, l))
    return non_stat


def non_stat_kernel_gradient(x1, x2, x0, w, l):
    """
    Non-stationary kernel gradient.
    kernel = g(x1) g(x2)

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).
    x0 : np.ndarray
        Numpy array of the basis function locations.
    w : np.ndarray
        1d np.ndarray of weights. len(w) = len(x0).
    l : float
        Width measure of the basis functions.

    Return
    ------
    Covariance matrix : np.ndarray
    """
    dkdw = (np.einsum('ij,k->ijk', _dgdw(x1, x0, w, l), _g(x2, x0, w, l))
            + np.einsum('ij,k->ikj', _dgdw(x2, x0, w, l), _g(x1, x0, w, l)))
    dkdl = (np.outer(_dgdl(x1, x0, w, l), _g(x2, x0, w, l)) +
            np.outer(_dgdl(x2, x0, w, l), _g(x1, x0, w, l)).T)
    res = np.empty((len(w) + 1, len(x1), len(x2)))
    res[0:len(w)] = dkdw
    res[-1] = dkdl
    return res


def get_distance_matrix(x1, x2):
    """
    Function to calculate the pairwise distance matrix of
    points in x1 and x2.

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).

    Return
    ------
    distance matrix : np.ndarray
    """
    d = np.zeros((len(x1), len(x2)))
    for i in range(x1.shape[1]): d += (x1[:, i].reshape(-1, 1) - x2[:, i]) ** 2
    return np.sqrt(d)


def get_anisotropic_distance_matrix(x1, x2, hps):
    """
    Function to calculate the pairwise axial-anisotropic distance matrix of
    points in x1 and x2.

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).
    hps : np.ndarray
        1d array of values. The diagonal of the metric tensor describing the axial anisotropy.

    Return
    ------
    distance matrix : np.ndarray
    """
    d = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0])): d += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[i]) ** 2
    return np.sqrt(d)


def _g(x, x0, w, l):
    d = get_distance_matrix(x, x0)
    e = np.exp(-(d ** 2) / l)
    return np.sum(w * e, axis=1)


def _dgdw(x, x0, w, l):
    d = get_distance_matrix(x, x0)
    e = np.exp(-(d ** 2) / l).T
    return e


def _dgdl(x, x0, w, l):
    d = get_distance_matrix(x, x0)
    e = np.exp(-(d ** 2) / l)
    return np.sum(w * e * (d ** 2 / l ** 2), axis=1)


def wendland_anisotropic_gp2Scale_cpu(x1, x2, hps):
    """
    Function for the anisotropic Wendland kernel computed on the CPU.
    The Wendland kernel is compactly supported, leading to sparse covariance matrices.

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).
    hps : np.ndarray
        Array of hyperparameters. For this kernel we need D + 1 hyperparameters.

    Return
    ------
    Covariance matrix : np.ndarray
    """
    distance_matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0])): distance_matrix += (np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
    d = np.sqrt(distance_matrix)
    d[d > 1.] = 1.
    return _wendland_anisotropic_polynomial(d, hps[0])


def _wendland_anisotropic_polynomial(d, amplitude):
    return amplitude * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)


def _get_distance_matrix_gpu(x1, x2, device, hps):  # pragma: no cover
    import torch
    d = torch.zeros((len(x1), len(x2))).to(device, dtype=torch.float32)
    for i in range(x1.shape[1]):
        d += ((x1[:, i].reshape(-1, 1) - x2[:, i]) / hps[1 + i]) ** 2
    return torch.sqrt(d)


def wendland_anisotropic_gp2Scale_gpu(x1, x2, hps):  # pragma: no cover
    """
    Function for the anisotropic Wendland kernel computed on the GPU.
    Picks the first usable GPU backend (torch CUDA or MPS, else cupy); falls back
    to the CPU implementation with a UserWarning if no GPU backend is available.
    The Wendland kernel is compactly supported, leading to sparse covariance matrices.

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).
    hps : np.ndarray
        Array of hyperparameters. For this kernel we need D + 1 hyperparameters.

    Return
    ------
    Covariance matrix : np.ndarray
    """
    engine = _get_default_gpu_engine()
    if engine == "torch":
        import torch
        device = _get_torch_gpu_device()
        x1_dev = torch.as_tensor(x1, device=device, dtype=torch.float32)
        x2_dev = torch.as_tensor(x2, device=device, dtype=torch.float32)
        hps_dev = torch.as_tensor(hps, device=device, dtype=torch.float32)
        d = _get_distance_matrix_gpu(x1_dev, x2_dev, device, hps_dev)
        d = torch.clamp(d, max=1.0)
        kernel = hps_dev[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
        return kernel.detach().cpu().numpy()
    if engine == "cupy":
        import cupy as cp
        x1_dev = cp.asarray(x1, dtype=cp.float32)
        x2_dev = cp.asarray(x2, dtype=cp.float32)
        hps_dev = cp.asarray(hps, dtype=cp.float32)
        d = cp.zeros((len(x1), len(x2)), dtype=cp.float32)
        for i in range(x1.shape[1]):
            d += ((x1_dev[:, i].reshape(-1, 1) - x2_dev[:, i]) / hps_dev[1 + i]) ** 2
        d = cp.sqrt(d)
        d = cp.minimum(d, cp.float32(1.0))
        kernel = hps_dev[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
        return cp.asnumpy(kernel)

    warnings.warn(
        "No usable GPU backend was found for wendland_anisotropic_gp2Scale_gpu; "
        "falling back to the CPU Wendland implementation.",
        stacklevel=2,
    )
    return wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)


# --------------------------------------------------------------------------
# GPU backend selection
# --------------------------------------------------------------------------

def _get_torch_gpu_device():  # pragma: no cover
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
        return torch.device(f"cuda:{device_index}")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return None


def _cupy_gpu_available():  # pragma: no cover
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _get_default_gpu_engine():  # pragma: no cover
    if _get_torch_gpu_device() is not None:
        return "torch"
    if _cupy_gpu_available():
        return "cupy"
    return None


# --------------------------------------------------------------------------
# Support-aware (sparse-COO) anisotropic Wendland kernels for gp2Scale.
#
# These mirror wendland_anisotropic_gp2Scale_{cpu,gpu} but return a scipy.sparse
# COO block directly, built from an output-sensitive cKDTree neighbor search in
# the whitened (anisotropy-corrected) coordinates instead of a dense all-pairs
# evaluation.  Intended for gp2Scale workflows where the data has spatial
# locality and batches are sorted by a progression column.
# --------------------------------------------------------------------------

_GP2SCALE_SPARSE_SORT_WARNING_EMITTED = False


def _warn_gp2scale_sparse_kernel_sorting():
    global _GP2SCALE_SPARSE_SORT_WARNING_EMITTED
    if _GP2SCALE_SPARSE_SORT_WARNING_EMITTED:
        return
    warnings.warn(
        "The support-aware gp2Scale Wendland kernels work best when gp2Scale batches preserve locality. "
        "If your data have a progression column, such as day in a climate dataset, sort by that column before batching.",
        stacklevel=2,
    )
    _GP2SCALE_SPARSE_SORT_WARNING_EMITTED = True


def _wendland_triplets_from_neighbor_lists(neighbor_lists):
    row_parts = []
    col_parts = []
    for row_idx, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        cols = np.asarray(neighbors, dtype=np.int64)
        row_parts.append(np.full(cols.size, row_idx, dtype=np.int64))
        col_parts.append(cols)

    if not row_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    return np.concatenate(row_parts), np.concatenate(col_parts)


def _empty_gp2scale_sparse_block(x1, x2):
    return sparse.coo_matrix((len(x1), len(x2)))


def _whiten_gp2scale_points(x1, x2, hps):
    scales = np.asarray(hps[1:], dtype=float)
    z1 = np.asarray(x1, dtype=float) / scales
    z2 = np.asarray(x2, dtype=float) / scales
    return z1, z2


def _gp2scale_whitened_block_distance(z1, z2):
    mins_1 = np.min(z1, axis=0)
    maxs_1 = np.max(z1, axis=0)
    mins_2 = np.min(z2, axis=0)
    maxs_2 = np.max(z2, axis=0)
    gap = np.maximum(0.0, np.maximum(mins_1 - maxs_2, mins_2 - maxs_1))
    return np.linalg.norm(gap)


def _wendland_support_aware_cpu_triplets(x1, x2, hps):
    """
    Output-sensitive COO triplets for the anisotropic Wendland gp2Scale kernel.
    The support condition is an ellipsoid in the original coordinates and a unit
    ball in whitened coordinates, so block assembly can be written as a radius
    search rather than dense all-pairs evaluation.
    """
    if len(x1) == 0 or len(x2) == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    z1, z2 = _whiten_gp2scale_points(x1, x2, hps)
    if _gp2scale_whitened_block_distance(z1, z2) > 1.0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    tree1 = cKDTree(z1)
    tree2 = cKDTree(z2)
    neighbor_lists = tree1.query_ball_tree(tree2, r=1.0)

    rows, cols = _wendland_triplets_from_neighbor_lists(neighbor_lists)
    if rows.size == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    diff = z2[cols] - z1[rows]
    dist_sq = np.sum(diff * diff, axis=1)
    inside_mask = dist_sq <= 1.0
    if not np.all(inside_mask):
        rows = rows[inside_mask]
        cols = cols[inside_mask]
        dist_sq = dist_sq[inside_mask]
        if rows.size == 0:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

    distances = np.sqrt(np.minimum(dist_sq, 1.0))
    values = _wendland_anisotropic_polynomial(distances, hps[0])
    nonzero_mask = values != 0.0
    if not np.all(nonzero_mask):
        rows = rows[nonzero_mask]
        cols = cols[nonzero_mask]
        values = values[nonzero_mask]
        if rows.size == 0:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

    return values, rows, cols


def wendland_anisotropic_gp2Scale_cpu_sparse(x1, x2, hps):
    """
    Support-aware anisotropic Wendland kernel for gp2Scale.

    Preserves the usual kernel interface but performs block-local support checks
    and radius-search assembly internally, returning a sparse COO matrix
    directly. Intended for gp2Scale workflows where the batchwise kernel is
    assembled on sparse blocks. Sort batches by a locality-preserving column
    (e.g. time) for best performance.
    """
    _warn_gp2scale_sparse_kernel_sorting()
    values, rows, cols = _wendland_support_aware_cpu_triplets(x1, x2, hps)
    if values.size == 0:
        return _empty_gp2scale_sparse_block(x1, x2)
    return sparse.coo_matrix((values, (rows, cols)), shape=(len(x1), len(x2)))


def _wendland_support_aware_gpu_triplets(x1, x2, hps):  # pragma: no cover
    """
    Output-sensitive COO triplets for the anisotropic Wendland gp2Scale kernel
    with GPU-backed distance/polynomial evaluation when a usable GPU backend is
    available. Neighbor search remains host-side.
    """
    if len(x1) == 0 or len(x2) == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    z1, z2 = _whiten_gp2scale_points(x1, x2, hps)
    if _gp2scale_whitened_block_distance(z1, z2) > 1.0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    tree1 = cKDTree(z1)
    tree2 = cKDTree(z2)
    neighbor_lists = tree1.query_ball_tree(tree2, r=1.0)
    rows, cols = _wendland_triplets_from_neighbor_lists(neighbor_lists)
    if rows.size == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    diff = z2[cols] - z1[rows]
    engine = _get_default_gpu_engine()
    if engine == "torch":
        import torch
        device = _get_torch_gpu_device()
        diff_dev = torch.as_tensor(diff, device=device, dtype=torch.float32)
        dist_sq_dev = torch.sum(diff_dev * diff_dev, dim=1)
        inside_mask = (dist_sq_dev <= 1.0).detach().cpu().numpy()
        if not np.all(inside_mask):
            rows = rows[inside_mask]
            cols = cols[inside_mask]
            dist_sq_dev = dist_sq_dev[inside_mask]
            if rows.size == 0:
                return (
                    np.empty(0, dtype=float),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )
        distances = torch.sqrt(torch.clamp(dist_sq_dev, max=1.0))
        values = (_wendland_anisotropic_polynomial(distances, float(hps[0]))).detach().cpu().numpy()
    elif engine == "cupy":
        import cupy as cp
        diff_dev = cp.asarray(diff, dtype=cp.float32)
        dist_sq_dev = cp.sum(diff_dev * diff_dev, axis=1)
        inside_mask = cp.asnumpy(dist_sq_dev <= cp.float32(1.0))
        if not np.all(inside_mask):
            rows = rows[inside_mask]
            cols = cols[inside_mask]
            dist_sq_dev = dist_sq_dev[inside_mask]
            if rows.size == 0:
                return (
                    np.empty(0, dtype=float),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )
        distances = cp.sqrt(cp.minimum(dist_sq_dev, cp.float32(1.0)))
        values = cp.asnumpy(_wendland_anisotropic_polynomial(distances, float(hps[0])))
    else:
        warnings.warn(
            "No usable GPU backend was found for the support-aware GPU Wendland kernel; "
            "falling back to the CPU sparse Wendland kernel.",
            stacklevel=2,
        )
        return _wendland_support_aware_cpu_triplets(x1, x2, hps)

    nonzero_mask = values != 0.0
    if not np.all(nonzero_mask):
        rows = rows[nonzero_mask]
        cols = cols[nonzero_mask]
        values = values[nonzero_mask]

    return values, rows, cols


def wendland_anisotropic_gp2Scale_gpu_sparse(x1, x2, hps):  # pragma: no cover
    """
    GPU-backed support-aware anisotropic Wendland kernel for gp2Scale.

    Neighbor discovery remains host-side; only the distance/polynomial
    evaluation on the discovered support graph is GPU-backed when a usable
    backend (torch CUDA/MPS or cupy) is available. Falls back to the CPU
    sparse variant with a UserWarning otherwise.
    """
    _warn_gp2scale_sparse_kernel_sorting()
    values, rows, cols = _wendland_support_aware_gpu_triplets(x1, x2, hps)
    if values.size == 0:
        return _empty_gp2scale_sparse_block(x1, x2)
    return sparse.coo_matrix((values, (rows, cols)), shape=(len(x1), len(x2)))


def wasserstein_1d(a, b):
    """
    The 1d Wasserstein distance.

    Parameters
    ----------
    a : np.ndarray
        1d Numpy array. Input distribution.
    b : np.ndarray
        1d Numpy array. Input distribution.

    Return
    ------
    Wasserstein distance : float
    """
    a = a / np.sum(a)
    b = b / np.sum(b)
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    return np.mean(np.abs(a_sorted - b_sorted))


def wasserstein_1d_outer_vec(a, b):
    """
    Vectorized pairwise 1-D Wasserstein distance between all rows of ``a`` and ``b``.

    Parameters
    ----------
    a : np.ndarray
        Array of shape (M, K); each row is an unnormalized 1-D measure.
    b : np.ndarray
        Array of shape (N, K); each row is an unnormalized 1-D measure.

    Returns
    -------
    W : np.ndarray
        Distance matrix of shape (M, N).
    """
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)

    a_sorted = np.sort(a, axis=1)
    b_sorted = np.sort(b, axis=1)
    s = a_sorted[:, None, :] - b_sorted[None, :, :]
    return np.mean(np.abs(s), axis=2)


def bump(d, r, beta=1., ampl=1.):
    """
    Smooth compactly-supported bump function evaluated over a distance array.

    Parameters
    ----------
    d : np.ndarray
        Distance values (non-negative).
    r : float
        Support radius; the function is zero for ``d >= r``.
    beta : float, optional
        Sharpness parameter (default 1).
    ampl : float, optional
        Amplitude scale (default 1).

    Returns
    -------
    bump : np.ndarray
        Bump values, same shape as ``d``.
    """
    #x_new = x - x0
    #d = np.linalg.norm(x_new, axis = 1)
    a = np.zeros(d.shape)
    a = 1.0 - (d**2/r**2)
    i = np.where(a > 0.0)
    bump = np.zeros(a.shape)
    e = np.exp((-beta/a[i])+beta)
    bump[i] = ampl * e
    return bump


def sle_kernel(x1, x2, hps, args):
    """
    Sparse-Landmark-Embedding (SLE) kernel.

    Embeds points via a bump-function basis centered on the training set, then
    applies a squared-exponential kernel on the embedded space.
    Requires ``args["x_data"]`` (the training locations) to construct the basis.

    Parameters
    ----------
    x1 : np.ndarray
        Query points, shape (N1, D).
    x2 : np.ndarray
        Query points, shape (N2, D).
    hps : np.ndarray
        Hyperparameters ``[amplitude, radius, beta, length_scale]``.
    args : dict
        Must contain key ``"x_data"`` with the training point locations.

    Returns
    -------
    K : np.ndarray
        Covariance matrix of shape (N1, N2).
    """
    # Distance to each training point (using all dimensions)
    x_data = args["x_data"]
    d1 = get_distance_matrix(x1, x_data)  # (n1, m) - can be geodesic/non-cnd!
    d2 = get_distance_matrix(x2, x_data)  # (n2, m)

    # Apply bump to create embedding
    phi_x1 = bump(d1, hps[1], beta=hps[2], ampl=1.0)  # (n1, m)
    phi_x2 = bump(d2, hps[1], beta=hps[2], ampl=1.0)  # (n2, m)

    # RBF on embeddings
    D_mat = cdist(phi_x1, phi_x2, metric='euclidean')
    return hps[0] * np.exp(-D_mat ** 2 / hps[3])
