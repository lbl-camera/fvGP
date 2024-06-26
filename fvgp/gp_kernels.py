import numpy as np


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
    kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
        -(np.sqrt(3.0) * distance) / length

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


def matern_kernel_diff1_robust(distance, phi):
    """
    Function for the Matern kernel, order of differentiability = 1, robust version.
    kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi**2))) * np.exp(
        -(np.sqrt(3.0) * distance) * (phi**2))

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
    kernel = (
        1.0
        + ((np.sqrt(5.0) * distance) / (length))
        + ((5.0 * distance ** 2) / (3.0 * length ** 2))
    ) * np.exp(-(np.sqrt(5.0) * distance) / length)

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
    kernel = (
        1.0
        + ((np.sqrt(5.0) * distance) * (phi**2))
        + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
    ) * np.exp(-(np.sqrt(5.0) * distance) * (phi**2))

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
    Function for a polynomial kernel.
    kernel = (1.0+x1.T @ x2)**p

    Parameters
    ----------
    x1 : np.ndarray
        Point 1.
    x2 : np.ndarray
        Point 2.
    p : float
        Power hyperparameter.

    Return
    ------
    Kernel output : same as distance parameter.
    """
    kernel = (1.0 + x1.T @ x2) ** p
    return p


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
    obj : object instance
        GP object instance.

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
    distance_matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0])): distance_matrix += (np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
    d = np.sqrt(distance_matrix)
    d[d > 1.] = 1.
    kernel = hps[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return kernel


def _get_distance_matrix_gpu(x1, x2, device, hps):  # pragma: no cover
    import torch
    d = torch.zeros((len(x1), len(x2))).to(device, dtype=torch.float32)
    for i in range(x1.shape[1]):
        d += ((x1[:, i].reshape(-1, 1) - x2[:, i]) / hps[1 + i]) ** 2
    return torch.sqrt(d)


def wendland_anisotropic_gp2Scale_gpu(x1, x2, hps):  # pragma: no cover
    import torch
    cuda_device = torch.device("cuda:0")
    x1_dev = torch.from_numpy(x1).to(cuda_device, dtype=torch.float32)
    x2_dev = torch.from_numpy(x2).to(cuda_device, dtype=torch.float32)
    hps_dev = torch.from_numpy(hps).to(cuda_device, dtype=torch.float32)
    d = _get_distance_matrix_gpu(x1_dev, x2_dev, cuda_device, hps_dev)
    d[d > 1.] = 1.
    kernel = hps[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return kernel.cpu().numpy()


