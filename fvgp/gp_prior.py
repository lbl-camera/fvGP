import numpy as np
from .gp_kernels import *
import dask.distributed as distributed
import warnings
import itertools
from functools import partial
import scipy.sparse as sparse
from scipy.sparse import block_array
import time
from loguru import logger


class GPprior:
    def __init__(self,
                 index_set_dim,
                 x_data,
                 Euclidean,
                 gp_kernel_function=None,
                 gp_kernel_function_grad=None,
                 gp_mean_function=None,
                 gp_mean_function_grad=None,
                 hyperparameters=None,
                 ram_economy=False,
                 compute_device='cpu',
                 constant_mean=0.0,
                 gp2Scale=False,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000
                 ):

        assert callable(gp_kernel_function) or gp_kernel_function is None
        assert callable(gp_mean_function) or gp_mean_function is None
        assert isinstance(hyperparameters, np.ndarray)
        assert np.ndim(hyperparameters) == 1
        assert isinstance(constant_mean, float)

        self.index_set_dim = index_set_dim
        self.Euclidean = Euclidean
        self.gp_kernel_function = gp_kernel_function
        self.gp_mean_function = gp_mean_function
        self.hyperparameters = hyperparameters
        self.ram_economy = ram_economy
        self.constant_mean = constant_mean
        self.gp2Scale = gp2Scale
        self.client = gp2Scale_dask_client
        self.batch_size = gp2Scale_batch_size

        if not self.Euclidean and not callable(gp_kernel_function):
            raise Exception(
                "For GPs on non-Euclidean input spaces you need a user-defined kernel and initial hyperparameters.")

        if gp2Scale:
            if not callable(gp_kernel_function):
                warnings.warn("You have chosen to activate gp2Scale. A powerful tool!"
                              "But you have not supplied a kernel that is compactly supported."
                              "I will use an anisotropic Wendland kernel for now.",
                              stacklevel=2)
                if compute_device == "cpu":
                    gp_kernel_function = wendland_anisotropic_gp2Scale_cpu
                elif compute_device == "gpu":
                    gp_kernel_function = wendland_anisotropic_gp2Scale_gpu
            worker_info = list(self.client.scheduler_info()["workers"].keys())
            if not worker_info: raise Exception("No workers available")
            self.compute_workers = list(worker_info)

        # kernel
        if callable(gp_kernel_function):
            self.kernel = gp_kernel_function
        elif gp_kernel_function is None:
            self.kernel = self._default_kernel
        else:
            raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self._d_gp_kernel_dx
        if callable(gp_kernel_function_grad):
            self.dk_dh = gp_kernel_function_grad
        else:
            if self.ram_economy is True:
                self.dk_dh = self._gp_kernel_derivative
            else:
                self.dk_dh = self._gp_kernel_gradient

        # prior mean
        if callable(gp_mean_function):
            self.mean_function = gp_mean_function
            self.default_m = False
        else:
            self.mean_function = self._default_mean_function
            self.default_m = True

        if callable(gp_mean_function_grad):
            self.dm_dh = gp_mean_function_grad
        elif callable(gp_mean_function):
            self.dm_dh = self._finitediff_dm_dh
        else:
            self.dm_dh = self._default_dm_dh

        self.m, self.K = self._compute_prior(x_data)

    def augment_data(self, x_old, x_new, constant_mean=0.0):
        self.constant_mean = constant_mean
        self.m, self.K = self._update_prior(x_old, x_new)

    def update_data(self, x_data, constant_mean=0.0):
        self.constant_mean = constant_mean
        self.m, self.K = self._compute_prior(x_data)

    def update_hyperparameters(self, x_data, hyperparameters):
        self.hyperparameters = hyperparameters
        self.m, self.K = self._compute_prior(x_data)

    def _compute_prior(self, x_data):
        m = self.compute_mean(x_data)
        K = self.compute_prior_covariance_matrix(x_data)
        assert np.ndim(m) == 1
        assert np.ndim(K) == 2
        return m, K

    def _update_prior(self, x_old, x_new):
        m = self._update_mean(x_new)
        K = self._update_prior_covariance_matrix(x_old, x_new)
        assert np.ndim(m) == 1
        assert np.ndim(K) == 2
        return m, K

    def compute_prior_covariance_matrix(self, x, hyperparameters=None):
        """computes the covariance matrix from the kernel"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if self.gp2Scale:
            K = self._compute_prior_covariance_gp2Scale(x, hyperparameters)
        else:
            K = self.kernel(x, x, hyperparameters)
        return K

    def _update_prior_covariance_matrix(self, x_old, x_new):
        """This updated K based on new data"""
        if self.gp2Scale:
            K = self._update_prior_covariance_gp2Scale(x_old, x_new, self.hyperparameters)
        else:
            k = self.kernel(x_old, x_new, self.hyperparameters)
            kk = self.kernel(x_new, x_new, self.hyperparameters)
            K = np.block([
                [self.K, k],
                [k.T, kk]
            ])
        return K

    def compute_mean(self, x_data, hyperparameters=None):
        """computes the covariance matrix from the kernel"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        m = self.mean_function(x_data, hyperparameters)
        return m

    def _update_mean(self, x_new):
        if self.default_m: self.m[:] = self.constant_mean
        m = np.append(self.m, self.mean_function(x_new, self.hyperparameters))
        return m

    @staticmethod
    def ranges(N, nb):
        """ splits a range(N) into nb chunks defined by chunk_start, chunk_end """
        if nb == 0: nb = 1
        step = N / nb
        return [(round(step * i), round(step * (i + 1))) for i in range(nb)]

    def _compute_prior_covariance_gp2Scale(self, x_data, hyperparameters):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        st = time.time()
        client = self.client
        point_number = len(x_data)
        num_batches = point_number // self.batch_size
        NUM_RANGES = num_batches
        logger.debug("client id: {}", client.id)

        self.x_data_scatter_future = client.scatter(
            x_data, workers=self.compute_workers, broadcast=True)
        ranges = self.ranges(len(x_data), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
        ranges_ij = list(
            itertools.product(ranges, ranges))  # all i/j ranges as ((i_start, i_end), (j_start, j_end)) pairs of tuples
        ranges_ij = [range_ij for range_ij in ranges_ij if range_ij[0][0] <= range_ij[1][0]]  # filter lower diagonal
        logger.debug("        gp2Scale covariance matrix init done after {} seconds.", time.time() - st)

        results = list(map(self.harvest_result, distributed.as_completed(client.map(
            partial(kernel_function,
                    hyperparameters=hyperparameters,
                    kernel=self.kernel),
            ranges_ij,
            [self.x_data_scatter_future] * len(ranges_ij),
            [self.x_data_scatter_future] * len(ranges_ij)),
            with_results=True)))

        logger.debug("        gp2Scale covariance matrix result written after {} seconds.", time.time() - st)

        # reshape the result set into COO components
        data, i_s, j_s = map(np.hstack, zip(*results))
        # mirror across diagonal
        diagonal_mask = i_s != j_s
        data, i_s, j_s = np.hstack([data, data[diagonal_mask]]), \
            np.hstack([i_s, j_s[diagonal_mask]]), \
            np.hstack([j_s, i_s[diagonal_mask]])
        K = sparse.coo_matrix((data, (i_s, j_s)))
        K.resize(len(x_data), len(x_data))
        logger.debug("        gp2Scale covariance matrix assembled after {} seconds.", time.time() - st)
        return K

    def _update_prior_covariance_gp2Scale(self, x_old, x_new, hyperparameters):
        client = self.client
        """computes the covariance matrix from the kernel on HPC in sparse format"""

        self.x_new_scatter_future = client.scatter(
            x_new, workers=self.compute_workers, broadcast=True)
        self.x_old_scatter_future = client.scatter(
            x_old, workers=self.compute_workers, broadcast=True)

        point_number = len(x_old)
        num_batches = point_number // self.batch_size
        NUM_RANGES = num_batches
        ranges_data = self.ranges(len(x_old), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
        num_batches2 = len(x_new) // self.batch_size
        ranges_input = self.ranges(len(x_new), num_batches2)
        ranges_ij = list(itertools.product(ranges_data, ranges_input))

        # K = np.block([[self.K, B],
        #               [B,      C]])
        # Calculate B

        results = list(map(self.harvest_result,
                           distributed.as_completed(client.map(
                               partial(kernel_function_update,
                                       hyperparameters=hyperparameters,
                                       kernel=self.kernel),
                               ranges_ij,
                               [self.x_old_scatter_future] * len(ranges_ij),
                               [self.x_new_scatter_future] * len(ranges_ij)),
                               with_results=True)))

        data, i_s, j_s = map(np.hstack, zip(*results))
        B = sparse.coo_matrix((data, (i_s, j_s)))
        B.resize(len(x_old), len(x_new))

        # mirror across diagonal
        ranges_ij2 = list(itertools.product(ranges_input, ranges_input))
        ranges_ij2 = [range_ij2 for range_ij2 in ranges_ij2 if
                      range_ij2[0][0] <= range_ij2[1][0]]  # filter lower diagonal

        results = list(map(self.harvest_result,
                           distributed.as_completed(client.map(
                               partial(kernel_function,
                                       hyperparameters=hyperparameters,
                                       kernel=self.kernel),
                               ranges_ij2,
                               [self.x_new_scatter_future] * len(ranges_ij2),
                               [self.x_new_scatter_future] * len(ranges_ij2)),
                               with_results=True)))
        data, i_s, j_s = map(np.hstack, zip(*results))
        diagonal_mask = i_s != j_s
        data, i_s, j_s = np.hstack([data, data[diagonal_mask]]), \
            np.hstack([i_s, j_s[diagonal_mask]]), \
            np.hstack([j_s, i_s[diagonal_mask]])
        D = sparse.coo_matrix((data, (i_s, j_s)))
        D.resize(len(x_new), len(x_new))

        res = block_array([[self.K, B],
                           [B.transpose(), D]])

        return res

    @staticmethod
    def harvest_result(future_result):
        future, result = future_result
        future.release()
        return result

    ####################################################
    ####################################################
    ####################################################
    ####################################################

    def _default_kernel(self, x1, x2, hyperparameters):
        """
        Function for the default kernel, a Matern kernel of first-order differentiability.

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
        for i in range(len(x1[0])):
            distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        distance_matrix = np.sqrt(distance_matrix)
        return hps[0] * matern_kernel_diff1(distance_matrix, 1)

    def _d_gp_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-8
        new_points[:, direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters)
        b = self.kernel(points1, points2, hyperparameters)
        derivative = (a - b) / epsilon
        return derivative

    def _gp_kernel_gradient(self, points1, points2, hyperparameters):
        gradient = np.empty((len(hyperparameters), len(points1), len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def _gp_kernel_derivative(self, points1, points2, direction, hyperparameters):
        # gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        derivative = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return derivative

    def _dkernel_dh(self, points1, points2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-8
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyperparameters1)
        b = self.kernel(points1, points2, new_hyperparameters2)
        derivative = (a - b) / (2.0 * epsilon)
        return derivative

    def _default_mean_function(self, x, hyperparameters):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)))
        mean[:] = self.constant_mean
        return mean

    def _finitediff_dm_dh(self, x, hps):
        gr = np.empty((len(hps), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.mean_function(x, temp_hps1)
            b = self.mean_function(x, temp_hps2)
            gr[i] = (a - b) / 2e-6
        return gr

    def _default_dm_dh(self, x, hps):
        gr = np.zeros((len(hps), len(x)))
        return gr


########################################################
########################################################
########################################################


def kernel_function(range_ij, x1_future, x2_future, hyperparameters, kernel):
    """
    Essentially, parameters other than range_ij are static across calls. range_ij defines the region of the
    covariance matrix being calculated.
    Rather than return a sparse array in local coordinates, we can return the COO components in global coordinates.
    """

    hps = hyperparameters
    range_i, range_j = range_ij
    x1 = x1_future[range_i[0]:range_i[1]]
    x2 = x2_future[range_j[0]:range_j[1]]
    k = kernel(x1, x2, hps)
    k_sparse = sparse.coo_matrix(k)

    data, rows, cols = k_sparse.data, k_sparse.row + range_i[0], k_sparse.col + range_j[0]

    # mask lower triangular values when current chunk spans diagonal
    if range_i[0] == range_j[0]:
        mask = [row <= col for (row, col) in zip(rows, cols)]
        return data[mask], rows[mask], cols[mask]
    else:
        return data, rows, cols


def kernel_function_update(range_ij, x1_future, x2_future, hyperparameters, kernel):
    """
    Essentially, parameters other than range_ij are static across calls. range_ij defines the region of the
    covariance matrix being calculated.
    Rather than return a sparse array in local coordinates, we can return the COO components in global coordinates.
    """

    hps = hyperparameters
    range_i, range_j = range_ij
    x1 = x1_future[range_i[0]:range_i[1]]
    x2 = x2_future[range_j[0]:range_j[1]]
    k = kernel(x1, x2, hps)
    k_sparse = sparse.coo_matrix(k)

    data, rows, cols = k_sparse.data, k_sparse.row + range_i[0], k_sparse.col + range_j[0]

    return data, rows, cols
