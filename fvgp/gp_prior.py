from typing import List

import numpy as np
from .kernels import *
import warnings
import itertools
import scipy.sparse as sparse
import time
from loguru import logger

from .utils import log_time


class GPprior:
    def __init__(self,
                 data,
                 kernel_function=None,
                 kernel_function_grad=None,
                 prior_mean_function=None,
                 prior_mean_function_grad=None,
                 hyperparameters=None,
                 ram_economy=False,
                 compute_device='cpu',
                 gp2Scale=False,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000
                 ):

        assert callable(kernel_function) or kernel_function is None
        assert callable(prior_mean_function) or prior_mean_function is None
        assert isinstance(hyperparameters, np.ndarray)
        assert np.ndim(hyperparameters) == 1

        self.Euclidean = data.Euclidean
        self.kernel_function = kernel_function
        self.prior_mean_function = prior_mean_function
        self.hyperparameters = hyperparameters
        self.ram_economy = ram_economy
        self.gp2Scale = gp2Scale
        self.client = gp2Scale_dask_client
        self.batch_size = gp2Scale_batch_size
        self.data = data

        if not self.Euclidean and not callable(kernel_function):
            raise Exception(
                "For GPs on non-Euclidean input spaces you need a user-defined kernel and initial hyperparameters.")

        if gp2Scale:
            if not callable(kernel_function):
                warnings.warn("You have chosen to activate gp2Scale. A powerful tool!"
                              "But you have not supplied a kernel that is compactly supported."
                              "I will use an anisotropic Wendland kernel for now.",
                              stacklevel=2)
                if compute_device == "cpu":
                    kernel_function = wendland_anisotropic_gp2Scale_cpu
                elif compute_device == "gpu":
                    kernel_function = wendland_anisotropic_gp2Scale_gpu
            worker_info = list(self.client.scheduler_info()["workers"].keys())
            if not worker_info: raise Exception("No workers available")
            self.compute_workers = list(worker_info)

        # kernel
        if callable(kernel_function):
            self.kernel = kernel_function
        elif kernel_function is None:
            self.kernel = self._default_kernel
        else:
            raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self._d_kernel_dx
        if callable(kernel_function_grad):
            self.dk_dh = kernel_function_grad
        else:
            if self.ram_economy is True:
                self.dk_dh = self._kernel_derivative
            else:
                self.dk_dh = self._kernel_gradient

        # prior mean
        if callable(prior_mean_function):
            self.mean_function = prior_mean_function
            self.default_m = False
        else:
            self.mean_function = self._default_mean_function
            self.default_m = True

        if callable(prior_mean_function_grad):
            self.dm_dh = prior_mean_function_grad
        elif callable(prior_mean_function):
            self.dm_dh = self._finitediff_dm_dh
        else:
            self.dm_dh = self._default_dm_dh

        self.m, self.K = self._compute_prior(data.x_data)

    def augment_data(self, x_old, x_new):
        self.m, self.K = self._update_prior(x_old, x_new)

    def update_data(self):
        self.m, self.K = self._compute_prior(self.data.x_data)

    def update_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.m, self.K = self._compute_prior(self.data.x_data)

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
        if self.default_m:
            m = np.zeros((len(self.m) + len(x_new)))
            m[:] = np.mean(self.data.y_data)
        m = np.append(self.m, self.mean_function(x_new, self.hyperparameters))
        return m

    @staticmethod
    def _ranges(N, nb):
        """ splits a range(N) into nb chunks defined by chunk_start, chunk_end """
        if nb == 0: nb = 1
        step = N / nb
        return [(round(step * i), round(step * (i + 1))) for i in range(nb)]

    def _compute_prior_covariance_gp2Scale(self, x_data, hyperparameters):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        with log_time(cumulative_key='compute_prior_covariance_gp2Scale'):
            st = time.time()
            client = self.client
            point_number = len(x_data)
            num_batches = -(-point_number // self.batch_size)
            NUM_RANGES = num_batches
            logger.debug("client id: {}", client.id)

            self.x_data_scatter_future = client.scatter(
                x_data, workers=self.compute_workers, broadcast=True)
            ranges = self._ranges(len(x_data), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
            ranges_ij = list(
                itertools.product(ranges, ranges))  # all i/j ranges as ((i_start, i_end), (j_start, j_end)) pairs of tuples
            ranges = np.array(ranges_ij).reshape(NUM_RANGES, NUM_RANGES, 2, 2)
            logger.debug("        gp2Scale covariance matrix init done after {} seconds.", time.time() - st)

            dsk = {f'kernel_{i}_{j}': (kernel_function,
                                       ranges[i][j],
                                       self.x_data_scatter_future,
                                       self.x_data_scatter_future,
                                       hyperparameters,
                                       self.kernel)
                   for i in range(NUM_RANGES) for j in range(NUM_RANGES)
                   if i<=j
                   }

            dsk.update({f'stack_blocks_{r}':(self.stack_blocks,
                    [f'kernel_{i}_{r}' for i in range(0, r)], # blocks that need to be reflected up to upper triangle
                    [f'kernel_{r}_{j}' for j in range(r, NUM_RANGES)] # blocks in the row on upper triangle
                                             )
                        for r in range(NUM_RANGES)})

            dsk.update({f'make_csr_{r}':(self.make_csr, f'stack_blocks_{r}')
                        for r in range(NUM_RANGES)})

            dsk.update({'stack_csr':(self.stack_csr, [f'make_csr_{r}' for r in range(NUM_RANGES)])})

            K = client.get(dsk, 'stack_csr')

            logger.debug("        gp2Scale covariance matrix assembled after {} seconds.", time.time() - st)
            logger.debug("        gp2Scale covariance matrix sparsity = {}.", float(K.nnz) / float(K.shape[0] ** 2))
        return K

    def _update_prior_covariance_gp2Scale(self, x_old, x_new, hyperparameters):
        with log_time(cumulative_key='compute_prior_covariance_gp2Scale'):
            st = time.time()
            client = self.client
            """computes the covariance matrix from the kernel on HPC in sparse format"""

            self.x_new_scatter_future = client.scatter(
                x_new, workers=self.compute_workers, broadcast=True)
            self.x_old_scatter_future = client.scatter(
                x_old, workers=self.compute_workers, broadcast=True)

            point_number = len(x_old)
            num_batches = -(-point_number // self.batch_size)
            ranges_data = self._ranges(len(x_old), num_batches)  # the chunk ranges, as (start, end) tuples
            num_batches2 = -(-len(x_new) // self.batch_size)
            ranges_input = self._ranges(len(x_new), num_batches2)
            ranges_ij = list(itertools.product(ranges_data, ranges_input))

            ranges_ij2 = list(itertools.product(ranges_input, ranges_input))
            ranges_ij2 = [range_ij2 for range_ij2 in ranges_ij2 if
                          range_ij2[0][0] <= range_ij2[1][0]]  # filter lower diagonal

            ranges = np.array(ranges_ij).reshape(num_batches, num_batches2, 2, 2)
            ranges_corner = np.array(ranges_ij2).reshape(num_batches2, num_batches2, 2, 2)

            dsk = {f'kernel_{i}_{j}': (kernel_function,
                                   ranges[i][j],
                                   self.x_old_scatter_future,
                                   self.x_new_scatter_future,
                                   hyperparameters,
                                   self.kernel)
                   for i in range(num_batches) for j in range(num_batches2)
                   }

            dsk.update({f'kernel_corner_{i}_{j}': (kernel_function,
                                   ranges_corner[i][j],
                                   self.x_new_scatter_future,
                                   self.x_new_scatter_future,
                                   hyperparameters,
                                   self.kernel)
                   for i in range(num_batches2) for j in range(num_batches2)
                   })

            dsk.update({f'stack_blocks_upper_{r}':(self.stack_blocks,
                                             [], # blocks that need to be reflected up to upper triangle
                                             [f'kernel_{r}_{j}' for j in range(num_batches2)] # blocks in the row on upper triangle
                                             )
                        for r in range(num_batches)})

            dsk.update({f'stack_blocks_lower_{r}':(self.stack_blocks,
                                             [f'kernel_{j}_{r}' for j in range(num_batches)],
                                             []
                                             )
                        for r in range(num_batches2)})

            dsk.update({f'stack_blocks_corner_{r}': (self.stack_blocks,
                                                     [f'kernel_corner_{i}_{r}' for i in range(0, r)],
                                                     [f'kernel_corner_{r}_{j}' for j in range(r, num_batches2)]
                                                    )
                        for r in range(num_batches2)})

            dsk.update({f'make_csr_upper_{r}':(self.make_csr, f'stack_blocks_upper_{r}')
                        for r in range(num_batches)})

            dsk.update({f'make_csr_lower_{r}':(self.make_csr, f'stack_blocks_lower_{r}')
                        for r in range(num_batches2)})

            dsk.update({f'make_csr_corner_{r}':(self.make_csr, f'stack_blocks_corner_{r}')
                        for r in range(num_batches2)})

            dsk.update({'stack_csr_upper':(self.stack_csr, [f'make_csr_upper_{r}' for r in range(num_batches)])})

            dsk.update({'stack_csr_lower': (self.stack_csr, [f'make_csr_lower_{r}' for r in range(num_batches2)])})

            dsk.update({'stack_csr_corner': (self.stack_csr, [f'make_csr_corner_{r}' for r in range(num_batches2)])})

            B, B_T, C = client.get(dsk, ['stack_csr_upper', 'stack_csr_lower', 'stack_csr_corner'])

            K = sparse.block_array([[self.K, B],[B_T, C]], format='csr')

            logger.debug("        gp2Scale covariance matrix assembled after {} seconds.", time.time() - st)
            logger.debug("        gp2Scale covariance matrix sparsity = {}.", float(K.nnz) / float(K.shape[0] ** 2))

            return K

    @staticmethod
    def stack_blocks(symmetric_blocks: List[sparse.coo_matrix], row_blocks: List[sparse.coo_matrix]):
        transpose_blocks = [block.T for block in symmetric_blocks]
        blocks = [*transpose_blocks, *row_blocks]
        return sparse.hstack(blocks)

    @staticmethod
    def make_csr(block_row: sparse.coo_matrix):
        csr = block_row.tocsr()
        return csr

    @staticmethod
    def stack_csr(block_rows: List[sparse.coo_matrix]):
        data = np.hstack([block_row.data for block_row in block_rows])
        indices = np.hstack([block_row.indices for block_row in block_rows])

        indptr = []
        last_indptr = 0
        for block_row in block_rows:
            indptr.append(block_row.indptr[:-1]+last_indptr)
            last_indptr += block_row.indptr[-1]
        indptr.append([last_indptr])

        indptr = np.hstack(indptr)

        shape = (np.sum([block_row.shape[0] for block_row in block_rows]), block_rows[0].shape[1])


        csr = sparse.csr_matrix((data, indices, indptr), shape=shape)
        csr._has_sorted_indices = True
        return csr

    ####################################################
    ####################################################
    ####################################################
    ####################################################
    @staticmethod
    def _default_kernel(x1, x2, hyperparameters):
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

    def _d_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-8
        new_points[:, direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters)
        b = self.kernel(points1, points2, hyperparameters)
        derivative = (a - b) / epsilon
        return derivative

    def _kernel_gradient(self, points1, points2, hyperparameters):
        gradient = np.empty((len(hyperparameters), len(points1), len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def _kernel_derivative(self, points1, points2, direction, hyperparameters):
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
        mean[:] = np.mean(self.data.y_data)
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
    shape = (range_i[1] - range_i[0], range_j[1] - range_j[0])
    x1 = x1_future[range_i[0]:range_i[1]]
    x2 = x2_future[range_j[0]:range_j[1]]
    k = kernel(x1, x2, hps)
    k_sparse = sparse.coo_matrix(k)

    data, rows, cols = k_sparse.data, k_sparse.row, k_sparse.col

    return sparse.coo_matrix((data, (rows, cols)), shape=shape)


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
