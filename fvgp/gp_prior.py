import numpy as np
from .kernels import *
import dask.distributed as distributed
import warnings

warnings.simplefilter("once", UserWarning)
import itertools
from functools import partial
import scipy.sparse as sparse
from scipy.sparse import block_array
import time
from loguru import logger
from scipy.sparse import coo_matrix, vstack
import inspect


class GPprior:
    def __init__(self,
                 data,
                 trainer,
                 kernel=None,
                 prior_mean_function=None,
                 kernel_grad=None,
                 prior_mean_function_grad=None,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000,
                 ):

        self.kernel_function = kernel
        self.prior_mean_function = prior_mean_function
        self.client = gp2Scale_dask_client
        self.batch_size = gp2Scale_batch_size
        self.data = data
        self.trainer = trainer

        assert callable(kernel) or kernel is None
        assert callable(prior_mean_function) or prior_mean_function is None
        assert isinstance(self.hyperparameters, np.ndarray)
        assert np.ndim(self.hyperparameters) == 1

        if not self.Euclidean and not callable(kernel):
            raise Exception(
                "For GPs on non-Euclidean input spaces you need a user-defined kernel and initial hyperparameters.")

        if self.gp2Scale:
            if not callable(kernel):
                warnings.warn("You have chosen to activate gp2Scale. A powerful tool!"
                              "But you have not supplied a kernel that is compactly supported."
                              "I will use an anisotropic Wendland kernel for now.",
                              stacklevel=2)
                if self.compute_device == "cpu":
                    kernel = wendland_anisotropic_gp2Scale_cpu
                elif self.compute_device == "gpu":
                    kernel = wendland_anisotropic_gp2Scale_gpu
            if self.client is not None:
                worker_info = list(self.client.scheduler_info()["workers"].keys())
                self.compute_workers = list(worker_info)
            else:
                worker_info = False
            if not worker_info: logger.debug("No workers available")

        # kernel
        self.k_n_params = 3
        if callable(kernel):
            self.kernel = kernel
            self.k_n_params = len(inspect.signature(kernel).parameters)
        elif kernel is None:
            self.kernel = self._default_kernel
        else:
            raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self._d_kernel_dx
        if callable(kernel_grad):
            self._dk_dh = kernel_grad
        elif not callable(kernel):
            self._dk_dh = self._default_kernel_analytical_gradient
            self.ram_economy = False
        else:
            if self.ram_economy is True:
                self._dk_dh = self._kernel_derivative
            else:
                self._dk_dh = self._kernel_gradient

        # prior-mean
        self.m_n_params = 2
        if callable(prior_mean_function):
            self.mean_function = prior_mean_function
            self.m_n_params = len(inspect.signature(prior_mean_function).parameters)
        else:
            self.mean_function = self._default_mean_function

        if callable(prior_mean_function_grad):
            self._dm_dh = prior_mean_function_grad
        elif callable(prior_mean_function):
            self._dm_dh = self._finitediff_dm_dh
        else:
            self._dm_dh = self._default_dm_dh

        self.m, self.K = self._compute_prior(data.x_data, self.hyperparameters)
        logger.debug("Prior successfully initialized.")

    ##############################################################
    @property
    def args(self):
        return self.data.args

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters

    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def ram_economy(self):
        return self.data.ram_economy

    @ram_economy.setter
    def ram_economy(self, value):
        self.data.ram_economy = value

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    @property
    def Euclidean(self):
        return self.data.Euclidean

    @property
    def compute_device(self):
        return self.data.compute_device

    ################################################################
    #START: FUNCTIONS THAT ALLOW INTERACTING WITH THE CLASS
    def augment_state_data(self, x_old, x_new):
        self.m, self.K = self._update_prior(x_old, x_new, self.hyperparameters)
        logger.debug("Prior mean and covariance updated after data augmentation.")

    def update_state_data(self):
        self.m, self.K = self._compute_prior(self.x_data, self.hyperparameters)
        logger.debug("Prior mean and covariance updated after data change.")

    def update_state_hyperparameters(self):
        self.m, self.K = self._compute_prior(self.x_data, self.hyperparameters)
        logger.debug("Prior mean and covariance updated after hyperparameter change.")

    def compute_prior_covariance_matrix(self, x, hyperparameters):
        """computes the covariance matrix from the kernel"""
        if self.gp2Scale:
            K = self._compute_prior_covariance_gp2Scale(x, hyperparameters)
        else:
            K = self.compute_covariances(x, x, hyperparameters)
        return K

    def compute_covariances(self, x1, x2, hps):
        if self.k_n_params == 3:
            return self.kernel(x1, x2, hps)
        elif self.k_n_params == 4:
            return self.kernel(x1, x2, hps, self.args)
        else:
            raise Exception("No valid kernel function signature")

    def compute_mean(self, x, hyperparameters):
        """computes the covariance matrix from the kernel"""
        if self.m_n_params == 2:
            m = self.mean_function(x, hyperparameters)
        elif self.m_n_params == 3:
            m = self.mean_function(x, hyperparameters, self.args)
        else:
            raise Exception("No valid mean function signature")
        return m

    def dk_dh(self, x1, x2, hyperparameters, direction=None):
        if self.ram_economy:
            return self._dk_dh(x1, x2, hyperparameters, direction)
        else:
            return self._dk_dh(x1, x2, hyperparameters)

    def dm_dh(self, x_data, hyperparameters):
        return self._dm_dh(x_data, hyperparameters)

    #END: FUNCTIONS THAT ALLOW INTERACTING WITH THE CLASS
    #################################################################

    def _compute_prior(self, x_data, hyperparameters):
        m = self.compute_mean(x_data, hyperparameters)
        K = self.compute_prior_covariance_matrix(x_data, hyperparameters)
        assert np.ndim(m) == 1, "mean: " + str(m)
        assert np.ndim(K) == 2
        logger.debug("Prior mean and covariance matrix successfully computed.")
        return m, K

    def _update_prior(self, x_old, x_new, hyperparameters):
        m = self._update_mean(x_new, hyperparameters)
        K = self._update_prior_covariance_matrix(x_old, x_new, hyperparameters)
        assert np.ndim(m) == 1
        assert np.ndim(K) == 2
        return m, K

    def _update_prior_covariance_matrix(self, x_old, x_new, hyperparameters):
        """This updated K based on new data"""
        if self.gp2Scale:
            K = self._update_prior_covariance_gp2Scale(x_old, x_new, hyperparameters)
        else:
            k = self.compute_covariances(x_old, x_new, hyperparameters)
            kk = self.compute_covariances(x_new, x_new, hyperparameters)
            K = np.block([
                [self.K, k],
                [k.T, kk]
            ])
        return K

    def _update_mean(self, x_new, hyperparameters):
        if np.ndim(self.m) == 1:
            m = np.append(self.m, self.compute_mean(x_new, hyperparameters))
        elif np.ndim(self.m) == 2:
            raise Exception(
                "prior mean has to be a vector")  #m = np.vstack([self.m, self.compute_mean(x_new, hyperparameters)])
        else:
            raise Exception("Prior mean in wrong format")
        return m

    @staticmethod
    def _ranges(N, nb):
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
            x_data, workers=self.compute_workers, broadcast=True, direct=True)
        ranges = self._ranges(len(x_data), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
        ranges_ij = list(
            itertools.product(ranges, ranges))  # all i/j ranges as ((i_start, i_end), (j_start, j_end)) pairs of tuples
        ranges_ij = [range_ij for range_ij in ranges_ij if range_ij[0][0] <= range_ij[1][0]]  # filter lower diagonal
        logger.debug("        gp2Scale covariance matrix init done after {} seconds.", time.time() - st)

        results = list(map(self._harvest_result, distributed.as_completed(client.map(
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
        logger.debug("        gp2Scale covariance matrix result stacked after {} seconds.", time.time() - st)
        del results
        # mirror across diagonal
        diagonal_mask = i_s != j_s
        data, i_s, j_s = np.hstack([data, data[diagonal_mask]]), \
            np.hstack([i_s, j_s[diagonal_mask]]), \
            np.hstack([j_s, i_s[diagonal_mask]])
        K = sparse.coo_matrix((data, (i_s, j_s)), shape=(len(x_data), len(x_data)))
        del data
        logger.debug("        gp2Scale covariance matrix assembled after {} seconds.", time.time() - st)
        #K = self._coo_to_csr_chunked(i_s, j_s, data, (len(data), len(data)), int(len(data)/2))
        K = K.tocsr()
        logger.debug("        gp2Scale covariance matrix in CSR after {} seconds.", time.time() - st)
        logger.debug("        gp2Scale covariance matrix sparsity = {}.", float(K.nnz) / float(K.shape[0] ** 2))
        return K

    @staticmethod
    def _coo_to_csr_chunked(row, col, data, shape, chunk_size):  #pragma: no cover
        n_rows = shape[0]
        chunks = []
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            mask = (row >= start) & (row < end)
            r = row[mask] - start  # Normalize to chunk
            c = col[mask]
            d = data[mask]
            coo_chunk = coo_matrix((d, (r, c)), shape=(end - start, shape[1]))
            csr_chunk = coo_chunk.tocsr()
            chunks.append(csr_chunk)
        return vstack(chunks, format='csr')

    def _update_prior_covariance_gp2Scale(self, x_old, x_new, hyperparameters):
        client = self.client
        """computes the covariance matrix from the kernel on HPC in sparse format"""

        self.x_new_scatter_future = client.scatter(
            x_new, workers=self.compute_workers, broadcast=True, direct=True)
        self.x_old_scatter_future = client.scatter(
            x_old, workers=self.compute_workers, broadcast=True, direct=True)

        point_number = len(x_old)
        num_batches = point_number // self.batch_size
        NUM_RANGES = num_batches
        ranges_data = self._ranges(len(x_old), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
        num_batches2 = len(x_new) // self.batch_size
        ranges_input = self._ranges(len(x_new), num_batches2)
        ranges_ij = list(itertools.product(ranges_data, ranges_input))

        # K = np.block([[self.K, B],
        #               [B,      C]])
        # Calculate B

        results = list(map(self._harvest_result,
                           distributed.as_completed(client.map(
                               partial(kernel_function_update,
                                       hyperparameters=hyperparameters,
                                       kernel=self.kernel),
                               ranges_ij,
                               [self.x_old_scatter_future] * len(ranges_ij),
                               [self.x_new_scatter_future] * len(ranges_ij)),
                               with_results=True)))

        data, i_s, j_s = map(np.hstack, zip(*results))
        B = sparse.coo_matrix((data, (i_s, j_s)), shape=(len(x_old), len(x_new)))

        # mirror across diagonal
        ranges_ij2 = list(itertools.product(ranges_input, ranges_input))
        ranges_ij2 = [range_ij2 for range_ij2 in ranges_ij2 if
                      range_ij2[0][0] <= range_ij2[1][0]]  # filter lower diagonal

        results = list(map(self._harvest_result,
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
        D = sparse.coo_matrix((data, (i_s, j_s)), shape=(len(x_new), len(x_new)))

        res = block_array([[self.K, B],
                           [B.transpose(), D]])

        return res

    @staticmethod
    def _harvest_result(future_result):
        future, result = future_result
        future.release()
        return result

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
        logger.debug("Default kernel in use.")
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])):
            distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        distance_matrix = np.sqrt(distance_matrix)
        return hps[0] * matern_kernel_diff1(distance_matrix, 1)

    def _d_kernel_dx(self, x1, x2, direction, hyperparameters):
        new_points = np.array(x1)
        epsilon = 1e-8
        new_points[:, direction] += epsilon
        a = self.compute_covariances(new_points, x2, hyperparameters)
        b = self.compute_covariances(x1, x2, hyperparameters)
        derivative = (a - b) / epsilon
        return derivative

    def _kernel_gradient(self, x1, x2, hyperparameters):
        gradient = np.empty((len(hyperparameters), len(x1), len(x2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self._dkernel_dh(x1, x2, direction, hyperparameters)
        return gradient

    def _kernel_derivative(self, x1, x2, hyperparameters, direction):
        derivative = self._dkernel_dh(x1, x2, direction, hyperparameters)
        return derivative

    @staticmethod
    def _default_kernel_analytical_gradient(x1, x2, hyperparameters):
        gradient = np.zeros((len(hyperparameters), len(x1), len(x2)))
        hps = hyperparameters
        dm = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])): dm += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        dm = np.sqrt(dm)

        non_zero_ind = np.where(dm != 0.0)
        for direction in range(len(x1[0])):
            dddh = np.zeros(dm.shape)
            dddh[non_zero_ind] = -abs(np.subtract.outer(x1[:, direction], x2[:, direction]))[non_zero_ind] ** 2 / (
                    hps[direction + 1] ** 3 * dm[non_zero_ind])
            gradient[direction + 1] = hps[0] * matern_kernel_diff1_grad(dm, dddh)
        gradient[0] = matern_kernel_diff1(dm, 1)
        return gradient

    def _dkernel_dh(self, x1, x2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-8
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.compute_covariances(x1, x2, new_hyperparameters1)
        b = self.compute_covariances(x1, x2, new_hyperparameters2)
        derivative = (a - b) / (2.0 * epsilon)
        return derivative

    def _default_mean_function(self, x, hyperparameters):
        """evaluates the gp mean function at the data points """
        if np.ndim(self.y_data) == 1:
            raise Exception("y_data wrong format")
        elif np.ndim(self.y_data) == 2:
            mean = np.zeros((len(x)))
            #for i in range(mean.shape[1]): mean[:, i] = np.mean(self.y_data[:, i])
            mean[:] = np.mean(self.y_data)
        else:
            raise Exception("Wrong dim in default mean function")
        return mean

    def _finitediff_dm_dh(self, x, hps):
        gr = np.empty((len(hps), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.compute_mean(x, temp_hps1)
            b = self.compute_mean(x, temp_hps2)
            gr[i] = (a - b) / 2e-6
        return gr

    @staticmethod
    def _default_dm_dh(x, hps):
        gr = np.zeros((len(hps), len(x)))
        return gr

    def __getstate__(self):
        state = dict(
            kernel_function=self.kernel_function,
            prior_mean_function=self.prior_mean_function,
            m_n_params=self.m_n_params,
            k_n_params=self.k_n_params,
            batch_size=self.batch_size,
            data=self.data,
            trainer=self.trainer,
            kernel=self.kernel,
            d_kernel_dx=self.d_kernel_dx,
            _dk_dh=self._dk_dh,
            mean_function=self.mean_function,
            _dm_dh=self._dm_dh,
            m=self.m,
            K=self.K,
            client=None,
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


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
