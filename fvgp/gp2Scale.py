############################################################
############################################################
############################################################

import itertools
import time
from functools import partial
import dask.distributed as distributed
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import block_array
class gp2Scale():
    def __init__(
        self,
        #x_data,
        batch_size=10000,
        gp_kernel_function=None,
        covariance_dask_client=None,
        info=False,
    ):
        """
        The constructor for the gp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        assert covariance_dask_client is not None
        self.batch_size = batch_size
        self.point_number = None
        self.num_batches = None
        self.info = info
        #self.x_data = x_data
        self.kernel = gp_kernel_function
        self.number_of_workers = len(covariance_dask_client.scheduler_info()['workers'])
        self.cov_initialized = False

        worker_info = list(covariance_dask_client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        self.compute_workers = list(worker_info)

        self.x_data_scatter_future = None
        self.x_new_scatter_future = None
        self.x_data = None

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    def _total_number_of_batches(self):
        Db = float(self.num_batches)
        return 0.5 * Db * (Db + 1.)

    @staticmethod
    def ranges(N, nb):
        """ splits a range(N) into nb chunks defined by chunk_start, chunk_end """
        if nb == 0: nb = 1
        step = N / nb
        return [(round(step * i), round(step * (i + 1))) for i in range(nb)]

    def update_x_data(self, x_new):
        if isinstance(x_new, list) and isinstance(self.x_data, list):
            self.x_data = self.x_data + x_new
        elif isinstance(x_new, np.ndarray) and isinstance(self.x_data, np.ndarray):
            self.x_data = np.row_stack([self.x_data, x_new])
        else: raise Exception("x_data or x_new is not of a permissible format.")

    def compute_covariance(self, x_data, hyperparameters, client):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        self.cov_initialized = True
        self.point_number = len(x_data)
        self.num_batches = self.point_number // self.batch_size
        NUM_RANGES = self.num_batches
        self.x_data = x_data
        self.x_data_scatter_future = client.scatter(
            x_data, workers=self.compute_workers, broadcast=False)
        ranges = self.ranges(len(x_data), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
        ranges_ij = list(
            itertools.product(ranges, ranges))  # all i/j ranges as ((i_start, i_end), (j_start, j_end)) pairs of tuples
        ranges_ij = [range_ij for range_ij in ranges_ij if range_ij[0][0] <= range_ij[1][0]]  # filter lower diagonal
        kernel_caller = kernel_function

        results = list(map(self.harvest_result, distributed.as_completed(client.map(
                        partial(kernel_caller,
                                hyperparameters=hyperparameters,
                                kernel=self.kernel),
                                ranges_ij,
                                [self.x_data_scatter_future] * len(ranges_ij),
                                [self.x_data_scatter_future] * len(ranges_ij)),
                              with_results=True)))

        #reshape the result set into COO components
        data, i_s, j_s = map(np.hstack, zip(*results))
        # mirror across diagonal
        diagonal_mask = i_s != j_s
        data, i_s, j_s = np.hstack([data, data[diagonal_mask]]), \
                         np.hstack([i_s, j_s[diagonal_mask]]), \
                         np.hstack([j_s, i_s[diagonal_mask]])

        return sparse.coo_matrix((data, (i_s, j_s)))

    def update_covariance(self, x_new, hyperparameters, client, cov):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        if not self.cov_initialized:
            raise Exception("Before updating the covariance, you have to compute a covariance.")
        self.x_new_scatter_future = client.scatter(
            x_new, workers=self.compute_workers, broadcast=False)
        NUM_RANGES = self.num_batches
        ranges_data = self.ranges(len(self.x_data), NUM_RANGES)  # the chunk ranges, as (start, end) tuples
        num_batches2 = len(x_new) // self.batch_size
        ranges_input = self.ranges(len(x_new), num_batches2)
        ranges_ij = list(itertools.product(ranges_data, ranges_input))
        ranges_ij2 = list(itertools.product(ranges_input, ranges_input))

        kernel_caller = kernel_function_update

        #K = np.block([[A, B],
        #             [B,C]])
        #Calculate B
        results = list(map(self.harvest_result,
                          distributed.as_completed(client.map(
                              partial(kernel_caller,
                                      hyperparameters=hyperparameters,
                                      kernel=self.kernel),
                              ranges_ij,
                              [self.x_data_scatter_future] * len(ranges_ij),
                              [self.x_new_scatter_future] * len(ranges_ij)),
                              with_results=True)))

        data, i_s, j_s = map(np.hstack, zip(*results))
        B = sparse.coo_matrix((data, (i_s, j_s)))

        # mirror across diagonal
        ranges_ij2 = [range_ij2 for range_ij2 in ranges_ij2 if range_ij2[0][0] <= range_ij2[1][0]]  # filter lower diagonal
        kernel_caller = kernel_function
        results = list(map(self.harvest_result,
                          distributed.as_completed(client.map(
                              partial(kernel_caller,
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

        res = block_array([[cov,           B],
                           [B.transpose(), D]])

        self.update_x_data(x_new)
        return res

    @staticmethod
    def harvest_result(future_result):
        future, result = future_result
        future.release()
        return result

    def calculate_sparse_noise_covariance(self, vector):
        diag = sparse.eye(len(vector), format="coo")
        diag.setdiag(vector)
        return diag


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
class gpm2Scale(gp2Scale):  # pragma: no cover
    def __init__(self, input_space_dim,):
        self.input_space = input_space_dim

    def train(self,
              hyperparameter_bounds,
              method="global",
              init_hyperparameters=None,
              max_iter=120,
              ):
        """
        This function finds the maximum of the log_likelihood and therefore trains the fvGP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be be 'hgdl' and
        providing a dask client

        inputs:
            hyperparameter_bounds (2d numpy array)
        optional inputs:
            init_hyperparameters (1d numpy array):  default = None (= use earlier initialization)
            method = "global": "global"/"local"/"hgdl"/callable f(obj,optimization_dict)
            optimization_dict = None: if optimizer is callable, the this will be passed as dict
            pop_size = 20
            tolerance = 0.0001
            max_iter: default = 120
            local_optimizer = "L-BFGS-B"  important for local and hgdl optimization
            global_optimizer = "genetic"
            deflation_radius = None        for hgdl
            dask_client = None (will use local client, only for hgdl optimization)

        output:
            None, just updates the class with the new hyperparameters
        """
        ############################################
        if init_hyperparameters is None:
            init_hyperparameters = np.array(self.hyperparameters)
        print("fvGP training started with ", len(self.x_data), " data points")
        ######################
        #####TRAINING#########
        ######################
        self.hyperparameters = self.optimize_log_likelihood(
            init_hyperparameters,
            np.array(hyperparameter_bounds),
            max_iter,
            method
        )
        # print("computing the prior")
        # self.compute_prior_fvGP_pdf(self.covariance_dask_client)
        self.y_data = self.hyperparameters[0:-2].reshape(self.point_number, self.output_dim)
        np.save("output_points", self.y_data)
        ######################
        ######################
        ######################

    def optimize_log_likelihood(self,
                                starting_hps,
                                hp_bounds,
                                max_iter,
                                method
                                ):

        # start_log_likelihood = self.log_likelihood(starting_hps, recompute_xK = False)
        if method == "mcmc":
            print("MCMC started in fvGP")
            print('bounds are', hp_bounds)
            res = mcmc(self.log_likelihood, hp_bounds, max_iter=max_iter, x0=starting_hps)
            hyperparameters = np.array(res["x"])
            self.mcmc_info = res
            print("MCMC has found solution: ", hyperparameters, "with log marginal_likelihood ", res["f(x)"])
        elif method == "global":
            res = differential_evolution(self.neg_log_likelihood, hp_bounds)
        self.hyperparameters = hyperparameters
        return hyperparameters

    def log_likelihood(self, y):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        client = self.covariance_dask_client
        dim = float(self.input_dim)
        y1 = y2 = x[0:-2].reshape(self.point_number, self.output_dim)
        self.SparsePriorCovariance.reset_prior().result()
        hps = x[-2:]
        self.compute_covariance(y1, y2, hps, self.variances, client)
        logdet = self.SparsePriorCovariance.logdet().result()
        n = len(y)
        x = self.x_data
        traceKXX = self.SparsePriorCovariance.traceKXX(x).result()
        res = -(0.5 * traceKXX) - (dim * 0.5 * logdet) - (0.5 * dim * n * np.log(2.0 * np.pi))
        return res

    def neg_log_likelihood(self, y):
        """
        computes the marginal log-likelihood
        input:
            hyperparameters
        output:
            negative marginal log-likelihood (scalar)
        """

        return -log_likelihood(y)


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
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
    k = kernel(x1, x2, hps, None)
    k_sparse = sparse.coo_matrix(k)

    #print("kernel compute time: ", time.time() - st, flush = True)
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
    #print(x1, flush=True)
    k = kernel(x1, x2, hps, None)
    k_sparse = sparse.coo_matrix(k)

    #print("kernel compute time: ", time.time() - st, flush = True)
    data, rows, cols = k_sparse.data, k_sparse.row + range_i[0], k_sparse.col + range_j[0]

    return data, rows, cols
