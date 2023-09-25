############################################################
############################################################
############################################################

import dask.distributed as distributed
import numpy as np
from distributed import wait

from .sparse_matrix import gp2ScaleSparseMatrix
import time
import scipy.sparse as sparse
from functools import partial
from itertools import islice


def batched(iterable, n): # pragma: no cover

    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class gp2Scale():  # pragma: no cover

    """
    This class allows the user to scale GPs up to millions of datapoints. There is full high-performance-computing
    support through DASK.

    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space.
    x_data : np.ndarray
        The point positions. Shape (V x D), where D is the `input_space_dim`.
    batch_size : int
        The covariance is divided into batches of the defined size for distributed computing.
    variances : np.ndarray, optional
        An numpy array defining the uncertainties in the data `y_data`. Shape (V x 1) or (V). Note: if no
        variances are provided they will be set to `abs(np.mean(y_data) / 100.0`.
    limit_workers : int, optional
        If given as integer only the workers up to the limit will be used.
    LUtimeout : int, optional (future release)
        Controls the timeout for the LU decomposition.
    gp_kernel_function : Callable, optional
        A function that calculates the covariance between datapoints. It accepts as input x1 (a V x D array of positions),
        x2 (a U x D array of positions), hyperparameters (a 1-D array of length D+1 for the default kernel), and a
        `gpcam.gp_optimizer.GPOptimizer` instance. The default is a stationary anisotropic kernel
        (`fvgp.gp.GP.default_kernel`).
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at an input position. It accepts as input
        an array of positions (of size V x D), hyperparameters (a 1-D array of length D+1 for the default kernel)
        and a `gpcam.gp_optimizer.GPOptimizer` instance. The return value is a 1-D array of length V. If None is provided,
        `fvgp.gp.GP.default_mean_function` is used.
        a finite difference scheme is used.
    covariance_dask_client : dask.distributed.client, optional
        The client used for the covariance computation. If none is provided a local client will be used.
    info : bool, optional
        Controls the output of the algorithm for tests. The default is False
    args : user defined, optional
        These optional arguments will be available as attribute in kernel and mean function definitions.

    """

    def __init__(
        self,
        x_data,
        batch_size = 10000,
        gp_kernel_function = None,
        LUtimeout = 100,
        covariance_dask_client = None,
        info = False,
        ):
        """
        The constructor for the gp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        #self.input_space_dim = input_space_dim
        self.batch_size = batch_size
        self.point_number = len(x_data)
        self.num_batches = self.point_number // self.batch_size
        self.info = info
        self.LUtimeout = LUtimeout
        self.x_data = x_data
        self.kernel = gp_kernel_function



        covariance_dask_client, self.compute_worker_set, self.actor_worker = self._init_dask_client(covariance_dask_client)
        ###initiate actor that is a future contain the covariance and methods
        self.SparsePriorCovariance = covariance_dask_client.submit(gp2ScaleSparseMatrix,self.point_number, actor=True, workers=self.actor_worker).result()
        #self.covariance_dask_client = covariance_dask_client

        scatter_data = {"x_data":self.x_data} ##data that can be scattered
        self.scatter_future = covariance_dask_client.scatter(scatter_data,workers = self.compute_worker_set)               ##scatter the data

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

    def compute_covariance(self, hyperparameters,client):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        ###scatter data
        start_time = time.time()
        count = 0
        num_batches_i = len(self.x_data) // self.batch_size
        num_batches_j = len(self.x_data) // self.batch_size
        last_batch_size_i = len(self.x_data) % self.batch_size
        last_batch_size_j = len(self.x_data) % self.batch_size
        if last_batch_size_i != 0: num_batches_i += 1
        if last_batch_size_j != 0: num_batches_j += 1

        igrid, jgrid = np.mgrid[0:num_batches_i, 0:num_batches_j]

        COLLECT_BATCH_SIZE = 1000  # number of batches to compute before collecting results

        collect_batches = batched(zip(igrid.ravel(), jgrid.ravel()), COLLECT_BATCH_SIZE)  # split batches into chunks
        for batch in collect_batches:  # for each chunk
            futures = list(map(partial(self.submit_kernel_function, hyperparameters=hyperparameters, client=client),
                               batch))  # submit kernel function for each i,j in the chunk
            wait(futures)
            self.SparsePriorCovariance.get_future_results(futures)

        # TODO: use loguru over prints
        if self.info:
            print("All tasks submitted after ", time.time() - start_time, flush=True)
            # print("number of computed batches: ", count)
            print("total prior covariance compute time: ", time.time() - start_time, "Non-zero count: ", self.SparsePriorCovariance.get_result().result().count_nonzero())
            print("Sparsity: ", self.SparsePriorCovariance.get_result().result().count_nonzero() / float(self.point_number) ** 2)

    def submit_kernel_function(self, ij, hyperparameters, client):
        i, j = ij
        beg_i = i * self.batch_size
        end_i = min((i+1) * self.batch_size, self.point_number)
        beg_j = j * self.batch_size
        end_j = min((j+1) * self.batch_size, self.point_number)

        data = {"scattered_data": self.scatter_future, "hps": hyperparameters, "kernel": self.kernel,
                "range_i": (beg_i, end_i), "range_j": (beg_j, end_j), "mode": "prior", "gpu": 0}

        future = client.submit(kernel_function, data)

        return future

    def get_future_results(self, futures, info=False):
        res = []
        for future in futures:
            SparseCov_sub, ranges, ketime, worker = future.result()
            if info: print("Collected Future ", future.key, " has finished its work in", ketime," seconds. time stamp: ",time.time() - self.st, flush = True)
            res.append((SparseCov_sub,ranges[0],ranges[1]))

        self.SparsePriorCovariance.insert_many(res)
        if info: print("    Size of the current covariance matrix: ", self.SparsePriorCovariance.K.count_nonzero(), flush=True)

    def calculate_sparse_noise_covariance(self,vector):
        diag = sparse.eye(len(vector), format="coo")
        diag.setdiag(vector)
        return diag

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################DASK########################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    def _init_dask_client(self,dask_client):
        if dask_client is None: dask_client = distributed.Client()
        worker_info = list(dask_client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        actor_worker = worker_info[0]
        compute_worker_set = set(worker_info[1:])
        print("We have ", len(compute_worker_set)," compute workers ready to go.")
        print("Actor on", actor_worker)
        print("Scheduler Address: ", dask_client.scheduler_info()["address"])
        return dask_client, compute_worker_set,actor_worker

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def kernel_function(data):  # pragma: no cover

    st = time.time()
    hps= data["hps"]
    mode = data["mode"]
    kernel = data["kernel"]
    worker = distributed.get_worker()
    if mode == "prior":
        x1 = data["scattered_data"]["x_data"][data["range_i"][0]:data["range_i"][1]]
        x2 = data["scattered_data"]["x_data"][data["range_j"][0]:data["range_j"][1]]
        range1 = data["range_i"]
        range2 = data["range_j"]
        k = kernel(x1,x2,hps, None)
    else:
        x1 = data["x_data"]
        x2 = data["x_data"]
        k = kernel(x1,x2,hps, None)
    k_sparse = sparse.coo_matrix(k)
    return k_sparse, (data["range_i"][0],data["range_j"][0]), time.time() - st, worker.address
