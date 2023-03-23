import time
from dask.distributed import wait
import scipy.sparse as sparse
import scipy.sparse.linalg as solve
import numpy as np
import dask.distributed as distributed
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
from scipy.optimize import differential_evolution
from scipy.sparse import coo_matrix
import gc
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spilu
from .mcmc import mcmc
import torch
from dask.distributed import Variable
from .sparse_matrix import gp2ScaleSparseMatrix
import gc


def sparse_stat_kernel(x1,x2, hps):
    d = 0
    for i in range(len(x1[0])): d += abs(np.subtract.outer(x1[:,i],x2[:,i]))**2
    d = np.sqrt(d)
    d[d == 0.0] = 1e-16
    d[d > hps[1]] = hps[1]
    kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
    ((3.0*(d/hps[1])**2*np.log((d/hps[1])/(1+np.sqrt(1.0 - (d/hps[1])**2))))+\
    ((2.0*(d/hps[1])**2 + 1.0) * np.sqrt(1.0-(d/hps[1])**2)))
    return hps[0] * kernel

def sparse_stat_kernel_robust(x1,x2, hps):
    d = 0
    for i in range(len(x1[0])): d += abs(np.subtract.outer(x1[:,i],x2[:,i]))**2
    d = np.sqrt(d)
    d[d == 0.0] = 1e-16
    d[d > 1./hps[1]**2] = 1./hps[1]**2
    kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
    ((3.0*(d*hps[1]**2)**2*np.log((d*hps[1]**2)/(1+np.sqrt(1.0 - (d*hps[1]**2)**2))))+\
    ((2.0*(d*hps[1]**2)**2 + 1.0) * np.sqrt(1.0-(d*hps[1]**2)**2)))
    return (hps[0]**2) * kernel



############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################


class gp2Scale():
    """
    This class allows the user to scale GPs up to millions of datapoints. There is full high-performance-computing
    support through DASK.
    
    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space.
    x_data : np.ndarray
        The point positions. Shape (V x D), where D is the `input_space_dim`.
    y_data : np.ndarray
        The values of the data points. Shape (V,output_number).
    init_hyperparameters : np.ndarray
        Vector of hyperparameters used by the GP initially. The class provides methods to train hyperparameters.
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
        input_space_dim,
        x_data,
        y_data,
        init_hyperparameters,
        batch_size,
        variances = None,
        limit_workers = None,
        LUtimeout = 100,
        gp_kernel_function = sparse_stat_kernel,
        gp_mean_function = None,
        covariance_dask_client = None,
        info = False,
        ):
        """
        The constructor for the gp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        if input_space_dim != len(x_data[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(y_data) == 2: y_data = y_data[:,0]

        self.input_dim = input_space_dim
        self.x_data = x_data
        self.point_number = len(self.x_data)
        self.y_data = y_data
        self.batch_size = batch_size
        self.num_batches = self.point_number // self.batch_size
        self.limit_workers = limit_workers
        self.info = info
        self.LUtimeout = LUtimeout
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.y_data.shape)) * \
                    abs(np.mean(self.y_data) / 100.0)
            print("CAUTION: you have not provided data variances in fvGP,")
            print("they will be set to 1 percent of the data values!")
        elif np.ndim(variances) == 2:
            self.variances = variances[:,0]
        elif np.ndim(variances) == 1:
            self.variances = np.array(variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        if len(self.variances[self.variances < 0]) > 0: raise Exception("Negative measurement variances communicated to fvgp.")
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if gp_kernel_function == "robust": self.kernel = sparse_stat_kernel_robust
        elif callable(gp_kernel_function): self.kernel = gp_kernel_function
        else: raise Exception("A kernel callable has to be provided!")

        self.gp_mean_function = gp_mean_function
        if  callable(gp_mean_function): self.mean_function = gp_mean_function
        else: self.mean_function = self.default_mean_function

        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = np.array(init_hyperparameters)
        ##########################################
        #compute the prior########################
        ##########################################
        covariance_dask_client, self.compute_worker_set, self.actor_worker = self._init_dask_client(covariance_dask_client)
        ###initiate actor that is a future contain the covariance and methods
        self.SparsePriorCovariance = covariance_dask_client.submit(gp2ScaleSparseMatrix,self.point_number, actor=True, workers=self.actor_worker).result()# Create Actor
        self.covariance_dask_client = covariance_dask_client
        scatter_data = {"x1_data":self.x_data, "x2_data":self.x_data} ##data that can be scattered
        self.scatter_future = covariance_dask_client.scatter(scatter_data,workers = self.compute_worker_set)               ##scatter the data

        self.st = time.time()
        self.compute_prior_fvGP_pdf(covariance_dask_client)
        if self.info:
            sp = self.SparsePriorCovariance.get_result().result()
            print("gp2Scale successfully initiated, here is some info about the prior covariance matrix:")
            print("non zero elements: ", sp.count_nonzero())
            print("Size in GBits:     ", sp.data.nbytes/1e9)
            print("Sparsity: ",sp.count_nonzero()/float(self.point_number)**2)
            if self.point_number <= 5000:
                print("Here is an image:")
                plt.imshow(sp.toarray())
                plt.show()

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    def compute_prior_fvGP_pdf(self,client):
        """
        This function computes the important entities, namely the prior covariance and
        its product with the (values - prior_mean) and returns them and the prior mean
        input:
            none
        return:
            prior mean
            prior covariance
            covariance value product
        """
        self.prior_mean_vec = np.zeros((self.point_number)) #self.mean_function(self,self.x_data,self.hyperparameters)
        cov_y = self._compute_covariance_value_product(
                self.hyperparameters,
                self.y_data,
                self.variances,
                self.prior_mean_vec,client)
        self.covariance_value_prod = cov_y

    def _compute_covariance_value_product(self, hyperparameters,y_data, variances, mean, client):
        self.compute_covariance(self.x_data,self.x_data,hyperparameters, variances,client)
        y = y_data - mean
        #try: success = self.SparsePriorCovariance.compute_LU().result(timeout=self.LUtimeout)
        #except: print("LU failed")
        x = self.SparsePriorCovariance.solve(y).result()
        return x

    def total_number_of_batches(self):
        Db = float(self.num_batches)
        return 0.5 * Db * (Db + 1.)

    def compute_covariance(self, x1,x2,hyperparameters, variances,client):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        ###initialize futures
        futures = []           ### a list of futures
        actor_futures = []
        finished_futures = set()
        ###get workers
        compute_workers = set(self.compute_worker_set)
        idle_workers = set(compute_workers)
        ###future_worker_assignments
        self.future_worker_assignments = {}
        ###scatter data
        start_time = time.time()
        count = 0
        num_batches_i = len(x1) // self.batch_size
        num_batches_j = len(x2) // self.batch_size
        last_batch_size_i = len(x1) % self.batch_size
        last_batch_size_j = len(x2) % self.batch_size
        if last_batch_size_i != 0: num_batches_i += 1
        if last_batch_size_j != 0: num_batches_j += 1

        for i in range(num_batches_i):
            beg_i = i * self.batch_size
            end_i = min((i+1) * self.batch_size, self.point_number)
            batch1 = self.x_data[beg_i: end_i]
            for j in range(i,num_batches_j):
                beg_j = j * self.batch_size
                end_j = min((j+1) * self.batch_size, self.point_number)
                batch2 = self.x_data[beg_j : end_j]
                ##make workers available that are not actively computing
                while not idle_workers:
                    idle_workers, futures, finished_futures = self.free_workers(futures, finished_futures)
                    time.sleep(0.1)

                ####collect finished workers but only if actor is not busy, otherwise do it later
                if len(finished_futures) >= 1000:
                    actor_futures.append(self.SparsePriorCovariance.get_future_results(set(finished_futures)))
                    finished_futures = set()

                #get idle worker and submit work
                current_worker = self.get_idle_worker(idle_workers)
                data = {"scattered_data": self.scatter_future,"hps": hyperparameters, "kernel" :self.kernel,  "range_i": (beg_i,end_i), "range_j": (beg_j,end_j), "mode": "prior","gpu": 0}
                futures.append(client.submit(kernel_function, data, workers = current_worker))
                self.assign_future_2_worker(futures[-1].key,current_worker)
                if self.info:
                    print("    submitted batch. i:", beg_i,end_i,"   j:",beg_j,end_j, "to worker ",current_worker, " Future: ", futures[-1].key)
                    print("    current time stamp: ", time.time() - start_time," percent finished: ",float(count)/self.total_number_of_batches())
                    print("")
                count += 1

        if self.info:
            print("All tasks submitted after ",time.time() - start_time,flush = True)
            print("number of computed batches: ", count)

        actor_futures.append(self.SparsePriorCovariance.get_future_results(finished_futures.union(futures)))
        #actor_futures[-1].result()
        client.gather(actor_futures)
        actor_futures.append(self.SparsePriorCovariance.add_to_diag(variances)) ##add to diag on actor
        actor_futures[-1].result()
        #clean up
        #del futures
        #del actor_futures
        #del finished_futures
        #del scatter_future

        #########
        if self.info: 
            print("total prior covariance compute time: ", time.time() - start_time, "Non-zero count: ", self.SparsePriorCovariance.get_result().result().count_nonzero())
            print("Sparsity: ",self.SparsePriorCovariance.get_result().result().count_nonzero()/float(self.point_number)**2)


    def free_workers(self, futures, finished_futures):
        free_workers = set()
        remaining_futures = []
        for future in futures:
            if future.status == "cancelled": print("WARNING: cancelled futures encountered!")
            if future.status == "finished":
                finished_futures.add(future)
                free_workers.add(self.future_worker_assignments[future.key])
                del self.future_worker_assignments[future.key]
            else: remaining_futures.append(future)
        return free_workers, remaining_futures, finished_futures

    def assign_future_2_worker(self, future_key, worker_address):
        self.future_worker_assignments[future_key] = worker_address

    def get_idle_worker(self,idle_workers):
        return idle_workers.pop()
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
        if self.limit_workers: worker_info = worker_info[0:self.limit_workers]
        actor_worker = worker_info[0]
        compute_worker_set = set(worker_info[1:])
        print("We have ", len(compute_worker_set)," compute workers ready to go.")
        print("Actor on", actor_worker)
        print("Scheduler Address: ", dask_client.scheduler_info()["address"])
        return dask_client, compute_worker_set,actor_worker

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################TRAINING####################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    def train(self,
        hyperparameter_bounds,
        init_hyperparameters = None,
        max_iter = 120,
        dask_client = None
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
        print("fvGP training started with ",len(self.x_data)," data points")
        ######################
        #####TRAINING#########
        ######################
        self.hyperparameters = self.optimize_log_likelihood(
            init_hyperparameters,
            np.array(hyperparameter_bounds),
            max_iter,
            )
        print("computing the prior")
        self.compute_prior_fvGP_pdf(self.covariance_dask_client)
        np.save("latest_hps", self.hyperparameters)
        ######################
        ######################
        ######################



    def optimize_log_likelihood(self,
        starting_hps,
        hp_bounds,
        max_iter,
        ):

        #start_log_likelihood = self.log_likelihood(starting_hps, recompute_xK = False)
        print("MCMC started in fvGP")
        print('bounds are',hp_bounds)
        res = mcmc(self.log_likelihood,hp_bounds,max_iter = max_iter, x0 = starting_hps)
        hyperparameters = np.array(res["x"])
        self.mcmc_info = res
        print("MCMC has found solution: ", hyperparameters, "with log marginal_likelihood ",res["f(x)"])
        self.hyperparameters = hyperparameters
        return hyperparameters

    def log_likelihood(self,hyperparameters = None):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        client = self.covariance_dask_client
        mean = np.zeros((self.point_number))   #self.mean_function(self,self.x_data,hyperparameters) * 0.0
        #if mean.ndim > 1: raise Exception("Your mean function did not return a 1d numpy array!")
        if hyperparameters is None: x,K = self.covariance_value_prod,self.SparsePriorCovariance
        else:
            self.SparsePriorCovariance.reset_prior().result()
            x = self._compute_covariance_value_product(hyperparameters,self.y_data, self.variances, mean,client)
        y = self.y_data - mean
        logdet = self.SparsePriorCovariance.logdet().result()
        n = len(y)
        res = -(0.5 * (y.T @ x)) - (0.5 * logdet) - (0.5 * n * np.log(2.0*np.pi))
        return res

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################gp MEAN AND  KERNEL#########################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################


    def default_mean_function(self,gp_obj,x,hyperparameters):
        """evaluates the gp mean function at the data points """
        #, requires_grad = True
        mean = torch.ones((len(x)), dtype = float) + torch.mean(self.y_data, dtype = float)
        return mean
    def get_distance_matrix_robust(self,x1,x2,hps):
        d = np.zeros((len(x1),len(x2)))
        for i in range(x1.shape[1]):
            d += ((x1[:,i].reshape(-1, 1) - x2[:,i])*hps[i+1])**2
        return np.sqrt(d)
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################POSTERIOR###################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    def posterior_mean(self, x_iset):
        """
        function to compute the posterior mean
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x":    the input points,
             "f(x)": the posterior mean vector (1d numpy array)}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.x_data[0]): p = np.column_stack([p,np.zeros((len(p)))])
        k = self.kernel(self.x_data,p,self.hyperparameters)
        A = k.T @ self.covariance_value_prod
        #posterior_mean = self.mean_function(self,p,self.hyperparameters) + A
        posterior_mean = A
        return {"x": p,
                "f(x)": posterior_mean}


    def posterior_covariance(self, x_iset, umfpack = True):
        """
        Function to compute the posterior covariance.
        Parameters
        ----------
        x_iset : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        variance_only : bool, optional
            If True the compuation of the posterior covariance matrix is avoided which can save compute time.
            In that case the return will only provide the variance at the input points.
            Default = False.
        Return
        ------
        solution dictionary : dict
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.x_data[0]): p = np.column_stack([p,np.zeros((len(p)))])
        if len(p) == len(self.x_data): raise Warning("The number of prediction points and data points coincide which can lead to instabilities when user-defined kernels are used.")

        k = self.kernel(self.x_data,p,self.hyperparameters)
        kk = self.kernel(p, p,self.hyperparameters)
        k_cov_prod = np.empty(k.shape)
        for i in range(len(k[0])): k_cov_prod[:,i] = spsolve(self.SparsePriorCovariance.get_result().result(),k[:,i], use_umfpack = umfpack)

        S = kk - (k_cov_prod.T @ k)
        v = np.array(np.diag(S))
        if np.any(v < -0.001):
            logger.warning(inspect.cleandoc("""#
            Negative variances encountered. That normally means that the model is unstable.
            Rethink the kernel definitions, add more noise to the data,
            or double check the hyperparameter optimization bounds. This will not
            terminate the algorithm, but expect anomalies."""))
            v[v<0.0] = 0.0
            if not variance_only:
                np.fill_diagonal(S, v)

        return {"x": p,
                "v(x)": v,
                "S(x)": S}





#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
class gpm2Scale(gp2Scale):
    def __init__(self,input_space_dim,
                 output_space_dim,
                 x_data, ##data in the original input space
                 init_hyperparameters,
                 batch_size,
                 variances  = None,
                 init_y_data = None, #initial latent space positions
                 gp_kernel_function = sparse_stat_kernel,
                 gp_mean_function = None, covariance_dask_client = None,
                 info = False):

        if input_space_dim != len(x_data[0]): raise ValueError("input space dimensions are not in agreement with the point positions given")
        self.input_dim = input_space_dim
        self.output_dim = output_space_dim
        self.x_data = x_data
        self.point_number = len(self.x_data)
        if init_y_data is None: init_y_data = np.random.rand(len(x_data), self.output_dim)
        self.y_data = init_y_data
        self.batch_size = batch_size
        self.num_batches = self.point_number // self.batch_size
        self.info = info
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.x_data.shape[0])) * \
                    abs(np.mean(self.x_data) / 100.0)
            print("CAUTION: you have not provided data variances in fvGP,")
            print("they will be set to 1 percent of the data values!")
        if len(self.variances[self.variances < 0]) > 0: raise Exception("Negative measurement variances communicated to fvgp.")
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if gp_kernel_function == "robust": self.kernel = sparse_stat_kernel_robust
        elif callable(gp_kernel_function): self.kernel = gp_kernel_function
        else: raise Exception("A kernel callable has to be provided!")

        self.gp_mean_function = gp_mean_function
        if  callable(gp_mean_function): self.mean_function = gp_mean_function
        else: self.mean_function = self.default_mean_function

        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = np.array(init_hyperparameters)
        ##########################################
        #compute the prior########################
        ##########################################
        covariance_dask_client, self.compute_worker_set, self.actor_worker = self._init_dask_client(covariance_dask_client)
        ###initiate actor that is a future contain the covariance and methods
        self.SparsePriorCovariance = covariance_dask_client.submit(gp2ScaleSparseMatrix,self.point_number, actor=True, workers=self.actor_worker).result()# Create Actor

        self.covariance_dask_client = covariance_dask_client
        scatter_data = {"x1_data":self.x_data, "x2_data":self.x_data} ##data that can be scattered
        self.scatter_future = covariance_dask_client.scatter(scatter_data,workers = self.compute_worker_set)               ##scatter the data

        self.st = time.time()
        self.compute_covariance(self.y_data,self.y_data,self.hyperparameters,variances,covariance_dask_client)
        if self.info:
            sp = self.SparsePriorCovariance.get_result().result()
            print("gp2Scale successfully initiated, here is some info about the prior covariance matrix:")
            print("non zero elements: ", sp.count_nonzero())
            print("Size in GBits:     ", sp.data.nbytes/1e9)
            print("Sparsity: ",sp.count_nonzero()/float(self.point_number)**2)
            if self.point_number <= 5000:
                print("Here is an image:")
                plt.imshow(sp.toarray())
                plt.show()



    def total_number_of_batches(self):
        Db = float(self.num_batches)
        return 0.5 * Db * (Db + 1.)

    def compute_covariance(self, x1,x2,hyperparameters, variances,client):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        ###initialize futures
        futures = []           ### a list of futures
        actor_futures = []
        finished_futures = set()
        ###get workers
        compute_workers = set(self.compute_worker_set)
        idle_workers = set(compute_workers)
        ###future_worker_assignments
        self.future_worker_assignments = {}
        ###scatter data
        start_time = time.time()
        count = 0
        num_batches_i = len(x1) // self.batch_size
        num_batches_j = len(x2) // self.batch_size
        last_batch_size_i = len(x1) % self.batch_size
        last_batch_size_j = len(x2) % self.batch_size
        if last_batch_size_i != 0: num_batches_i += 1
        if last_batch_size_j != 0: num_batches_j += 1

        for i in range(num_batches_i):
            beg_i = i * self.batch_size
            end_i = min((i+1) * self.batch_size, self.point_number)
            batch1 = self.x_data[beg_i: end_i]
            for j in range(i,num_batches_j):
                beg_j = j * self.batch_size
                end_j = min((j+1) * self.batch_size, self.point_number)
                batch2 = self.x_data[beg_j : end_j]
                ##make workers available that are not actively computing
                while not idle_workers:
                    idle_workers, futures, finished_futures = self.free_workers(futures, finished_futures)
                    time.sleep(0.1)

                ####collect finished workers but only if actor is not busy, otherwise do it later
                if len(finished_futures) >= 1000:
                    actor_futures.append(self.SparsePriorCovariance.get_future_results(set(finished_futures)))
                    finished_futures = set()

                #get idle worker and submit work
                current_worker = self.get_idle_worker(idle_workers)
                data = {"scattered_data": self.scatter_future,"hps": hyperparameters, "kernel" :self.kernel,  "range_i": (beg_i,end_i), "range_j": (beg_j,end_j), "mode": "prior","gpu": 0}
                futures.append(client.submit(kernel_function, data, workers = current_worker))
                self.assign_future_2_worker(futures[-1].key,current_worker)
                if self.info:
                    print("    submitted batch. i:", beg_i,end_i,"   j:",beg_j,end_j, "to worker ",current_worker, " Future: ", futures[-1].key)
                    print("    current time stamp: ", time.time() - start_time," percent finished: ",float(count)/self.total_number_of_batches())
                    print("")
                count += 1

        if self.info:
            print("All tasks submitted after ",time.time() - start_time,flush = True)
            print("number of computed batches: ", count)

        actor_futures.append(self.SparsePriorCovariance.get_future_results(finished_futures.union(futures)))
        #actor_futures[-1].result()
        client.gather(actor_futures)
        actor_futures.append(self.SparsePriorCovariance.add_to_diag(variances)) ##add to diag on actor
        actor_futures[-1].result()
        #clean up
        #del futures
        #del actor_futures
        #del finished_futures
        #del scatter_future

        #########
        if self.info: 
            print("total prior covariance compute time: ", time.time() - start_time, "Non-zero count: ", self.SparsePriorCovariance.get_result().result().count_nonzero())
            print("Sparsity: ",self.SparsePriorCovariance.get_result().result().count_nonzero()/float(self.point_number)**2)


    def free_workers(self, futures, finished_futures):
        free_workers = set()
        remaining_futures = []
        for future in futures:
            if future.status == "cancelled": print("WARNING: cancelled futures encountered!")
            if future.status == "finished":
                finished_futures.add(future)
                free_workers.add(self.future_worker_assignments[future.key])
                del self.future_worker_assignments[future.key]
            else: remaining_futures.append(future)
        return free_workers, remaining_futures, finished_futures

    def assign_future_2_worker(self, future_key, worker_address):
        self.future_worker_assignments[future_key] = worker_address

    def get_idle_worker(self,idle_workers):
        return idle_workers.pop()
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

 
    def train(self,
        hyperparameter_bounds,
        method = "global",
        init_hyperparameters = None,
        max_iter = 120,
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
        print("fvGP training started with ",len(self.x_data)," data points")
        ######################
        #####TRAINING#########
        ######################
        self.hyperparameters = self.optimize_log_likelihood(
            init_hyperparameters,
            np.array(hyperparameter_bounds),
            max_iter,
            method
            )
        #print("computing the prior")
        #self.compute_prior_fvGP_pdf(self.covariance_dask_client)
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

        #start_log_likelihood = self.log_likelihood(starting_hps, recompute_xK = False)
        if method == "mcmc":
            print("MCMC started in fvGP")
            print('bounds are',hp_bounds)
            res = mcmc(self.log_likelihood,hp_bounds,max_iter = max_iter, x0 = starting_hps)
            hyperparameters = np.array(res["x"])
            self.mcmc_info = res
            print("MCMC has found solution: ", hyperparameters, "with log marginal_likelihood ",res["f(x)"])
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
        self.compute_covariance(y1,y2,hps, self.variances,client)
        logdet = self.SparsePriorCovariance.logdet().result()
        n = len(y)
        x = self.x_data
        traceKXX = self.SparsePriorCovariance.traceKXX(x).result()
        res = -(0.5 * traceKXX) - (dim * 0.5 * logdet) - (0.5 * dim *n * np.log(2.0*np.pi))
        return res


    def neg_log_likelihood(self, y):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        client = self.covariance_dask_client
        dim = float(self.input_dim)
        y1 = y2 = y[0:-2].reshape(self.point_number, self.output_dim)
        self.SparsePriorCovariance.reset_prior().result()
        hps = y[-2:]
        self.compute_covariance(y1,y2,hps, self.variances,client)
        logdet = self.SparsePriorCovariance.logdet().result()
        n = len(y)
        x = self.x_data
        traceKXX = self.SparsePriorCovariance.traceKXX(x).result()
        res = (0.5 * traceKXX) + (dim * 0.5 * logdet) + (0.5 * dim *n * np.log(2.0*np.pi))
        print("res")
        print("")
        return res


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def kernel_function(data):
    st = time.time()
    hps= data["hps"]
    mode = data["mode"]
    kernel = data["kernel"]
    worker = distributed.get_worker()
    if mode == "prior":
        x1 = data["scattered_data"]["x1_data"][data["range_i"][0]:data["range_i"][1]]
        #x1 = data["x1_data"][data["range_i"][0]:data["range_i"][1]]
        x2 = data["scattered_data"]["x2_data"][data["range_j"][0]:data["range_j"][1]]
        #x2 = data["x2_data"][data["range_j"][0]:data["range_j"][1]]
        range1 = data["range_i"]
        range2 = data["range_j"]
        k = kernel(x1,x2,hps)
    else:
        x1 = data["x1_data"]
        x2 = data["x2_data"]
        k = kernel(x1,x2,hps)
    k_sparse = sparse.coo_matrix(k)
    return k_sparse, (data["range_i"][0],data["range_j"][0]), time.time() - st, worker.address
