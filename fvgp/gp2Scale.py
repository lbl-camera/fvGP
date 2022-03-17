import time
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
#from .sparse_matrix import UpdateSM
import gc

class gp2Scale():
    """
    gp2Scale class: Provides tools for a Large-Scale single-task GP.

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space

    Attributes:
        input_space_dim (int):         dim1
        points (N x dim1 numpy array): 2d numpy array of points
        values (N dim numpy array):    2d numpy array of values
        init_hyperparameters:          1d numpy array (>0)

    Optional Attributes:
        variances (N dim numpy array):                  variances of the values, default = array of shape of points
                                                        with 1 % of the values
        gp_kernel_function(callable):                   None/function defining the 
                                                        kernel def name(x1,x2,hyperparameters,self), 
                                                        make sure to return a 2d numpy array, default = None uses default kernel
        gp_mean_function(callable):                     None/function def name(gp_obj, x, hyperparameters), 
                                                        make sure to return a 1d numpy array, default = None
        sparse (bool):                                  default = False
        normalize_y:                                    default = False, normalizes the values \in [0,1]

    Example:
        obj = fvGP(3,np.array([[1,2,3],[4,5,6]]),
                         np.array([2,4]),
                         np.array([2,3,4,5]),
                         variances = np.array([0.01,0.02]),
                         gp_kernel_function = kernel_function,
                         gp_mean_function = some_mean_function
        )
    """

    def __init__(
        self,
        input_space_dim,
        points,
        values,
        init_hyperparameters,
        batch_size,
        target_worker_count,
        variances = None,
        workerFrac2Start = 0.5,
        LUtimeout = 100,
        gp_kernel_function = None,
        gp_mean_function = None,
        covariance_dask_client = None,
        info = False
        ):
        """
        The constructor for the gp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        if input_space_dim != len(points[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(values) == 2: values = values[:,0]

        self.input_dim = input_space_dim
        self.x_data = points
        self.point_number = len(self.x_data)
        self.y_data = values
        self.batch_size = batch_size
        self.num_batches = self.point_number // self.batch_size

        self.info = info
        self.LUtimeout = LUtimeout
        self.target_worker_count = target_worker_count
        self.workerFrac2Start = workerFrac2Start
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.y_data.shape)) * \
                    abs(np.mean(self.y_data) / 100.0)
            print("CAUTION: you have not provided data variances in fvGP,")
            print("they will be set to 1 percent of the data values!")
        elif variances.dim() == 2:
            self.variances = variances[:,0]
        elif variances.dim() == 1:
            self.variances = np.array(variances)#, requires_grad = True)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        if len(self.variances[self.variances < 0]) > 0: raise Exception("Negative measurement variances communicated to fvgp.")
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if callable(gp_kernel_function): self.kernel = gp_kernel_function
        #else: raise Exception("A kernel callable has to be provided!")
        #self.d_kernel_dx = self.d_gp_kernel_dx

        self.gp_mean_function = gp_mean_function
        if  callable(gp_mean_function): self.mean_function = gp_mean_function
        else: self.mean_function = self.default_mean_function

        #if callable(gp_kernel_function_grad): self.dk_dh = gp_kernel_function_grad
        #else:
        #    if self.ram_economy is True: self.dk_dh = self.gp_kernel_derivative
        #    else: self.dk_dh = self.gp_kernel_gradient

        #if callable(gp_mean_function_grad): self.dm_dh = gp_mean_function_grad
        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = np.array(init_hyperparameters) #,requires_grad = True)
        ##########################################
        #compute the prior########################
        ##########################################
        covariance_dask_client, self.compute_worker_set, self.actor_worker = self._init_dask_client(covariance_dask_client)
        ###initiate actor that is a future contain the covariance and methods
        self.SparsePriorCovariance = covariance_dask_client.submit(gp2ScaleSparseMatrix,self.point_number, actor=True, workers=self.actor_worker).result()# Create Actor
        self.covariance_dask_client = covariance_dask_client
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

    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean, client):
        np.save("latest_hps", hyperparameters)
        self.compute_covariance(self.x_data,self.x_data,hyperparameters, variances,client)
        y = values - mean
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
        scatter_data = {"x1_data":x1, "x2_data":x2,"hps": hyperparameters, "kernel" : self.kernel} ##data that can be scattered
        scatter_future = client.scatter(scatter_data,workers = compute_workers)               ##scatter the data
        ###############
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
                    time.sleep(0.01)

                ####collect finished workers but only if actor is not busy, otherwise do it later
                if len(finished_futures) >= 1000:
                    actor_futures.append(self.SparsePriorCovariance.get_future_results(set(finished_futures)))
                    finished_futures = set()

                #get idle worker and submit work
                current_worker = self.get_idle_worker(idle_workers)
                data = {"scattered_data": scatter_future, "range_i": (beg_i,end_i), "range_j": (beg_j,end_j), "mode": "prior","gpu": 0}
                futures.append(client.submit(kernel_function, data, workers = current_worker))
                self.assign_future_2_worker(futures[-1].key,current_worker)
                if self.info:
                    print("    submitted batch. i:", beg_i,end_i,"   j:",beg_j,end_j, "to worker ",current_worker, " Future: ", futures[-1].key)
                    print("    current time stamp: ", time.time() - start_time," percent finished: ",float(count)/self.total_number_of_batches())
                    print("")
                count += 1

        if self.info:
            print("All tasks submitted after ",time.time() - start_time,flush = True)
            print("actual number of computed batches: ", count)
            print("still have to gather ",len(futures)," results",flush = True)
            print("also have to gather ",len(finished_futures)," results",flush = True)

        actor_futures.append(self.SparsePriorCovariance.get_future_results(finished_futures.union(futures)))
        actor_futures.append(self.SparsePriorCovariance.add_to_diag(variances)) ##add to diag on actor
        #clean up
        actor_futures[-1].result()
        #########
        if self.info: 
            print("total prior covariance compute time: ", time.time() - start_time, "Non-zero count: ", self.SparsePriorCovariance.get_result().result().count_nonzero())
            print("Sparsity: ",self.SparsePriorCovariance.get_result().result().count_nonzero()/float(self.point_number)**2)
        #client.run(gc.collect)


    def free_workers(self, futures, finished_futures):
        free_workers = set()
        remaining_futures = []
        for future in futures:
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

    #def add_idle_worker(self,worker,idle_workers):
    #    return idle_workers.add(worker)


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
        try: dask_client.wait_for_workers(n_workers=int(self.target_worker_count * self.workerFrac2Start), timeout = 600)
        except: time.sleep(100)

        worker_info = list(dask_client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        actor_worker = worker_info[0]
        compute_worker_set = set(worker_info[1:])
        print("We have ", len(compute_worker_set)," compute workers ready to go.")
        print("Actor on", actor_worker)
        print("Scheduler Address: ", dask_client.scheduler_info()["address"])
        return dask_client, compute_worker_set,actor_worker

    #def _update_worker_set(self,client, current_worker_set):
    #    worker_info = list(client.scheduler_info()["workers"].keys())
    #    if not worker_info: raise Exception("No workers available")
    #    new_worker_set = set(worker_info).difference(current_worker_set)
    #    print("updated workers. new workers: ", new_worker_set)
    #    return new_worker_set
       

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
        method = "global",
        optimization_dict = None,
        pop_size = 20,
        tolerance = 0.0001,
        max_iter = 120,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        deflation_radius = None,
        dask_client = None):
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
            method,
            optimization_dict,
            max_iter,
            pop_size,
            tolerance,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client
            )
        print("computing the prior")
        self.compute_prior_fvGP_pdf(self.covariance_dask_client)
        ######################
        ######################
        ######################



    def optimize_log_likelihood(self,starting_hps,
        hp_bounds,method,optimization_dict,max_iter,
        pop_size,tolerance,
        local_optimizer,
        global_optimizer,
        deflation_radius,
        constraints = None,
        dask_client = None):

        #start_log_likelihood = self.log_likelihood(starting_hps, recompute_xK = False)
        print("MCMC started in fvGP")
        print('bounds are',hp_bounds)
        res = mcmc(self.log_likelihood,hp_bounds,max_iter = max_iter, x0 = starting_hps)
        hyperparameters = np.array(res["x"])
        print("MCMC has found solution: ", hyperparameters, "with neg. marginal_likelihood ",res["f(x)"])
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
        #if recompute_xK is True:
        if hyperparameters is None: x,K = self.covariance_value_prod,self.SparsePriorCovariance
        else:
            self.SparsePriorCovariance.reset_prior().result()
            x = self._compute_covariance_value_product(hyperparameters,self.y_data, self.variances, mean,client)
        y = self.y_data - mean
        logdet = self.SparsePriorCovariance.logdet().result()
        n = len(y)
        res = (0.5 * (y.T @ x)) + (0.5 * logdet) + (0.5 * n * np.log(2.0*np.pi))
        return res


    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################LINEAR ALGEBRA##############################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    #def slogdet(self, A):
    #    """
    #    fvGPs slogdet method based on torch
    #    """
    #    sign = 1.
    #    B = splu(A.tocsc())
    #    upper_diag = abs(B.U.diagonal())
    #    res = np.sum(np.log(upper_diag))
    #    return sign, res


    #def solve(self, A, b):
    #    #####for sparsity:
    #    try:
    #        x,info = solve.cg(A,b, maxiter = 20)
    #    except Exception as e:
    #        #print("fvGP: Sparse solve did not work out.")
    #        #print("reason: ", str(e))
    #        info = 1
    #    if info > 0:
    #        #print("cg did not work out, let's do a minres")
    #        x,info = solve.minres(A,b, show = self.info)
    #    return x
    ##################################################################################
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
        return np.sqrt(d + 1e-16)

    def default_kernel(self,x1,x2,hyperparameters,obj):
        ################################################################
        ###standard anisotropic kernel in an input space with l2########
        ################################################################
        """
        x1: 2d numpy array of points
        x2: 2d numpy array of points
        obj: object containing kernel definition

        Return:
        -------
        Kernel Matrix
        """
        distance_matrix = self.get_distance_matrix_robust(x1,x2,hyperparameters)
        #return   hyperparameters[0]**2 *  obj.matern_kernel_diff1(distance_matrix,1)
        return hyperparameters[0]**2  *  torch.exp(-distance_matrix)



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
        k = kernel_function({"x_data":self.x_data,"x2":p, "hps" : self.hyperparameters, "mode" : "post"})[0]
        A = k.T @ self.covariance_value_prod
        #posterior_mean = self.mean_function(self,p,self.hyperparameters) + A
        posterior_mean = A
        return {"x": p,
                "f(x)": posterior_mean}

def kernel_function(data):
    st = time.time()
    hps= data["scattered_data"]["hps"]
    mode = data["mode"]
    kernel = data["scattered_data"]["kernel"]
    worker = distributed.get_worker()
    if mode == "prior":
        x1 = data["scattered_data"]["x1_data"][data["range_i"][0]:data["range_i"][1]]
        x2 = data["scattered_data"]["x2_data"][data["range_j"][0]:data["range_j"][1]]
        range1 = data["range_i"]
        range2 = data["range_j"]
        k = kernel(x1,x2,hps, None)
    else:
        x1 = data["x1_data"]
        x2 = data["x2_data"]
        k = kernel(x1,x2,hps, None)
    k_sparse = sparse.coo_matrix(k)
    return k_sparse, (data["range_i"][0],data["range_j"][0]), time.time() - st, worker.address
