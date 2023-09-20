############################################################
############################################################
############################################################

import dask.distributed as distributed
from .sparse_matrix import gp2ScaleSparseMatrix
import time
import scipy.sparse as sparse
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

    def compute_covariance(self, x1,x2,hyperparameters,client):
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
                if len(finished_futures) >= 2000:
                    actor_futures.append(self.SparsePriorCovariance.get_future_results(set(finished_futures), info = self.info))
                    finished_futures = set()

                #get idle worker and submit work
                current_worker = self.get_idle_worker(idle_workers)
                data = {"scattered_data": self.scatter_future,"hps": hyperparameters, "kernel" :self.kernel,  "range_i": (beg_i,end_i), "range_j": (beg_j,end_j), "mode": "prior","gpu": 0}
                futures.append(client.submit(kernel_function, data, workers = current_worker))
                self.assign_future_2_worker(futures[-1].key,current_worker)
                if self.info and ( i % 10) == 0.:
                    print("    submitted batch. i:", beg_i,end_i,"   j:",beg_j,end_j, "to worker ",current_worker, " Future: ", futures[-1].key,flush = True)
                    print("    current time stamp: ", time.time() - start_time," percent finished: ",int(100. * float(count)/self._total_number_of_batches()),flush = True)
                    #print("    Size of the current covariance matrix: ", self.SparsePriorCovariance.get_result().result().count_nonzero(), flush = True)
                    print("",flush = True)
                count += 1

        if self.info:
            print("All tasks submitted after ",time.time() - start_time,flush = True)
            print("number of computed batches: ", count)

        actor_futures.append(self.SparsePriorCovariance.get_future_results(finished_futures.union(futures), info = self.info))
        client.gather(actor_futures)
        #actor_futures.append(self.SparsePriorCovariance.add_to_diag(variances)) ##add to diag on actor
        actor_futures[-1].result()

        #########
        if self.info:
            print("total prior covariance compute time: ", time.time() - start_time, "Non-zero count: ", self.SparsePriorCovariance.get_result().result().count_nonzero(),flush = True)
            print("Sparsity: ",self.SparsePriorCovariance.get_result().result().count_nonzero()/float(self.point_number)**2,flush = True)


    def free_workers(self, futures, finished_futures):
        free_workers = set()
        remaining_futures = []
        for future in futures:
            if future.status == "cancelled": print("WARNING: cancelled futures encountered!",flush = True)
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
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
class gpm2Scale(gp2Scale): # pragma: no cover
    def __init__(self,input_space_dim,
                 output_space_dim,
                 x_data, ##data in the original input space
                 init_hyperparameters,
                 batch_size,
                 variances  = None,
                 init_y_data = None, #initial latent space positions
                 gp_kernel_function = None,
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
        scatter_data = {"x_data":self.x_data} ##data that can be scattered
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
    #print(time.time() - st, flush = True)
    return k_sparse, (data["range_i"][0],data["range_j"][0]), time.time() - st, worker.address
