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
        compute_device:                                 cpu/gpu, default = cpu
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
        variances = None,
        entry_limit = 2e9,
        ram_limit = 1e10,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_mean_function = None,
        covariance_dask_client = None,
        gpus_per_worker = 0,
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
        self.compute_device = compute_device
        self.batch_size = batch_size
        self.num_batches = self.point_number // self.batch_size
        last_batch_size = self.point_number % self.batch_size
        if last_batch_size != 0: self.num_batches += 1

        self.entry_limit = entry_limit
        self.ram_limit = ram_limit
        self.gpus_per_worker = gpus_per_worker
        self.info = info
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            #, requires_grad = True) *
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
        covariance_dask_client = self._init_dask_client(covariance_dask_client)
        self.covariance_dask_client = covariance_dask_client
        self.st = time.time()
        self.compute_prior_fvGP_pdf(covariance_dask_client)
        if self.info:
            print("gpLG successfully initiated, here is some info about the prior covariance matrix:")
            print("non zero elements: ", self.SparsePriorCovariance.count_nonzero())
            print("Size in GBits:     ", self.SparsePriorCovariance.data.nbytes/1e9)
            print("Sparsity: ",self.SparsePriorCovariance.count_nonzero()/float(self.point_number)**2)
            if self.point_number <= 5000:
                print("Here is an image:")
                plt.imshow(self.SparsePriorCovariance.toarray())
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
        cov_y,K = self._compute_covariance_value_product(
                self.hyperparameters,
                self.y_data,
                self.variances,
                self.prior_mean_vec,client)
        self.SparsePriorCovariance = K
        self.covariance_value_prod = cov_y
        return K, cov_y

    ##################################################################################
    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean,client):
        K = self.compute_covariance(hyperparameters, variances,client)
        if self.info: print("Covariance computed with sparsity: ",K.count_nonzero()/float(self.point_number)**2)
        y = values - mean
        x = self.solve(K.tocsc(), y)
        return x,K

    def total_number_of_batches(self):
        Db = float(self.num_batches)
        return 0.5 * Db * (Db + 1.)


    def compute_covariance(self, hyperparameters, variances,client):
        """computes the covariance matrix from the kernel on HPC in sparse format"""
        futures = []           ### a list of futures
        compute_workers = self.workers["worker"][1:]
        insert_worker = self.workers["worker"][0]
        idle_workers = set(compute_workers)
        #gpu_assignment = {}
        #for worker in idle_workers: gpu_assignment[worker] = set([i for i in range(self.gpus_per_worker)])
        sparse_sub_cov_set = []
        count = 0
        SparsePriorCovariance = client.submit(gp2ScaleSparseMatrix,self.point_number,actor=True, workers=insert_worker).result()# Create Actor
        scatter_data = {"x_data":self.x_data, "hps": hyperparameters, "kernel" : self.kernel} ##data that can be scattered
        scatter_future = client.scatter(scatter_data,workers = compute_workers)        ##scatter the data
        f = []

        start_time = time.time()
        for i in range(self.num_batches):
            beg_i = i * self.batch_size
            end_i = min((i+1) * self.batch_size, self.point_number)
            batch1 = self.x_data[beg_i: end_i]
            for j in range(i,self.num_batches):
                beg_j = j * self.batch_size
                end_j = min((j+1) * self.batch_size, self.point_number)
                batch2 = self.x_data[beg_j : end_j]
                #while not idle_workers:
                #    #futures = self.collect_submatrices(futures, idle_workers, sparse_sub_cov_set)
                #    futures = self.free_workers(futures, idle_workers)
                #    time.sleep(0.01)
                #this_worker = idle_workers.pop()
                #this_gpu = gpu_assignment[this_worker].pop()
                #this_gpu = 0
                data = {"scattered_data": scatter_future, "range_i": (beg_i,end_i), "range_j": (beg_j,end_j), "mode": "prior","gpu": 0}
                #futures.append(client.submit(kernel_function, data, workers = this_worker))
                futures.append(client.submit(kernel_function, data))
                if self.info: print("submitted batch. i:", beg_i,end_i,"   j:",beg_j,end_j, "to worker ", this_worker, "Future: ", futures[-1].key)
                #if SparsePriorCovariance.get_thread_status().result():
                finished_futures,futures = self.get_finished_futures(futures)
                f.append(SparsePriorCovariance.collect_submatrices(finished_futures))
                finished_futures = []

                if self.info: print("current time stamp: ", time.time() - start_time," percent finished: ",float(count)/self.total_number_of_batches(), flush = True)
                count += 1

        if self.info: print("All tasks submitted after ",time.time() - start_time,flush = True)
        if self.info: print("actual number of computed batches: ", count)
        while futures:
            finished_futures, futures = self.get_finished_futures(futures)
            f.append(SparsePriorCovariance.collect_submatrices(finished_futures))
            finished_futures = []
            time.sleep(0.1)

        client.gather(f)  ##let the insertion finish
        end = SparsePriorCovariance.get().result() ##get the current Prior Covariance
        #self.collect_remaining_submatrices(futures, idle_workers, sparse_sub_cov_set) ##let rest of futures finsish
        #SparsePriorCovariance = self.coalesce(end,sparse_sub_cov_set) ##fill them into the Prior Covariance

        client.cancel(futures) ##make sure allf utures are cancelled
        diag = sparse.eye(self.point_number, format="coo") ##make variance
        diag.setdiag(variances) ##make variance
        SparsePriorCovariance = end + diag  ##add variance

        #plt.imshow(SparsePriorCovariance.toarray())
        #plt.show()

        if self.info: print("total prior covariance compute time: ", time.time() - start_time)

        return SparsePriorCovariance
    
    def get_finished_futures(self,futures):
        finished_futures = []
        remaining_futures = []
        for future in futures:
            if future.status == "finished":
                finished_futures.append(future)
            else: remaining_futures.append(future)
        futures = remaining_futures
        return finished_futures, futures

    #def free_workers(self,futures, idle_workers):
    #    for future in futures:
    #        if future.status == "finished":
    #            idle_workers.add(worker)


    #def collect_submatrices(self,futures, idle_workers, sparse_sub_cov_set):
    #    new_futures = []
    #    for future in futures:
    #        if future.status == "finished":
    #            SparseCov_sub, ranges,ketime, worker = future.result()
    #            if self.info: print("Future", future.key, " has finished its work in", ketime," seconds.")
    #            if SparseCov_sub.count_nonzero()/float(self.batch_size)**2 > 0.1:
    #                print("WARNING: Collected submatrix not sparse; sparsity: ", SparseCov_sub.count_nonzero()/float(self.batch_size)**2)
    #            idle_workers.add(worker)
    #            sparse_sub_cov_set.append((SparseCov_sub, ranges[0], ranges[1]))
    #        else: new_futures.append(future)
    #    futures = new_futures
    #    return futures

    def collect_remaining_submatrices(self,futures, idle_workers, sparse_sub_cov_set):
        while futures:
            futures = self.collect_submatrices(futures, idle_workers, sparse_sub_cov_set)


    def coalesce(self, bg, sparse_sub_cov_set):
        for entry in sparse_sub_cov_set:
            sm = entry[0]
            i  = entry[1]
            j  = entry[2]

            if i != j:
                row = np.concatenate([bg.row,sm.row + i, sm.col + j])
                col = np.concatenate([bg.col,sm.col + j, sm.row + i])
                bg = coo_matrix((np.concatenate([bg.data,sm.data,sm.data]),(row,col)), shape = bg.shape )
            else:
                row = np.concatenate([bg.row,sm.row + i])
                col = np.concatenate([bg.col,sm.col + j])
                bg = coo_matrix((np.concatenate([bg.data,sm.data]),(row,col)), shape = bg.shape)
                if bg.data.nbytes > self.ram_limit:
                    for future in futures: client.cancel(futures); client.shutdown()
                    raise Exception("RAM limit exceeded, EXIT")
        return bg


    #def insert(self, bg,sm, i ,j):
    #    if i != j:
    #        row = np.concatenate([bg.row,sm.row + i, sm.col + j])
    #        col = np.concatenate([bg.col,sm.col + j, sm.row + i])
    #        res = coo_matrix((np.concatenate([bg.data,sm.data,sm.data]),(row,col)), shape = bg.shape )
    #    else:
    #        row = np.concatenate([bg.row,sm.row + i])
    #        col = np.concatenate([bg.col,sm.col + j])
    #        res = coo_matrix((np.concatenate([bg.data,sm.data]),(row,col)), shape = bg.shape)
    #    return res
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
        if dask_client is None:
            dask_client = distributed.Client()
            print("No dask client provided to gp2Scale. Using the local client", flush = True)
        else: print("dask client provided to gp2Scale", flush = True)
        client = dask_client
        worker_info = list(client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        self.workers = {#"host": worker_info[0],
                "worker": worker_info[0:]}
        print("We have ", len(self.workers["worker"])," workers ready to go.")
        print("all the workers: ",self.workers["worker"])
        print("the scheduler: ", client.scheduler_info()["address"])
        self.number_of_workers = len(self.workers["worker"])
        return client

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

        ############################
        ####global optimization:##
        ############################
        #def constraint(v):
        #    return np.array(np.sum(v[3:]))

        #nlc = NonlinearConstraint(constraint,0,10000)

        #if method == "global":
        #    print("fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
        #    print("maximum number of iterations: ", max_iter)
        #    print("termination tolerance: ", tolerance)
        #    print("bounds: ", hp_bounds)
        #    res = differential_evolution(
        #        self.log_likelihood,
        #        hp_bounds,
        #        disp=True,
        #        maxiter=max_iter,
        #        popsize = pop_size,
        #        tol = tolerance,
        #        workers = 1, constraints = (nlc),
        #    )
        #    hyperparameters = np.array(res["x"])
        #    Eval = self.log_likelihood(hyperparameters)
        #    print("fvGP found hyperparameters ",hyperparameters," with likelihood ",
        #        Eval," via global optimization")
        ############################
        ####local optimization:#####
        ############################
        #else:
        #    raise ValueError("No optimization mode specified in fvGP")
        ###################################################
        #if start_log_likelihood < self.log_likelihood(hyperparameters):
        #    hyperparameters = np.array(starting_hps)
        #    print("fvGP: Optimization returned smaller log likelihood; resetting to old hyperparameters.")
        #    print("New hyperparameters: ",
        #    hyperparameters,
        #    "with log likelihood: ",
        #    self.log_likelihood(hyperparameters))
        return hyperparameters


    def log_likelihood(self,hyperparameters, recompute_xK = True):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        #client = detach_from_self(self.covariance_dask_client)
        client = self.covariance_dask_client
        mean = np.zeros((self.point_number))   #self.mean_function(self,self.x_data,hyperparameters) * 0.0
        if mean.ndim > 1: raise Exception("Your mean function did not return a 1d numpy array!")
        if recompute_xK is True: x,K = self._compute_covariance_value_product(hyperparameters,self.y_data, self.variances, mean,client)
        else: x,K = self.covariance_value_prod,self.SparsePriorCovariance
        y = self.y_data - mean
        sign, logdet = self.slogdet(K.tocsc())
        n = len(y)
        if sign == 0.0: res = (0.5 * (y.T @ x)) + (0.5 * n * np.log(2.0*np.pi))
        else: res = (0.5 * (y.T @ x)) + (0.5 * sign * logdet) + (0.5 * n * np.log(2.0*np.pi))
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
    def slogdet(self, A):
        """
        fvGPs slogdet method based on torch
        """
        sign = 1.
        B = splu(A.tocsc())
        upper_diag = abs(B.U.diagonal())
        res = np.sum(np.log(upper_diag))
        return sign, res


    def solve(self, A, b):
        #####for sparsity:
        try:
            x,info = solve.cg(A,b, maxiter = 20)
        except Exception as e:
            #print("fvGP: Sparse solve did not work out.")
            #print("reason: ", str(e))
            info = 1
        if info > 0:
            #print("cg did not work out, let's do a minres")
            x,info = solve.minres(A,b, show = self.info)
        return x
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

#def insert(sm_dict):
#    sm = sm_dict["sp_matrix_tuple"][0]
#    i  = sm_dict["sp_matrix_tuple"][1]
#    j  = sm_dict["sp_matrix_tuple"][2]
    #bg = sm_dict["SparsePriorCovariance"]
#    bg = global_var.get()
#    if i != j:
#        row = np.concatenate([bg.row,sm.row + i, sm.col + j])
#        col = np.concatenate([bg.col,sm.col + j, sm.row + i])
#        res = coo_matrix((np.concatenate([bg.data,sm.data,sm.data]),(row,col)), shape = bg.shape )
#    else:
#        row = np.concatenate([bg.row,sm.row + i])
#        col = np.concatenate([bg.col,sm.col + j])
#        res = coo_matrix((np.concatenate([bg.data,sm.data]),(row,col)), shape = bg.shape)
    #sm_dict["SparsePriorCovariance"] = res
#    global_var.set(res)
#    return res

def kernel_function(data):
    st = time.time()
    hps= data["scattered_data"]["hps"]
    mode = data["mode"]
    kernel = data["scattered_data"]["kernel"]
    worker = distributed.get_worker()
    if mode == "prior":
        x1 = data["scattered_data"]["x_data"][data["range_i"][0]:data["range_i"][1]]
        x2 = data["scattered_data"]["x_data"][data["range_j"][0]:data["range_j"][1]]
        range1 = data["range_i"]
        range2 = data["range_j"]
        k = kernel(x1,x2,hps, None)
    else: 
        x1 = data["x_data"]
        x2 = data["x2"]
        k = kernel(x1,x2,hps, None)
    k_sparse = sparse.coo_matrix(k)
    return k_sparse, (data["range_i"][0],data["range_j"][0]), time.time() - st, worker.address
