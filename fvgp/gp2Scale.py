import time
import scipy.sparse as sparse
import scipy.sparse.linalg as solve
import numpy as np
from hgdl.hgdl import HGDL
import dask.distributed as distributed
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
from scipy.optimize import differential_evolution
from scipy.sparse import coo_matrix


class gp2Scale():
    """
    gp2Scale class: Provides tools for a Lagre-Scale single-task GP.

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
        covariance_dask_client = None
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
        self.num_batches = (self.point_number // self.batch_size) + 1
        self.last_batch_size= self.point_number % self.batch_size
        self.entry_limit = entry_limit
        self.ram_limit = ram_limit
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            #, requires_grad = True) *
            self.variances = np.ones((self.y_data.shape)) * \
                    abs(self.y_data / 100.0)
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
        else: raise Exception("A kernel callbale has to be provided!")
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
        self.client = self._init_dask_client(covariance_dask_client)
        self.compute_prior_fvGP_pdf()
        print("gpLG successfully initiated, here is some info about the prior covarianve matrix:")
        print("non zero elements: ", self.SparsePriorCovariance.count_nonzero())
        print("Size in GBits:     ", self.SparsePriorCovariance.data.nbytes/1000000000)
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
    def compute_prior_fvGP_pdf(self):
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
                self.prior_mean_vec)
        self.SparsePriorCovariance = K
        self.covariance_value_prod = cov_y
        return K, cov_y

    ##################################################################################
    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean):
        K = self.compute_covariance(hyperparameters, variances)
        y = values - mean
        x = self.solve(K.tocsc(), y)
        #if self.use_inv is True: x = self.K_inv @ y
        #else: x = self.solve(K, y)
        if x.ndim == 2: x = x[:,0]
        return x,K
    ##################################################################################
    def compute_covariance(self, hyperparameters, variances):
        """computes the covariance matrix from the kernel"""
        SparsePriorCovariance = sparse.eye(self.point_number, format="lil")
        tasks = []
        count = 0
        fd = []
        for i in range(self.num_batches):
            #print("(",i,") of ", self.num_batches - 1)
            beg_i = i * self.batch_size
            end_i = min((i+1) * self.batch_size, self.point_number)
            if beg_i == end_i: continue
            batch1 = self.x_data[beg_i: end_i]
            for j in range(i,self.num_batches):
                worker = self.workers["worker"][int(count - ((count//self.number_of_workers) * self.number_of_workers))]
                beg_j = j * self.batch_size
                end_j = min((j+1) * self.batch_size, self.point_number)
                if beg_j == end_j: continue
                batch2 = self.x_data[beg_j : end_j]
                #if self.point_number >= 100000: 
                print("submitted batch. i:", beg_i,end_i,"   j:",beg_j,end_j, "to worker ", worker)
                data = {"batch1":batch1,"batch2": batch2, "hps" : hyperparameters, "range_i": (beg_i,end_i), "range_j": (beg_j,end_j), "mode": "prior"}
                #fd.append(self.client.scatter(data, workers = worker))
                tasks.append(self.client.submit(self.kernel,data, workers = worker))
                time.sleep(1)
                count += 1
                SparsePriorCovariance, tasks = self.collect_submatrices(tasks, SparsePriorCovariance)
                if SparsePriorCovariance.count_nonzero() > self.entry_limit or SparsePriorCovariance.data.nbytes > self.ram_limit:
                    for future in tasks: self.client.cancel(tasks); self.client.shutdown()
                    raise Exception("Matrix is not sparse enough. We are running the risk of a total crash. exit()")
                #else: print("Sparsity: ",SparsePriorCovariance.count_nonzero()/float(self.point_number)**2,"  ",SparsePriorCovariance.count_nonzero(),"elements of allowed ", self.entry_limit)

        SparsePriorCovariance = self.collect_remaining_submatrices(tasks, SparsePriorCovariance)
        self.client.cancel(tasks)
        diag = sparse.eye(self.point_number, format="lil")
        diag.setdiag(variances)
        SparsePriorCovariance = SparsePriorCovariance + diag
        #plt.figure(figsize = (15,15))
        #plt.imshow(SparsePriorCovariance.toarray())
        #plt.show()

        return SparsePriorCovariance

    def collect_submatrices(self,futures, SparsePriorCovariance):
        #get a part of the covariance, and fit into the sparse one, but only the vales needed
        #throw warning if too many values are not zero
        new_futures = []
        for future in futures:
            if future.status == "finished":
                if self.point_number >= 100000: print("Future", future, " has finished its work")
                #print("reading results ...", flush  = True)
                CoVariance_sub, data = future.result()
                #print("setting small stuff to zero ...")
                zero_indices = np.where(CoVariance_sub < 1e-16)
                CoVariance_sub[zero_indices] = 0.0
                #print("making sub matrix sparse...")
                SparseCov_sub = sparse.lil_matrix(CoVariance_sub)
                #print("substituting...")
                SparsePriorCovariance = self.insert(SparsePriorCovariance,SparseCov_sub, data["range_i"][0], data["range_j"][0])
                #SparsePriorCovariance[data["range_i"][0]:data["range_i"][1],data["range_j"][0]:data["range_j"][1]] = SparseCov_sub
                #SparsePriorCovariance[data["range_j"][0]:data["range_j"][1],data["range_i"][0]:data["range_i"][1]] = SparseCov_sub.transpose()
                #
                #print("done")
                #plt.imshow(SparsePriorCovariance.toarray())
                #plt.show()
                #input()
                #
            else: new_futures.append(future)

        return SparsePriorCovariance, new_futures


    def collect_remaining_submatrices(self, futures, SparsePriorCovariance):
        results = self.client.gather(futures)
        for result in results:
            CoVariance_sub, data = result
            zero_indices = np.where(CoVariance_sub < 1e-16)
            CoVariance_sub[zero_indices] = 0.0
            SparseCov_sub = sparse.lil_matrix(CoVariance_sub)
            SparsePriorCovariance = self.insert(SparsePriorCovariance,SparseCov_sub, data["range_i"][0], data["range_j"][0])
            #SparsePriorCovariance[data["range_i"][0]:data["range_i"][1],data["range_j"][0]:data["range_j"][1]] = SparseCov_sub
            #SparsePriorCovariance[data["range_j"][0]:data["range_j"][1],data["range_i"][0]:data["range_i"][1]] = SparseCov_sub.transpose()
        return SparsePriorCovariance
    
    def insert(self, B,s, i ,j):
        bg = B.tocoo()
        sm = s.tocoo()
        if i != j:
            row = np.concatenate([bg.row,sm.row + i, sm.col + j])
            col = np.concatenate([bg.col,sm.col + j, sm.row + i])
            res = coo_matrix((np.concatenate([bg.data,sm.data,sm.data]),(row,col)), shape = B.shape )
        else:
            row = np.concatenate([bg.row,sm.row + i])
            col = np.concatenate([bg.col,sm.col + j])
            res = coo_matrix((np.concatenate([bg.data,sm.data]),(row,col)), shape = B.shape )

        return res.tolil()
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
        else: print("dask client provided to HGDL", flush = True)
        client = dask_client
        worker_info = list(client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        self.workers = {"host": worker_info[0],
                "worker": worker_info[1:]}
        print("Host ",self.workers["host"]," has ", len(self.workers["worker"])," workers.")
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
        self.compute_prior_fvGP_pdf()
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

        start_log_likelihood = self.log_likelihood(starting_hps)

        print(
            "fvGP hyperparameter tuning in progress. Old hyperparameters: ",
            starting_hps, " with old log likelihood: ", start_log_likelihood)
        print("method: ", method)

        ############################
        ####global optimization:##
        ############################
        def constraint(v):
            return np.array(np.sum(v[3:]))

        nlc = NonlinearConstraint(constraint,0,10000)

        if method == "global":
            print("fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            print("maximum number of iterations: ", max_iter)
            print("termination tolerance: ", tolerance)
            print("bounds: ", hp_bounds)
            res = differential_evolution(
                self.log_likelihood,
                hp_bounds,
                disp=True,
                maxiter=max_iter,
                popsize = pop_size,
                tol = tolerance,
                workers = 1, constraints = (nlc),
            )
            hyperparameters = np.array(res["x"])
            Eval = self.log_likelihood(hyperparameters)
            print("fvGP found hyperparameters ",hyperparameters," with likelihood ",
                Eval," via global optimization")
        ############################
        ####local optimization:#####
        ############################
        else:
            raise ValueError("No optimization mode specified in fvGP")
        ###################################################
        if start_log_likelihood < self.log_likelihood(hyperparameters):
            hyperparameters = np.array(starting_hps)
            print("fvGP: Optimization returned smaller log likelihood; resetting to old hyperparameters.")
            print("New hyperparameters: ",
            hyperparameters,
            "with log likelihood: ",
            self.log_likelihood(hyperparameters))
        return hyperparameters


    def log_likelihood(self,hyperparameters):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        mean = np.zeros((self.point_number))   #self.mean_function(self,self.x_data,hyperparameters) * 0.0
        if mean.ndim > 1: raise Exception("Your mean function did not return a 1d numpy array!")
        x,K = self._compute_covariance_value_product(hyperparameters,self.y_data, self.variances, mean)
        y = self.y_data - mean
        sign, logdet = self.slogdet(K.tocsc())
        n = len(y)
        if sign == 0.0: res = (0.5 * (y.T @ x)) + (0.5 * n * np.log(2.0*np.pi))
        else: res = (0.5 * (y.T @ x)) + (0.5 * sign * logdet) + (0.5 * n * np.log(2.0*np.pi))
        print("Evaluating marginal log-likelihood", res)
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

    def minimumSwaps(self,arr):
        a = dict(enumerate(arr))
        b = {v:k for k,v in a.items()}
        count = 0
        for i in a:
            x = a[i]
            if x!=i:
                y = b[i]
                a[y] = x
                b[x] = y
                count+=1
        return count

    def slogdet(self, A):
        """
        fvGPs slogdet method based on torch
        """
        lu = splu(A)
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        diagL = diagL.astype(np.complex128)
        diagU = diagU.astype(np.complex128)
        logdet= np.real(np.nansum(np.log(diagL)) + np.nansum(np.log(diagU)))
        swap_sign = self.minimumSwaps(lu.perm_r)
        sign = np.sign(np.real(swap_sign*np.sign(diagL).prod()*np.sign(diagU).prod()))
        return sign, logdet
        #s,l = np.linalg.slogdet(A)
        #return s,l
        #if self.compute_device == "cpu":
        #    sign, logdet = torch.slogdet(A)
        #    logdet = torch.nan_to_num(logdet)
        #    return sign, logdet
        #elif self.compute_device == "gpu" or self.compute_device == "multi-gpu":
        #    sign, logdet = torch.slogdet(A)
        #    sign = sign.cpu()
        #    logdet = logdet.cpu()
        #    logdet = torch.nan_to_num(logdet)
        #    return sign, logdet

    def inv(self, A):
            B = torch.inverse(A)
            return B

    def solve(self, A, b):
        #####for sparsity:
        #zero_indices = np.where(A < 1e-16)
        #A[zero_indices] = 0.0
        #if self.is_sparse(A):
        try:
            x = spsolve(A,b)
            return x
        except Exception as e:
            print("fvGP: Sparse solve did not work out.")
            print("reason: ", str(e))
    ##################################################################################
    def add_to_diag(self,Matrix, Vector):
        d = torch.einsum("ii->i", Matrix)
        d += Vector
        return Matrix
    #def is_sparse(self,A):
    #    if float(np.count_nonzero(A))/float(len(A)**2) < 0.01: return True
    #    else: return False
    #def how_sparse_is(self,A):
    #    return float(np.count_nonzero(A))/float(len(A)**2)

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
        k = self.kernel({"batch1": self.x_data,"batch2":p,"hps" : self.hyperparameters, "mode" : "post"})[0]
        A = k.T @ self.covariance_value_prod
        #posterior_mean = self.mean_function(self,p,self.hyperparameters) + A
        posterior_mean = A
        return {"x": p,
                "f(x)": posterior_mean}


