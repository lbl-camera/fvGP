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
        print("gpLG successfully initiated")


    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
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
        #self.SparsePriorCovariance = sparse.csc_matrix((self.point_number, self.point_number), dtype=np.float32)
        self.SparsePriorCovariance = sparse.eye(self.point_number, format="csc")
        cov_y,K = self._compute_covariance_value_product(
                self.hyperparameters,
                self.y_data,
                self.variances,
                self.prior_mean_vec)
        self.prior_covariance = K
        self.covariance_value_prod = cov_y
    ##################################################################################
    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean):
        K = self.compute_covariance(hyperparameters, variances)
        y = values - mean
        x = self.solve(K, y)
        #if self.use_inv is True: x = self.K_inv @ y
        #else: x = self.solve(K, y)
        if x.ndim == 2: x = x[:,0]
        return x,K
    ##################################################################################
    def compute_covariance(self, hyperparameters, variances):
        """computes the covariance matrix from the kernel"""
        tasks = []
        ind   = []
        #self.SparsePriorCovariance
        print("creating the covariance")
        for i in range(self.num_batches):
            b = i * self.batch_size
            e = min((i+1) * self.batch_size, self.point_number)
            #print("i from ",b," to ", e)
            if b == e: continue
            batch1 = self.x_data[i * self.batch_size : e]
            for j in range(i,self.num_batches):
                b = j * self.batch_size
                print("(",i,j,") of ", self.num_batches)
                e = min((j+1) * self.batch_size, self.point_number)
                if b == e: continue
                #print("j from ",b ," to ", e)
                batch2 = self.x_data[j * self.batch_size : (j+1) * self.batch_size]
                #print(batch1)
                #print(batch2)
                #print("=======")
                #input()
                data = {"batch1":batch1,"batch2": batch2, "hps" : hyperparameters}
                ind.append(np.array([i*self.batch_size,j*self.batch_size]))
                tasks.append(self.client.submit(self.kernel,data))
                self.collect_submatrices(tasks, ind)
        #self.add_to_diag(variances)
        self.SparsePriorCovariance = self.SparsePriorCovariance + (sparse.eye(self.point_number, format="csc") * variances[0])
        #plt.imshow(self.SparsePriorCovariance.toarray())
        #plt.show()
        if len(self.SparsePriorCovariance.data) > 0.1 * self.point_number**2:
            print("Matrix Not Sparse, Sparsety Coefficient ", len(self.SparsePriorCovariance.data)/float(self.point_number)**2)
        return self.SparsePriorCovariance

    def collect_submatrices(self,futures, ind):
        #get a part of the covariance, and fit into the sparse one, but only the vales needed
        #throw warning if too many values are not zero
        for i in range(len(futures)):
            if futures[i].status == "finished":
                CoVariance_sub = futures[i].result()
                zero_indices = np.where(CoVariance_sub < 1e-16)
                CoVariance_sub[zero_indices] = 0.0
                SparseCov_sub = sparse.csc_matrix(CoVariance_sub)
                self.SparsePriorCovariance[ind[i][0]:ind[i][0] + len(CoVariance_sub),ind[i][1]:ind[i][1] + len(CoVariance_sub[0])] = SparseCov_sub

                CoVariance_sub = CoVariance_sub.T
                SparseCov_sub = sparse.csc_matrix(CoVariance_sub)
                self.SparsePriorCovariance[ind[i][1]:ind[i][1] + len(CoVariance_sub),ind[i][0]:ind[i][0] + len(CoVariance_sub[0])] = SparseCov_sub
                #print("this is our matrix: ", self.SparsePriorCovariance.toarray())
                #plt.imshow(self.SparsePriorCovariance.toarray())
                #plt.show()

    def _init_dask_client(self,dask_client):
        if dask_client is None: 
            dask_client = distributed.Client()
            print("No dask client provided to gp2Scale. Using the local client", flush = True)
        else: print("dask client provided to HGDL", flush = True)
        client = dask_client
        worker_info = list(client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        self.workers = {"host": worker_info[0],
                "walkers": worker_info[1:]}
        print("Host ",self.workers["host"]," has ", len(self.workers["walkers"])," workers.")
        self.number_of_walkers = len(self.workers["walkers"])
        return client



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
        sign, logdet = self.slogdet(K)
        n = len(y)
        if sign == 0.0: return (0.5 * (y.T @ x)) + (0.5 * n * np.log(2.0*np.pi))
        return (0.5 * (y.T @ x)) + (0.5 * sign * logdet) + (0.5 * n * np.log(2.0*np.pi))


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
        logdet = np.log(diagL).sum() + np.log(diagU).sum()
        swap_sign = self.minimumSwaps(lu.perm_r)
        sign = swap_sign*np.sign(diagL).prod()*np.sign(diagU).prod()
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
        k = self.kernel({"batch1": self.x_data,"batch2":p,"hps" : self.hyperparameters})
        A = k.T @ self.covariance_value_prod
        #posterior_mean = self.mean_function(self,p,self.hyperparameters) + A
        posterior_mean = A
        return {"x": p,
                "f(x)": posterior_mean}

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
        dask_client = None):

        start_log_likelihood = self.log_likelihood(starting_hps)

        print(
            "fvGP hyperparameter tuning in progress. Old hyperparameters: ",
            starting_hps, " with old log likelihood: ", start_log_likelihood)
        print("method: ", method)

        ############################
        ####global optimization:##
        ############################
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
                workers = 1,
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

