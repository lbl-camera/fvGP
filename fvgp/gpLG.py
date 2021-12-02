import time
import torch
from hgdl.hgdl import HGDL



class gpHGDL():
    """
    gpLG class: Provides tools for a Lagre-Scale single-task GP.

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
        self.x_data = torch.tensor(points, dtype = float)#, requires_grad = True)
        self.point_number = len(self.x_data)
        self.y_data = torch.tensor(values, dtype = float)#, requires_grad = True)
        self.compute_device = compute_device

        if self.normalize_y is True: self._normalize_y_data()
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            #, requires_grad = True) *
            self.variances = torch.ones((self.y_data.shape), dtype = float) * \
                    abs(self.y_data / 100.0)
            print("CAUTION: you have not provided data variances in fvGP,")
            print("they will be set to 1 percent of the data values!")
        elif variances.dim() == 2:
            self.variances = variances[:,0]
        elif variances.dim() == 1:
            self.variances = torch.tensor(variances, dtype = float)#, requires_grad = True)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        if len(self.variances[self.variances < 0]) > 0: raise Exception("Negative measurement variances communicated to fvgp.")
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if callable(gp_kernel_function): self.kernel = gp_kernel_function
        else: self.kernel = self.default_kernel
        self.d_kernel_dx = self.d_gp_kernel_dx

        self.gp_mean_function = gp_mean_function
        if  callable(gp_mean_function): self.mean_function = gp_mean_function
        else: self.mean_function = self.default_mean_function

        if callable(gp_kernel_function_grad): self.dk_dh = gp_kernel_function_grad
        else:
            if self.ram_economy is True: self.dk_dh = self.gp_kernel_derivative
            else: self.dk_dh = self.gp_kernel_gradient

        if callable(gp_mean_function_grad): self.dm_dh = gp_mean_function_grad
        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = torch.tensor(init_hyperparameters, dtype = float) #,requires_grad = True)
        ##########################################
        #compute the prior########################
        ##########################################
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
        self.prior_mean_vec = self.mean_function(self,self.x_data,self.hyperparameters)
        cov_y,K = self._compute_covariance_value_product(
                self.hyperparameters,
                self.y_data,
                self.variances,
                self.prior_mean_vec)
        self.prior_covariance = K
        if self.use_inv is True: self.K_inv = self.inv(K)
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
        for i in range(num_batches):
            for j in range(i,num_batches):
                data = {"batch1":batch[i],"batch2": batch[j], "hps" : hyperparameters}
                tasks.append(self.client.submit(self.kernel,data))
        for i in range(len(tasks)):
            #get a part of the covariance, and fit into the sparse one, but only the vales needed
            #throw warning if too many values are not zero
            CoVariance_block = client.gather(tasks[i])

        self.add_to_diag(CoVariance, variances)
        return CoVariance

    def slogdet(self, A):
        """
        fvGPs slogdet method based on torch
        """
        #s,l = np.linalg.slogdet(A)
        #return s,l
        if self.compute_device == "cpu":
            sign, logdet = torch.slogdet(A)
            logdet = torch.nan_to_num(logdet)
            return sign, logdet
        elif self.compute_device == "gpu" or self.compute_device == "multi-gpu":
            sign, logdet = torch.slogdet(A)
            sign = sign.cpu()
            logdet = logdet.cpu()
            logdet = torch.nan_to_num(logdet)
            return sign, logdet

    def inv(self, A):
            B = torch.inverse(A)
            return B

    def solve(self, A, b):
        """
        fvGPs slogdet method based on torch
        """
        #x = np.linalg.solve(A,b)
        #return x
        #if b.dim() == 1: b = np.expand_dims(b,axis = 1)
        if self.compute_device == "cpu":
        #    #####for sparsity:
        #    if self.sparse == True:
        #        zero_indices = np.where(A < 1e-16)
        #        A[zero_indices] = 0.0
        #        if self.is_sparse(A):
        #            try:
        #                A = scipy.sparse.csr_matrix(A)
        #                x = scipy.sparse.spsolve(A,b)
        #                return x
        #            except Exceprion as e:
        #                print("fvGP: Sparse solve did not work out.")
        #                print("reason: ", str(e))
            ##################
            try:
                x = torch.linalg.solve(A,b)
                return x
            except Exception as e:
                try:
                    print("fvGP: except statement invoked: torch.solve() on cpu did not work")
                    print("reason: ", str(e))
                    #x, qr = torch.lstsq(b,A)
                    x, qr = torch.linalg.lstsq(A,b)
                except Exception as e:
                    print("fvGP: except statement 2 invoked: torch.solve() and torch.lstsq() on cpu did not work")
                    print("falling back to numpy.lstsq()")
                    print("reason: ", str(e))
                    x,res,rank,s = torch.linalg.lstsq(A,b)
                    return x
            return x
        elif self.compute_device == "gpu" or A.ndim < 3:
            A = A.to(device = "cuda")
            b = b.to(device = "cuda")
            try:
                x = torch.linalg.solve(A, b)
            except Exception as e:
                print("fvGP: except statement invoked: torch.solve() on gpu did not work")
                print("reason: ", str(e))
                x,res,rank,s = torch.linalg.lstsq(A,b)
            return x.cpu()
        #elif self.compute_device == "multi-gpu":
        #    n = min(len(A), torch.cuda.device_count())
        #    split_A = torch.tensor_split(A,n)
        #    split_b = torch.tensor_split(b,n)
        #    results = []
        #    for i, (tmp_A,tmp_b) in enumerate(zip(split_A,split_b)):
        #        cur_device = torch.device("cuda:"+str(i))
        #        tmp_A = tmp_A.to(device = cur_device)
        #        tmp_b = tmp_b.to(device = cur_device)
        #        results.append(torch.linalg.solve(tmp_A,tmp_b)[0])
        #    total = results[0].cpu()
        #    for i in range(1,len(results)):
        #        total = np.append(total, results[i].cpu().numpy(), 0)
        #    return total
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

