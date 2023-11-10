#!/usr/bin/env python
import inspect
import time
import warnings
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import minres
import dask.distributed as distributed
import numpy as np
import scipy
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from loguru import logger
from .mcmc import mcmc
from hgdl.hgdl import HGDL
from .gp2Scale import gp2Scale as gp2S
from dask.distributed import Client
from scipy.stats import norm
from imate import logdet

# TODO:


class GP():
    """
    This class provides all the tools for a single-task Gaussian Process (GP).
    Use fvGP for multi-task GPs. However, the fvGP class inherits all methods from this class.
    This class allows for full HPC support for training via the HGDL package.

    V ... number of input points

    D ... input space dimensionality

    N ... arbitrary integers (N1, N2,...)


    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space (D). If the input is non-Euclidean, the input dimensionality will be ignored.
    x_data : np.ndarray or set or list
        The input point positions. Shape (V x D), where D is the `input_space_dim`. If dealing with non-Euclidean inputs
        x_data should be a set or iterable, not a numpy array.
    y_data : np.ndarray
        The values of the data points. Shape (V,1) or (V).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is a random draw from a uniform distribution
        within hyperparameter_bounds, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD). If sparse_node or gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    hyperparameter_bounds : np.ndarray, optional
        A 2d numpy array of shape (N x 2), where N is the number of needed hyperparameters.
        The default is None, in which case the hyperparameter_bounds are estimated from the domain size
        and the initial y_data. If normalize_y is True or the data changes significantly,
        the hyperparameters and the bounds should be changed/retrained. Initial hyperparameters and bounds
        can also be set in the train calls. The default only works for the default kernels.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the data
        `y_data` in form of a point-wise variance. Shape (len(y_data), 1) or (len(y_data)).
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data) / 100.0`. If
        noise covariances are required, also make use of the gp_noise_function.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the compute_device
        becomes much more important. In that case, the default kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive semi-definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input x1 is a N1 x D array of positions, x2 is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        user-defined length for other kernels
        obj is an `fvgp.GP` instance. The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a covariance matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `'gp_kernel_function'` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input x1 (a N1 x D array of positions),
        x2 (a N2 x D array of positions),
        hyperparameters (a 1d array of length D+1 for the default kernel), and a
        `fvgp.GP` instance. The default is a finite difference calculation.
        If 'ram_economy' is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If 'ram economy' is False,the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See 'ram_economy'.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `'gp_mean_function'` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D), hyperparameters
        (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `'gp_mean_function'` is provided.
    gp_noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input x is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a fvgp.GP instance.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `'gp_noise_function'`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D),
        hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 3-D array of
        shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters. If `'gp_noise_function'` is provided but no gradient function,
        a finite-difference approximation will be used.
        The same rules regarding ram economy as for the kernel definition apply here.
    normalize_y : bool, optional
        If True, the data values `'y_data'` will be normalized to max(y_data) = 1, min(y_data) = 0.
        The default is False.
        Variances will be updated accordingly.
    sparse_mode : bool, optional
        When sparse_mode is enabled, the algorithm will use a user-defined kernel
        function or, if that's not provided, an anisotropic Wendland kernel
        and check for sparsity in the prior covariance. If sparsity is present,
        sparse operations will be used to speed up computations.
        Caution: the covariance is still stored at first in a dense format.
        For more extreme scaling, check out the gp2Scale option.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. The noise function will have
        to return a scipy.sparse matrix instead of a numpy array. There are a few more
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the compute_device option should be revisited.
        The kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale to distribute covariance computations over. Has to contain at least 3 workers.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    store_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster.
        For larger problems (>2000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        True. Note, the training will always use Cholesky or LU decomposition instead of the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the marginal log_likelihood is/are used for the training.
        If True, components of the derivative of the marginal log-likelihood are
        calculated subsequently, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (or noise function) with
        respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for ram_economy=True it should be
        of the form f(x1[, x2], direction, hyperparameters, obj)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ram_economy=False, the function should be of the form f(x1[, x2,] hyperparameters, obj)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.
    info : bool, optional
        Provides a way how to see the progress of gp2Scale, Default is False



    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation (co)variances
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP
    KVinv : np.ndarray
        If enabled, the inverse of the prior covariance + nose matrix V
        inv(K+V)
    KVlogdet : float
        logdet(K+V)
    V : np.ndarray
        the noise covariance matrix
    """

    def __init__(
        self,
        input_space_dim,
        x_data,
        y_data,
        init_hyperparameters=None,
        hyperparameter_bounds=None,
        noise_variances=None,
        compute_device="cpu",
        gp_kernel_function=None,
        gp_kernel_function_grad=None,
        gp_noise_function=None,
        gp_noise_function_grad=None,
        gp_mean_function=None,
        gp_mean_function_grad=None,
        sparse_mode=False,
        gp2Scale=False,
        gp2Scale_dask_client=None,
        gp2Scale_batch_size=10000,
        normalize_y=False,
        store_inv=True,
        ram_economy=False,
        args=None,
        info=False
    ):
        ########################################
        ###assign and check some attributes#####
        ########################################
        if isinstance(x_data, np.ndarray):
            if np.ndim(x_data) == 1: x_data = x_data.reshape(-1, 1)
            if input_space_dim != len(x_data[0]):
                raise ValueError("The input space dimension is not in agreement with the provided x_data.")
            self.non_Euclidean = False
        else:
            if not callable(gp_kernel_function):
                raise Exception(
                    "For GPs on non-Eucledian input spaces you need a user-defined kernel, hyperparameters, \
                    and hyperparameter_bounds.")
            input_space_dim = 1
            self.non_Euclidean = True
        if self.non_Euclidean and init_hyperparameters is None:
            raise Exception(
                "You are running fvGP on non-Euclidean inputs. Please provide initial hyperparameters.")
        if np.ndim(y_data) == 2: y_data = y_data[:, 0]
        if compute_device == 'gpu':
            try:
                import torch
            except:
                raise Exception(
                    "You have specified the 'gpu' as your compute device. You need to install pytorch\
                     manually for this to work.")
        self.normalize_y = normalize_y
        self.input_space_dim = input_space_dim
        self.x_data = x_data
        self.point_number = len(self.x_data)
        self.y_data = y_data
        self.compute_device = compute_device
        self.ram_economy = ram_economy
        self.args = args
        self.info = info
        self.sparse_mode = sparse_mode
        self.store_inv = store_inv
        self.KVinv = None
        self.mcmc_info = None
        self.gp2Scale = gp2Scale
        self.hyperparameter_bounds = hyperparameter_bounds

        if (callable(gp_kernel_function) or callable(gp_mean_function) or callable(
            gp_noise_function)) and init_hyperparameters is None:
            warnings.warn(
                "You have provided callables for kernel, mean, or noise \
                functions but no initial hyperparameters.",
                stacklevel=2)
            warnings.warn("It is likely they have to be defined for a success initialization.", stacklevel=2)

        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        if self.hyperparameter_bounds is None:
            if self.non_Euclidean:
                if callable(gp_kernel_function):
                    warnings.warn("You have not provided hyperparameter bounds but a kernel function. \n \
                    Make sure you provide them tot he training.")
            else:
                hyperparameter_bounds = np.zeros((input_space_dim + 1, 2))
                hyperparameter_bounds[0] = np.array([np.var(y_data) / 100., np.var(y_data) * 10.])
                for i in range(input_space_dim):
                    range_xi = np.max(x_data[:, i]) - np.min(x_data[:, i])
                    hyperparameter_bounds[i + 1] = np.array([range_xi / 100., range_xi * 10.])
                self.hyperparameter_bounds = hyperparameter_bounds

        if init_hyperparameters is None:
            if self.hyperparameter_bounds is None: raise Exception("hyperparameter_bounds non available.")
            init_hyperparameters = np.random.uniform(low=self.hyperparameter_bounds[:, 0],
                                                     high=self.hyperparameter_bounds[:, 1],
                                                     size=len(self.hyperparameter_bounds))
        self.hyperparameters = init_hyperparameters
        ##########################################
        ##############preps for sparse mode#######
        ##########################################
        if self.sparse_mode and self.store_inv:
            warnings.warn(
                "sparse_mode and store_inv enabled but they should not be used together. \
                I'll set store_inv = False.",
                stacklevel=2)
            self.store_inv = False
        if self.sparse_mode and not callable(gp_kernel_function):
            warnings.warn("You have chosen to activate sparse mode. Great! \n \
                        But you have not supplied a kernel that is compactly supported. \n I will use an \
                        anisotropic Wendland kernel for now.",
                          stacklevel=2)
            gp_kernel_function = self.wendland_anisotropic

        ###########################################
        #####gp2Scale##############################
        ###########################################
        if gp2Scale:
            if gp2Scale_dask_client is None:
                gp2Scale_dask_client = Client()
                warnings.warn("gp2Scale needs a 'gp2Scale_dask_client'. \
                Set to distributed.Client().", stacklevel=2)
            self.gp2Scale_dask_client = gp2Scale_dask_client

            if not callable(gp_kernel_function):
                warnings.warn("You have chosen to activate gp2SCale. A powerful tool! \n \
                        But you have not supplied a kernel that is compactly supported. \n I will use an anisotropic Wendland kernel for now.",
                              stacklevel=2)
                if compute_device == "cpu":
                    gp_kernel_function = wendland_anisotropic_gp2Scale_cpu
                elif compute_device == "gpu":
                    gp_kernel_function = wendland_anisotropic_gp2Scale_gpu

            self.gp2Scale_obj = gp2S(x_data, batch_size=gp2Scale_batch_size,
                                     gp_kernel_function=gp_kernel_function,
                                     covariance_dask_client=gp2Scale_dask_client,
                                     info=info)
            self.store_inv = False
            warnings.warn("WARNING: gp2Scale activated. Only training via MCMC will be performed. \
                    Only noise variances (no noise covariances can be considered). \
                    A customed sparse kernel should be used, otherwise an anisotropic Wendland kernel is used.",
                          stacklevel=2)

        ###########################################
        ###assign kernel, mean and noise functions#
        ###########################################
        # noise
        if noise_variances is not None and callable(gp_noise_function):
            warnings.warn("Noise function and measurement noise provided. noise_variances set to None.", stacklevel=2)
            noise_variances = None
        if callable(gp_noise_function):
            self.noise_function = gp_noise_function
        elif noise_variances is not None:
            self.noise_function = None
        else:
            warnings.warn(
                "No noise function or measurement noise provided. Noise variances will be set to 1% of mean(y_data).",
                stacklevel=2)
            self.noise_function = self._default_noise_function
        if callable(gp_noise_function_grad):
            self.noise_function_grad = gp_noise_function_grad
        elif callable(gp_noise_function):
            if self.ram_economy is True:
                self.noise_function_grad = self._finitediff_dnoise_dh_econ
            else:
                self.noise_function_grad = self._finitediff_dnoise_dh
        else:
            if self.ram_economy is True:
                self.noise_function_grad = self._default_dnoise_dh_econ
            else:
                self.noise_function_grad = self._default_dnoise_dh

        # kernel
        if callable(gp_kernel_function):
            self.kernel = gp_kernel_function
        elif gp_kernel_function is None:
            self.kernel = self.default_kernel
        else:
            raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self._d_gp_kernel_dx
        if callable(gp_kernel_function_grad):
            self.dk_dh = gp_kernel_function_grad
        else:
            if self.ram_economy is True:
                self.dk_dh = self._gp_kernel_derivative
            else:
                self.dk_dh = self._gp_kernel_gradient

        # prior mean
        if callable(gp_mean_function):
            self.mean_function = gp_mean_function
        else:
            self.mean_function = self._default_mean_function
        if callable(gp_mean_function_grad):
            self.dm_dh = gp_mean_function_grad
        elif callable(gp_mean_function):
            self.dm_dh = self._finitediff_dm_dh
        else:
            self.dm_dh = self._default_dm_dh
        if gp_kernel_function is None and (gp_noise_function is not None or gp_mean_function is not None):
            warnings.warn("You are using the default kernel but a user-defined mean or noise function. \
                    Make sure the right hyperparameter indices are used in all functions. \
                    The default kernel function uses the first D + 1 hyperparameters", stacklevel=2)

        ##########################################
        #######prepare noise covariances##########
        ##########################################
        if noise_variances is None:
            ##noise covariances are always a square matrix
            self.V = self.noise_function(self.x_data, init_hyperparameters, self)
            if isinstance(self.V, np.ndarray) and np.ndim(self.V) == 1:
                raise Exception(
                    "Your noise function did not return a square matrix, it should though, the noise can be correlated.")
            elif isinstance(self.V, np.ndarray) and self.V.shape[0] != self.V.shape[1]:
                raise Exception("Your noise function return is not a square matrix")
        elif gp2Scale:
            self.V = gp2Scale_obj.calculate_sparse_noise_covariance(noise_variances)
        elif np.ndim(noise_variances) == 2:
            if any(noise_variances <= 0.0): raise Exception(
                "Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.V = np.diag(noise_variances[:, 0])
        elif np.ndim(noise_variances) == 1:
            if any(noise_variances <= 0.0): raise Exception(
                "Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.V = np.diag(noise_variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")

        #########################################
        ###########normalization#################
        #########################################
        if self.normalize_y:
            warnings.warn(
                "y_data and noise normalized. Make sure your hyperparameters and their bounds are still valid. They will not be recomputed.")
            self.y_data, self.y_min, self.y_max = self._normalize_y_data(self.y_data)
            self.V = (1. / (self.y_max - self.y_min) ** 2) * self.V

        ##########################################
        # compute the prior########################
        ##########################################
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V = self._compute_GPpriorV(
            self.x_data, self.y_data, self.hyperparameters, calc_inv=self.store_inv)

    def update_gp_data(
        self,
        x_data,
        y_data,
        noise_variances=None,
    ):
        """
        This function updates the data in the gp object instance.
        The data will NOT be appended but overwritten!
        Please provide the full updated data set.

        Parameters
        ----------
        x_data : np.ndarray
            The point positions. Shape (V x D), where D is the `input_space_dim`.
        y_data : np.ndarray
            The values of the data points. Shape (V,1) or (V).
        noise_variances : np.ndarray, optional
            An numpy array defining the uncertainties in the data `y_data` in form of a point-wise variance.
            Shape (len(y_data), 1) or (len(y_data)).
            Note: if no variances are provided here, the noise_covariance
            callable will be used; if the callable is not provided the noise variances
            will be set to `'abs(np.mean(y_data)) / 100.0'`. If you provided a noise function,
            the noise_variances will be ignored.
        """
        if isinstance(x_data, np.ndarray):
            if np.ndim(x_data) == 1: x_data = x_data.reshape(-1, 1)
            if self.input_space_dim != len(x_data[0]): raise ValueError(
                "The input space dimension id not in agreement with the provided x_data.")
        if np.ndim(y_data) == 2: y_data = y_data[:, 0]

        self.x_data = x_data
        self.point_number = len(self.x_data)
        self.y_data = y_data
        ###########################################
        #####gp2Scale##############################
        ###########################################
        if self.gp2Scale:
            self.gp2Scale_obj = gp2S(x_data, batch_size=gp2Scale_batch_size,
                                     gp_kernel_function=self.kernel,
                                     covariance_dask_client=self.gp2Scale_dask_client,
                                     info=self.info)
        ##########################################
        #######prepare noise covariances##########
        ##########################################
        if noise_variances is not None and callable(self.noise_function):
            warnings.warn("Noise function and measurement noise provided. noise_variances set to None", stacklevel=2)
            noise_variances = None

        if noise_variances is None:
            ##noise covariances are always a square matrix
            if not callable(self.noise_function): raise Exception(
                "You originally provided noise in the initialization; please provide noise for the updated data.")
            self.V = self.noise_function(self.x_data, self.hyperparameters, self)
        elif np.ndim(noise_variances) == 2:
            if any(noise_variances <= 0.0): raise Exception(
                "Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.V = np.diag(noise_variances[:, 0])
        elif np.ndim(noise_variances) == 1:
            if any(noise_variances <= 0.0): raise Exception(
                "Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.V = np.diag(noise_variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array.")

        #########################################
        ###########normalization#################
        #########################################
        if self.normalize_y:
            warnings.warn(
                "y_data and noise normalized. Make sure your hyperparameters and their bounds are still valid. They will not be recomputed.")
            self.y_data, self.y_min, self.y_max = self._normalize_y_data(self.y_data)
            self.V = (1. / (self.y_max - self.y_min) ** 2) * self.V
        ######################################
        #####transform to index set###########
        ######################################
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V = self._compute_GPpriorV(
            self.x_data, self.y_data, self.hyperparameters, calc_inv=self.store_inv)

    ###################################################################################
    ###################################################################################
    ###################################################################################
    #################TRAINING##########################################################
    ###################################################################################
    def train(self,
              hyperparameter_bounds=None,
              init_hyperparameters=None,
              method="global",
              pop_size=20,
              tolerance=0.0001,
              max_iter=120,
              local_optimizer="L-BFGS-B",
              global_optimizer="genetic",
              constraints=(),
              dask_client=None):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be 'hgdl' and
        providing a dask client. However, in that case fvgp.GP.train_async() is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization.
            The default is an array of bounds of the length of the initial hyperparameters
            with all bounds defined practically as [0.00001, inf].
            The initial hyperparameters are either defined by the user at initialization, or in this function call,
            or are defined as np.ones((input_space_dim + 1)).
            This choice is only recommended in very basic scenarios and
            can lead to suboptimal results. It is better to provide
            hyperparameter bounds.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the bounds.
        method : str or Callable, optional
            The method used to train the hyperparameters.
            The options are `'global'`, `'local'`, `'hgdl'`, `'mcmc'`, and a callable.
            The callable gets a gp.GP instance and has to return a 1d np.ndarray of hyperparameters.
            The default is `'global'` (scipy's differential evolution).
            If method = "mcmc",
            the attribute fvgp.GP.mcmc_info is updated and contains convergence and distribution information.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `'hgdl'` see `'hgdl.readthedocs.io'`.
            If the optimizer is a scipy optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.
        """
        if hyperparameter_bounds is None:
            if self.hyperparameter_bounds is None: raise Exception("Please provide hyperparameter_bounds")
            hyperparameter_bounds = self.hyperparameter_bounds.copy()
        if init_hyperparameters is None: init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                                                  high=hyperparameter_bounds[:, 1],
                                                                                  size=len(hyperparameter_bounds))

        self.hyperparameters = self._optimize_log_likelihood(
            init_hyperparameters,
            np.array(hyperparameter_bounds),
            method,
            max_iter,
            pop_size,
            tolerance,
            constraints,
            local_optimizer,
            global_optimizer,
            dask_client
        )
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V = self._compute_GPpriorV(
            self.x_data, self.y_data, self.hyperparameters, calc_inv=self.store_inv)

    ##################################################################################
    def train_async(self,
                    hyperparameter_bounds=None,
                    init_hyperparameters=None,
                    max_iter=10000,
                    local_optimizer="L-BFGS-B",
                    global_optimizer="genetic",
                    constraints=(),
                    dask_client=None):
        """
        This function asynchronously finds the maximum of the log marginal likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to `'fvgp.GP.update_hyperparameters()'`,
        which will automatically update the GP prior with the new hyperparameters.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization.
            The default is an array of bounds for the default kernel D = input_space_dim + 1
            with all bounds defined practically as [0.00001, inf].
            This choice is only recommended in very basic scenarios.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the bounds.
        max_iter : int, optional
            Maximum number of epochs for HGDL. Default = 10000.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of hgdl.NonLinearConstraint instances, optional
            Equality and inequality constraints for the optimization. See `'hgdl.readthedocs.io'`
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.

        Return
        ------
        Optimization object that can be given to `fvgp.GP.update_hyperparameters()`
        to update the prior GP : object instance
        """
        if self.gp2Scale: raise Exception("gp2Scale does not allow asynchronous training!")
        if dask_client is None: dask_client = distributed.Client()
        if hyperparameter_bounds is None:
            if self.hyperparameter_bounds is None: raise Exception("Please provide hyperparameter_bounds")
            hyperparameter_bounds = self.hyperparameter_bounds.copy()
        if init_hyperparameters is None: init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                                                  high=hyperparameter_bounds[:, 1],
                                                                                  size=len(hyperparameter_bounds))

        opt_obj = self._optimize_log_likelihood_async(
            init_hyperparameters,
            hyperparameter_bounds,
            max_iter,
            constraints,
            local_optimizer,
            global_optimizer,
            dask_client
        )
        return opt_obj

    ##################################################################################
    def stop_training(self, opt_obj):
        """
        This function stops the training if HGDL is used. It leaves the dask client alive.

        Parameters
        ----------
        opt_obj : object
            An object returned form the `fvgp.GP.train_async()` function.
        """
        try:
            opt_obj.cancel_tasks()
            logger.debug("fvGP successfully cancelled the current training.")
        except:
            warnings.warn("No asynchronous training to be cancelled in fvGP, no training is running.", stacklevel=2)

    ###################################################################################
    def kill_training(self, opt_obj):
        """
        This function stops the training if HGDL is used, and kills the dask client.

        Parameters
        ----------
        opt_obj : object
            An object returned form the `fvgp.GP.train_async()` function.
        """

        try:
            opt_obj.kill_client()
            logger.debug("fvGP successfully killed the training.")
        except:
            warnings.warn("No asynchronous training to be killed, no training is running.", stacklevel=2)

    ##################################################################################
    def update_hyperparameters(self, opt_obj):
        """
        This function asynchronously finds the maximum of the marginal log_likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function just submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`, which will automatically
        update the GP prior with the new hyperparameters.

        Parameters
        ----------
        object : HGDL class instance
            HGDL class instance returned by `fvgp.GP.train_async()`

        Return
        ------
        The current hyperparameters : np.ndarray
        """

        try:
            res = opt_obj.get_latest()[0]["x"]
        except:
            logger.debug("      The optimizer object could not be queried")
            logger.debug("      That probably means you are not optimizing the hyperparameters asynchronously")
        else:
            try:
                l_n = self.neg_log_likelihood(res)
                l_o = self.neg_log_likelihood(self.hyperparameters)
                if l_n - l_o < 0.000001:
                    self.hyperparameters = res
                    self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V = self._compute_GPpriorV(
                        self.x_data, self.y_data, self.hyperparameters, calc_inv=self.store_inv)
                    logger.debug("    fvGP async hyperparameter update successful")
                    logger.debug("    Latest hyperparameters: {}", self.hyperparameters)
                else:
                    logger.debug(
                        "    The update was attempted but the new hyperparameters led to a lower likelihood, so I kept the old ones")
                    logger.debug(f"Old likelihood: {-l_o} at {self.hyperparameters}")
                    logger.debug(f"New likelihood: {-l_n} at {res}")
            except Exception as e:
                logger.debug("    Async Hyper-parameter update not successful in fvGP. I am keeping the old ones.")
                logger.debug("    hyperparameters: {}", self.hyperparameters)

        return self.hyperparameters

    ##################################################################################
    def set_hyperparameters(self, hps):
        """
        Function to set hyperparameters.

        Parameters:
        -----------
        hps : np.ndarray
            A 1-d numpy array of hyperparameters.
        """
        self.hyperparameters = np.array(hps)
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V = self._compute_GPpriorV(
            self.x_data, self.y_data, self.hyperparameters, calc_inv=self.store_inv)

    ##################################################################################
    def get_hyperparameters(self):
        """
        Function to get the current hyperparameters.

        Parameters: None
        -----------

        Return:
        -------
        hyperparameters : np.ndarray
        """

        return self.hyperparameters

    ##################################################################################
    def get_prior_pdf(self):
        """
        Function to get the current prior covariance matrix.

        Parameters:
        -----------
        None

        Return:
        -------
        A dictionary containing information about the GP prior distribution : dict
        """

        return {"prior corvariance (K)": self.K, "log(|KV|)": self.KVlogdet, "inv(KV)": self.KVinv,
                "prior mean": self.prior_mean_vec}

    ##################################################################################
    def _optimize_log_likelihood_async(self,
                                       starting_hps,
                                       hp_bounds,
                                       max_iter,
                                       constraints,
                                       local_optimizer,
                                       global_optimizer,
                                       dask_client):

        logger.debug("fvGP hyperparameter tuning in progress. Old hyperparameters: {}", starting_hps)
        if not self._in_bounds(starting_hps, hp_bounds):
            raise Exception("Starting positions outside of optimization bounds.")

        opt_obj = HGDL(self.neg_log_likelihood,
                       self.neg_log_likelihood_gradient,
                       hp_bounds,
                       hess=self.neg_log_likelihood_hessian,
                       local_optimizer=local_optimizer,
                       global_optimizer=global_optimizer,
                       num_epochs=max_iter,
                       constraints=constraints)

        logger.debug("HGDL successfully initialized. Calling optimize()")
        opt_obj.optimize(dask_client=dask_client, x0=np.array(starting_hps).reshape(1, -1))
        logger.debug("optimize() called")
        return opt_obj

    ##################################################################################
    def _optimize_log_likelihood(self,
                                 starting_hps,
                                 hp_bounds,
                                 method,
                                 max_iter,
                                 pop_size,
                                 tolerance,
                                 constraints,
                                 local_optimizer,
                                 global_optimizer,
                                 dask_client=None):

        if not self._in_bounds(starting_hps, hp_bounds):
            raise Exception("Starting positions outside of optimization bounds.")

        if self.gp2Scale:
            method = 'mcmc'
        else:
            start_log_likelihood = self.log_likelihood(starting_hps)
            logger.debug("fvGP hyperparameter tuning in progress. Old hyperparameters: ",
                         starting_hps, " with old log likelihood: ", start_log_likelihood)
        logger.debug("method: ", method)

        ############################
        ####global optimization:##
        ############################
        if method == "global":
            logger.debug(
                "fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hp_bounds)
            res = differential_evolution(
                self.neg_log_likelihood,
                hp_bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=tolerance,
                disp=self.info,
                polish=False,
                x0=starting_hps.reshape(1, -1),
                constraints=constraints,
                workers=1,
            )
            hyperparameters = np.array(res["x"])
            Eval = self.neg_log_likelihood(hyperparameters)
            logger.debug(f"fvGP found hyperparameters {hyperparameters} with likelihood {Eval} via global optimization")
        ############################
        ####local optimization:#####
        ############################
        elif method == "local":
            logger.debug("fvGP is performing a local update of the hyper parameters.")
            logger.debug("starting hyperparameters: {}", starting_hps)
            logger.debug("Attempting a BFGS optimization.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hp_bounds)
            OptimumEvaluation = minimize(
                self.neg_log_likelihood,
                starting_hps,
                method=local_optimizer,
                jac=self.neg_log_likelihood_gradient,
                hess=self.neg_log_likelihood_hessian,
                bounds=hp_bounds,
                tol=tolerance,
                callback=None,
                constraints=constraints,
                options={"maxiter": max_iter})

            if OptimumEvaluation["success"] == True:
                logger.debug(f"fvGP local optimization successfully concluded with result: "
                             f"{OptimumEvaluation['fun']} at {OptimumEvaluation['x']}")
            else:
                logger.debug("fvGP local optimization not successful.")
            hyperparameters = OptimumEvaluation["x"]
        ############################
        ####hybrid optimization:####
        ############################
        elif method == "hgdl":
            logger.debug("fvGP submitted HGDL optimization")
            logger.debug("starting hyperparameters: {}", starting_hps)
            logger.debug('bounds are', hp_bounds)

            opt_obj = HGDL(self.neg_log_likelihood,
                           self.neg_log_likelihood_gradient,
                           hp_bounds,
                           hess=self.neg_log_likelihood_hessian,
                           local_optimizer=local_optimizer,
                           global_optimizer=global_optimizer,
                           num_epochs=max_iter,
                           constraints=constraints)

            opt_obj.optimize(dask_client=dask_client, x0=starting_hps.reshape(1, -1))
            hyperparameters = opt_obj.get_final()[0]["x"]
            opt_obj.kill_client()
        elif method == "mcmc":
            logger.debug("MCMC started in fvGP")
            logger.debug('bounds are {}', hp_bounds)
            res = mcmc(self.log_likelihood, hp_bounds, x0=starting_hps, n_updates=max_iter, info=self.info)
            hyperparameters = np.array(res["distribution mean"])
            self.mcmc_info = res
        elif callable(method):
            hyperparameters = method(self)
        else:
            raise ValueError("No optimization mode specified in fvGP")
        ###################################################
        if method != 'mcmc': new_likelihood = self.log_likelihood(hyperparameters)
        if method != 'mcmc' and start_log_likelihood > new_likelihood:
            logger.debug(
                f"New hyperparameters: {hyperparameters} with log likelihood: {self.log_likelihood(hyperparameters)}")
            warning_str = "Old log marginal likelihood: " + str(
                start_log_likelihood) + " New log marginal likelihood: " + str(new_likelihood)
            warnings.warn(f"Old log marginal likelihood:  {start_log_likelihood}", stacklevel=2)
            warnings.warn(f"New log marginal likelihood:  {new_likelihood}", stacklevel=2)
            hyperparameters = starting_hps
        return hyperparameters

    ##################################################################################
    def log_likelihood(self, hyperparameters=None):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Return
        ------
        log marginal likelihood of the data : float
        """
        if hyperparameters is None:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V
        else:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv=False)
        n = len(self.y_data)
        return -(0.5 * ((self.y_data - mean).T @ KVinvY)) - (0.5 * KVlogdet) - (0.5 * n * np.log(2.0 * np.pi))

    ##################################################################################
    def neg_log_likelihood(self, hyperparameters=None):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N)
            If not provided, the covariance will not be recomputed.

        Return
        ------
        negative log marginal likelihood of the data : float
        """
        return -self.log_likelihood(hyperparameters=hyperparameters)

    ##################################################################################
    def neg_log_likelihood_gradient(self, hyperparameters=None):
        """
        Function that computes the gradient of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Return
        ------
        Gradient of the negative log marginal likelihood : np.ndarray
        """
        logger.debug("log-likelihood gradient is being evaluated...")
        if hyperparameters is None:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.V
        else:
            K, KV, KVinvY, KVlogdet, FO, KVinv, mean, cov = \
                self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv=False)

        b = KVinvY
        y = self.y_data - mean
        if self.ram_economy is False:
            try:
                dK_dH = self.dk_dh(self.x_data, self.x_data, hyperparameters, self) + self.noise_function_grad(
                    self.x_data, hyperparameters, self)
            except Exception as e:
                raise Exception(
                    "The gradient evaluation dK/dh + dNoise/dh was not successful. \n That normally means the combination of ram_economy and definition of the gradient function is wrong. ",
                    str(e))
            KV = np.array([KV, ] * len(hyperparameters))
            a = self._solve(KV, dK_dH)
        bbT = np.outer(b, b.T)
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.dm_dh(self.x_data, hyperparameters, self)

        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if self.ram_economy is False:
                matr = a[i]
            else:
                try:
                    dK_dH = self.dk_dh(self.x_data, self.x_data, i, hyperparameters, self) + self.noise_function_grad(
                        self.x_data, i, hyperparameters, self)
                except:
                    raise Exception(
                        "The gradient evaluation dK/dh + dNoise/dh was not successful. \n That normally means the combination of ram_economy and definition of the gradient function is wrong.")
                matr = np.linalg.solve(KV, dK_dH)
            if dL_dHm[i] == 0.0:
                if self.ram_economy is False:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH[i])
                else:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH)
                dL_dH[i] = - 0.5 * (mtrace - np.trace(matr))
            else:
                dL_dH[i] = 0.0
        logger.debug("gradient norm: {}", np.linalg.norm(dL_dH + dL_dHm))
        return dL_dH + dL_dHm

    ##################################################################################
    def neg_log_likelihood_hessian(self, hyperparameters=None):
        """
        Function that computes the Hessian of the marginal log-likelihood.
        It does so by a first-order approximation of the exact gradient.

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Return
        ------
        Hessian of the negative log marginal likelihood : np.ndarray
        """
        ##implemented as first-order approximation
        len_hyperparameters = len(hyperparameters)
        d2L_dmdh = np.zeros((len_hyperparameters, len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.neg_log_likelihood_gradient(hyperparameters=hyperparameters)
        for i in range(len_hyperparameters):
            hps_temp = np.array(hyperparameters)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i, i:] = ((self.neg_log_likelihood_gradient(hyperparameters=hps_temp) - grad_at_hps) / epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))

    def test_log_likelihood_gradient(self, hyperparameters):
        thps = np.array(hyperparameters)
        grad = np.empty((len(thps)))
        eps = 1e-6
        for i in range(len(thps)):
            thps_aux = np.array(thps)
            thps_aux[i] = thps_aux[i] + eps
            grad[i] = (self.log_likelihood(hyperparameters=thps_aux) - self.log_likelihood(hyperparameters=thps)) / eps
        analytical = -self.neg_log_likelihood_gradient(hyperparameters=thps)
        if np.linalg.norm(grad - analytical) > np.linalg.norm(grad) / 100.0:
            print("Gradient possibly wrong")
            print("finite diff appr: ", grad)
            print("analytical      : ", analytical)
        else:
            print("Gradient correct")
            print("finite diff appr: ", grad)
            print("analytical      : ", analytical)
        assert np.linalg.norm(grad - analytical) < np.linalg.norm(grad) / 100.0

        return grad, analytical

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
    ##################################################################################
    ##################################################################################
    def _compute_GPpriorV(self, x_data, y_data, hyperparameters, calc_inv=False):
        # get the prior mean
        prior_mean_vec = self.mean_function(x_data, hyperparameters, self)

        # get the latest noise
        if callable(self.noise_function):
            V = self.noise_function(x_data, hyperparameters, self)
        else:
            V = self.V

        # get K
        if self.gp2Scale:
            st = time.time()
            K = self.gp2Scale_obj.compute_covariance(hyperparameters, self.gp2Scale_dask_client)
            Ksparsity = float(K.nnz) / float(len(x_data) ** 2)
            if self.info: print("Transferring the covariance matrix to host done after ", time.time() - st,
                                " seconds. sparsity = ", Ksparsity, flush=True)
        else:
            K = self._compute_K(hyperparameters)

        # check if shapes are correct
        if K.shape != V.shape: raise Exception("Noise covariance and prior covariance not of the same shape.")

        # get K + V
        KV = K + V

        # get Kinv/KVinvY, LU, Chol, logdet(KV)
        if self.gp2Scale:
            # try fast but RAM intensive SuperLU first
            if len(x_data) < 50000 and Ksparsity < 0.0001:
                try:
                    LU = splu(KV.tocsc())
                    factorization_obj = ("LU", LU)
                    KVinvY = LU.solve(y_data - prior_mean_vec)
                    upper_diag = abs(LU.U.diagonal())
                    KVlogdet = np.sum(np.log(upper_diag))
                # if that did not work, do random lin algebra magic
                except:
                    KVinvY, exit_code = minres(KV.tocsc(), y_data - prior_mean_vec)
                    factorization_obj = ("gp2Scale", None)
                    if exit_code != 0:
                        warnings.warn("CG solve not successful in gp2Scale. Trying MINRES")
                        KVinvY, exit_code = minres(KV.tocsc(), y_data - prior_mean_vec)
                    if self.compute_device == "gpu":
                        gpu = True
                    else:
                        gpu = False
                    KVlogdet, info_slq = logdet(KV, method='slq', min_num_samples=10, max_num_samples=100,
                                                lanczos_degree=20, error_rtol=0.1, gpu=False,
                                                return_info=True, plot=False, verbose=self.info)
            # if the problem is large go with rand. lin. algebra straight away
            else:
                if self.info: print("MINRES solve in progress ...", time.time() - st, "seconds.")
                factorization_obj = ("gp2Scale", None)
                KVinvY, exit_code = minres(KV.tocsc(), y_data - prior_mean_vec)
                if self.info: print("MINRES solve done after ", time.time() - st, "seconds.")

                if self.compute_device == "gpu":
                    gpu = True
                else:
                    gpu = False
                if self.info: print("logdet() in progress ... ", time.time() - st, "seconds.")
                KVlogdet, info_slq = logdet(KV, method='slq', min_num_samples=10, max_num_samples=100,
                                            lanczos_degree=20, error_rtol=0.1, orthogonalize=0, gpu=False,
                                            return_info=True, plot=False, verbose=self.info)
                if self.info: print("logdet/LU done after ", time.time() - st, "seconds.")

            KVinv = None
        elif self.sparse_mode and self._is_sparse(KV):
            KV = csc_matrix(KV)
            LU = splu(KV)
            factorization_obj = ("LU", LU)
            KVinvY = lu.solve(y_data - prior_mean_vec)
            upper_diag = abs(lu.u.diagonal())
            KVlogdet = np.sum(np.log(upper_diag))
            KVinv = None
        else:
            c, l = cho_factor(KV)
            factorization_obj = ("Chol", c, l)
            KVinvY = cho_solve((c, l), y_data - prior_mean_vec)
            upper_diag = abs(c.diagonal())
            KVlogdet = 2.0 * np.sum(np.log(upper_diag))
            if calc_inv:
                KVinv = self._inv(KV)
            else:
                KVinv = None

        return K, KV, KVinvY, KVlogdet, factorization_obj, KVinv, prior_mean_vec, V

    ##################################################################################
    def _compute_K(self, hyperparameters):
        """computes the covariance matrix from the kernel"""
        K = self.kernel(self.x_data, self.x_data, hyperparameters, self)
        return K

    ##################################################################################
    def _KVsolve(self, b):
        if self.factorization_obj[0] == "LU":
            LU = self.factorization_obj[1]
            return LU.solve(b)
        elif self.factorization_obj[0] == "Chol":
            c, l = self.factorization_obj[1], self.factorization_obj[2]
            return cho_solve((c, l), b)
        else:
            res = np.empty((len(b), b.shape[1]))
            if b.shape[1] > 100: warnings.warn(
                "You want to predict at >100 points. \n When using gp2Scale, this takes a while. \n Better predict at only a handful of points.")
            for i in range(b.shape[1]):
                res[:, i], exit_status = minres(self.KV, b[:, i])
                # if exit_status != 0: res[:,i], exit_status = minres(self.KV,b[:,i])
            return res

    def _logdet(self, A, factorization_obj=None):
        if self.compute_device == "cpu":
            s, logdet = np.linalg.slogdet(A)
            return logdet
        elif self.compute_device == "gpu":
            try:
                import torch
                A = torch.from_numpy(A).cuda()
                sign, logdet = torch.slogdet(A)
                sign = sign.cpu().numpy()
                logdet = logdet.cpu().numpy()
                logdet = np.nan_to_num(logdet)
                return logdet
            except Exception as e:
                warnings.warn(
                    "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
                s, logdet = np.linalg.slogdet(A)
                return logdet
        else:
            sign, logdet = np.linalg.slogdet(A)
            return logdet

    def _inv(self, A):
        if self.compute_device == "cpu":
            return np.linalg.inv(A)
        elif self.compute_device == "gpu":
            import torch
            A = torch.from_numpy(A)
            B = torch.inverse(A)
            return B.numpy()
        else:
            return np.linalg.inv(A)

    def _solve(self, A, b):
        if np.ndim(b) == 1: b = np.expand_dims(b, axis=1)
        if self.compute_device == "cpu":
            try:
                x = np.linalg.solve(A, b)
            except:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return x
        elif self.compute_device == "gpu" or A.ndim < 3:
            try:
                import torch
                A = torch.from_numpy(A).cuda()
                b = torch.from_numpy(b).cuda()
                x = torch.linalg.solve(A, b)
                return x.cpu().numpy()
            except Exception as e:
                warnings.warn(
                    "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
                try:
                    x = np.linalg.solve(A, b)
                except:
                    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
                return x
        elif self.compute_device == "multi-gpu":
            try:
                import torch
                n = min(len(A), torch.cuda.device_count())
                split_A = np.array_split(A, n)
                split_b = np.array_split(b, n)
                results = []
                for i, (tmp_A, tmp_b) in enumerate(zip(split_A, split_b)):
                    cur_device = torch.device("cuda:" + str(i))
                    tmp_A = torch.from_numpy(tmp_A).cuda(cur_device)
                    tmp_b = torch.from_numpy(tmp_b).cuda(cur_device)
                    results.append(torch.linalg.solve(tmp_A, tmp_b)[0])
                total = results[0].cpu().numpy()
                for i in range(1, len(results)):
                    total = np.append(total, results[i].cpu().numpy(), 0)
                return total
            except Exception as e:
                warnings.warn(
                    "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
                try:
                    x = np.linalg.solve(A, b)
                except:
                    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
                return x
        else:
            raise Exception("No valid solve method specified")

    ##################################################################################
    def _is_sparse(self, A):
        if float(np.count_nonzero(A)) / float(len(A) ** 2) < 0.01:
            return True
        else:
            return False

    def _how_sparse_is(self, A):
        return float(np.count_nonzero(A)) / float(len(A) ** 2)

    def _default_mean_function(self, x, hyperparameters, gp_obj):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)))
        mean[:] = np.mean(self.y_data)
        return mean

    def _default_noise_function(self, x, hyperparameters, gp_obj):
        noise = np.ones((len(x))) * (np.mean(abs(self.y_data)) / 100.0)
        if self.gp2Scale:
            return self.gp2Scale_obj.calculate_sparse_noise_covariance(noise)
        else:
            return np.diag(noise)

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###############################gp prediction###############################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        """
        This function calculates the posterior mean for a set of input points.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        hyperparameters : np.ndarray, optional
            A numpy array of the correct size depending on the kernel. This is optional in case the posterior mean
            has to be computed with given hyperparameters, which is, for instance, the case if the posterior mean is
            a constraint during training. The default is None which means the initialized or trained hyperparameters
            are used.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution points and function values : dict
        """
        x_data, y_data, KVinvY = self.x_data.copy(), self.y_data.copy(), self.KVinvY.copy()
        if hyperparameters is not None:
            hps = hyperparameters
            K, KV, KVinvY, logdet, FO, KVinv, mean, cov = self._compute_GPpriorV(x_data, y_data, hps, calc_inv=False)
        else:
            hps = self.hyperparameters

        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, hps, self)
        A = k.T @ KVinvY
        posterior_mean = self.mean_function(x_pred, hps, self) + A

        return {"x": x_pred,
                "f(x)": posterior_mean}

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None):
        """
        This function calculates the gradient of the posterior mean for a set of input points.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        hyperparameters : np.ndarray, optional
            A numpy array of the correct size depending on the kernel. This is optional in case the posterior mean
            has to be computed with given hyperparameters, which is, for instance, the case if the posterior mean is
            a constraint during training. The default is None which means the initialized or trained hyperparameters
            are used.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        direction : int, optional
            Direction of derivative, If None (default) the whole gradient will be computed.

        Return
        ------
        Solution : dict
        """
        x_data, y_data, KVinvY = self.x_data.copy(), self.y_data.copy(), self.KVinvY.copy()
        if hyperparameters is not None:
            hps = hyperparameters
            K, KV, KVinvY, logdet, FO, KVinv, mean, cov = self._compute_GPpriorV(x_data, y_data, hps, calc_inv=False)
        else:
            hps = self.hyperparameters

        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, hps, self)
        f = self.mean_function(x_pred, hps, self)
        eps = 1e-6
        if direction is not None:
            x1 = np.array(x_pred)
            x1[:, direction] = x1[:, direction] + eps
            mean_der = (self.mean_function(x1, hps, self) - f) / eps
            k = self.kernel(x_data, x_pred, hps, self)
            k_g = self.d_kernel_dx(x_pred, x_data, direction, hps)
            posterior_mean_grad = mean_der + (k_g @ KVinvY)
        else:
            posterior_mean_grad = np.zeros((x_pred.shape))
            for direction in range(len(x_pred[0])):
                x1 = np.array(x_pred)
                x1[:, direction] = x1[:, direction] + eps
                mean_der = (self.mean_function(x1, hps, self) - f) / eps
                k = self.kernel(x_data, x_pred, hps, self)
                k_g = self.d_kernel_dx(x_pred, x_data, direction, hps)
                posterior_mean_grad[:, direction] = mean_der + (k_g @ KVinvY)
            direction = "ALL"

        return {"x": x_pred,
                "direction": direction,
                "df/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        """
        Function to compute the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        variance_only : bool, optional
            If True the computation of the posterior covariance matrix is avoided which can save compute time.
            In that case the return will only provide the variance at the input points.
            Default = False.
        add_noise : bool, optional
            If True the noise variances will be added to the posterior variances. Default = False.

        Return
        ------
        Solution : dict
        """

        x_data = self.x_data.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
        if self.KVinv is not None:
            if variance_only:
                S = None
                v = np.diag(kk) - np.einsum('ij,jk,ki->i', k.T, self.KVinv, k)
            else:
                S = kk - (k.T @ self.KVinv @ k)
                v = np.array(np.diag(S))
        else:
            k_cov_prod = self._KVsolve(k)
            S = kk - (k_cov_prod.T @ k)
            v = np.array(np.diag(S))
        if np.any(v < -0.001):
            logger.warning(inspect.cleandoc("""#
            Negative variances encountered. That normally means that the model is unstable.
            Rethink the kernel definitions, add more noise to the data,
            or double check the hyperparameter optimization bounds. This will not
            terminate the algorithm, but expect anomalies."""))
            v[v < 0.0] = 0.0
            if not variance_only:
                np.fill_diagonal(S, v)

        if add_noise and callable(self.noise_function):
            noise = self.noise_function(x_pred, self.hyperparameters, self)
            if scipy.sparse.issparse(noise): noise = noise.toarray()
            v = v + np.diag(noise)
            if S is not None: S = S + noise

        return {"x": x_pred,
                "v(x)": v,
                "S": S}

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        """
        Function to compute the gradient of the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        direction : int, optional
            Direction of derivative, If None (default) the whole gradient will be computed.

        Return
        ------
        Solution : dict
        """
        x_data = self.x_data.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        k_covariance_prod = self._KVsolve(k)
        if direction is not None:
            k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
            kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
            x1 = np.array(x_pred)
            x2 = np.array(x_pred)
            eps = 1e-6
            x1[:, direction] = x1[:, direction] + eps
            kk_g = (self.kernel(x1, x1, self.hyperparameters, self) - \
                    self.kernel(x2, x2, self.hyperparameters, self)) / eps
            a = kk_g - (2.0 * k_g.T @ k_covariance_prod)
            return {"x": x_pred,
                    "dv/dx": np.diag(a),
                    "dS/dx": a}
        else:
            grad_v = np.zeros((len(x_pred), len(x_pred[0])))
            for direction in range(len(x_pred[0])):
                k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
                kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
                x1 = np.array(x_pred)
                x2 = np.array(x_pred)
                eps = 1e-6
                x1[:, direction] = x1[:, direction] + eps
                kk_g = (self.kernel(x1, x1, self.hyperparameters, self) - \
                        self.kernel(x2, x2, self.hyperparameters, self)) / eps
                grad_v[:, direction] = np.diag(kk_g - (2.0 * k_g.T @ k_covariance_prod))
            return {"x": x_pred,
                    "dv/dx": grad_v}

    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out=None):
        """
        Function to compute the joint prior over f (at measured locations) and f_pred at x_pred.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution : dict
        """

        x_data, K, prior_mean_vec = self.x_data.copy(), self.K.copy(), self.prior_mean_vec.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
        post_mean = self.mean_function(x_pred, self.hyperparameters, self)
        joint_gp_prior_mean = np.append(prior_mean_vec, post_mean)
        return {"x": x_pred,
                "K": K,
                "k": k,
                "kappa": kk,
                "prior mean": joint_gp_prior_mean,
                "S": np.block([[K, k], [k.T, kk]])}

    ###########################################################################
    def joint_gp_prior_grad(self, x_pred, direction, x_out=None):
        """
        Function to compute the gradient of the data-informed prior.

        Parameters
        ------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        direction : int
            Direction of derivative.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
        """
        x_data, K, prior_mean_vec = self.x_data.copy(), self.K.copy(), self.prior_mean_vec.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
        k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        eps = 1e-6
        x1[:, direction] = x1[:, direction] + eps
        x2[:, direction] = x2[:, direction] - eps
        kk_g = (self.kernel(x1, x1, self.hyperparameters, self) - self.kernel(x2, x2, self.hyperparameters, self)) / (
                2.0 * eps)
        post_mean = self.mean_function(x_pred, self.hyperparameters, self)
        mean_der = (self.mean_function(x1, self.hyperparameters, self) - self.mean_function(x2, self.hyperparameters,
                                                                                            self)) / (2.0 * eps)
        full_gp_prior_mean_grad = np.append(np.zeros((prior_mean_vec.shape)), mean_der)
        prior_cov_grad = np.zeros(K.shape)
        return {"x": x_pred,
                "K": K,
                "dk/dx": k_g,
                "d kappa/dx": kk_g,
                "d prior mean/x": full_gp_prior_mean_grad,
                "dS/dx": np.block([[prior_cov_grad, k_g], [k_g.T, kk_g]])}

    ###########################################################################
    def entropy(self, S):
        """
        Function computing the entropy of a normal distribution
        res = entropy(S); S is a 2d np.ndarray array, a covariance matrix which is non-singular.

        Parameters:
        -----------
        S : np.ndarray
            A covariance matrix.

        Return:
        -------
        Entropy : float
        """
        dim = len(S[0])
        logdet = self._logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * logdet)

    ###########################################################################
    def gp_entropy(self, x_pred, x_out=None):
        """
        Function to compute the entropy of the gp prior probability distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Entropy : float
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        priors = self.joint_gp_prior(x_pred, x_out=None)
        S = priors["S"]
        dim = len(S[0])
        logdet = self._logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * logdet)

    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out=None):
        """
        Function to compute the gradient of entropy of the prior in a given direction.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        direction : int
            Direction of derivative.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Entropy gradient in given direction : float
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        priors1 = self.joint_gp_prior(x_pred, x_out=None)
        priors2 = self.joint_gp_prior_grad(x_pred, direction, x_out=None)
        S1 = priors1["S"]
        S2 = priors2["dS/dx"]
        return 0.5 * np.trace(self._inv(S1) @ S2)

    ###########################################################################
    def _kl_div_grad(self, mu1, dmu1dx, mu2, S1, dS1dx, S2):
        """
        This function computes the gradient of the KL divergence between two normal distributions
        when the gradients of the mean and covariance are given.
        a = kl_div(mu1, dmudx,mu2, S1, dS1dx, S2); S1, S2 are 2d numpy arrays, matrices have to be non-singular,
        mu1, mu2 are mean vectors, given as 2d arrays
        """
        logdet1 = self._logdet(S1)
        logdet2 = self._logdet(S2)
        x1 = self._solve(S2, dS1dx)
        mu = np.subtract(mu2, mu1)
        x2 = self._solve(S2, mu)
        x3 = self._solve(S2, -dmu1dx)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + ((x3.T @ mu) + (x2.T @ -dmu1dx)) - np.trace(np.linalg.inv(S1) @ dS1dx))
        return kld

    ###########################################################################
    def kl_div(self, mu1, mu2, S1, S2):
        """
        Function to compute the KL divergence between two Gaussian distributions.

        Parameters
        ----------
        mu1 : np.ndarray
            Mean vector of distribution 1.
        mu1 : np.ndarray
            Mean vector of distribution 2.
        S1 : np.ndarray
            Covariance matrix of distribution 1.
        S2 : np.ndarray
            Covariance matrix of distribution 2.

        Return
        ------
        KL divergence : float
        """
        logdet1 = self._logdet(S1)
        logdet2 = self._logdet(S2)
        x1 = self._solve(S2, S1)
        mu = np.subtract(mu2, mu1)
        x2 = self._solve(S2, mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu)[0] - float(dim) + (logdet2 - logdet1))
        if kld < -1e-4:
            warnings.warn("Negative KL divergence encountered. That happens when \n \
                    one of the covariance matrices is close to positive semi definite \n\
                    and therefore the logdet() calculation becomes unstable.\n \
                    Returning abs(KLD)")
            logger.debug("Negative KL divergence encountered")
        return abs(kld)

    ###########################################################################
    def gp_kl_div(self, x_pred, comp_mean, comp_cov, x_out=None):
        """
        Function to compute the kl divergence of a posterior at given points
        and a given normal distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        comp_mean : np.array
            Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
        comp_cov : np.array
            Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        res = self.posterior_mean(x_pred, x_out=None)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_pred, x_out=None)["S"]

        return {"x": x_pred,
                "gp posterior mean": gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}

    ###########################################################################
    def gp_kl_div_grad(self, x_pred, comp_mean, comp_cov, direction, x_out=None):
        """
        Function to compute the gradient of the kl divergence of a posterior at given points.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        comp_mean : np.ndarray
            Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
        comp_cov : np.ndarray
            Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
        direction: int
            The direction in which the gradient will be computed.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution : dict
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        gp_mean = self.posterior_mean(x_pred, x_out=None)["f(x)"]
        gp_mean_grad = self.posterior_mean_grad(x_pred, direction=direction, x_out=None)["df/dx"]
        gp_cov = self.posterior_covariance(x_pred, x_out=None)["S"]
        gp_cov_grad = self.posterior_covariance_grad(x_pred, direction=direction, x_out=None)["dS/dx"]

        return {"x": x_pred,
                "gp posterior mean": gp_mean,
                "gp posterior mean grad": gp_mean_grad,
                "gp posterior covariance": gp_cov,
                "gp posterior covariance grad": gp_cov_grad,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div grad": self._kl_div_grad(gp_mean, gp_mean_grad, comp_mean, gp_cov, gp_cov_grad, comp_cov)}

    ###########################################################################
    def mutual_information(self, joint, m1, m2):
        """
        Function to calculate the mutual information between two normal distributions, which is
        equivalent to the KL divergence(joint, marginal1 * marginal1).

        Parameters
        ----------
        joint : np.ndarray
            The joint covariance matrix.
        m1 : np.ndarray
            The first marginal distribution
        m2 : np.ndarray
            The second marginal distribution

        Return
        ------
        Mutual information : float
        """
        return self.entropy(m1) + self.entropy(m2) - self.entropy(joint)

    ###########################################################################
    def gp_mutual_information(self, x_pred, x_out=None):
        """
        Function to calculate the mutual information between
        the random variables f(x_data) and f(x_pred).
        The mutual information is always positive, as it is a KL divergence, and is bounded
        from below by 0. The maxima are expected at the data points. Zero is expected far from the
        data support.
        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
        """
        x_data, K = self.x_data.copy(), self.K.copy() + (np.identity(len(self.K)) * 1e-9)
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self) + (np.identity(len(x_pred)) * 1e-9)

        joint_covariance = \
            np.asarray(np.block([[K, k], \
                                 [k.T, kk]]))
        return {"x": x_pred,
                "mutual information": self.mutual_information(joint_covariance, kk, K)}

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None):
        """
        Function to calculate the interaction information between
        the random variables f(x_data) and f(x_pred). This is the mutual information
        of each f(x_pred) with f(x_data). It is also called the Multiinformation.
        It is best used when several prediction points are supposed to be mutually aware.
        The total correlation is always positive, as it is a KL divergence, and is bounded
        from below by 0. The maxima are expected at the data points. Zero is expected far from the
        data support.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            Total correlation between prediction points, as a collective.
        """
        x_data, K = self.x_data.copy(), self.K.copy() + (np.identity(len(self.K)) * 1e-9)
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self) + (np.identity(len(x_pred)) * 1e-9)
        joint_covariance = np.asarray(np.block([[K, k], \
                                                [k.T, kk]]))

        prod_covariance = np.asarray(np.block([[K, k * 0.], \
                                               [k.T * 0., kk * np.identity(len(kk))]]))

        return {"x": x_pred,
                "total correlation": self.kl_div(np.zeros((len(joint_covariance))), np.zeros((len(joint_covariance))),
                                                 joint_covariance, prod_covariance)}

    ###########################################################################
    def gp_relative_information_entropy(self, x_pred, x_out=None):
        """
        Function to compute KL divergence and therefore relative information entropy
        of the prior distribution over-predicted function values and the posterior distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            Relative information entropy of prediction points, as a collective.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self) + (np.identity(len(x_pred)) * 1e-9)
        post_cov = self.posterior_covariance(x_pred, x_out=None)["S"] + (np.identity(len(x_pred)) * 1e-9)
        return {"x": x_pred,
                "RIE": self.kl_div(np.zeros((len(x_pred))), np.zeros((len(x_pred))), kk, post_cov)}

    ###########################################################################
    def gp_relative_information_entropy_set(self, x_pred, x_out=None):
        """
        Function to compute KL divergence and therefore relative information entropy
        if the prior distribution over-predicted function values and the posterior distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            Relative information entropy of prediction points, but not as a collective.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")
        RIE = np.zeros((len(x_pred)))
        for i in range(len(x_pred)):
            RIE[i] = self.gp_relative_information_entropy(x_pred[i].reshape(1, len(x_pred[i])), x_out=None)["RIE"]

        return {"x": x_pred,
                "RIE": RIE}

    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out=None):
        """
        Function to compute probability of a probabilistic quantity of interest,
        given the GP posterior at a given point.

        Parameters
        ----------
        x_pred: 1d or 2d numpy array of points, note, these are elements of the
                index set which results from a cartesian product of input and output space
        comp_mean: a vector of mean values, same length as x_pred
        comp_cov: covarianve matrix, in R^{len(x_pred)xlen(x_pred)}
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            The probability of a probabilistic quantity of interest, given the GP posterior at a given point.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        res = self.posterior_mean(x_pred, x_out=None)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_pred, x_out=None)["S"]
        gp_cov_inv = self._inv(gp_cov)
        comp_cov_inv = self._inv(comp_cov)
        cov = self._inv(gp_cov_inv + comp_cov_inv)
        mu = cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        logdet1 = self._logdet(cov)
        logdet2 = self._logdet(gp_cov)
        logdet3 = self._logdet(comp_cov)
        dim = len(mu)
        C = 0.5 * (((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T \
                    @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean)) \
                   - (gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0 * np.pi) ** (dim / 2.0)) + (0.5 * (logdet2 + logdet3)))
        return {"mu": mu,
                "covariance": cov,
                "probability":
                    np.exp(ln_p)
                }

    def posterior_probability_grad(self, x_pred, comp_mean, comp_cov, direction, x_out=None):
        """
        Function to compute the gradient of the probability of a probabilistic quantity of interest,
        given the GP posterior at a given point.

        Parameters
        ----------
        x_pred: 1d or 2d numpy array of points, note, these are elements of the
                index set which results from a cartesian product of input and output space
        comp_mean: a vector of mean values, same length as x_pred
        comp_cov: covarianve matrix, in R^{len(x_pred)xlen(x_pred)}
        direction : int
            The direction to compute the gradient in.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            The gradient of the probability of a probabilistic quantity of interest,
            given the GP posterior at a given point.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        x1[:, direction] = x1[:, direction] + 1e-6
        x2[:, direction] = x2[:, direction] - 1e-6

        probability_grad = (self.posterior_probability(x1, comp_mean, comp_cov, x_out=None)["probability"] - \
                            self.posterior_probability(x2, comp_mean, comp_cov, x_out=None)["probability"]) / 2e-6
        return {"probability grad": probability_grad}

    ###########################################################################
    def _int_gauss(self, S):
        return ((2.0 * np.pi) ** (len(S) / 2.0)) * np.sqrt(np.linalg.det(S))

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Kernels#####################################################
    ##################################################################################
    ##################################################################################
    def squared_exponential_kernel(self, distance, length):
        """
        Function for the squared exponential kernel.
        kernel = np.exp(-(distance ** 2) / (2.0 * (length ** 2)))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """
        kernel = np.exp(-(distance ** 2) / (2.0 * (length ** 2)))
        return kernel

    def squared_exponential_kernel_robust(self, distance, phi):
        """
        Function for the squared exponential kernel (robust version)
        kernel = np.exp(-(distance ** 2) * (phi ** 2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        phi : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """
        kernel = np.exp(-(distance ** 2) * (phi ** 2))
        return kernel

    def exponential_kernel(self, distance, length):
        """
        Function for the exponential kernel.
        kernel = np.exp(-(distance) / (length))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        kernel = np.exp(-(distance) / (length))
        return kernel

    def exponential_kernel_robust(self, distance, phi):
        """
        Function for the exponential kernel (robust version)
        kernel = np.exp(-(distance) * (phi**2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        phi : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        kernel = np.exp(-(distance) * (phi ** 2))
        return kernel

    def matern_kernel_diff1(self, distance, length):
        """
        Function for the Matern kernel, order of differentiability = 1.
        kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
            -(np.sqrt(3.0) * distance) / length

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
            -(np.sqrt(3.0) * distance) / length
        )
        return kernel

    def matern_kernel_diff1_robust(self, distance, phi):
        """
        Function for the Matern kernel, order of differentiability = 1, robust version.
        kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi**2))) * np.exp(
            -(np.sqrt(3.0) * distance) * (phi**2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        phi : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """
        ##1/l --> phi**2
        kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi ** 2))) * np.exp(
            -(np.sqrt(3.0) * distance) * (phi ** 2))
        return kernel

    def matern_kernel_diff2(self, distance, length):
        """
        Function for the Matern kernel, order of differentiability = 2.
        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) / (length))
            + ((5.0 * distance ** 2) / (3.0 * length ** 2))
        ) * np.exp(-(np.sqrt(5.0) * distance) / length)

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        kernel = (
                     1.0
                     + ((np.sqrt(5.0) * distance) / (length))
                     + ((5.0 * distance ** 2) / (3.0 * length ** 2))
                 ) * np.exp(-(np.sqrt(5.0) * distance) / length)
        return kernel

    def matern_kernel_diff2_robust(self, distance, phi):
        """
        Function for the Matern kernel, order of differentiability = 2, robust version.
        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) * (phi**2))
            + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
        ) * np.exp(-(np.sqrt(5.0) * distance) * (phi**2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        phi : scalar
            The length scale hyperparameters.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        kernel = (
                     1.0
                     + ((np.sqrt(5.0) * distance) * (phi ** 2))
                     + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
                 ) * np.exp(-(np.sqrt(5.0) * distance) * (phi ** 2))
        return kernel

    def sparse_kernel(self, distance, radius):
        """
        Function for a compactly supported kernel.

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        radius : scalar
            Radius of support.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d > radius] = radius
        kernel = (np.sqrt(2.0) / (3.0 * np.sqrt(np.pi))) * \
                 ((3.0 * (d / radius) ** 2 * np.log((d / radius) / (1 + np.sqrt(1.0 - (d / radius) ** 2)))) + \
                  ((2.0 * (d / radius) ** 2 + 1.0) * np.sqrt(1.0 - (d / radius) ** 2)))
        return kernel

    def periodic_kernel(self, distance, length, p):
        """
        Function for a periodic kernel.
        kernel = np.exp(-(2.0/length**2)*(np.sin(np.pi*distance/p)**2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            Length scale.
        p : scalar
            Period.

        Return
        ------
        Kernel output : same as distance parameter.
        """

        kernel = np.exp(-(2.0 / length ** 2) * (np.sin(np.pi * distance / p) ** 2))
        return kernel

    def linear_kernel(self, x1, x2, hp1, hp2, hp3):
        """
        Function for a linear kernel.
        kernel = hp1 + (hp2*(x1-hp3)*(x2-hp3))

        Parameters
        ----------
        x1 : float
            Point 1.
        x2 : float
            Point 2.
        hp1 : float
            Hyperparameter.
        hp2 : float
            Hyperparameter.
        hp3 : float
            Hyperparameter.

        Return
        ------
        Kernel output : same as distance parameter.
        """
        kernel = hp1 + (hp2 * (x1 - hp3) * (x2 - hp3))
        return kernel

    def dot_product_kernel(self, x1, x2, hp, matrix):
        """
        Function for a dot-product kernel.
        kernel = hp + x1.T @ matrix @ x2

        Parameters
        ----------
        x1 : np.ndarray
            Point 1.
        x2 : np.ndarray
            Point 2.
        hp : float
            Offset hyperparameter.
        matrix : np.ndarray
            PSD matrix defining the inner product.

        Return
        ------
        Kernel output : same as distance parameter.
        """
        kernel = hp + x1.T @ matrix @ x2
        return kernel

    def polynomial_kernel(self, x1, x2, p):
        """
        Function for a polynomial kernel.
        kernel = (1.0+x1.T @ x2)**p

        Parameters
        ----------
        x1 : np.ndarray
            Point 1.
        x2 : np.ndarray
            Point 2.
        p : float
            Power hyperparameter.

        Return
        ------
        Kernel output : same as distance parameter.
        """
        kernel = (1.0 + x1.T @ x2) ** p
        return p

    def default_kernel(self, x1, x2, hyperparameters, obj):
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
        obj : object instance
            GP object instance.

        Return
        ------
        Covariance matrix : np.ndarray
        """
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])):
            distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        distance_matrix = np.sqrt(distance_matrix)
        return hps[0] * obj.matern_kernel_diff1(distance_matrix, 1)

    def wendland_anisotropic(self, x1, x2, hyperparameters, obj):
        """
        Function for the Wendland kernel, default kernel if `'sparse_mode'` is enabled.
        The Wendland kernel is compactly supported, leading to sparse covariance matrices.

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D).
        x2 : np.ndarray
            Numpy array of shape (V x D).
        hyperparameters : np.ndarray
            Array of hyperparameters. For this kernel we need D + 1 hyperparameters.
        obj : object instance
            GP object instance.

        Return
        ------
        Covariance matrix : np.ndarray
        """
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])): distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        d = np.sqrt(distance_matrix)
        d[d > 1.] = 1.
        kernel = (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
        return hps[0] * kernel

    def non_stat_kernel(self, x1, x2, x0, w, l):
        """
        Non-stationary kernel.
        kernel = g(x1) g(x2)

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D).
        x2 : np.ndarray
            Numpy array of shape (V x D).
        x0 : np.ndarray
            Numpy array of the basis function locations.
        w : np.ndarray
            1d np.ndarray of weights. len(w) = len(x0).
        l : float
            Width measure of the basis functions.

        Return
        ------
        Covariance matrix : np.ndarray
        """
        non_stat = np.outer(self._g(x1, x0, w, l), self._g(x2, x0, w, l))
        return non_stat

    def non_stat_kernel_gradient(self, x1, x2, x0, w, l):
        """
        Non-stationary kernel gradient.
        kernel = g(x1) g(x2)

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D).
        x2 : np.ndarray
            Numpy array of shape (V x D).
        x0 : np.ndarray
            Numpy array of the basis function locations.
        w : np.ndarray
            1d np.ndarray of weights. len(w) = len(x0).
        l : float
            Width measure of the basis functions.

        Return
        ------
        Covariance matrix : np.ndarray
        """
        dkdw = np.einsum('ij,k->ijk', self._dgdw(x1, x0, w, l), self._g(x2, x0, w, l)) + np.einsum('ij,k->ikj',
                                                                                                   self._dgdw(x2, x0, w,
                                                                                                              l),
                                                                                                   self._g(x1, x0, w,
                                                                                                           l))
        dkdl = np.outer(self._dgdl(x1, x0, w, l), self._g(x2, x0, w, l)) + np.outer(self._dgdl(x2, x0, w, l),
                                                                                    self._g(x1, x0, w, l)).T
        res = np.empty((len(w) + 1, len(x1), len(x2)))
        res[0:len(w)] = dkdw
        res[-1] = dkdl
        return res

    def _get_distance_matrix(self, x1, x2):
        d = np.zeros((len(x1), len(x2)))
        for i in range(x1.shape[1]): d += (x1[:, i].reshape(-1, 1) - x2[:, i]) ** 2
        return np.sqrt(d)

    def _g(self, x, x0, w, l):
        d = self._get_distance_matrix(x, x0)
        e = np.exp(-(d ** 2) / l)
        return np.sum(w * e, axis=1)

    def _dgdw(self, x, x0, w, l):
        d = self._get_distance_matrix(x, x0)
        e = np.exp(-(d ** 2) / l).T
        return e

    def _dgdl(self, x, x0, w, l):
        d = self._get_distance_matrix(x, x0)
        e = np.exp(-(d ** 2) / l)
        return np.sum(w * e * (d ** 2 / l ** 2), axis=1)

    ##################################################################################
    ##################################################################################
    ###################Kernel and Mean Function Derivatives###########################
    ##################################################################################
    def _d_gp_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-8
        new_points[:, direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters, self)
        b = self.kernel(points1, points2, hyperparameters, self)
        derivative = (a - b) / epsilon
        return derivative

    def _dkernel_dh(self, points1, points2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-8
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyperparameters1, self)
        b = self.kernel(points1, points2, new_hyperparameters2, self)
        derivative = (a - b) / (2.0 * epsilon)
        return derivative

    def _gp_kernel_gradient(self, points1, points2, hyperparameters, obj):
        gradient = np.empty((len(hyperparameters), len(points1), len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def _gp_kernel_derivative(self, points1, points2, direction, hyperparameters, obj):
        # gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        derivative = self._dkernel_dh(points1, points2, direction, hyperparameters)
        return derivative

    def _default_dm_dh(self, x, hps, gp_obj):
        gr = np.zeros((len(hps), len(x)))
        return gr

    def _default_dnoise_dh(self, x, hps, gp_obj):
        gr = np.zeros((len(hps), len(x), len(x)))
        return gr

    def _default_dnoise_dh_econ(self, x, i, hps, gp_obj):
        gr = np.zeros((len(x), len(x)))
        return gr

    def _finitediff_dm_dh(self, x, hps, gp_obj):
        gr = np.empty((len(hps), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.mean_function(x, temp_hps1, self)
            b = self.mean_function(x, temp_hps2, self)
            gr[i] = (a - b) / 2e-6
        return gr

    ##########################
    def _finitediff_dnoise_dh(self, x, hps, gp_obj):
        gr = np.zeros((len(hps), len(x), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.noise_function(x, temp_hps1, self)
            b = self.noise_function(x, temp_hps2, self)
            gr[i] = (a - b) / 2e-6
        return gr

    ##########################
    def _finitediff_dnoise_dh_econ(self, x, i, hps, gp_obj):
        temp_hps1 = np.array(hps)
        temp_hps1[i] = temp_hps1[i] + 1e-6
        temp_hps2 = np.array(hps)
        temp_hps2[i] = temp_hps2[i] - 1e-6
        a = self.noise_function(x, temp_hps1, self)
        b = self.noise_function(x, temp_hps2, self)
        gr = (a - b) / 2e-6
        return gr

    def _normalize(self, data):
        min_d = np.min(data)
        max_d = np.max(data)
        data = (data - min_d) / (max_d - min_d)
        return data, min_d, max_d

    def _normalize_y_data(self, y_data):
        return self._normalize(self.y_data)

    def _normalize_x_data(self, x_data):
        n_x = np.empty(x_data.shape)
        x_min = np.empty((len(x_data)))
        x_max = np.empty((len(x_data)))
        for i in range(len(self.x_data[0])):
            n_x[:, i], x_min[i], x_max[i] = self._normalize(x_data[:, i])
        return n_x, x_min, x_max

    def _normalize_x_pred(self, x_pred, x_min, x_max):
        new_x_pred = np.empty(x_pred.shape)
        for i in range(len(x_pred[0])):
            new_x_pred[:, i] = (x_pred[:, i] - x_min[i]) / (x_max[i] - x_min[i])
        return new_x_pred

    def _cartesian_product_euclid(self, x, y):
        """
        Input x,y have to be 2d numpy arrays
        The return is the cartesian product of the two sets
        """
        res = []
        for i in range(len(y)):
            for j in range(len(x)):
                res.append(np.append(x[j], y[i]))
        return np.array(res)

    def _cartesian_product_noneuclid(self, x, y):
        """
        Input x,y have to be 2d numpy arrays
        The return is the cartesian product of the two sets
        """
        res = []
        for i in range(len(y)):
            for j in range(len(x)):
                res.append([x[j], y[i]])
        return res

    ####################################################################################
    ####################################################################################
    #######################VALIDATION###################################################
    ####################################################################################
    def _crps_s(self, x, mu, sigma):
        return sigma * ((1. / np.sqrt(np.pi))
                        - 2. * norm.pdf((x - mu) / sigma)
                        - (((x - mu) / sigma) * (2. * norm.cdf((x - mu) / sigma) - 1.)))

    def crps(self, x_test, y_test):
        """
        This function calculates the continuous rank probability score.
        Note that in the multi-task setting the user should perform their
        input point transformation beforehand.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape (V x 1). These are the y data to compare against.

        Return
        ------
        CRPS : float
        """

        mean = self.posterior_mean(x_test)["f(x)"]
        sigma = self.posterior_covariance(x_test)["v(x)"]
        r = np.empty((len(x_test)))
        for i in range(len(x_test)): r[i] = self._crps_s(y_test[i], mean[i], sigma[i])
        return np.sum(r) / len(r)

    def rmse(self, x_test, y_test):
        """
        This function calculates the root mean squared error.
        Note that in the multi-task setting the user should perform their
        input point transformation beforehand.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape (V x 1). These are the y data to compare against

        Return
        ------
        RMSE : float
        """

        v1 = y_test
        v2 = self.posterior_mean(x_test)["f(x)"]
        return np.sqrt(np.sum((v1 - v2) ** 2) / len(v1))

    def make_2d_x_pred(self, bx, by, resx=100, resy=100):  # pragma: no cover
        """
        This is a purely convenience-driven function calculating prediction points
        on a grid.

        Parameters
        ----------
        bx : np.ndarray
            A numpy array of shape (2) defining lower and upper bounds in x direction.
        by : np.ndarray
            A numpy array of shape (2) defining lower and upper bounds in y direction.
        resx : int, optional
            Resolution in x direction. Default = 100.
        resy : int, optional
            Resolution in y direction. Default = 100.

        Return
        ------
        prediction points : np.ndarray
        """

        x = np.linspace(bx[0], bx[1], resx)
        y = np.linspace(by[0], by[1], resy)
        from itertools import product
        x_pred = np.array(list(product(x, y)))
        return x_pred

    def make_1d_x_pred(self, b, res=100):  # pragma: no cover
        """
        This is a purely convenience-driven function calculating prediction points
        on a 1d grid.

        Parameters
        ----------
        b : np.ndarray
            A numpy array of shape (2) defineing lower and upper bounds
        res : int, optional
            Resolution. Default = 100

        Return
        ------
        prediction points : np.ndarray
        """

        x_pred = np.linspace(b[0], b[1], res).reshape(res, -1)
        return x_pred

    def _in_bounds(self, v, bounds):
        if any(v < bounds[:, 0]) or any(v > bounds[:, 1]): return False
        return True


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
def wendland_anisotropic_gp2Scale_cpu(x1, x2, hps, obj):
    distance_matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0])): distance_matrix += (np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
    d = np.sqrt(distance_matrix)
    d[d > 1.] = 1.
    kernel = hps[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return kernel


def wendland_anisotropic_gp2Scale_gpu(x1, x2, hps, obj):  # pragma: no cover
    import torch
    cuda_device = torch.device("cuda:0")
    x1_dev = torch.from_numpy(x1).to(cuda_device, dtype=torch.float32)
    x2_dev = torch.from_numpy(x2).to(cuda_device, dtype=torch.float32)
    hps_dev = torch.from_numpy(hps).to(cuda_device, dtype=torch.float32)
    d = _get_distance_matrix_gpu(x1_dev, x2_dev, cuda_device, hps_dev)
    d[d > 1.] = 1.
    kernel = hps[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return kernel.cpu().numpy()


def _get_distance_matrix_gpu(x1, x2, device, hps):  # pragma: no cover
    import torch
    d = torch.zeros((len(x1), len(x2))).to(device, dtype=torch.float32)
    for i in range(x1.shape[1]):
        d += ((x1[:, i].reshape(-1, 1) - x2[:, i]) / hps[1 + i]) ** 2
    return torch.sqrt(d)
