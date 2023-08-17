#!/usr/bin/env python
import inspect
import time
import itertools
from functools import partial
import math
import warnings
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import dask.distributed as distributed
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from loguru import logger
from .mcmc import mcmc
from hgdl.hgdl import HGDL



#TODO:
#   check ALL docs
#   change and run all test scripts


class GP():
    """
    This class provides all the tools for a single-task Gaussian Process (GP).
    Use fvGP for multi task GPs. However, the fvGP class inherits all methods from this class.
    This class allows for full HPC support for training via the HGDL package.
    
    V ... number of input points
    D ... input space dimensionality
    N ... arbitrary integers (N1, N2,...)

    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space (D).
    x_data : np.ndarray
        The input point positions. Shape (V x D), where D is the `input_space_dim`.
    y_data : np.ndarray
        The values of the data points. Shape (V,1) or (V).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        The class provides methods to train hyperparameters.
        The default is an array of ones, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD).
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the data
        `y_data` in form of a point-wise variance. Shape (len(y_data), 1) or (len(y_data)).
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data) / 100.0`. If
        noise covariances are required, also make use of the gp_noise_function.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
        For "gpu", pytoch has to be installed manually.
    gp_kernel_function : Callable, optional
        A symmetric positive semi-definite covariance function (a kernel) 
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input x1 is a N1 x D array of positions, x2 is a N2 x D
        array of positions, the hyperparameters argument 
        is a 1d array of length D+1 for the default kernel and of a different
        user-defined length for other kernels
        obj is an `fvgp.gp.GP` instance. The default is a stationary anisotropic kernel
        (`fvgp.gp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a covariance matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``gp_kernel_function'' with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input x1 (a N1 x D array of positions),
        x2 (a N2 x D array of positions), 
        hyperparameters (a 1d array of length D+1 for the default kernel), and a
        `fvgp.gp.GP` instance. The default is a finite difference calculation.
        If 'ram_economy' is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and a
        `fvgp.gp.GP` instance, and the output
        is a numpy array of shape (V x U).
        If 'ram economy' is False,the function's input is x1, x2, hyperparameters, and a
        `fvgp.gp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See 'ram_economy'.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.gp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.gp.GP._default_mean_function` is used.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_mean_function'' at a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.gp.GP` instance. The return value is a 2d array of shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparametes, or a finite-difference approximation
        is used if ``gp_mean_function'' is provided.
    gp_noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_noise_function'' at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.gp.GP` instance. The return value is a 3-D array of shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not dpeend on hyperparametes. If ``gp_noise_function'' is provided but no gradient function,
        a finite-difference approximation will be used.
    normalize_y : bool, optional
        If True, the data values ``y_data'' will be normalized to max(y_data) = 1, min(y_data) = 0. The default is False.
    store_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster.
        For larger problems (>2000 data points), the use of inversion should be avoided due to computational instability and costs. The default is
        True. Note, the training will always use Cholesky or LU decomposition instead of the inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the marginal log_likelihood is/are used for the training.
        If True, components of the derivative of the marginal log-likelihood are calculated subsequently, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel with respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for ram_economy=True it should be of the form f(x1, x2, direction, hyperparameters)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ram_economy=False, the function should be of the form f(points1, points2, hyperparameters) and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters. CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.



    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation (co)variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP
    KVinv : np.ndarray
        If enabled, the inverse of the prior covariance + nosie matrix V
        inv(K+V)
    KVlogdet : float
        logdet(K_V)
    """
    def __init__(
        self,
        input_space_dim,
        x_data,
        y_data,
        init_hyperparameters = None,
        noise_variances = None,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_kernel_function_grad = None,
        gp_noise_function = None,
        gp_noise_function_grad = None,
        gp_mean_function = None,
        gp_mean_function_grad = None,
        sparse_mode = False,
        normalize_y = False,
        store_inv = True,
        ram_economy = False,
        args = None
        ):
        ########################################
        ###assign and check some attributes#####
        ########################################
        if np.ndim(x_data) == 1: x_data = x_data.reshape(-1,1)
        if input_space_dim != len(x_data[0]): raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(y_data) == 2: y_data = y_data[:,0]
        if compute_device == 'gpu':
            try: import torch
            except: raise Exception("You have specified the 'gpu' as your compute device. You need to install pytorch manually for this to work.")


        self.normalize_y = normalize_y
        self.input_space_dim = input_space_dim
        self.x_data = x_data
        self.point_number = len(self.x_data)
        self.y_data = y_data
        self.compute_device = compute_device
        self.ram_economy = ram_economy
        self.args = args
        self.sparse_mode = sparse_mode
        self.store_inv = store_inv
        if self.sparse_mode and self.store_inv:
            warnings.warn("sparse_mode and store_inv enabled but they should not be used together. I'll set store_inv = False.", stacklevel=2)
            self.store_inv = False
        if self.sparse_mode and not callable(gp_kernel_function):
                warnings.warn("You have chosen to activate sparse mode. Great! \n But you have not supplied a kernel that is compactly supported. \n I will use an anisotropic Wendland kernel for now.", stacklevel=2)
                gp_kernel_function = self.wendland_anisotropic

        self.KVinv = None
        self.mcmc_info = None
        ###########################################
        ###assign kernel, mean and noise functions#
        ###########################################
        if callable(gp_noise_function): self.noise_function = gp_noise_function
        elif noise_variances is not None: self.noise_function = None
        else:
            warnings.warn("No noise function or measurement noise provided. Noise variances will be set to 1% of mean(y_data).", stacklevel=2)
            self.noise_function = self._default_noise_function
        if noise_variances is not None and callable(gp_noise_function): raise Exception("Noise function and measurement noise provided. Only one should be given.")
        if callable(gp_noise_function_grad): self.noise_function_grad = gp_noise_function_grad
        elif callable(gp_noise_function):
            if self.ram_economy is True: self.noise_function_grad = self._finitediff_dnoise_dh_econ
            else: self.noise_function_grad = self._finitediff_dnoise_dh
        else:
            if self.ram_economy is True: self.noise_function_grad = self._default_dnoise_dh_econ
            else: self.noise_function_grad = self._default_dnoise_dh

        if callable(gp_kernel_function): self.kernel = gp_kernel_function
        elif gp_kernel_function is None: self.kernel = self.default_kernel
        else: raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self.d_gp_kernel_dx
        if callable(gp_kernel_function_grad): self.dk_dh = gp_kernel_function_grad
        else:
            if self.ram_economy is True: self.dk_dh = self.gp_kernel_derivative
            else: self.dk_dh = self.gp_kernel_gradient

        if  callable(gp_mean_function): self.mean_function = gp_mean_function
        else: self.mean_function = self._default_mean_function
        if callable(gp_mean_function_grad): self.dm_dh = gp_mean_function_grad
        elif callable(gp_mean_function): self.dm_dh = self._finitediff_dm_dh
        else: self.dm_dh = self._default_dm_dh
        ##########################################
        #######prepare noise covariances##########
        ##########################################
        if noise_variances is None:
            ##noise covariances are always a square matrix
            self.noise_covariances = self.noise_function(self.x_data, init_hyperparameters,self)
            if np.ndim(self.noise_covariances) == 1: raise Exception("Your noise function did not return a square matrix, it should though, the noise can be correlated.")
            elif self.noise_covariances.shape[0] != self.noise_covariances.shape[1]: raise Exception("Your noise function return is not a square matrix")
        elif np.ndim(noise_variances) == 2:
            if any(noise_variances <= 0.0): raise Exception("Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.noise_covariances = np.diag(noise_variances[:,0])
        elif np.ndim(noise_variances) == 1:
            if any(noise_variances <= 0.0): raise Exception("Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.noise_covariances = np.diag(noise_variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        #########################################
        ###########normalization#################
        #########################################
        if self.normalize_y:
            self.y_data, self.y_min, self.y_max = self._normalize_y_data(self.y_data)
            self.noise_covariances = (1./(self.y_max-self.y_min)**2) * self.noise_covariances
        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        if init_hyperparameters is None: init_hyperparameters  = np.ones((nput_space_dim + 1))
        self.hyperparameters = np.array(init_hyperparameters)
        ##########################################
        #compute the prior########################
        ##########################################
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.noise_covariances = self._compute_GPpriorV(self.x_data, self.y_data, self.hyperparameters, calc_inv = self.store_inv)

    def update_gp_data(
        self,
        x_data,
        y_data,
        noise_variances = None,
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
            An numpy array defining the uncertainties in the data `y_data` in form of a point-wise variance. Shape (len(y_data), 1) or (len(y_data)). 
            Note: if no variances are provided here, the noiase_covariance callable will be used; if the callable is not provided the noise variances
            will be set to `abs(np.mean(y_data) / 100.0`.
        """
        if np.ndim(x_data) == 1: x_data = x_data.reshape(-1,1)
        if self.input_space_dim != len(x_data[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(y_data) == 2: y_data = y_data[:,0]

        self.x_data = x_data
        self.point_number = len(self.x_data)
        self.y_data = y_data

        ##########################################
        #######prepare variances##################
        ##########################################
        if noise_variances is None:
            ##noise covariances are always a square matrix
            self.noise_covariances = self.noise_function(self.x_data, self.hyperparameters,self)
        elif np.ndim(noise_variances) == 2:
            if any(noise_variances <= 0.0): raise Exception("Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.noise_covariances = np.diag(noise_variances[:,0])
        elif np.ndim(noise_variances) == 1:
            if any(noise_variances <= 0.0): raise Exception("Negative or zero measurement variances communicated to fvgp or derived from the data.")
            self.noise_covariances = np.diag(noise_variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        #########################################
        ###########normalization#################
        #########################################
        if self.normalize_y:
            self.y_data, self.y_min, self.y_max = self._normalize_y_data(self.y_data)
            self.noise_covariances = (1./(self.y_max-self.y_min)**2) * self.noise_covariances
        ######################################
        #####transform to index set###########
        ######################################
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.noise_covariances = self._compute_GPpriorV(self.x_data, self.y_data, self.hyperparameters, calc_inv = self.store_inv)

    ###################################################################################
    ###################################################################################
    ###################################################################################
    #################TRAINING##########################################################
    ###################################################################################
    def train(self,
        hyperparameter_bounds = None,
        init_hyperparameters = None,
        method = "global",
        pop_size = 20,
        tolerance = 0.0001,
        max_iter = 120,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        constraints = (),
        dask_client = None):

        """
        This function finds the maximum of the marginal log_likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be be 'hgdl' and
        providing a dask client. The GP prior will automatically be updated with the new hyperparameters.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization. The default is an array of bounds for the default kernel D = input_space_dim + 1
            with all bounds defined practically as [0.00001, inf]. This choice is only recommended in very basic scenarios.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is reusing the initial hyperparameters given at initialization
        method : str or Callable, optional
            The method used to train the hyperparameters. The options are `'global'`, `'local'`, `'hgdl'`, `'mcmc'`, and a callable.
            The callable gets an gp.GP instance and has to return a 1d np.array of hyperparameters.
            The default is `'global'` (scipy's differential evolution). 
            If method = "mcmc",
            the attribute gp.GP.mcmc_info is updated and contains convergence and disttibution information.
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
            Equality and inequality constraints for the optimization. If the optimizer is ``hgdl'' see ``hgdl.readthedocs.io''.
            If the optimizer is a scipy optimizer, see scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.
        """
        ############################################
        if init_hyperparameters is None:
            init_hyperparameters = np.array(self.hyperparameters)
        if hyperparameter_bounds is None:
            hyperpameter_bounds = np.zeros((len(init_hyperparameters)))
            hyperparameter_bounds[0] = np.array([0.00001,1e8])
            hyperparameter_bounds[1:] = np.array([0.00001,1e8])

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
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.noise_covariances = self._compute_GPpriorV(self.x_data, self.y_data, self.hyperparameters, calc_inv = self.store_inv)
    ##################################################################################
    def train_async(self,
        hyperparameter_bounds = None,
        init_hyperparameters = None,
        max_iter = 10000,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        constraints = (),
        dask_client = None):
        """
        This function asynchronously finds the maximum of the log marginal likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function just submits the training and returns
        an object which can be given to `fvgp.gp.update_hyperparameters`, which will automatically update the GP prior with the new hyperparameters.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization. The default is an array of bounds for the default kernel D = input_space_dim + 1
            with all bounds defined practically as [0.00001, inf]. This choice is only recommended in very basic scenarios.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is reusing the initial hyperparameters given at initialization
        max_iter : int, optional
            Maximum number of epochs for HGDL. Default = 10000.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of hgdl.NonLinearConstraint instances, optional
            Equality and inequality constraints for the optimization. See ``hgdl.readthedocs.io''
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.

        Return
        ------
        Optimization object that can be given to `fvgp.gp.update_hyperparameters` to update the prior GP : object instance
        """
        if dask_client is None: dask_client = distributed.Client()
        if init_hyperparameters is None:
            init_hyperparameters = np.array(self.hyperparameters)
        if hyperparameter_bounds is None:
            hyperpameter_bounds = np.zeros((len(init_hyperparameters)))
            hyperparameter_bounds[0] = np.array([0.00001,1e8])
            hyperparameter_bounds[1:] = np.array([0.00001,1e8])

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
    def stop_training(self,opt_obj):
        """
        This function stops the training if HGDL is used. It leaves the dask client alive.

        Parameters
        ----------
        opt_obj : object
            An object returned form the `fvgp.gp.GP.train_async` function.
        """
        try:
            opt_obj.cancel_tasks()
            logger.debug("fvGP successfully cancelled the current training.")
        except:
            warnings.warn("No asynchronous training to be cancelled in fvGP, no training is running.")
    ###################################################################################
    def kill_training(self,opt_obj):
        """
        This function stops the training if HGDL is used, and kills the dask client.

        Parameters
        ----------
        opt_obj : object
            An object returned form the `fvgp.gp.GP.train_async` function.
        """

        try:
            opt_obj.kill_client()
            logger.debug("fvGP successfully killed the training.")
        except:
            warnings.warn("No asynchronous training to be killed, no training is running.")

    ##################################################################################
    def update_hyperparameters(self, opt_obj):
        """
        This function asynchronously finds the maximum of the marginal log_likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function just submits the training and returns
        an object which can be given to `fvgp.gp.update_hyperparameters`, which will automatically update the GP prior with the new hyperparameters.

        Parameters
        ----------
        object : HGDL class instance
            HGDL class instance returned by `fvgp.gp.train_async`

        Return
        ------
        The current hyperparameters : np.ndarray
        """
        success = False
        try:
            res = opt_obj.get_latest()[0]["x"]
            success = True
        except:
            logger.debug("      The optimizer object could not be queried")
            logger.debug("      That probably means you are not optimizing the hyperparameters asynchronously")
        if success is True:
            try:
                l_n = self.neg_log_likelihood(res)
                l_o = self.neg_log_likelihood(self.hyperparameters)
                if l_n - l_o < 0.000001:
                    self.hyperparameters = res
                    self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.noise_covariances = self._compute_GPpriorV(self.x_data, self.y_data, self.hyperparameters, calc_inv = self.store_inv)
                    logger.debug("    fvGP async hyperparameter update successful")
                    logger.debug("    Latest hyperparameters: {}", self.hyperparameters)
                else:
                    logger.debug("    The update was attempted but the new hyperparameters led to a lower likelihood, so I kept the old ones")
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
        hps : np.array
            A 1-d numpy array of hyperparameters.
        """
        self.hyperparameters = np.array(hps)
        self.K, self.KV, self.KVinvY, self.KVlogdet, self.factorization_obj, self.KVinv, self.prior_mean_vec, self.noise_covariances = self._compute_GPpriorV(self.x_data, self.y_data, self.hyperparameters, calc_inv = self.store_inv)
    ##################################################################################
    def get_hyperparameters(self):
        """
        Function to get the current hyperparameters.

        Parameters: None
        -----------

        Return:
        -------
        hyperparameters : np.array
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
        dict
        """

        return {"prior corvariance (K)":self.K, "log(|KV|)":self.KVlogdet, "inv(KV)":self.KVinv, "prior mean":self.prior_mean_vec}
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
        opt_obj = HGDL(self.neg_log_likelihood,
                    self.neg_log_likelihood_gradient,
                    hp_bounds,
                    hess = self.neg_log_likelihood_hessian,
                    local_optimizer = local_optimizer,
                    global_optimizer = global_optimizer,
                    num_epochs = max_iter,
                    constraints = constraints)

        logger.debug("HGDL successfully initialized. Calling optimize()")
        opt_obj.optimize(dask_client = dask_client, x0 = np.array(starting_hps).reshape(1,-1))
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
            dask_client = None):

        start_log_likelihood = self.log_likelihood(starting_hps)

        logger.debug(
            "fvGP hyperparameter tuning in progress. Old hyperparameters: ",
            starting_hps, " with old log likelihood: ", start_log_likelihood)
        logger.debug("method: ", method)

        ############################
        ####global optimization:##
        ############################
        if method == "global":
            logger.debug("fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hp_bounds)
            res = differential_evolution(
                self.neg_log_likelihood,
                hp_bounds,
                maxiter=max_iter,
                popsize = pop_size,
                tol = tolerance,
                constraints = constraints,
                workers = 1,
            )
            hyperparameters = np.array(res["x"])
            Eval = self.neg_log_likelihood(hyperparameters)
            logger.debug(f"fvGP found hyperparameters {hyperparameters} with likelihood {Eval} via global optimization")
        ############################
        ####local optimization:#####
        ############################
        elif method == "local":
            hyperparameters = np.array(starting_hps)
            logger.debug("fvGP is performing a local update of the hyper parameters.")
            logger.debug("starting hyperparameters: {}", hyperparameters)
            logger.debug("Attempting a BFGS optimization.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hp_bounds)
            OptimumEvaluation = minimize(
                self.neg_log_likelihood,
                hyperparameters,
                method= local_optimizer,
                jac=self.neg_log_likelihood_gradient,
                hess = self.neg_log_likelihood_hessian,
                bounds = hp_bounds,
                tol = tolerance,
                callback = None,
                constraints = constraints,
                options = {"maxiter": max_iter})

            if OptimumEvaluation["success"] == True:
                logger.debug(f"fvGP local optimization successfully concluded with result: "
                             f"{OptimumEvaluation['fun']} at {OptimumEvaluation['x']}")
                hyperparameters = OptimumEvaluation["x"]
            else:
                logger.debug("fvGP local optimization not successful.")
        ############################
        ####hybrid optimization:####
        ############################
        elif method == "hgdl":
            logger.debug("fvGP submitted HGDL optimization")
            logger.debug('bounds are',hp_bounds)

            opt_obj = HGDL(self.neg_log_likelihood,
                    self.neg_log_likelihood_gradient,
                    hp_bounds,
                    hess = self.neg_log_likelihood_hessian,
                    local_optimizer = local_optimizer,
                    global_optimizer = global_optimizer,
                    num_epochs = max_iter,
                    constraints = constraints)

            opt_obj.optimize(dask_client = dask_client, x0 = np.array(starting_hps).reshape(1,-1))
            hyperparameters = opt_obj.get_final()[0]["x"]
            opt_obj.kill_client()
        elif method == "mcmc":
            logger.debug("MCMC started in fvGP")
            logger.debug('bounds are {}', hp_bounds)
            res = mcmc(self.log_likelihood,hp_bounds, x0 = starting_hps, n_updates = max_iter)
            hyperparameters = np.array(res["distribution mean"])
            self.mcmc_info = res
        elif callable(method):
            hyperparameters = method(self)
        else:
            raise ValueError("No optimization mode specified in fvGP")
        ###################################################
        new_likelihood = self.log_likelihood(hyperparameters)
        if start_log_likelihood > new_likelihood and method != 'mcmc':
            logger.debug(f"New hyperparameters: {hyperparameters} with log likelihood: {self.log_likelihood(hyperparameters)}")
            warning_str = "Old log marginal likelihood: "+ str(start_log_likelihood) + " New log marginal likelihood: "+ str(new_likelihood)
            warnings.warn(f"Old log marginal likelihood:  {start_log_likelihood}")
            warnings.warn(f"New log marginal likelihood:  {new_likelihood}")


            hyperparameters = starting_hps
        return hyperparameters
    ##################################################################################
    def log_likelihood(self,hyperparameters):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (V)
        Return
        ------
            marginal log-likelihood : float
        """
        K, KV,  KVinvY, KVlogdet, FO, KVinv, mean, cov = self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv = False)
        n = len(self.y_data)
        return -(0.5 * ((self.y_data - mean).T @ KVinvY)) - (0.5 * KVlogdet) - (0.5 * n * np.log(2.0*np.pi))
    ##################################################################################
    def neg_log_likelihood(self,hyperparameters):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (V)
        Return
        ------
            negative marginal log-likelihood : float
        """
        return -self.log_likelihood(hyperparameters)
    ##################################################################################
    def neg_log_likelihood_gradient(self, hyperparameters):
        """
        Function that computes the gradient of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (V)
        Return
        ------
        Gradient of the negative marginal log-likelihood : np.ndarray
        """
        logger.debug("log-likelihood gradient is being evaluated...")
        K, KV,  KVinvY, KVlogdet, FO, KVinv, mean, cov = self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv = False)
        b = KVinvY
        y = self.y_data - mean
        if self.ram_economy is False:
            try: dK_dH = self.dk_dh(self.x_data,self.x_data, hyperparameters,self) + self.noise_function_grad(self.x_data, hyperparameters,self)
            except Exception as e: raise Exception("The gradient evaluation dK/dh + dNoise/dh was not successful. \n That normally means the combination of ram_economy and definition of the gradient function is wrong. ",str(e))
            KV = np.array([KV,] * len(hyperparameters))
            a = self._solve(KV,dK_dH)
        bbT = np.outer(b , b.T)
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.dm_dh(self.x_data,hyperparameters,self)


        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if self.ram_economy is False: matr = a[i]
            else:
                try: dK_dH = self.dk_dh(self.x_data,self.x_data, i,hyperparameters, self) + self.noise_function_grad(self.x_data, i,hyperparameters,self)
                except: raise Exception("The gradient evaluation dK/dh + dNoise/dh was not successful. \n That normally means the combination of ram_economy and definition of the gradient function is wrong.")
                matr = np.linalg.solve(KV,dK_dH)
            if dL_dHm[i] == 0.0:
                if self.ram_economy is False: mtrace = np.einsum('ij,ji->', bbT, dK_dH[i])
                else: mtrace = np.einsum('ij,ji->', bbT, dK_dH)
                dL_dH[i] = - 0.5 * (mtrace - np.trace(matr))
            else:
                dL_dH[i] = 0.0

        logger.debug("gradient norm: {}",np.linalg.norm(dL_dH + dL_dHm))
        return dL_dH + dL_dHm

    ##################################################################################
    def neg_log_likelihood_hessian(self, hyperparameters):
        """
        Function that computes the Hessian of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (V)
        Return
        ------
        Hessian of the negative marginal log-likelihood : np.ndarray
        """
        ##implemented as first-order approximation
        len_hyperparameters = len(hyperparameters)
        d2L_dmdh = np.zeros((len_hyperparameters,len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.neg_log_likelihood_gradient(hyperparameters)
        for i in range(len_hyperparameters):
            hps_temp = np.array(hyperparameters)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i,i:] = ((self.neg_log_likelihood_gradient(hps_temp) - grad_at_hps)/epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))

    def test_log_likelihood_gradient(self,hyperparameters):
        thps = np.array(hyperparameters)
        grad = np.empty((len(thps)))
        eps = 1e-6
        for i in range(len(thps)):
            thps_aux = np.array(thps)
            thps_aux[i] = thps_aux[i] + eps
            grad[i] = (self.log_likelihood(thps_aux) - self.log_likelihood(thps))/eps
        analytical = -self.neg_log_likelihood_gradient(thps)
        if np.linalg.norm(grad-analytical) > np.linalg.norm(grad)/100.0:
            print("Gradient possibly wrong")
            print("finite diff appr: ",grad)
            print("analytical      : ",analytical)
        else:
            print("Gradient correct")
            print("finite diff appr: ",grad)
            print("analytical      : ",analytical)
        assert np.linalg.norm(grad-analytical) < np.linalg.norm(grad)/100.0

        return grad, analytical
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
    ##################################################################################
    ##################################################################################
    def _compute_GPpriorV(self, x_data, y_data, hyperparameters, calc_inv = False):
        prior_mean_vec = self.mean_function(x_data,hyperparameters,self)
        if callable(self.noise_function): noise_covariances = self.noise_function(x_data,hyperparameters,self)
        else: noise_covariances = self.noise_covariances
        K, KV = self._compute_covariance(hyperparameters, noise_covariances) ###could be done in batches for RAM
        if self.sparse_mode and self._is_sparse(KV):
            #print("Sparsity detected: ", self._how_sparse_is(K), "hps: ", hyperparameters)
            KV = csc_matrix(KV)
            LU = splu(KV)
            factorization_obj = ("LU", LU)
            KVinvY = LU.solve(y_data - prior_mean_vec)
            upper_diag = abs(LU.U.diagonal())
            KVlogdet = np.sum(np.log(upper_diag))
            KVinv = None
        else:
            #if self.sparse_mode: print("Sparse mode enabled but no sparsity detected", self._how_sparse_is(K), "hps: ", hyperparameters)
            c, l = cho_factor(KV)
            factorization_obj = ("Chol",c,l)
            KVinvY = cho_solve((c, l), y_data - prior_mean_vec)
            upper_diag = abs(c.diagonal())
            KVlogdet = 2.0 * np.sum(np.log(upper_diag))
            if calc_inv: KVinv = self._inv(KV)
            else: KVinv = None

        return K, KV, KVinvY, KVlogdet, factorization_obj, KVinv, prior_mean_vec, noise_covariances

    def _compute_covariance(self, hyperparameters, noise_covariances):
        """computes the covariance matrix from the kernel and add noise"""
        PriorCovariance = self.kernel(self.x_data, self.x_data, hyperparameters, self)
        if PriorCovariance.shape != noise_covariances.shape: raise Exception("Noise covariance and prior covariance not of the same shape.")
        return PriorCovariance, PriorCovariance + noise_covariances
    ##################################################################################
    def _KVsolve(self, b):
        if self.factorization_obj[0] == "LU":
            LU = self.factorization_obj[1]
            return LU.solve(b)
        if self.factorization_obj[0] == "Chol":
            c,l = self.factorization_obj[1], self.factorization_obj[2]
            return cho_solve((c, l), b)

    def _logdet(self, A, factorization_obj = None):
        """
        fvGPs slogdet method based on torch for  or numpy
        for CPU

        Parameters:
        -----------
        A : np.ndarray
        Non-singular matrix.
        """
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
                warnings.warn("I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU. Error: ", e)
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
        """
        fvGPs slogdet method based on torch
        """



        if b.ndim == 1: b = np.expand_dims(b,axis = 1)
        if self.compute_device == "cpu":
            try: x = np.linalg.solve(A,b)
            except: x,res,rank,s = np.linalg.lstsq(A,b,rcond=None)
            return x
        elif self.compute_device == "gpu" or A.ndim < 3:
            try:
                import torch
                A = torch.from_numpy(A).cuda()
                b = torch.from_numpy(b).cuda()
                x = torch.linalg.solve(A, b)
                return x.cpu().numpy()
            except Exception as e:
                warnings.warn("I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU. Error: ", e)
                try: x = np.linalg.solve(A,b)
                except: x,res,rank,s = np.linalg.lstsq(A,b,rcond=None)
                return x
        elif self.compute_device == "multi-gpu":
            try:
                import torch
                n = min(len(A), torch.cuda.device_count())
                split_A = np.array_split(A,n)
                split_b = np.array_split(b,n)
                results = []
                for i, (tmp_A,tmp_b) in enumerate(zip(split_A,split_b)):
                    cur_device = torch.device("cuda:"+str(i))
                    tmp_A = torch.from_numpy(tmp_A).cuda(cur_device)
                    tmp_b = torch.from_numpy(tmp_b).cuda(cur_device)
                    results.append(torch.linalg.solve(tmp_A,tmp_b)[0])
                total = results[0].cpu().numpy()
                for i in range(1,len(results)):
                    total = np.append(total, results[i].cpu().numpy(), 0)
                return total
            except Exception as e:
                warnings.warn("I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU. Error: ", e)
                try: x = np.linalg.solve(A,b)
                except: x,res,rank,s = np.linalg.lstsq(A,b,rcond=None)
                return x
        else:
            raise Exception("No valid solve method specified")
    ##################################################################################
    def _is_sparse(self,A):
        if float(np.count_nonzero(A))/float(len(A)**2) < 0.01: return True
        else: return False

    def _how_sparse_is(self,A):
        return float(np.count_nonzero(A))/float(len(A)**2)

    def _default_mean_function(self,x,hyperparameters,gp_obj):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)))
        mean[:] = np.mean(self.y_data)
        return mean

    def _default_noise_function(self, x, hyperparameters, gp_obj):
        return np.diag(np.ones((self.y_data.shape)) * (np.mean(abs(self.y_data)) / 100.0))

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
    def posterior_mean(self, x_pred, hyperparameters = None, x_out = None): ######think about how this input would look
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
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L), where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        solution dictionary : {}
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")


        if hyperparameters is not None:
            hps = self.hyperparameters
            K, KV, KVinvY, logdet, FO, KVinv, mean, cov = self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv = False)
        else:
            hps = self.hyperparameters
            KVinvY = self.KVinvY

        k = self.kernel(self.x_data,x_pred,hps,self)
        A = k.T @ KVinvY
        posterior_mean = self.mean_function(x_pred,hps,self) + A


        return {"x": x_pred,
                "f(x)": posterior_mean}

    def posterior_mean_grad(self, x_pred, hyperparameters = None, x_out = None, direction = None):
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
            Output space in case of multi-task GP use.

        Return
        ------
        solution dictionary : dict
        """

        if hyperparameters is not None:
            hps = self.hyperparameters
            K, KV, KVinvY, logdet, FO, KVinv, mean, cov = self._compute_GPpriorV(self.x_data, self.y_data, hyperparameters, calc_inv = False)
        else:
            hps = self.hyperparameters
            KVinvY = self.KVinvY
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,hps,self)
        f = self.mean_function(x_pred,hps,self)
        eps = 1e-6
        if direction is not None:
            x1 = np.array(x_pred)
            x1[:,direction] = x1[:,direction] + eps
            mean_der = (self.mean_function(x1,hps,self) - f)/eps
            k = self.kernel(self.x_data,x_pred,hps,self)
            k_g = self.d_kernel_dx(x_pred,self.x_data, direction,hps)
            posterior_mean_grad = mean_der + (k_g @ KVinvY)
        else:
            posterior_mean_grad = np.zeros((x_pred.shape))
            for direction in range(len(x_pred[0])):
                x1 = np.array(x_pred)
                x1[:,direction] = x1[:,direction] + eps
                mean_der = (self.mean_function(x1,hps,self) - f)/eps
                k = self.kernel(self.x_data,x_pred,hps,self)
                k_g = self.d_kernel_dx(x_pred,self.x_data, direction,hps)
                posterior_mean_grad[:,direction] = mean_der + (k_g @ KVinvY)
            direction = "ALL"

        return {"x": x_pred,
                "direction":direction,
                "df/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out = None, variance_only = False):
        """
        Function to compute the posterior covariance.
        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output space in case of multi-task GP use.
        variance_only : bool, optional
            If True the compuation of the posterior covariance matrix is avoided which can save compute time.
            In that case the return will only provide the variance at the input points.
            Default = False.
        Return
        ------
        solution dictionary : dict
        """

        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,self.hyperparameters,self)
        kk = self.kernel(x_pred, x_pred,self.hyperparameters,self)
        if self.KVinv is not None:
            if variance_only:
                S = False
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
            v[v<0.0] = 0.0
            if not variance_only:
                np.fill_diagonal(S, v)

        return {"x": x_pred,
                "v(x)": v,
                "S(x)": S}

    def posterior_covariance_grad(self, x_pred, x_out = None, direction = None):
        """
        Function to compute the gradient of the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        Return
        ------
        solution dictionary : dict
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,self.hyperparameters,self)
        k_covariance_prod = self._KVsolve(k)
        if direction is not None:
            k_g = self.d_kernel_dx(x_pred,self.x_data, direction,self.hyperparameters).T
            kk =  self.kernel(x_pred, x_pred,self.hyperparameters,self)
            x1 = np.array(x_pred)
            x2 = np.array(x_pred)
            eps = 1e-6
            x1[:,direction] = x1[:,direction] + eps
            kk_g = (self.kernel(x1, x1,self.hyperparameters,self)-\
                    self.kernel(x2, x2,self.hyperparameters,self)) /eps
            a = kk_g - (2.0 * k_g.T @ k_covariance_prod)
            return {"x": x_pred,
                "dv/dx": np.diag(a),
                "dS/dx": a}
        else:
            grad_v = np.zeros((len(x_pred),len(x_pred[0])))
            for direction in range(len(x_pred[0])):
                k_g = self.d_kernel_dx(x_pred,self.x_data, direction,self.hyperparameters).T
                kk =  self.kernel(x_pred, x_pred,self.hyperparameters,self)
                x1 = np.array(x_pred)
                x2 = np.array(x_pred)
                eps = 1e-6
                x1[:,direction] = x1[:,direction] + eps
                kk_g = (self.kernel(x1, x1,self.hyperparameters,self)-\
                    self.kernel(x2, x2,self.hyperparameters,self)) /eps
                grad_v[:,direction] = np.diag(kk_g - (2.0 * k_g.T @ k_covariance_prod))
            return {"x": x_pred,
                    "dv/dx": grad_v}


    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out = None):
        """
        Function to compute the joint prior over f (at measured locations) and f__pred at x_pred
        Parameters
        ----------
            x_pred: np.ndarray
                1d or 2d numpy array of points, note, these are elements of the
                index set which results from a cartesian product of input and output space
            x_out: optional, np.ndarray
                points in th eoutput space
        Return
        ------
        solution dictionary : dict
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,self.hyperparameters,self)
        kk = self.kernel(x_pred, x_pred,self.hyperparameters,self)
        post_mean = self.mean_function(x_pred, self.hyperparameters,self)
        joint_gp_prior_mean = np.append(self.prior_mean_vec, post_mean)
        return  {"x": x_pred,
                 "K": self.K,
                 "k": k,
                 "kappa": kk,
                 "prior mean": joint_gp_prior_mean,
                 "S(x)": np.block([[self.K, k],[k.T, kk]])}
    ###########################################################################
    def gp_prior_grad(self, x_pred, direction, x_out = None):
        """
        function to compute the gradient of the data-informed prior
        Parameters
        ------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            direction: direction in which to compute the gradient
        Return
        -------
        solution dictionary : dict
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,self.hyperparameters,self)
        kk = self.kernel(x_pred, x_pred,self.hyperparameters,self)
        k_g = self.d_kernel_dx(x_pred,self.x_data, direction,self.hyperparameters).T
        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        eps = 1e-6
        x1[:,direction] = x1[:,direction] + eps
        x2[:,direction] = x2[:,direction] - eps
        kk_g = (self.kernel(x1, x1,self.hyperparameters,self)-self.kernel(x2, x2,self.hyperparameters,self)) /(2.0*eps)
        post_mean = self.mean_function(x_pred, self.hyperparameters,self)
        mean_der = (self.mean_function(x1,self.hyperparameters,self) - self.mean_function(x2,self.hyperparameters,self))/(2.0*eps)
        full_gp_prior_mean_grad = np.append(np.zeros((self.prior_mean_vec.shape)), mean_der)
        prior_cov_grad = np.zeros(self.K.shape)
        return  {"x": x_pred,
                 "K": self.K,
                 "dk/dx": k_g,
                 "d kappa/dx": kk_g,
                 "d prior mean/x": full_gp_prior_mean_grad,
                 "dS/dx": np.block([[prior_cov_grad, k_g],[k_g.T, kk_g]])}

    ###########################################################################

    def entropy(self, S):
        """
        function comuting the entropy of a normal distribution
        res = entropy(S); S is a 2d numpy array, matrix has to be non-singular
        """
        dim  = len(S[0])
        logdet = self._logdet(S)
        return (float(dim)/2.0) +  ((float(dim)/2.0) * np.log(2.0 * np.pi)) + (0.5 * logdet)
    ###########################################################################
    def gp_entropy(self, x_pred, x_out = None):
        """
        Function to compute the entropy of the prior.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        Return
        ------
        entropy : float
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        priors = self.gp_prior(x_pred)
        S = priors["S(x)"]
        dim  = len(S[0])
        s, logdet = self._logdet(S)
        return (float(dim)/2.0) +  ((float(dim)/2.0) * np.log(2.0 * np.pi)) + (0.5 * logdet)
    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out = None):
        """
        Function to compute the gradient of entropy of the prior in a given direction.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        direction : int
            0 <= direction <= D - 1
        Return
        ------
        entropy : float
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        priors1 = self.gp_prior(x_pred)
        priors2 = self.gp_prior_grad(x_pred,direction)
        S1 = priors1["S(x)"]
        S2 = priors2["dS/dx"]
        return 0.5 * np.trace(self._inv(S1) @ S2)
    ###########################################################################
    def kl_div(self,mu1, mu2, S1, S2):
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
        KL div : float
        """
        s1, logdet1 = self._logdet(S1)
        s2, logdet2 = self._logdet(S2)
        x1 = self._solve(S2,S1)
        mu = np.subtract(mu2,mu1)
        x2 = self._solve(S2,mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu) - dim + (logdet2-logdet1))
        if kld < -1e-4: logger.debug("Negative KL divergence encountered")
        return kld
    ###########################################################################
    def gp_kl_div(self, x_pred, comp_mean, comp_cov, x_out = None):
        """
        function to compute the kl divergence of a posterior at given points
        Parameters
        ----------
            x_pred : 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            comp_mean : np.array
                Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
            comp_cov : np.array
                Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
        Return
        -------
            solution dictionary : dict
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        res = self.posterior_mean(x_pred)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_pred)["S(x)"]

        return {"x": x_pred,
                "gp posterior mean" : gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}


    ###########################################################################
    def gp_kl_div_grad(self, x_pred, comp_mean, comp_cov, direction, x_out = None):
        """
        function to compute the gradient of the kl divergence of a posterior at given points
        Parameters
        ----------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            comp_mean : np.array
                Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
            comp_cov : np.array
                Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
            direction: direction in which the gradient will be computed
        Return
        -------
            solution dictionary : dict
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        gp_mean = self.posterior_mean(x_pred)["f(x)"]
        gp_mean_grad = self.posterior_mean_grad(x_pred,direction)["df/dx"]
        gp_cov  = self.posterior_covariance(x_pred)["S(x)"]
        gp_cov_grad  = self.posterior_covariance_grad(x_pred,direction)["dS/dx"]

        return {"x": x_pred,
                "gp posterior mean" : gp_mean,
                "gp posterior mean grad" : gp_mean_grad,
                "gp posterior covariance": gp_cov,
                "gp posterior covariance grad": gp_cov_grad,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div grad": self.kl_div_grad(gp_mean, gp_mean_grad,comp_mean, gp_cov, gp_cov_grad, comp_cov)}
    ###########################################################################
    def shannon_information_gain(self, x_pred, x_out = None):
        """
        Function to compute the shannon-information --- the predicted drop in posterior uncertainty --- given
        a set of points. The shannon_information gain is a scalar.
        Parameters
        ----------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space if
                    x_out is given.
        Return
        -------
        solution dictionary : dict
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,self.hyperparameters,self)
        kk = self.kernel(x_pred, x_pred,self.hyperparameters,self)


        full_gp_covariances = \
                np.asarray(np.block([[self.K,k],\
                            [k.T,kk]]))

        e1 = self.entropy(self.K)
        e2 = self.entropy(full_gp_covariances)
        sig = (e2 - e1)
        return {"x": x_pred,
                "prior entropy" : e1,
                "posterior entropy": e2,
                "sig":sig}
    ###########################################################################
    def shannon_information_gain_vec(self, x_pred, x_out = None):
        """
        Function to compute the shannon-information gain of a set of points, but per point, in comparison to GP.shannon_information_gain.
        In this case, the information_gain is a vector.
        Parameters
        ----------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
        Return
        -------
        solution dictionary : {}
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        k = self.kernel(self.x_data,x_pred,self.hyperparameters,self)
        kk = self.kernel(x_pred, x_pred,self.hyperparameters,self)

        full_gp_covariances = np.empty((len(x_pred),len(self.K)+1,len(self.K)+1))
        for i in range(len(x_pred)): full_gp_covariances[i] = np.block([[self.K,k[:,i].reshape(-1,1)],[k[:,i].reshape(1,-1),kk[i,i]]])
        e1 = self.entropy(self.K)
        e2 = self.entropy(full_gp_covariances)
        sig = (e2 - e1)
        return {"x": x_pred,
                "prior entropy" : e1,
                "posterior entropy": e2,
                "sig(x)":sig}

    ###########################################################################
    def shannon_information_gain_grad(self, x_pred, direction, x_out = None):
        """
        Function to compute the gradient if the shannon-information gain --- the predicted drop in posterior entropy --- given
        a set of points.

        Parameters
        ----------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            direction: direction in which to compute the gradient
        Return
        -------
        solution dictionary : {}
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        e2 = self.gp_entropy_grad(x_pred,direction)
        sig = e2
        return {"x": x_pred,
                "sig grad":sig}
    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out = None):
        """
        Function to compute the probability of an uncertain feature given the gp posterior.
        Parameters
        ----------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            comp_mean: a vector of mean values, same length as x_pred
            comp_cov: covarianve matrix, in R^{len(x_pred)xlen(x_pred)}

        Return
        -------
        solution dictionary : {}
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        res = self.posterior_mean(x_pred)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_pred)["S(x)"]
        gp_cov_inv = self._inv(gp_cov)
        comp_cov_inv = self._inv(comp_cov)
        cov = self._inv(gp_cov_inv + comp_cov_inv)
        mu =  cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        s1, logdet1 = self._logdet(cov)
        s2, logdet2 = self._logdet(gp_cov)
        s3, logdet3 = self._logdet(comp_cov)
        dim  = len(mu)
        C = 0.5*(((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T \
               @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean))\
               -(gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0*np.pi)**(dim/2.0)) + (0.5*(logdet2 + logdet3)))
        return {"mu": mu,
                "covariance": cov,
                "probability":
                np.exp(ln_p)
                }
    def posterior_probability_grad(self, x_pred, comp_mean, comp_cov, direction, x_out = None):
        """
        Function to compute the gradient of the probability of an uncertain feature given the gp posterior
        Parameters
        ----------
            x_pred: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            comp_mean: a vector of mean values, same length as x_pred
            comp_cov: covarianve matrix, in R^{len(x_pred)xlen(x_pred)}
            direction: direction in which to compute the gradient

        Return
        -------
        solution dictionary : {}
        """
        if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
        if x_out is not None: x_pred = self._cartesian_product(x_pred,x_out)
        if len(x_pred[0]) != self.input_space_dim: raise Exception("Wrong dimensionality of the input points x_pred.")

        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        x1[:,direction] = x1[:,direction] + 1e-6
        x2[:,direction] = x2[:,direction] - 1e-6

        probability_grad = (posterior_probability(x1, comp_mean_comp_cov) - posterior_probability(x2, comp_mean_comp_cov))/2e-6
        return {"probability grad": probability_grad}

    ###########################################################################
    def _int_gauss(self,S):
        return ((2.0*np.pi)**(len(S)/2.0))*np.sqrt(np.linalg.det(S))

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
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
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
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """
        kernel = np.exp(-(distance ** 2) * (phi ** 2))
        return kernel



    def exponential_kernel(self, distance, length):
        """
        Function for the exponential kernel
        kernel = np.exp(-(distance) / (length))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
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
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """

        kernel = np.exp(-(distance) * (phi**2))
        return kernel



    def matern_kernel_diff1(self, distance, length):
        """
        Function for the matern kernel, order of differentiablity = 1.
        kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
            -(np.sqrt(3.0) * distance) / length

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """

        kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
            -(np.sqrt(3.0) * distance) / length
        )
        return kernel


    def matern_kernel_diff1_robust(self, distance, phi):
        """
        Function for the matern kernel, order of differentiablity = 1, robust version.
        kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi**2))) * np.exp(
            -(np.sqrt(3.0) * distance) * (phi**2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        phi : scalar
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """
        ##1/l --> phi**2
        kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi**2))) * np.exp(
            -(np.sqrt(3.0) * distance) * (phi**2))
        return kernel



    def matern_kernel_diff2(self, distance, length):
        """
        Function for the matern kernel, order of differentiablity = 2.
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
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """

        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) / (length))
            + ((5.0 * distance ** 2) / (3.0 * length ** 2))
        ) * np.exp(-(np.sqrt(5.0) * distance) / length)
        return kernel


    def matern_kernel_diff2_robust(self, distance, phi):
        """
        Function for the matern kernel, order of differentiablity = 2, robust version.
        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) * (phi**2))
            + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
        ) * np.exp(-(np.sqrt(5.0) * distance) * (phi**2))

        Parameters
        ----------
        distance : scalar or np.ndarray
            Distance between a set of points.
        length : scalar
            The length scale hyperparameters

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """


        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) * (phi**2))
            + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
        ) * np.exp(-(np.sqrt(5.0) * distance) * (phi**2))
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
        A structure of the shape of the distance input parameter : float or np.ndarray
        """

        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d > radius] = radius
        kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
        ((3.0*(d/radius)**2*np.log((d/radius)/(1+np.sqrt(1.0 - (d/radius)**2))))+\
        ((2.0*(d/radius)**2+1.0)*np.sqrt(1.0-(d/radius)**2)))
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
        A structure of the shape of the distance input parameter : float or np.ndarray
        """

        kernel = np.exp(-(2.0/length**2)*(np.sin(np.pi*distance/p)**2))
        return kernel

    def linear_kernel(self, x1,x2, hp1,hp2,hp3):
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
        A structure of the shape of the distance input parameter : float
        """
        kernel = hp1 + (hp2*(x1-hp3)*(x2-hp3))
        return kernel

    def dot_product_kernel(self, x1,x2,hp,matrix):
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
        A structure of the shape of the distance input parameter : float
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
        A structure of the shape of the distance input parameter : float
        """
        kernel = (1.0+x1.T @ x2)**p
        return p

    def default_kernel(self,x1,x2,hyperparameters,obj):
        """
        Function for the default kernel, a Matern kernel of first-order differentiability.

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D)
        x2 : np.ndarray
            Numpy array of shape (V x D)
        hyperparameters : np.ndarray
            Array of hyperparameters. For this kernel we need D + 1 hyperparameters
        obj : object instance
            GP object instance.

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1),len(x2)))
        for i in range(len(x1[0])):
            distance_matrix += abs(np.subtract.outer(x1[:,i],x2[:,i])/hps[1+i])**2
        distance_matrix = np.sqrt(distance_matrix)
        return   hps[0] * obj.matern_kernel_diff1(distance_matrix,1)


    def wendland_anisotropic(self,x1,x2, hps, obj):
        distance_matrix = np.zeros((len(x1),len(x2)))
        for i in range(len(x1[0])): distance_matrix += abs(np.subtract.outer(x1[:,i],x2[:,i])/hps[1+i])**2
        d = np.sqrt(distance_matrix)
        d[d > 1.] = 1.
        kernel = (1.-d)**8 * (35.*d**3 + 25.*d**2 + 8.*d + 1.)
        return kernel


    def non_stat_kernel(self,x1,x2,x0,w,l):
        """
        Non-stationary kernel.
        kernel = g(x1) g(x2)

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D)
        x2 : np.ndarray
            Numpy array of shape (V x D)
        x0 : np.array
            Numpy array of the basis function locations
        w : np.ndarray
            1d np.array of weights. len(w) = len(x0)
        l : float
            Width measure of the basis functions.

        Return
        ------
        A structure of the shape of the distance input parameter : float or np.ndarray
        """
        non_stat = np.outer(self._g(x1,x0,w,l),self._g(x2,x0,w,l))
        return non_stat
    

    def non_stat_kernel_gradient(self,x1,x2,x0,w,l):
        dkdw = np.einsum('ij,k->ijk', self._dgdw(x1,x0,w,l), self._g(x2,x0,w,l)) + np.einsum('ij,k->ikj', self._dgdw(x2,x0,w,l), self._g(x1,x0,w,l))
        dkdl =  np.outer(self._dgdl(x1,x0,w,l), self._g(x2,x0,w,l)) + np.outer(self._dgdl(x2,x0,w,l), self._g(x1,x0,w,l)).T
        res = np.empty((len(w)+1,len(x1),len(x2)))
        res[0:len(w)] = dkdw
        res[-1] = dkdl
        return res

    def _get_distance_matrix(self,x1,x2):
        d = np.zeros((len(x1),len(x2)))
        for i in range(x1.shape[1]): d += (x1[:,i].reshape(-1, 1) - x2[:,i])**2
        return np.sqrt(d)

    def _g(self,x,x0,w,l):
        d = self._get_distance_matrix(x,x0)
        e = np.exp( -(d**2) / l)
        return np.sum(w * e,axis = 1)

    def _dgdw(self,x,x0,w,l):
        d = self._get_distance_matrix(x,x0)
        e = np.exp( -(d**2) / l).T
        return e

    def _dgdl(self,x,x0,w,l):
        d = self._get_distance_matrix(x,x0)
        e = np.exp( -(d**2) / l)
        return np.sum(w * e * (d**2 / l**2), axis = 1)

    ##################################################################################
    ##################################################################################
    ###################Kernel and Mean Function Derivatives###########################
    ##################################################################################
    def d_gp_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-8
        new_points[:,direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters,self)
        b = self.kernel(points1,    points2, hyperparameters,self)
        derivative = ( a - b )/epsilon
        return derivative

    def dkernel_dh(self, points1, points2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-8
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyperparameters1,self)
        b = self.kernel(points1, points2, new_hyperparameters2,self)
        derivative = ( a - b )/(2.0*epsilon)
        return derivative

    def gp_kernel_gradient(self, points1, points2, hyperparameters, obj):
        gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self.dkernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def gp_kernel_derivative(self, points1, points2, direction, hyperparameters, obj):
        #gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        derivative = self.dkernel_dh(points1, points2, direction, hyperparameters)
        return derivative

    def _default_dm_dh(self,x,hps,gp_obj):
        gr = np.zeros((len(hps),len(x)))
        return gr

    def _default_dnoise_dh(self,x1,x2,hps,gp_obj):
        gr = np.zeros((len(hps), len(x1),len(x2)))
        return gr

    def _default_dnoise_dh_econ(self,x1,x2,i,hps,gp_obj):
        gr = np.zeros((len(x1),len(x2)))
        return gr


    def _finitediff_dm_dh(self,x,hps,gp_obj):
        gr = np.empty((len(hps),len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.mean_function(x,temp_hps1,self)
            b = self.mean_function(x,temp_hps2,self)
            gr[i] = (a-b)/2e-6
        return gr

    ##########################
    def _finitediff_dnoise_dh(self,x,hps,gp_obj):
        gr = np.zeros((len(hps), len(x),len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.noise_function(x,temp_hps1,self)
            b = self.noise_function(x,temp_hps2,self)
            gr[i] = (a-b)/2e-6
        return gr
    ##########################
    def _finitediff_dnoise_dh_econ(self,x,i,hps,gp_obj):
        temp_hps1 = np.array(hps)
        temp_hps1[i] = temp_hps1[i] + 1e-6
        temp_hps2 = np.array(hps)
        temp_hps2[i] = temp_hps2[i] - 1e-6
        a = self.noise_function(x,temp_hps1,self)
        b = self.noise_function(x,temp_hps2,self)
        gr = (a-b)/2e-6
        return gr



    def _normalize(self,data):
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
            n_x[:,i],x_min[i],x_max[i] = self._normalize(x_data[:,i])
        return n_x, x_min,x_max

    def _normalize_x_pred(self, x_pred, x_min, x_max):
        new_x_pred = np.empty(x_pred.shape)
        for i in range(len(x_pred[0])):
            new_x_pred[:,i] = (x_pred[:,i] - x_min[i]) / (x_max[i] - x_min[i])
        return new_x_pred

    def _cartesian_product(self,x,y):
        """
        Input x,y have to be 2d numpy arrays
        The return is the cartesian product of the two sets
        """
        res = []
        for i in range(len(y)):
            for j in range(len(x)):
                res.append(np.append(x[j],y[i]))
        return np.array(res)

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

