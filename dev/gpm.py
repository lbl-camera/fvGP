#!/usr/bin/env python

import warnings
import numpy as np
from loguru import logger
from dask.distributed import Client
from scipy.stats import norm
from .gp_prior import GPprior
from .gp_data import GPdata
from .gp_marginal_density import GPMarginalDensity
from .gp_likelihood import GPlikelihood
from .gp_training import GPtraining
from .gp_posterior import GPposterior


# TODO: search below "TODO"
#   variational inference in fvgp?


class GPm:
    """
    This class provides all capabilities for gplvm and tools to do GP regression on smooth manifolds.
    Use fvGP for multi-task GPs. However, the fvGP class inherits all methods from this class.
    This class allows full HPC support for training via the `hgdl` package.

    V ... number of input points

    D ... input space dimensionality

    N ... arbitrary integers (N1, N2,...)


    Parameters
    ----------
    x_data : np.ndarray or list
        The input point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.index_set_dim`.
        For single-task GPs, the index set dimension = input space dimension.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the `gp_noise_function`.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the `compute_device`
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters).
        The input `x1` is a N1 x D array of positions, `x2` is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `gp_kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x D array of positions),
        `x2` (a N2 x D array of positions) and
        `hyperparameters` (a 1d array of length D+1 for the default kernel).
        The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2, direction (int), and hyperparameters (numpy array).
        The output is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D) and hyperparameters
        (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `gp_mean_function` is provided.
    gp_noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `gp_noise_function` is provided but no noise function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the `compute_device` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    gp2Scale_linalg_mode : str, optional
        One of `Chol`, `sparseLU`, `sparseCG`, `sparseMINRES`, `sparseSolve`, `sparseCGpre`
        (incomplete LU preconditioner), or `sparseMINRESpre`. The default is None which amounts to
        an automatic determination of the mode.
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (3-10 times).
        For larger problems (>2000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not use the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
        Caution: this option, together with `append=True` in `tell()` will mean that the inverse of
        the covariance is updated, not recomputed, which can lead to instability.
        In application where data is appended many times, it is recommended to either turn
        `calc_inv` off, or to regularly force the recomputation of the inverse via `gp_rank_n_update` in
        `update_gp_data`.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x, direction, hyperparameters)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.

    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation variances
    prior.hyperparameters : np.ndarray
        Current hyperparameters in use.
    prior.K : np.ndarray
        Current prior covariance matrix of the GP
    prior.m : np.ndarray
        Current prior mean vector.
    likelihood.V : np.ndarray
        the noise covariance matrix
    """

    def __init__(
        self,
        x_data,
        init_hyperparameters=None,
        noise_variances=None,
        compute_device="cpu",
        gp_kernel_function=None,
        gp_kernel_function_grad=None,
        gp_noise_function=None,
        gp_noise_function_grad=None,
        gp_mean_function=None,
        gp_mean_function_grad=None,
        gp2Scale=False,
        gp2Scale_dask_client=None,
        gp2Scale_batch_size=10000,
        gp2Scale_linalg_mode=None,
        calc_inv=False,
        ram_economy=False,
        args=None,
    ):
        self.compute_device = compute_device
        self.args = args
        self.calc_inv = calc_inv
        self.gp2Scale = gp2Scale
        hyperparameters = init_hyperparameters
        ########################################
        ###init data instance###################
        ########################################
        self.data = GPmdata(x_data, np.ones((len(x_data))), noise_variances)
        ########################################
        # prepare initial hyperparameters and bounds
        if callable(gp_kernel_function) or callable(gp_mean_function) or callable(gp_noise_function):
            if init_hyperparameters is None: raise Exception(
                "You have provided callables for kernel, mean, or noise functions but no "
                "initial hyperparameters.")
        else:
            if init_hyperparameters is None:
                hyperparameters = np.ones((self.data.index_set_dim + 1))
                warnings.warn("Hyperparameters initialized to a vector of ones.")

        # warn if they could not be prepared
        if hyperparameters is None:
            raise Exception("'init_hyperparameters' not provided and could not be calculated. Please provide them ")

        if gp2Scale:
            try:
                from imate import logdet as imate_logdet
            except:
                raise Exception(
                    "You have activated `gp2Scale`. You need to install imate"
                    " manually for this to work.")
            if gp2Scale_dask_client is None:
                logger.debug("Creating my own local client.")
                gp2Scale_dask_client = Client()

        if compute_device == 'gpu':
            try:
                import torch
            except:
                raise Exception(
                    "You have specified the 'gpu' as your compute device. You need to install pytorch "
                    "manually for this to work.")
        ########################################
        ###init prior instance##################
        ########################################
        self.prior = GPprior(self.data,
                             hyperparameters=hyperparameters,
                             gp_kernel_function=gp_kernel_function,
                             gp_mean_function=gp_mean_function,
                             gp_kernel_function_grad=gp_kernel_function_grad,
                             gp_mean_function_grad=gp_mean_function_grad,
                             gp2Scale=gp2Scale,
                             gp2Scale_dask_client=gp2Scale_dask_client,
                             gp2Scale_batch_size=gp2Scale_batch_size,
                             ram_economy=ram_economy
                             )
        ########################################
        ###init likelihood instance#############
        ########################################
        #self.likelihood = GPlikelihood(self.data,
        #                               hyperparameters=self.prior.hyperparameters,
        #                               gp_noise_function=gp_noise_function,
        #                               gp_noise_function_grad=gp_noise_function_grad,
        #                               ram_economy=ram_economy,
        #                               gp2Scale=gp2Scale
        #                               )

        ##########################################
        #######prepare marginal density###########
        ##########################################
        #self.marginal_density = GPMarginalDensity(
        #    self.data,
        #    self.prior,
        #    self.likelihood,
        #    calc_inv=calc_inv,
        #    gp2Scale=gp2Scale,
        #    gp2Scale_linalg_mode=gp2Scale_linalg_mode,
        #    compute_device=compute_device
        #)

        ##########################################
        #######prepare training###################
        ##########################################
        #self.trainer = GPtraining(gp2Scale=gp2Scale)

        ##########################################
        #######prepare posterior evaluations######
        ##########################################
        #self.posterior = GPposterior(self.data,
        #                             self.prior,
        #                             self.marginal_density,
        #                             self.likelihood
        #                             )
        self.x_data = self.data.x_data
        self.index_set_dim = self.prior.index_set_dim

    def log_likelihood(self, y):
        """
        computes the marginal log-likelihood for gplvm
        input:
            hyperparameters
        output:
            marginal log-likelihood (scalar)
        """
        dim = float(self.input_dim)
        y1 = y2 = x[0:-2].reshape(self.point_number, self.output_dim)
        self.SparsePriorCovariance.reset_prior().result()
        hps = x[-2:]
        self.compute_covariance(y1, y2, hps, self.variances, client)
        logdet = self.SparsePriorCovariance.logdet().result()
        n = len(y)
        x = self.x_data
        traceKXX = self.SparsePriorCovariance.traceKXX(x).result()
        res = -(0.5 * traceKXX) - (dim * 0.5 * logdet) - (0.5 * dim * n * np.log(2.0 * np.pi))
        return res

    def neg_log_likelihood(self, y):
        """
        computes the marginal log-likelihood
        input:
            hyperparameters
        output:
            negative marginal log-likelihood (scalar)
        """

        return -log_likelihood(y)


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
        logger.debug("log marginal likelihood is being evaluated")
        if hyperparameters is None:
            K, V, m = self._get_KVm()
            KVinvY = self.KVinvY
            KVlogdet = self.KVlinalg.logdet()
        else:
            st = time.time()
            K = self.prior_obj.compute_prior_covariance_matrix(self.data_obj.x_data, hyperparameters=hyperparameters)
            logger.debug("   Prior covariance matrix computed after {} seconds.", time.time() - st)
            V = self.likelihood_obj.calculate_V(hyperparameters)
            logger.debug("   V computed after {} seconds.", time.time() - st)
            m = self.prior_obj.compute_mean(self.data_obj.x_data, hyperparameters=hyperparameters)
            logger.debug("   Prior mean computed after {} seconds.", time.time() - st)
            KVinvY, KVlogdet = self.compute_new_KVlogdet_KVinvY(K, V, m)
            logger.debug("   KVinvY and logdet computed after {} seconds.", time.time() - st)

        n = len(self.data_obj.y_data)
