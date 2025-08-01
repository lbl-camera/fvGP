#!/usr/bin/env python

import warnings
import numpy as np
from loguru import logger
from distributed import Client
from scipy.stats import norm
from .gp_prior import GPprior
from .gp_data import GPdata
from .gp_marginal_density import GPMarginalDensity
from .gp_likelihood import GPlikelihood
from .gp_training import GPtraining
from .gp_posterior import GPposterior


# TODO: search below "TODO"
#   as work on functions is completed and functions are maintained, add verbose assert statements.


class GP:
    """
    This class provides all the tools for a single-task Gaussian Process (GP).
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
    y_data : np.ndarray
        The values of the data points. Shape (V).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the `noise_function`.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the `compute_device`
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
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
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x D array of positions),
        `x2` (a N2 x D array of positions) and
        `hyperparameters` (a 1d array of length D+1 for the default kernel).
        The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2, direction (int), and hyperparameters (numpy array).
        The output is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `prior_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D) and hyperparameters
        (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `prior_mean_function` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `noise_function` is provided but no noise function,
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
        an automatic determination of the mode. For advanced customization options
        this can also be an iterable with three callables: the first f(K), where K is the covariance matrix
        to compute a factorization object
        which is available in the second and third callable. The second being the linear solve f(obj, vec),
        and the third being the logdet=f(obj). If a factorization object is not required, the first callable
        can return the matrix itself (K).
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
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x, direction, hyperparameters)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args: dict, optional
        A dictionary of advances settings.


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
        y_data,
        init_hyperparameters=None,
        noise_variances=None,
        compute_device="cpu",
        kernel_function=None,
        kernel_function_grad=None,
        noise_function=None,
        noise_function_grad=None,
        prior_mean_function=None,
        prior_mean_function_grad=None,
        gp2Scale=False,
        gp2Scale_dask_client=None,
        gp2Scale_batch_size=10000,
        gp2Scale_linalg_mode=None,
        calc_inv=False,
        ram_economy=False,
        args=None
    ):
        assert isinstance(x_data, list) or isinstance(x_data, np.ndarray), "wrong format in x_data"
        assert isinstance(y_data, np.ndarray) and np.ndim(y_data) == 1, "wrong format in y_data"
        assert isinstance(noise_variances, np.ndarray) or noise_variances is None, "wrong format in noise_variances"
        assert init_hyperparameters is None or isinstance(init_hyperparameters,np.ndarray), "wrong init_hyperparameters"
        assert isinstance(compute_device, str), "wrong format in compute_device"
        assert callable(kernel_function) or kernel_function is None, "wrong format in kernel_function"
        assert callable(kernel_function_grad) or kernel_function_grad is None, "wrong format in kernel_function"
        assert callable(noise_function) or noise_function is None, "wrong format in noise_function"
        assert callable(noise_function_grad) or noise_function_grad is None, "wrong format in noise_function"
        assert callable(prior_mean_function) or prior_mean_function is None, "wrong format in prior_mean_function"
        assert callable(prior_mean_function_grad) or prior_mean_function_grad is None, \
            "wrong format in prior_mean_function"
        assert len(x_data) == len(y_data), "x_data and y_data do not have the same lengths."

        self.compute_device = compute_device
        self.calc_inv = calc_inv
        self.gp2Scale = gp2Scale
        if args is None: self.args = {}
        else: self.args = args
        hyperparameters = init_hyperparameters

        ########################################
        ###init data instance###################
        ########################################
        self.data = GPdata(x_data, y_data, noise_variances)
        ########################################
        # prepare initial hyperparameters and bounds
        if self.data.Euclidean:
            if callable(kernel_function) or callable(prior_mean_function) or callable(noise_function):
                if init_hyperparameters is None: raise Exception(
                    "You have provided callables for kernel, mean, or noise functions but no "
                    "initial hyperparameters.")
            else:
                if init_hyperparameters is None:
                    hyperparameters = np.ones((self.data.index_set_dim + 1))
                    warnings.warn("Hyperparameters initialized to a vector of ones.")
        else:
            hyperparameters = init_hyperparameters

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
                             kernel=kernel_function,
                             prior_mean_function=prior_mean_function,
                             kernel_grad=kernel_function_grad,
                             prior_mean_function_grad=prior_mean_function_grad,
                             gp2Scale=gp2Scale,
                             gp2Scale_dask_client=gp2Scale_dask_client,
                             gp2Scale_batch_size=gp2Scale_batch_size,
                             ram_economy=ram_economy,
                             args=self.args
                             )
        ########################################
        ###init likelihood instance#############
        ########################################
        self.likelihood = GPlikelihood(self.data,
                                       hyperparameters=self.prior.hyperparameters,
                                       noise_function=noise_function,
                                       noise_function_grad=noise_function_grad,
                                       ram_economy=ram_economy,
                                       gp2Scale=gp2Scale,
                                       args=self.args
                                       )

        ##########################################
        #######prepare marginal density###########
        ##########################################
        self.marginal_density = GPMarginalDensity(
            self.data,
            self.prior,
            self.likelihood,
            calc_inv=calc_inv,
            gp2Scale=gp2Scale,
            gp2Scale_linalg_mode=gp2Scale_linalg_mode,
            compute_device=compute_device,
            args=self.args
        )

        ##########################################
        #######prepare training###################
        ##########################################
        self.trainer = GPtraining(gp2Scale=gp2Scale, args=self.args)

        ##########################################
        #######prepare posterior evaluations######
        ##########################################
        self.posterior = GPposterior(self.data,
                                     self.prior,
                                     self.marginal_density,
                                     self.likelihood,
                                     args=self.args
                                     )
        self.x_data = self.data.x_data
        self.y_data = self.data.y_data
        self.noise_variances = self.data.noise_variances
        self.index_set_dim = self.data.index_set_dim

    def update_gp_data(
        self,
        x_new,
        y_new,
        noise_variances_new=None,
        append=True,
        gp_rank_n_update=None
    ):
        """
        This function updates the data in the gp object instance.
        The data will only be overwritten if `append=False`, otherwise
        the data will be appended. This is a change from earlier versions.
        Now, the default is not to overwrite the existing data.


        Parameters
        ----------
        x_new : np.ndarray or list
            The point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.index_set_dim`.
            If dealing with non-Euclidean inputs
            `x_new` should be a list, not a numpy array.
        y_new : np.ndarray
            The values of the data points. Shape (V).
        noise_variances_new : np.ndarray, optional
            An numpy array defining the uncertainties in the data `y_data` in form of a point-wise variance.
            Shape (len(y_data)).
            Note: if no variances are provided here, the noise_covariance
            callable will be used; if the callable is not provided the noise variances
            will be set to `abs(np.mean(y_data)) / 100.0`. If you provided a noise function at initialization,
            the `noise_variances_new` will be ignored.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default=True.
            In the default case, data will be appended.
        gp_rank_n_update : bool, optional
            Indicates whether the GP marginal should be rank-n updated or recomputed. The default
            is `gp_rank_n_update=append`, meaning if data is only appended, the rank_n_update will
            be performed.
        """
        assert isinstance(x_new, list) or isinstance(x_new, np.ndarray), "wrong format in x_new"
        assert isinstance(y_new, np.ndarray) and np.ndim(y_new) == 1, "wrong format in y_new"
        assert isinstance(noise_variances_new, np.ndarray) or noise_variances_new is None, "wrong format in noise_variances_new"
        assert len(x_new) == len(y_new), "updated x and y do not have the same lengths."
        old_x_data = self.data.x_data.copy()
        if gp_rank_n_update is None: gp_rank_n_update = append
        # update data
        self.data.update(x_new, y_new, noise_variances_new, append=append)

        # update prior
        if append:
            self.prior.augment_data(old_x_data, x_new)
        else:
            self.prior.update_data()

        # update likelihood
        self.likelihood.update(self.prior.hyperparameters)

        # update marginal density
        self.marginal_density.update_data(gp_rank_n_update)
        ##########################################
        self.x_data = self.data.x_data
        self.y_data = self.data.y_data

    def set_args(self, args):
        self.args = args
        self.prior.args = args
        self.likelihood.args = args
        self.marginal_density.args = args
        self.trainer.args = args
        self.posterior.args = args
        self.marginal_density.KVlinalg.args = args

    def get_args(self):
        return self.args

    def _get_default_hyperparameter_bounds(self):
        """
        This function will create hyperparameter bounds for the default kernel based
        on the data only.


        Return:
        --------
        hyperparameter bounds for the default kernel : np.ndarray
        """
        if not self.data.Euclidean: raise Exception("Please provide custom hyperparameter bounds to "
                                                    "the training in the non-Euclidean setting")
        if len(self.prior.hyperparameters) != self.data.index_set_dim + 1:
            raise Exception("Please provide custom hyperparameter_bounds when kernel, mean or noise"
                            " functions are customized")
        hyperparameter_bounds = np.zeros((self.data.index_set_dim + 1, 2))
        hyperparameter_bounds[0] = np.array([np.var(self.data.y_data) / 100., np.var(self.data.y_data) * 10.])
        for i in range(self.data.index_set_dim):
            range_xi = np.max(self.data.x_data[:, i]) - np.min(self.data.x_data[:, i])
            hyperparameter_bounds[i + 1] = np.array([range_xi / 100., range_xi * 10.])
        assert isinstance(hyperparameter_bounds, np.ndarray) and np.ndim(hyperparameter_bounds) == 2
        return hyperparameter_bounds

    ###################################################################################
    ###################################################################################
    ###################################################################################
    #################TRAINING##########################################################
    ###################################################################################
    def train(self,
              hyperparameter_bounds=None,
              objective_function=None,
              objective_function_gradient=None,
              objective_function_hessian=None,
              init_hyperparameters=None,
              method="mcmc",
              pop_size=20,
              tolerance=0.0001,
              max_iter=200,
              local_optimizer="L-BFGS-B",
              global_optimizer="genetic",
              constraints=(),
              dask_client=None,
              info=False):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be `hgdl` and
        providing a dask client. However, in that case `fvgp.GP.train_async()` is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.


        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default means inferring the bounds from the communicated dataset.
            This only works for the default kernel.
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The Hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the Hessian of the negative log marginal likelihood.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
        method : str or Callable, optional
            The method used to train the hyperparameters.
            The options are `global`, `local`, `hgdl`, `mcmc`, and a callable.
            The callable gets a `gp.GP` instance and has to return a 1d np.ndarray of hyperparameters.
            The default is `mcmc` (scipy's differential evolution).
            If method = `mcmc`,
            the attribute `fvgp.GP.mcmc_info` is updated and contains convergence and distribution information.
            For `hgdl`, please provide a `distributed.Client()`.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to `method = hgdl`. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a `scipy` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if `hgdl` is used.
        info : bool, optional
            Provides a way how to access information reports during training of the GP. The default is False.
            If other information is needed please utilize `logger` as described in the online
            documentation (separately for HGDL and fvgp if needed).


        Return
        ------
        optimized hyperparameters (only fyi, gp is already updated) : np.ndarray
        """
        if self.gp2Scale: method = 'mcmc'
        if method == "hgdl" and dask_client is None: raise Exception("Please provide a dask_client for method =`hgdl`")
        if hyperparameter_bounds is None:
            hyperparameter_bounds = self._get_default_hyperparameter_bounds()
            warnings.warn("Default hyperparameter_bounds initialized because none were provided. "
                          "This will fail for custom kernel,"
                          " mean, or noise functions")

        if init_hyperparameters is None:
            if out_of_bounds(self.prior.hyperparameters, hyperparameter_bounds):
                init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                         high=hyperparameter_bounds[:, 1],
                                                         size=len(hyperparameter_bounds))
            else:
                init_hyperparameters = self.prior.hyperparameters
        else:
            if out_of_bounds(init_hyperparameters, hyperparameter_bounds):
                warnings.warn("Your init_hyperparameters are out of bounds. They will be over-written")
                init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                         high=hyperparameter_bounds[:, 1],
                                                         size=len(hyperparameter_bounds))

        if objective_function is not None and method == 'mcmc':
            warnings.warn("MCMC will ignore the user-defined objective function")
        if objective_function is not None and objective_function_gradient is None and (method == 'local' or 'hgdl'):
            raise Exception("For user-defined objective functions and local or hybrid optimization, a gradient and \
                             Hessian function of the objective function have to be defined.")
        if method == 'mcmc': objective_function = self.marginal_density.log_likelihood
        if objective_function is None: objective_function = self.marginal_density.neg_log_likelihood
        if objective_function_gradient is None:
            objective_function_gradient = self.marginal_density.neg_log_likelihood_gradient
        if objective_function_hessian is None:
            objective_function_hessian = self.marginal_density.neg_log_likelihood_hessian

        logger.debug("objective function: {}", objective_function)
        logger.debug("method: {}", method)

        hyperparameters = self.trainer.train(
            objective_function=objective_function,
            objective_function_gradient=objective_function_gradient,
            objective_function_hessian=objective_function_hessian,
            hyperparameter_bounds=hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            method=method,
            pop_size=pop_size,
            tolerance=tolerance,
            max_iter=max_iter,
            local_optimizer=local_optimizer,
            global_optimizer=global_optimizer,
            constraints=constraints,
            dask_client=dask_client,
            info=info
        )

        self.prior.update_hyperparameters(hyperparameters)
        self.likelihood.update(self.prior.hyperparameters)
        self.marginal_density.update_hyperparameters()
        assert isinstance(hyperparameters, np.ndarray) and np.ndim(hyperparameters) == 1
        return hyperparameters

    ##################################################################################
    def train_async(self,
                    hyperparameter_bounds=None,
                    objective_function=None,
                    objective_function_gradient=None,
                    objective_function_hessian=None,
                    init_hyperparameters=None,
                    max_iter=10000,
                    local_optimizer="L-BFGS-B",
                    global_optimizer="genetic",
                    constraints=(),
                    dask_client=None):
        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP asynchronously.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`,
        which will automatically update the GP with the new hyperparameters.


        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default means inferring the bounds from the communicated dataset.
            This only works for the default kernel.
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The Hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the Hessian of the negative log marginal likelihood.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to `method = hgdl`. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a `scipy` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training. If None is provided, a new
            `dask.distributed.Client` instance is constructed. It should be closed down manually when no longer needed.


        Return
        ------
        Optimization object that can be given to `fvgp.GP.update_hyperparameters()` to update the GP : object instance
        """
        if self.gp2Scale: raise Exception("gp2Scale does not allow asynchronous training!")
        if hyperparameter_bounds is None:
            hyperparameter_bounds = self._get_default_hyperparameter_bounds()
            warnings.warn("Default hyperparameter_bounds initialized because none were provided. "
                          "This will fail for custom kernel,"
                          " mean, or noise functions")

        if init_hyperparameters is None:
            if out_of_bounds(self.prior.hyperparameters, hyperparameter_bounds):
                init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                         high=hyperparameter_bounds[:, 1],
                                                         size=len(hyperparameter_bounds))
            else:
                init_hyperparameters = self.prior.hyperparameters
        else:
            if out_of_bounds(init_hyperparameters, hyperparameter_bounds):
                warnings.warn("Your init_hyperparameters are out of bounds. They will be over-written")
                init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                         high=hyperparameter_bounds[:, 1],
                                                         size=len(hyperparameter_bounds))

        if objective_function is None: objective_function = self.marginal_density.neg_log_likelihood
        if objective_function_gradient is None: objective_function_gradient = (
            self.marginal_density.neg_log_likelihood_gradient)
        if objective_function_hessian is None: objective_function_hessian = (
            self.marginal_density.neg_log_likelihood_hessian)

        opt_obj = self.trainer.train_async(
            objective_function=objective_function,
            objective_function_gradient=objective_function_gradient,
            objective_function_hessian=objective_function_hessian,
            hyperparameter_bounds=hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            max_iter=max_iter,
            local_optimizer=local_optimizer,
            global_optimizer=global_optimizer,
            constraints=constraints,
            dask_client=dask_client
        )
        return opt_obj

    ##################################################################################
    def stop_training(self, opt_obj):
        """
        Function to stop an asynchronous `hgdl` training.
        This leaves the :py:class:`distributed.client.Client` alive.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        self.trainer.stop_training(opt_obj)

    ###################################################################################
    def kill_client(self, opt_obj):
        """
        Function to kill an asynchronous training client. This shuts down the
        associated :py:class:`distributed.client.Client`.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        self.trainer.kill_client(opt_obj)

    ##################################################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters if an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.

        Return
        ------
        hyperparameters : np.ndarray
        """

        res = self.trainer.update_hyperparameters(opt_obj)
        if res is not None:
            l_n = self.marginal_density.neg_log_likelihood(res)
            l_o = self.marginal_density.neg_log_likelihood()
            if l_n - l_o < 0.000001:
                hyperparameters = res
                self.prior.update_hyperparameters(hyperparameters)
                self.likelihood.update(self.prior.hyperparameters)
                self.marginal_density.update_hyperparameters()
                logger.debug("    fvGP async hyperparameter update successful")
                logger.debug("    Latest hyperparameters: {}", hyperparameters)
            else:
                logger.debug(
                    "    The update was attempted but the new hyperparameters led to a \n \
                    lower likelihood, so I kept the old ones")
                logger.debug(f"Old likelihood: {-l_o} at {self.prior.hyperparameters}")
                logger.debug(f"New likelihood: {-l_n} at {res}")
        else:
            logger.debug("    Async Hyper-parameter update not successful in fvGP. I am keeping the old ones.")
            logger.debug("    hyperparameters: {}", self.prior.hyperparameters)

        return self.prior.hyperparameters

    ##################################################################################
    def set_hyperparameters(self, hps):
        """
        Function to set hyperparameters.


        Parameters
        ----------
        hps : np.ndarray
            A 1-d numpy array of hyperparameters.
        """
        assert isinstance(hps, np.ndarray), "wrong format in hyperparameters"
        assert np.ndim(hps) == 1, "wrong format in hyperparameters"
        self.prior.update_hyperparameters(hps)
        self.likelihood.update(self.prior.hyperparameters)
        self.marginal_density.update_hyperparameters()

    ##################################################################################
    def get_hyperparameters(self):
        """
        Function to get the current hyperparameters.


        Parameters
        ----------


        Return
        ------
        hyperparameters : np.ndarray
        """

        return self.prior.hyperparameters

    ##################################################################################
    def get_prior_pdf(self):
        """
        Function to get the current prior covariance matrix.


        Parameters
        ----------


        Return
        ------
        A dictionary containing information about the GP prior distribution : dict
        """

        return {"prior covariance (K)": self.prior.K,
                "prior mean": self.prior.m}

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
        if hyperparameters is not None:
            assert isinstance(hyperparameters, np.ndarray), "wrong format in hyperparameters"
            assert np.ndim(hyperparameters) == 1, "wrong format in hyperparameters"
        return self.marginal_density.log_likelihood(hyperparameters=hyperparameters)

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
        return self.marginal_density.log_likelihood(hyperparameters=hyperparameters)

    def test_log_likelihood_gradient(self, hyperparameters):
        """
        Function to test your gradient of the log-likelihood and therefore of the kernel function.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).

        Return
        ------
        analytical and finite difference gradients to compare
        """
        assert isinstance(hyperparameters, np.ndarray), "wrong format in hyperparameters"
        assert np.ndim(hyperparameters) == 1, "wrong format in hyperparameters"
        return self.marginal_density.test_log_likelihood_gradient(hyperparameters)

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        """
        This function calculates the posterior mean for a set of input points.

        Parameters
        ----------
        x_pred : np.ndarray or list
            A numpy array of shape (V x D), interpreted as an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        hyperparameters : np.ndarray, optional
            A numpy array of the correct size depending on the kernel. This is optional in case the posterior mean
            has to be computed with given hyperparameters, which is, for instance, the case if the posterior mean is
            a constraint during training. The default is None which means the initialized or trained hyperparameters
            are used.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Solution points and function values : dict
        """
        return self.posterior.posterior_mean(x_pred, hyperparameters=hyperparameters, x_out=x_out)

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None):
        """
        This function calculates the gradient of the posterior mean for a set of input points.

        Parameters
        ----------
        x_pred : np.ndarray or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        hyperparameters : np.ndarray, optional
            A numpy array of the correct size depending on the kernel. This is optional in case the posterior mean
            has to be computed with given hyperparameters, which is, for instance, the case if the posterior mean is
            a constraint during training. The default is None which means the initialized or trained hyperparameters
            are used.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        direction : int, optional
            Direction of derivative, If None (default) the whole gradient will be computed.

        Return
        ------
        Solution : dict
        """
        return self.posterior.posterior_mean_grad(x_pred, hyperparameters=hyperparameters,
                                                  x_out=x_out, direction=direction)

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        """
        Function to compute the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        variance_only : bool, optional
            If True the computation of the posterior covariance matrix is avoided which can save compute time.
            In that case the return will only provide the variance at the input points.
            Default = False. This is only relevant if `calc_inv` at initialization is True.
        add_noise : bool, optional
            If True the noise variances will be added to the posterior variances. Default = False.

        Return
        ------
        Solution : dict
        """
        return self.posterior.posterior_covariance(x_pred, x_out=x_out, variance_only=variance_only,
                                                   add_noise=add_noise)

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        """
        Function to compute the gradient of the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        direction : int, optional
            Direction of derivative, If None (default) the whole gradient will be computed.

        Return
        ------
        Solution : dict
        """
        return self.posterior.posterior_covariance_grad(x_pred, x_out=x_out, direction=direction)

    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out=None):
        """
        Function to compute the joint prior over f (at measured locations) and f_pred at x_pred.

        Parameters
        ----------
        x_pred : np.ndarray or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Solution : dict
        """

        return self.posterior.joint_gp_prior(x_pred, x_out=x_out)

    ###########################################################################
    def joint_gp_prior_grad(self, x_pred, direction, x_out=None):
        """
        Function to compute the gradient of the data-informed prior.

        Parameters
        ------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        direction : int
            Direction of derivative.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Solution : dict
        """
        return self.posterior.joint_gp_prior_grad(x_pred, direction, x_out=x_out)

    ###########################################################################
    def gp_entropy(self, x_pred, x_out=None):
        """
        Function to compute the entropy of the gp prior probability distribution.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Entropy : float
        """
        return self.posterior.gp_entropy(x_pred, x_out=x_out)

    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out=None):
        """
        Function to compute the gradient of entropy of the prior in a given direction.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        direction : int
            Direction of the derivative.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Entropy gradient in given direction : float
        """
        return self.posterior.gp_entropy_grad(x_pred, direction, x_out=x_out)

    ###########################################################################
    def gp_kl_div(self, x_pred, comp_mean, comp_cov, x_out=None):
        """
        Function to compute the kl divergence of a posterior at given points
        and a given normal distribution.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        comp_mean : np.ndarray
            Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
        comp_cov : np.ndarray
            Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Solution : dict
        """
        return self.posterior.gp_kl_div(x_pred, comp_mean, comp_cov, x_out=x_out)

    ###########################################################################
    def gp_mutual_information(self, x_pred, x_out=None, add_noise=False):
        """
        Function to calculate the mutual information between
        the random variables f(x_data) and f(x_pred).
        The mutual information is always positive, as it is a KL divergence, and is bounded
        from below by 0. The maxima are expected at the data points. Zero is expected far from the
        data support.
        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        add_noise : bool, optional
            If True the noise variances will be added to the prior over the prediction points. Default = False.

        Return
        ------
        Solution : dict
        """
        return self.posterior.gp_mutual_information(x_pred, x_out=x_out, add_noise=add_noise)

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None, add_noise=False):
        """
        Function to calculate the interaction information between
        the random variables f(x_data) and f(x_pred). This is the mutual information
        of each f(x_pred) with f(x_data). It is also called the Multi-information.
        It is best used when several prediction points are supposed to be mutually aware.
        The total correlation is always positive, as it is a KL divergence, and is bounded
        from below by 0. The maxima are expected at the data points. Zero is expected far from the
        data support.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        add_noise : bool, optional
            If True the noise variances will be added to the prior over the prediction points. Default = False.

        Return
        ------
        Solution : dict
            Total correlation between prediction points, as a collective.
        """
        return self.posterior.gp_total_correlation(x_pred, x_out=x_out, add_noise=add_noise)

    ###########################################################################
    def gp_relative_information_entropy(self, x_pred, x_out=None, add_noise=False):
        """
        Function to compute the KL divergence and therefore the relative information entropy
        of the prior distribution defined over predicted function values and the posterior distribution.
        The value is a reflection of how much information is predicted to be gained
        through observing a set of data points at x_pred.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        add_noise : bool, optional
            If True the noise variances will be added to the posterior covariance. Default = False.

        Return
        ------
        Solution : dict
            Relative information entropy of prediction points, as a collective.
        """
        return self.posterior.gp_relative_information_entropy(x_pred, x_out=x_out, add_noise=add_noise)

    ###########################################################################
    def gp_relative_information_entropy_set(self, x_pred, x_out=None, add_noise=False):
        """
        Function to compute the KL divergence and therefore the relative information entropy
        of the prior distribution over predicted function values and the posterior distribution.
        The value is a reflection of how much information is predicted to be gained
        through observing each data point in x_pred separately, not all
        at once as in `gp_relative_information_entropy`.


        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).
        add_noise : bool, optional
            If True the noise variances will be added to the posterior covariance. Default = False.

        Return
        ------
        Solution : dict
            Relative information entropy of prediction points, but not as a collective.
        """
        return self.posterior.gp_relative_information_entropy_set(x_pred, x_out=x_out, add_noise=add_noise)

    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out=None):
        """
        Function to compute probability of a probabilistic quantity of interest,
        given the GP posterior at given points.

        Parameters
        ----------
        x_pred : np.ndarray  or list
            A numpy array of shape (V x D), interpreted as  an array of input point positions, or a list for
            GPs on non-Euclidean input spaces.
        comp_mean: np.ndarray
            A vector of mean values, same length as x_pred.
        comp_cov: np.nparray
            Covariance matrix, in R^{len(x_pred) x len(x_pred)}
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Return
        ------
        Solution : dict
            The probability of a probabilistic quantity of interest, given the GP posterior at a given point.
        """
        return self.posterior.posterior_probability(x_pred, comp_mean, comp_cov, x_out=x_out)

    ####################################################################################
    ####################################################################################
    #######################VALIDATION###################################################
    ####################################################################################
    @staticmethod
    def _crps_s(x, mu, sigma):
        res = abs(sigma * ((1. / np.sqrt(np.pi))
                           - 2. * norm.pdf((x - mu) / sigma)
                           - (((x - mu) / sigma) * (2. * norm.cdf((x - mu) / sigma) - 1.))))
        return np.mean(res), np.sqrt(np.var(res))

    def crps(self, x_test, y_test):  # correct, tested
        """
        This function calculates the continuous rank probability score.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape (V x No) in the multi-output case. These are the y data to compare against.

        Return
        ------
        CRPS, standard dev. of CRPS : (float, float)
        """

        mean = self.posterior_mean(x_test)["m(x)"]
        sigma = np.sqrt(self.posterior_covariance(x_test)["v(x)"])
        assert mean.shape == sigma.shape == y_test.shape, (mean.shape, sigma.shape, y_test.shape)
        r = self._crps_s(y_test, mean, sigma)
        return r

    def rmse(self, x_test, y_test):  # correct, tested
        """
        This function calculates the root mean squared error.
        Note that in the multi-task setting the user should perform their
        input point transformation beforehand.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Return
        ------
        RMSE : float
        """

        v1 = y_test
        v2 = self.posterior_mean(x_test)["m(x)"]
        assert v1.shape == v2.shape, (v1.shape, v2.shape)
        return np.sqrt(np.sum((v1 - v2) ** 2) / v1.size)

    def nlpd(self, x_test, y_test):  # correct, tested
        """
        This function calculates the Negative log predictive density.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Return
        ------
        NLPD : float
        """

        mean = self.posterior_mean(x_test)["m(x)"]
        sigma = np.sqrt(self.posterior_covariance(x_test)["v(x)"])

        assert mean.shape == sigma.shape == y_test.shape, (mean.shape, sigma.shape, y_test.shape)

        g = self.gaussian_1d(y_test, mean, sigma)
        g[g == 0.] = 1e-16
        g = np.log(g)
        return -np.mean(g)

    def r2(self, x_test, y_test):
        """
        This function calculates the R2 prediction score.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Return
        ------
        R2 : float
        """
        y_pred_mean = self.posterior_mean(x_test)["m(x)"]
        assert y_pred_mean.shape == y_test.shape, (y_pred_mean.shape, y_test.shape)
        ss_res = np.sum((y_test - y_pred_mean) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1. - ss_res / ss_tot

    @staticmethod
    def gaussian_1d(x, mu, sigma):
        """
        Evaluates a 1D Gaussian (Normal) distribution at a point x.

        Parameters
        ----------
        x : np.ndarray
            The points where you want to evaluate the Gaussian.
        mu : np.ndarray
            The mean of the Gaussian (default 0.0).
        sigma : np.ndarray
            The standard deviation of the Gaussians.

        Return
        ------
        Evaluations of the Gaussian : np.ndarray
        """
        # Gaussian function formula
        coefficient = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
        return coefficient * np.exp(exponent)

    @staticmethod
    def make_2d_x_pred(bx, by, resx=100, resy=100):  # pragma: no cover
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

    @staticmethod
    def make_1d_x_pred(b, res=100):  # pragma: no cover
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

    def get_gp2Scale_exec_time(self, time_per_worker_execution, number_of_workers):
        """
        This function calculates the estimated time gp2Scale takes to calculate the covariance matrix
        as a function of the number of workers and their speed calculating a block.

        Parameters
        ----------
        time_per_worker_execution : float
            The time one worker takes to compute a block of the covariance matrix.
        number_of_workers : int
            The number of dask workers the covariance matrix calculation is distributed over.

        Return
        ------
        estimated execution time : float
        """
        b = self.prior.batch_size
        D = len(self.x_data)
        tb = time_per_worker_execution
        n = number_of_workers
        return (D ** 2 * tb) / (2. * n * b ** 2)


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def out_of_bounds(x, bounds):
    assert isinstance(x, np.ndarray)
    assert isinstance(bounds, np.ndarray)
    assert np.ndim(bounds) == 2
    for i in range(len(x)):
        if x[i] < bounds[i, 0] or x[i] > bounds[i, 1]:
            return True
    return False
