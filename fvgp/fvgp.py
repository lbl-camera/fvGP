#!/usr/bin/env python
import numpy as np
from .gp import GP


class fvGP(GP):
    """
    This class provides all the tools for a multi-task Gaussian Process (GP).
    This class allows for full HPC support for training. After initialization, this
    class provides all the methods described for the GP (:py:class:`fvgp.GP`) class.

    V ... number of input points

    Di... input space dimensionality

    No... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The main logic of :doc:`fvgp <fvgp:index>` is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best
    way to give the user optimal control and power. At various instances, for instances
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input `x` is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [[0],[1]], the input to the mean, kernel, and noise function might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]

    This has to be understood and taken into account when customizing :doc:`fvgp <fvgp:index>` for multi-task
    use. The examples will provide deeper insights.

    Parameters
    ----------
    x_data : np.ndarray or list
        The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_space_dim`.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
    y_data : np.ndarray
        The values of the data points. Shape (V,No).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). The task direction is
        simply considered a separate dimension. If sparse_node or gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    output_positions : np.ndarray, optional
        A 2d numpy array of shape (V x output_number), so that for each measurement position, the outputs
        are clearly defined by their positions in the output space. The default is
        np.array([[0,1,2,3,...,output_number - 1],[0,1,2,3,...,output_number - 1],...]).
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V, No).
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the gp_noise_function.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear system solves are executed. The default is `cpu`.
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the compute_device
        becomes much more important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters).
        The input `x1` a N1 x Di+1 array of positions, `x2` is a N2 x Di+1
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperparameters are initialized.
        The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD). The task
        direction is simply considered an additional dimension. This kernel should only be used for tests and in the
        simplest of cases.
        The output is a matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `gp_kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x Di + 1 array of positions),
        `x2` (a N2 x Di + 1 array of positions) and
        `hyperparameters` (a 1d array of length Di+2 for the default kernel).
        The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2,
        direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+1) and
         hyperparameters (a 1d array of length Di+2 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x Di+1) and hyperparameters
        (a 1d array of length Di+2 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `gp_mean_function` is provided.
    gp_noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input `x` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 3-D array of
        shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `gp_noise_function` is provided but no gradient function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. The noise function will have
        to return a `scipy.sparse` matrix instead of a numpy array. There are a few more
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the `compute_device` option should be revisited.
        The kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    gp2Scale_linalg_mode : str, optional
        One of `Chol`, `sparseLU`, `sparseCG`, or `sparseMINRES`. The default is None which amounts to
        an automatic determination of the mode.
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (3-10 times).
        The default is False. Note, the training will not use the
        inverse for stability reasons. Storing the inverse is
        a good option when the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x1[, x2], direction, hyperparameters)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x1[, x2,] hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.
    info : bool, optional
        Provides a way how to access various information reports. The default is False.

    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation (co)variances.
    prior.hyperparameters : np.ndarray
        Current hyperparameters in use.
    prior.K : np.ndarray
        Current prior covariance matrix of the GP
    prior.m : np.ndarray
        Current prior mean vector.
    marginal_density.KVinv : np.ndarray
        If enabled, the inverse of the prior covariance + nose matrix V
        inv(K+V)
    marginal_density.KVlogdet : float
        logdet(K+V)
    likelihood.V : np.ndarray
        the noise covariance matrix


    This class inherits all capabilities from :py:class:`fvgp.GP`.
    Check there for a full list of capabilities. Here are the most important.

    Base-GP Methods:

    :py:meth:`fvgp.GP.train`

    :py:meth:`fvgp.GP.train_async`

    :py:meth:`fvgp.GP.stop_training`

    :py:meth:`fvgp.GP.kill_training`

    :py:meth:`fvgp.GP.update_hyperparameters`

    :py:meth:`fvgp.GP.set_hyperparameters`

    :py:meth:`fvgp.GP.get_hyperparameters`

    Posterior Evaluations:

    :py:meth:`fvgp.GP.posterior_mean`

    :py:meth:`fvgp.GP.posterior_covariance`

    :py:meth:`fvgp.GP.posterior_mean_grad`

    :py:meth:`fvgp.GP.posterior_covariance_grad`

    :py:meth:`fvgp.GP.joint_gp_prior`

    :py:meth:`fvgp.GP.joint_gp_prior_grad`

    :py:meth:`fvgp.GP.gp_entropy`

    :py:meth:`fvgp.GP.gp_entropy_grad`

    :py:meth:`fvgp.GP.gp_kl_div`

    :py:meth:`fvgp.GP.gp_kl_div_grad`

    :py:meth:`fvgp.GP.gp_mutual_information`

    :py:meth:`fvgp.GP.gp_total_correlation`

    :py:meth:`fvgp.GP.gp_relative_information_entropy`

    :py:meth:`fvgp.GP.gp_relative_information_entropy_set`

    :py:meth:`fvgp.GP.posterior_probability`

    :py:meth:`fvgp.GP.posterior_probability_grad`

    Validation Methods:

    :py:meth:`fvgp.GP.crps`

    :py:meth:`fvgp.GP.rmse`

    :py:meth:`fvgp.GP.make_2d_x_pred`

    :py:meth:`fvgp.GP.make_1d_x_pred`

    :py:meth:`fvgp.GP.log_likelihood`

    :py:meth:`fvgp.GP.test_log_likelihood_gradient`
    """

    def __init__(
        self,
        x_data,
        y_data,
        init_hyperparameters=None,
        output_positions=None,
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
        info=False,
    ):

        if isinstance(x_data, np.ndarray):
            assert np.ndim(x_data) == 2
            self.input_space_dim = x_data.shape[1]
        else: self.input_space_dim = 1

        self.output_num = y_data.shape[1]
        output_space_dim = 1
        ###check the output dims

        if np.ndim(y_data) == 1:
            raise ValueError("The output number is 1, you can use GP for single-task GPs")
        if output_space_dim == 1 and isinstance(output_positions, np.ndarray) is False:
            self.output_positions = self._compute_standard_output_positions(len(x_data))
        else:
            self.output_positions = output_positions

        assert isinstance(self.output_positions, np.ndarray) and np.ndim(self.output_positions) == 2
        ####transform the space
        self.fvgp_x_data = x_data
        self.fvgp_y_data = y_data
        self.fvgp_noise_variances = noise_variances
        self.index_set_dim = self.input_space_dim + output_space_dim
        x_data, y_data, noise_variances = self._transform_index_set(x_data, y_data, noise_variances,
                                                                    self.output_positions)

        ####init GP
        super().__init__(
            x_data,
            y_data,
            init_hyperparameters=init_hyperparameters,
            noise_variances=noise_variances,
            compute_device=compute_device,
            gp_kernel_function=gp_kernel_function,
            gp_kernel_function_grad=gp_kernel_function_grad,
            gp_mean_function=gp_mean_function,
            gp_mean_function_grad=gp_mean_function_grad,
            gp_noise_function=gp_noise_function,
            gp_noise_function_grad=gp_noise_function_grad,
            gp2Scale=gp2Scale,
            gp2Scale_dask_client=gp2Scale_dask_client,
            gp2Scale_batch_size=gp2Scale_batch_size,
            gp2Scale_linalg_mode=gp2Scale_linalg_mode,
            calc_inv=calc_inv,
            ram_economy=ram_economy,
            args=args,
            info=info)
        if self.data.Euclidean: assert self.index_set_dim == self.input_space_dim + output_space_dim

    def update_gp_data(
        self,
        x_new,
        y_new,
        noise_variances_new=None,
        append=True,
        output_positions_new=None,
    ):

        """
        This function updates the data in the gp object instance.
        The data will NOT be appended but overwritten!
        Please provide the full updated data set.

        Parameters
        ----------
        x_new : np.ndarray
            The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_space_dim`.
            For multi-task GPs, the index set dimension = input space dimension + 1.
            If dealing with non-Euclidean inputs
            x_new should be a list, not a numpy array.
        y_new : np.ndarray
            The values of the data points. Shape (V,No).
        noise_variances_new : np.ndarray, optional
            An numpy array defining the uncertainties in the data `y_data` in form of a point-wise variance.
            Shape (len(y_new)).
            Note: if no variances are provided here, the noise_covariance
            callable will be used; if the callable is not provided the noise variances
            will be set to `abs(np.mean(y_data)) / 100.0`. If you provided a noise function,
            the noise_variances_new will be ignored.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default = True.
            In the default case, data will be appended.
        output_positions_new : np.ndarray, optional
            A 2d numpy array of shape (V x output_number), so that for each measurement position, the outputs
            are clearly defined by their positions in the output space. The default is
            np.array([[0,1,2,3,...,output_number - 1],[0,1,2,3,...,output_number - 1],...]).
        """
        ##########################################
        #######prepare value positions############
        ##########################################
        if not isinstance(output_positions_new, np.ndarray):
            output_positions_new = self._compute_standard_output_positions(len(x_new))
        ######################################
        #####transform to index set###########
        ######################################
        x_data, y_data, noise_variances = self._transform_index_set(x_new, y_new, noise_variances_new,
                                                                    output_positions_new)
        super().update_gp_data(x_data, y_data, noise_variances, append=append)
        self.output_positions = np.row_stack([self.output_positions, output_positions_new])

    ################################################################################################
    def _compute_standard_output_positions(self, point_number):
        value_pos = np.zeros((point_number, self.output_num))
        for j in range(self.output_num):
            value_pos[:, j] = j
        return value_pos

    ################################################################################################
    def _transform_index_set(self, x_data, y_data, noise_variances, output_positions):
        point_number = len(x_data)
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list)
        if isinstance(x_data, np.ndarray):
            new_points = np.zeros((point_number * self.output_num, self.index_set_dim))
        else:
            new_points = [0.] * point_number * self.output_num
        new_values = np.zeros((point_number * self.output_num))
        if noise_variances is not None:
            new_variances = np.zeros((point_number * self.output_num))
        else:
            new_variances = None
        for i in range(self.output_num):
            if isinstance(x_data, np.ndarray):
                new_points[i * point_number: (i + 1) * point_number] = np.column_stack([x_data, output_positions[:, i]])
            if isinstance(x_data, list):
                for j in range(len(x_data)):
                    new_points[i * point_number + j] = [x_data[j], output_positions[j, i]]
            new_values[i * point_number: (i + 1) * point_number] = y_data[:, i]
            if noise_variances is not None:
                new_variances[i * point_number: (i + 1) * point_number] = noise_variances[:, i]

        return new_points, new_values, new_variances

    ################################################################################################

