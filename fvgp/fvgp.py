#!/usr/bin/env python
import numpy as np
from .gp import GP


class fvGP(GP):
    """
    This class provides all the tools for a multi-task Gaussian Process (GP).
    This class allows for full HPC support for training. After initialization, this
    class provides all the methods described for the GP (:py:class:`fvgp.GP`) class.
    This class allows full HPC support for training via the `hgdl` package.

    V ... number of input points

    Di... input space dimensionality

    No... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The main logic of :doc:`fvgp <fvgp:index>` is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best
    way to give the user optimal control and power. At various instances, for example
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input `x` is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [0,1], the input to the mean, kernel, and noise functions might be

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
    y_data : np.ndarray or list
        The values of the data points. Shape (V,No) if `y_data` is an array.
        It is possible that not every entry in `x_data`
        has all corresponding tasks available. In that case `y_data` can be a list. In that case make sure
        that every entry in `y_data` has a corresponding `output_position` of the same shape.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). The task direction is
        simply considered a separate dimension. If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    output_positions : list, optional
        A list of 1d numpy arrays indicating which `task` measurements are available,
        so that for each measurement position, the outputs
        are clearly defined by their positions in the output space. The default is
        [[0,1,2,3,...,output_number - 1],[0,1,2,3,...,output_number - 1],...].
        It is possible that for certain inputs tasks are missing, e.g.,
        output_positions = [[0,1],[1]].
    noise_variances : np.ndarray or list, optional
        An numpy array or list defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V, No) if np.ndarray.
        If `y_data` is a list then the `noise_variances` should be a list.
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
        direction (int), and hyperparameters (numpy array).
        The output is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
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
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The input `x` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1) and
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
    info : bool, optional
        Provides a way how to access various information reports. The default is False.

    Attributes
    ----------
    x_data : np.ndarray or list
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation variances.
    fvgp_x_data : np.ndarray or list
        Data points from the fvgp point of view.
    fvgp_y_data : np.ndarray
        The data values from the fvgp point of view.
    fvgp_noise_variances : np.ndarray
        Observation variances from the fvgp point of view.
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
        assert isinstance(y_data, list) or isinstance(y_data, np.ndarray)
        assert output_positions is None or isinstance(output_positions, list)

        if isinstance(x_data, np.ndarray):
            assert np.ndim(x_data) == 2
            self.input_space_dim = x_data.shape[1]
        else: self.input_space_dim = 1

        self.output_num = len(y_data[0])
        ###check the output dims

        if isinstance(y_data, np.ndarray) and np.ndim(y_data) == 1:
            raise ValueError("The output number is 1, you can use the GP class for single-task GPs")
        if output_positions is None:
            self.output_positions = self._compute_standard_output_positions(len(x_data))
        else:
            self.output_positions = output_positions

        assert isinstance(self.output_positions, list)
        assert len(x_data) == len(y_data)
        ####transform the space
        self.fvgp_x_data = x_data
        self.fvgp_y_data = y_data
        self.fvgp_noise_variances = noise_variances
        self.index_set_dim = self.input_space_dim + 1
        x_data, y_data, noise_variances = self._transform_index_set2(x_data, y_data, noise_variances,
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

        if self.data.Euclidean: assert self.index_set_dim == self.input_space_dim + 1

    def update_gp_data(
        self,
        x_new,
        y_new,
        noise_variances_new=None,
        append=True,
        output_positions_new=None,
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
            The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_space_dim`.
            For multi-task GPs, the index set dimension = input space dimension + 1.
            If dealing with non-Euclidean inputs
            x_new should be a list, not a numpy array.
        y_new : np.ndarray or list
            The values of the data points. Shape (V,No) if `y_new`  is an array.
            It is possible that not every entry in `x_new`
            has all corresponding tasks available. In that case `y_new` can be a list. In that case make sure
            that every entry in `y_new` has a corresponding `output_position_new` of the same shape.
        noise_variances_new : np.ndarray or list, optional
            An numpy array or list defining the uncertainties/noise in the
            `y_data` in form of a point-wise variance. Shape (V, No) if np.ndarray.
            If `y_new` is a list then the `noise_variances` should be a list.
            Note: if no noise_variances are provided here, the gp_noise_function
            callable will be used; if the callable is not provided, the noise variances
            will be set to `abs(np.mean(y_data)) / 100.0`. If
            noise covariances are required (correlated noise), make use of the gp_noise_function.
            Only provide a noise function OR `noise_variances`, not both.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default = True.
            In the default case, data will be appended.
        output_positions_new : list, optional
            A list of 1d numpy arrays indicating which `task` measurements are available,
            so that for each measurement position, the outputs
            are clearly defined by their positions in the output space. The default is
            [[0,1,2,3,...,output_number - 1],[0,1,2,3,...,output_number - 1],...].
            The output_number is defined by the first entry in `y_data`.
        gp_rank_n_update : bool , optional
            Indicates whether the GP marginal should be rank-n updated or recomputed. The default
            is `gp_rank_n_update=append`, meaning if data is only appended, the rank_n_update will
            be performed.
        """
        assert isinstance(x_new, np.ndarray) or isinstance(x_new, list)
        assert isinstance(y_new, np.ndarray) or isinstance(y_new, list)
        assert len(x_new) == len(y_new)
        if append:
            if noise_variances_new is not None:
                assert isinstance(noise_variances_new, np.ndarray) or isinstance(noise_variances_new, list)
                if isinstance(noise_variances_new, np.ndarray): self.fvgp_noise_variances = (
                    np.row_stack([self.fvgp_noise_variances, noise_variances_new]))
                if isinstance(noise_variances_new, list): self.fvgp_noise_variances = (
                    self.fvgp_noise_variances + noise_variances_new)
            if isinstance(x_new, np.ndarray): self.fvgp_x_data = np.row_stack([self.fvgp_x_data, x_new])
            if isinstance(x_new, list): self.fvgp_x_data = self.fvgp_x_data + x_new
            if isinstance(y_new, np.ndarray): self.fvgp_y_data = np.row_stack([self.fvgp_y_data, y_new])
            if isinstance(y_new, list): self.fvgp_y_data = self.fvgp_y_data + y_new
        else:
            self.fvgp_noise_variances = noise_variances_new
            self.fvgp_x_data = x_new
            self.fvgp_y_data = y_new

        ##########################################
        #######prepare value positions############
        ##########################################
        if output_positions_new is None:
            output_positions_new = self._compute_standard_output_positions(len(x_new))
        ######################################
        #####transform to index set###########
        ######################################
        x_data, y_data, noise_variances = self._transform_index_set2(x_new, y_new, noise_variances_new,
                                                                     output_positions_new)
        super().update_gp_data(x_data, y_data, noise_variances, append=append, gp_rank_n_update=gp_rank_n_update)
        self.output_positions = self.output_positions + output_positions_new

    ################################################################################################
    def _compute_standard_output_positions(self, point_number):
        value_pos = []
        for j in range(point_number):
            value_pos.append(np.arange(0, self.output_num))
        return value_pos

    ################################################################################################
    @staticmethod
    def _transform_index_set2(x_data, y_data, noise_variances, output_positions):
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list)
        assert isinstance(y_data, np.ndarray) or isinstance(y_data, list)
        assert isinstance(output_positions, list)
        assert len(x_data) == len(y_data) == len(output_positions)
        if noise_variances is not None: assert len(noise_variances) == len(y_data)
        new_x_data = []
        new_y_data = []
        if noise_variances is not None: new_variances = []
        else: new_variances = None
        for i in range(len(x_data)):
            assert len(y_data[i]) == len(output_positions[i])
            if noise_variances is not None: assert len(noise_variances[i]) == len(output_positions[i])
            for j in range(len(y_data[i])):
                if isinstance(x_data, np.ndarray): new_x_data.append(np.append(x_data[i], output_positions[i][j]))
                else: new_x_data.append([x_data[i], output_positions[i][j]])
                new_y_data.append(y_data[i][j])
                if new_variances is not None:
                    new_variances.append(noise_variances[i][j])
        if isinstance(x_data, np.ndarray): new_x_data = np.asarray(new_x_data)
        new_y_data = np.asarray(new_y_data)
        if new_variances is not None: new_variances = np.asarray(new_variances)

        assert isinstance(new_x_data, list) or isinstance(new_x_data, np.ndarray)
        assert isinstance(new_y_data, np.ndarray) and np.ndim(new_y_data) == 1
        if new_variances is not None: assert isinstance(new_variances, np.ndarray) and np.ndim(new_variances) == 1
        return new_x_data, new_y_data, new_variances

    ################################################################################################
