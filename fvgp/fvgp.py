#!/usr/bin/env python
import numpy as np
import warnings
from .gp import GP


class fvGP(GP):
    """
    This class provides all the tools for a multi-task Gaussian Process (GP).
    This class allows for full HPC support for training. After initialization, this
    class provides all the methods described for the GP class.

    V ... number of input points

    Di... input space dimensionality

    Do... output space dimensionality

    No... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The main logic of fvGP is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best
    way to give the user optimal control and power. At various instances, for instances
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input ``x'' is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [[0],[1]], the input to the mean, kernel, and noise function might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]

    This has to be understood and taken into account when customizing fvGP for multi-task
    use.

    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space (D). If the input is non-Euclidean, the input dimensionality will be ignored.
    output_space_dim : int
        Integer specifying the number of dimensions of the output space. Most often 1.
        This is not the number of outputs/tasks.
        For instance, a spectrum as output at each input is itself a function over a 1d space but has many outputs.
    output_number : int
        Number of output values.
    x_data : np.ndarray
        The input point positions. Shape (V x D), where D is the `'input_space_dim'`.
    y_data : np.ndarray
        The values of the data points. Shape (V,No).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is an array that specifies the right number of
        initial hyperparameters for the default kernel, which is
        a deep kernel with two layers of width
        fvgp.fvGP.gp_deep_kernel_layer_width. If you specify
        another kernel, please provide
        init_hyperparameters.
    hyperparameter_bounds : np.ndarray, optional
        A 2d numpy array of shape (N x 2), where N is the number of needed hyperparameters.
        The default is None, in that case hyperparameter_bounds have to be specified
        in the train calls or default bounds are used. Those only work for the default kernel.
    output_positions : np.ndarray, optional
        A 3-D numpy array of shape (U x output_number x output_dim), so that for each measurement position, the outputs
        are clearly defined by their positions in the output space. The default is
        np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
        point in the input space. The default is only permissible if output_dim is 1.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the data
        `y_data` in form of a point-wise variance. Shape y_data.shape.
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
        The input x1 is a N1 x Di+Do array of positions, x2 is a N2 x Di+Do
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperparameters are initialized, and
        obj is an `fvgp.GP` instance. The default is a deep kernel with 2 hidden layers and
        a width of fvgp.fvGP.gp_deep_kernel_layer_width.
    gp_deep_kernel_layer_width : int, optional
        If no kernel is provided, fvGP will use a deep kernel of depth 2 and width gp_deep_kernel_layer_width.
        If a user defined kernel is provided this parameter is irrelevant. The default is 5.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``gp_kernel_function'' with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input x1 (a N1 x Di+Do array of positions),
        x2 (a N2 x Di+Do array of positions),
        hyperparameters, and a
        `fvgp.GP` instance. The default is a finite difference calculation.
        If 'ram_economy' is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If 'ram economy' is False,the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See 'ram_economy'.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+Do), hyperparameters
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_mean_function'' at a set of input positions with respect to
        the hyperparameters. It accepts as input an array of positions (of size N1 x Di+Do), hyperparameters
        and a `fvgp.GP` instance. The return value is a 2d array of shape (len(hyperparameters) x N1). If None is
        provided, either zeros are returned since the default mean function does not depend on hyperparameters, or a
        finite-difference approximation is used if ``gp_mean_function'' is provided.
    gp_noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input x is a numpy array of shape (N x Di+Do). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a fvgp.fvGP instance.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_noise_function'' at an input position with respect
        to the hyperparameters. It accepts as input an array of positions (of size N x Di+Do),
        hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 3-D array of shape
        (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not depend on hyperparameters.
        If ``gp_noise_function'' is provided but no gradient function,
        a finite-difference approximation will be used.
        The same rules regarding ram economy as for the kernel definition apply here.
    normalize_y : bool, optional
        If True, the data values ``y_data'' will be normalized to max(y_data) = 1, min(y_data) = 0.
        The default is False.
        Variances will be updated accordingly.
    sparse_mode : bool, optional
        When sparse_mode is enabled, the algorithm will use a user-defined kernel function or,
        if that's not provided, an anisotropic Wendland kernel
        and check for sparsity in the prior covariance. If sparsity is present,
        sparse operations will be used to speed up computations.
        Caution: the covariance is still stored at first in a dense format. For more extreme scaling,
        check out the gp2Scale option.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million datapoints. If gp2Scale is used, the default kernel is an anisotropic Wendland
        kernel which is compactly supported. The noise function will have
        to return a scipy.sparse matrix instead of a numpy array. There are a few more things
        to consider (read on); this is an advanced option.
        If no kernel is provided, the compute_device option should be revisited. The kernel will
        use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale to distribute covariance computations over. Has to contain at least 3 workers.
        On HPC architecture, this client is provided by the jobscript. Please have a look at the examples.
        A local client is used as default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    store_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance matrix
        after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster.
        For larger problems (>2000 data points), the use of inversion should be avoided due to
        computational instability and costs. The default is
        True. Note, the training will always use Cholesky or LU decomposition instead of the inverse
        for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
        If sparse_mode or gp2Scale is used, store_inv will be set to False.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the marginal log_likelihood
        is/are used for the training.
        If True, components of the derivative of the marginal log-likelihood are calculated
        subsequently, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (or noise function) with
        respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for ram_economy=True it should be of
        the form f(x1[, x2], direction, hyperparameters, obj)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ram_economy=False, the function should be of the form f(x1[, x2,] hyperparameters, obj)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters. CAUTION:
        This array will be stored and is very large.
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
    fvgp_x_data : np.ndarray
        Datapoint positions as seen by fvgp
    fvgp_y_data : np.ndarray
        Datapoint values as seen by fvgp
    noise_variances : np.ndarray
        Datapoint observation (co)variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP
    KVinv : np.ndarray
        If enabled, the inverse of the prior covariance + nose matrix V
        inv(K+V)
    KVlogdet : float
        logdet(K+V)
    """

    def __init__(
        self,
        input_space_dim,
        output_space_dim,
        output_number,
        x_data,
        y_data,
        init_hyperparameters=None,
        hyperparameter_bounds=None,
        output_positions=None,
        noise_variances=None,
        compute_device="cpu",
        gp_kernel_function=None,
        gp_deep_kernel_layer_width=5,
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
        info=False,
        ):

        self.orig_input_space_dim = input_space_dim
        self.output_num, self.output_dim = output_number, output_space_dim
        ###check the output dims

        if not isinstance(x_data,np.ndarray):
            raise Exception("Multi-task GPs on non-Euclidean input spaces are not implemented yet.")


        if np.ndim(y_data) == 1: raise ValueError("The output number is 1, you can use GP for single-task GPs")
        if output_number != len(y_data[0]): raise ValueError("The output number is not in agreement with the data values given")
        if output_space_dim == 1 and isinstance(output_positions, np.ndarray) == False:
            self.output_positions = self._compute_standard_output_positions(len(x_data))
        elif self.output_dim > 1 and not isinstance(output_positions, np.ndarray):
            raise Exception(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the fvGP class")
        else:
            self.output_positions = output_positions

        self.iset_dim = self.orig_input_space_dim + self.output_dim
        ####transform the space
        self.fvgp_x_data = x_data
        self.fvgp_y_data = y_data
        self.fvgp_noise_variances = noise_variances
        x_data, y_data, noise_variances = self._transform_index_set(x_data,y_data,noise_variances,self.output_positions)
        init_hps = init_hyperparameters

        if gp_kernel_function is None:
            gp_kernel_function = self._default_multi_task_kernel
            try:
                from .deep_kernel_network import Network
            except: raise Exception("You have not specified a kernel and the default kernel will be used. \n \
                    The default kernel needs pytorch to be installed manually.")
            self.gp_deep_kernel_layer_width = gp_deep_kernel_layer_width
            self.n = Network(self.iset_dim, gp_deep_kernel_layer_width)
            number_of_hps = int(2. * self.iset_dim * gp_deep_kernel_layer_width + gp_deep_kernel_layer_width**2 + 2.*gp_deep_kernel_layer_width + self.iset_dim + 2.)
            self.hps_bounds = np.zeros((number_of_hps,2))
            self.hps_bounds[0] = np.array([np.var(y_data)/10.,np.var(y_data)*10.])
            self.hps_bounds[1] = np.array([(np.max(x_data) - np.min(x_data)) / 100., (np.max(x_data) - np.min(x_data)) * 100.])
            self.hps_bounds[2:] = np.array([-1.,1.])
            init_hps = np.random.uniform(low = self.hps_bounds[:,0], high = self.hps_bounds[:,1],size = len(self.hps_bounds))
            warnings.warn("Hyperparameter bounds have been initialized automatically \
                    \n for the default kernel in fvgp. They will automatically used for the training.\
                    \n However, you can also define and provide new bounds.")
            hyperparameter_bounds = self.hps_bounds


        ####init GP
        super().__init__(
                self.iset_dim,
                x_data,
                y_data,
                init_hyperparameters=init_hps,
                hyperparameter_bounds=hyperparameter_bounds,
                noise_variances=noise_variances,
                compute_device=compute_device,
                gp_kernel_function=gp_kernel_function,
                gp_kernel_function_grad=gp_kernel_function_grad,
                gp_mean_function=gp_mean_function,
                gp_mean_function_grad=gp_mean_function_grad,
                gp_noise_function=gp_noise_function,
                gp_noise_function_grad=gp_noise_function_grad,
                sparse_mode=sparse_mode,
                gp2Scale=gp2Scale,
                gp2Scale_dask_client=gp2Scale_dask_client,
                gp2Scale_batch_size=gp2Scale_batch_size,
                store_inv=store_inv,
                normalize_y=normalize_y,
                ram_economy=ram_economy,
                args=args,
                info=info)

   ################################################################################################
    def update_gp_data(
        self,
        x_data,
        y_data,
        output_positions = None,
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
            The values of the data points. Shape (V,Do).
        output_positions : np.ndarray, optional
            A 3-D numpy array of shape (U x output_number x output_dim), so that
            for each measurement position, the outputs
            are clearly defined by their positions in the output space.
            The default is np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
            point in the input space. The default is only permissible if output_dim is 1.
        noise_variances : np.ndarray, optional
            An numpy array defining the uncertainties in the data `y_data`
            in form of a point-wise variance. Shape (y_data).
            Note: if no variances are provided here, the noise_covariance callable
            will be used; if the callable is not provided the noise variances
            will be set to `abs(np.mean(y_data)) / 100.0`. If you provided a noise function,
            the noise_variances will be ignored.
        """
        self.fvgp_x_data = np.array(x_data)
        self.fvgp_y_data = np.array(y_data)
        ##########################################
        #######prepare value positions############
        ##########################################
        if self.output_dim == 1 and isinstance(output_positions, np.ndarray) == False:
            self.output_positions = self._compute_standard_output_positions(len(x_data))
        elif self.output_dim > 1 and isinstance(output_positions, np.ndarray) == False:
            raise ValueError(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the fvGP class. EXIT"
            )
        else:
            self.output_positions = output_positions
        ######################################
        #####transform to index set###########
        ######################################
        x_data, y_data, noise_variances = self._transform_index_set(x_data,y_data,noise_variances, self.output_positions)
        super().update_gp_data(self.x_data, self.y_data, noise_variances)

   ################################################################################################
    def _compute_standard_output_positions(self, point_number):
        value_pos = np.zeros((point_number, self.output_num, self.output_dim))
        for j in range(self.output_num):
            value_pos[:, j, :] = j
        return value_pos

   ################################################################################################
    def _transform_index_set(self, x_data, y_data, noise_variances, output_positions):
        point_number = len(x_data)
        new_points = np.zeros((point_number * self.output_num, self.iset_dim))
        new_values = np.zeros((point_number * self.output_num))
        if noise_variances is not None: new_variances = np.zeros((point_number * self.output_num))
        else: new_variances = None
        for i in range(self.output_num):
            new_points[i * point_number : (i + 1) * point_number] = \
            np.column_stack([x_data, output_positions[:, i, :]])
            new_values[i * point_number : (i + 1) * point_number] = \
            y_data[:, i]
            if noise_variances is not None: new_variances[i * point_number : (i + 1) * point_number] = \
            noise_variances[:, i]
        return new_points, new_values, new_variances

   ################################################################################################
    def _default_multi_task_kernel(self, x1, x2, hps, obj):  # pragma: no cover
        signal_var = hps[0]
        length_scale = hps[1]
        hps_nn = hps[2:]
        w1_indices = np.arange(0,self.gp_deep_kernel_layer_width * self.iset_dim)
        last = self.gp_deep_kernel_layer_width * self.iset_dim
        w2_indices = np.arange(last, last + self.gp_deep_kernel_layer_width**2)
        last = last + self.gp_deep_kernel_layer_width**2
        w3_indices = np.arange(last,last + self.gp_deep_kernel_layer_width * self.iset_dim)
        last = last + self.gp_deep_kernel_layer_width * self.iset_dim
        b1_indices = np.arange(last,last + self.gp_deep_kernel_layer_width)
        last = last + self.gp_deep_kernel_layer_width
        b2_indices = np.arange(last,last + self.gp_deep_kernel_layer_width)
        last = last + self.gp_deep_kernel_layer_width
        b3_indices = np.arange(last, last + self.iset_dim)

        self.n.set_weights(hps_nn[w1_indices].reshape(self.gp_deep_kernel_layer_width ,self.iset_dim),
                           hps_nn[w2_indices].reshape(self.gp_deep_kernel_layer_width , self.gp_deep_kernel_layer_width),
                           hps_nn[w3_indices].reshape(self.iset_dim,self.gp_deep_kernel_layer_width))
        self.n.set_biases(hps_nn[b1_indices].reshape(self.gp_deep_kernel_layer_width ),
                          hps_nn[b2_indices].reshape(self.gp_deep_kernel_layer_width ),
                          hps_nn[b3_indices].reshape(self.iset_dim))
        x1_nn = self.n.forward(x1)
        x2_nn = self.n.forward(x2)
        d = obj._get_distance_matrix(x1_nn,x2_nn)
        k = signal_var * obj.matern_kernel_diff1(d,length_scale)
        return k
