import warnings
import weakref
import numpy as np
from loguru import logger
from distributed import Client
from scipy.stats import norm
from .gp_prior import GPprior
from .gp_data import GPdata
from .gp_marginal_likelihood import GPMarginalLikelihood
from .gp_likelihood import GPlikelihood
from .gp_training import GPtraining
from .gp_posterior import GPposterior
from .gp_kv import GPkv
import importlib
warnings.simplefilter("once", UserWarning)

# Tracks live GP instances per dask client (gp2Scale mode only).  Used to detect
# the case where a user creates a second GP on a client that still has a live GP,
# which triggers race conditions between the new init scatter and the pending
# `_dec_ref` callbacks from the previous GP's scatter activity.
_GP_INSTANCES_PER_CLIENT = weakref.WeakValueDictionary()

# TODO: also search below "TODO"
# Appends and rank_n_updates for gp2Scale are not yet fully tested. Have to check the compute graph and test (what does rank_n_update even mean for the different modes? ). 
# Caching a preconditioner should depend on how much the hps changed

class GP:
    """
    This class provides all the tools for a single-task Gaussian Process (GP).
    Use fvGP for multi-task GPs. However, the fvGP class inherits all methods from this class.
    This class allows full HPC distributed training via the ``hgdl`` package, large-scale sparse GPs
    via ``gp2Scale`` and offers GPU support.

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
        In this case, both the index set and the input space dim are set to 1.
    y_data : np.ndarray
        The values of the data points. Shape (V) or (V, N). If shape (V,N) the algorithm will run N independent GPs.
        This is not to be confused with multi-task learning. In this case, all GPs have to have the same prior
        mean function.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). If ``gp2Scale`` is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
        The full hyperparameter vector is passed to the kernel, mean, and noise callables,
        but the index ranges used by each callable are **disjoint and user-defined**.
        Each callable must only read the indices reserved for it. The gradient
        computation relies on this: when a hyperparameter index belongs to the mean
        function its kernel derivative is assumed zero, and vice versa.
    noise_variances : np.ndarray, optional
        A numpy array defining the uncertainties/noise in the
        ``y_data`` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to ``abs(np.mean(y_data)) / 100.0``. If
        noise covariances are required (correlated noise), make use of the ``noise_function``.
        Only provide a noise function OR ``noise_variances``, not both.
        If the shape of ``y_data`` is (V,N) the noise is still of shape (V), e.g., the outputs
        must have the same noise in this scenario.
    compute_device : str, optional
        One of ``cpu`` or ``gpu``, determines how linear algebra computations are executed. The default is ``cpu``.
        For ``gpu``, pytorch or cupy has to be installed manually. For advanced options see ``args``.
        If ``gp2Scale`` is enabled but no kernel is provided, the choice of the ``compute_device``
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, [args]).
        ``args`` is optional and is used to make :py:attr:`fvgp.GP.args` available.
        The input ``x1`` is a N1 x D array of positions, ``x2`` is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        The default is a stationary anisotropic kernel
        (:py:meth:`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a matrix, an N1 x N2 numpy array.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the kernel (disjoint from mean and noise indices).
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``kernel_function`` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input ``x1`` (a N1 x D array of positions),
        ``x2`` (a N2 x D array of positions) and
        ``hyperparameters`` (a 1d array of length D+1 for the default kernel).
        The default is an analytical gradient for the default kernel or a finite difference calculation otherwise.
        If ``ram_economy`` is True, the function's input is x1, x2, hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If ``ram_economy`` is ``False``, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See ``ram_economy``.
    prior_mean_function : Callable, optional
        A function f(x, hyperparameters, [args]) that evaluates the prior mean at a set of input position.
        It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        Optionally, the third argument ``args`` can be defined.
        The return value is a 1d array of length N1.
        If prior_mean_function is not provided, :py:meth:`fvgp.GP._default_mean_function` is used,
        which is the average of the ``y_data``.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the mean function (disjoint from kernel and noise indices).
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``prior_mean_function`` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D) and hyperparameters
        (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if ``prior_mean_function`` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters, [args]) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The third argument ``args`` is optional.
        The input ``x`` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the noise function (disjoint from kernel and mean indices).
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``noise_function``
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if ``noise_function`` is provided but no noise function gradient,
        a finite-difference approximation will be used.
        The same rules regarding ``ram_economy`` as for the kernel definition apply here.
        That means the function will have an additional ``direction`` parameter.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the ``compute_device`` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale, asynchronous training,a nd certain linear algebra operations.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    linalg_mode : str, optional
        Controls the linear-algebra backend used to solve (K+V)x=b and compute log|K+V|.
        The default is ``None``, which selects ``"Chol"`` for standard GPs and automatically
        picks the best sparse mode for gp2Scale GPs.

        **Recommended for standard (non-gp2Scale) GPs:**

        * ``"Chol"`` *(default)* — Cholesky factorization; numerically stable and memory-efficient.
        * ``"CholInv"`` — Cholesky factorization, then explicitly stores the inverse; speeds up posterior
          covariance evaluation 3–10×. Avoid for datasets larger than ~5 000 points due to memory
          and numerical cost. Training always uses the Cholesky factor for stability.
        * ``"Inv"`` — computes and stores the explicit inverse directly (no Cholesky) even during training. Only suitable for
          very small datasets where posterior covariance is computed many times.

        **Specialized for gp2Scale (sparse covariance matrices):**

        * ``"sparseLU"`` — sparse LU factorization; good default for sparse systems up to ~50 000 points.
        * ``"sparseCG"`` — sparse conjugate-gradient iterative solver.
        * ``"sparseMINRES"`` — sparse MINRES iterative solver.
        * ``"sparseSolve"`` — direct sparse solve via scipy.
        * ``"sparseCGpre"`` — preconditioned conjugate-gradient. The preconditioner type
          is selected by ``args["sparse_preconditioner_type"]`` (default ``"ilu"``;
          also compiled incomplete Cholesky ``"ichol"``/``"ic"``/``"incomplete_cholesky"``,
          zero-fill ``"ichol0"``, legacy Python IC(0) ``"native_ic"``/``"native_ichol"``,
          ``"block_jacobi"``, ``"schwarz"``/``"additive_schwarz"``, or ``"amg"``
          (requires pyamg). The compiled IC options require the optional ``ilupp`` package.
        * ``"sparseMINRESpre"`` — preconditioned MINRES; same preconditioner choices.
        * ``"sparseCGpre_<type>"`` / ``"sparseMINRESpre_<type>"`` — shortcut that sets
          ``args["sparse_preconditioner_type"]`` to ``<type>`` (e.g. ``"sparseCGpre_amg"``).

        **Custom solver (any GP):**

        Pass an iterable of three callables ``[f_factor, f_solve, f_logdet]``:

        * ``f_factor(K)`` — receives the covariance matrix and returns a factorization object
          (or the matrix itself if no factorization is needed).
        * ``f_solve(obj, b)`` — solves the linear system and returns the solution vector.
        * ``f_logdet(obj)`` — returns the log-determinant as a scalar.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for ``ram_economy=True`` it should be
        of the form f(x, hyperparameters, direction)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ``ram_economy=False``, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args: dict, optional
        Advanced options. Recognized keys are:

        Stochastic-Lanczos logdet (sparse modes):

        - "random_logdet_lanczos_degree" : int; default = 20
        - "random_logdet_error_rtol" : float; default = 0.01
        - "random_logdet_verbose" : True/False; default = False
        - "random_logdet_print_info" : True/False; default = False
        - "random_logdet_lanczos_compute_device" : str; default = "cpu"/"gpu"

        Sparse iterative solver tolerances and iteration limits:

        - "sparse_cg_tol" : float; default = 1e-5
        - "sparse_minres_tol" : float; default = 1e-5
        - "sparse_cg_maxiter" : int; default = None (use scipy default)
        - "sparse_minres_maxiter" : int; default = None (use scipy default)
        - "sparse_krylov_maxiter" : int; default = None (applies to both if the
          solver-specific key is not set)
        - "sparse_block_krylov" : True/False; default = False — use a block CG
          variant when there are multiple RHS columns
        - "sparse_krylov_mode" : "single"/"block"; equivalent toggle
        - "sparse_krylov_block_size" : int — RHS block size for block CG

        Iterative-solver acceleration (``sparseCG``/``sparseMINRES`` and the
        ``*pre`` variants):

        - "sparse_krylov_warm_start" : True/False; default = False — feed the
          previous training iteration's ``KVinvY`` as ``x0`` to the next solve
        - "sparse_preconditioner_type" : str; default = "ilu". One of "ilu",
          "ichol"/"ic"/"incomplete_cholesky", "ichol0", "native_ic"/"native_ichol",
          "block_jacobi", "schwarz"/"additive_schwarz", "amg" (requires pyamg)
        - "sparse_preconditioner_refresh_interval" : int; default = 1 —
          reuse the cached preconditioner for up to N consecutive solves
          before rebuilding. ``set_KV`` always force-refreshes.
        - "sparse_preconditioner_block_size" : int — block size for block_jacobi
          and additive_schwarz partitions
        - "sparse_preconditioner_schwarz_overlap" : int — overlap layers for
          additive Schwarz
        - "sparse_preconditioner_drop_tol" / "sparse_preconditioner_fill_factor"
          — forwarded to scipy ``spilu`` for "ilu"
        - "sparse_preconditioner_ichol_fill_in" / "sparse_preconditioner_ichol_threshold"
          — forwarded to ``ilupp`` thresholded IC for "ichol"
        - "sparse_preconditioner_amg_*" — forwarded to pyamg
          (``max_levels``, ``max_coarse``, ``strength``, ``cycle``, etc.)
        - "sparse_preconditioner_shift" / "_growth" / "_attempts" — diagonal
          shift retry knobs for legacy "native_ic"/"native_ichol" / "block_jacobi" / "additive_schwarz" when
          a local Cholesky encounters a non-PD block

        Practical sparse-solver guidance:

        - Keep compact-support covariance matrices genuinely sparse before using global
          factor-based preconditioners. If the kernel support is too broad, ILU/IC
          failures are usually memory/fill failures, not proof that Krylov solvers are bad.
        - Prefer ``sparseCGpre_*`` as the primary path for covariance systems, which are
          symmetric positive definite in the usual case. ``MINRES`` can be useful as a
          comparison, but it may need a stricter ``sparse_minres_tol`` to satisfy a raw
          relative residual check.
        - Sweep preconditioner parameters at the target scale. For ILU, ``drop_tol`` and
          ``fill_factor`` control a memory/solve-time tradeoff; a slightly more expressive
          factor can cost only marginally more to build but reduce solve time substantially.
        - For repeated nearby K+V updates, enable ``sparse_krylov_warm_start`` and reuse
          preconditioners with ``sparse_preconditioner_refresh_interval``. The best refresh
          interval is problem-dependent because preconditioner build cost can be comparable
          to an unconditioned solve.

        Cholesky compute-device routing:

        - "Chol_factor_compute_device" : str; default = "cpu"/"gpu"
        - "update_Chol_factor_compute_device": str; default = "cpu"/"gpu"
        - "Chol_solve_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_logdet_compute_device" : str; default = "cpu"/"gpu"

        GPU backend:

        - "GPU_engine" : "torch"/"cupy"; default = first available
        - "GPU_device" : str; e.g. "cuda:1" or "mps"
        - "GPU_device_index" : int — explicit CUDA device index

        All other keys will be stored and are available as part of the object instance and
        in kernel, mean, and noise functions.


    Attributes
    ----------
    x_data : np.ndarray or list
        Datapoint positions.
    y_data : np.ndarray
        Datapoint values.
    noise_variances : np.ndarray
        Datapoint observation variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP.
    m : np.ndarray
        Current prior mean vector.
    V : np.ndarray
        the noise covariance matrix or a vector.
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
        dask_client=None,
        gp2Scale_batch_size=10000,
        linalg_mode=None,
        ram_economy=False,
        args=None
    ):

        assert isinstance(noise_variances, np.ndarray) or noise_variances is None, "wrong format in noise_variances"
        assert init_hyperparameters is None or isinstance(init_hyperparameters,
                                                          np.ndarray), "wrong init_hyperparameters"
        assert isinstance(compute_device, str), "wrong format in compute_device"
        assert callable(kernel_function) or kernel_function is None, "wrong format in kernel_function"
        assert callable(kernel_function_grad) or kernel_function_grad is None, "wrong format in kernel_function_grad"
        assert callable(noise_function) or noise_function is None, "wrong format in noise_function"
        assert callable(noise_function_grad) or noise_function_grad is None, "wrong format in noise_function"
        assert callable(prior_mean_function) or prior_mean_function is None, "wrong format in prior_mean_function"
        assert callable(prior_mean_function_grad) or prior_mean_function_grad is None, \
            "wrong format in prior_mean_function"
        assert len(x_data) == len(y_data), "x_data and y_data do not have the same lengths."

        if args is None: args = {}
        hyperparameters = init_hyperparameters
        # Check gp2Scale
        dask_client = self.initialize_gp2Scale_dask_client(gp2Scale, dask_client)

        # Race-condition guard: in gp2Scale mode, only one GP can be alive per dask
        # client.  Sharing a client between live GPs causes the new GP's init scatter
        # to race against the previous GP's pending `_dec_ref` callbacks, surfacing as
        # `FutureCancelledError` or `KeyError` from the scheduler.
        if gp2Scale and dask_client is not None:
            existing = _GP_INSTANCES_PER_CLIENT.get(dask_client.id)
            if existing is not None and existing is not self:
                raise Exception(
                    f"Another GP instance is already active on this dask client "
                    f"(client.id={dask_client.id!r}). Sharing a dask client between "
                    f"multiple live GPs in gp2Scale mode triggers race conditions "
                    f"in the scheduler's scatter reference counting.\n"
                    f"To reuse the same client for a sequence of GPs, destroy the "
                    f"previous one first:\n"
                    f"    import gc\n"
                    f"    del previous_gp\n"
                    f"    gc.collect()\n"
                    f"    client.run(lambda: None)  # flush pending releases\n"
                    f"Or use a fresh dask client per GP."
                )

        ########################################
        ###init data instance [tier 1]##########
        ########################################
        self.data = GPdata(x_data, y_data,
                           args=args,
                           noise_variances=noise_variances,
                           ram_economy=ram_economy,
                           gp2Scale=gp2Scale,
                           compute_device=compute_device,
                           dask_client=dask_client)
        ########################################
        # prepare initial hyperparameters and bounds
        if self.data.Euclidean:
            if callable(kernel_function) or callable(prior_mean_function) or callable(noise_function):
                if init_hyperparameters is None: raise Exception(
                    "You have provided callables for kernel, mean, or noise functions but no "
                    "initial hyperparameters.")
            else:
                if init_hyperparameters is None:
                    hyperparameters = np.ones((self.index_set_dim + 1))
                    warnings.warn("Hyperparameters initialized to a vector of ones.")
        else:
            hyperparameters = init_hyperparameters

        # warn if they could not be prepared
        if hyperparameters is None:
            raise Exception("'init_hyperparameters' not provided and could not be calculated. Please provide them ")

        if compute_device == 'gpu':
            if not importlib.util.find_spec("torch") and not importlib.util.find_spec("cupy"):
                warnings.warn("You have specified the 'gpu' as your compute device. You need to install pytorch or cupy"
                              " manually for this to work.")

        ##########################################
        #######prepare training [tier 2]##########
        ##########################################
        self.trainer = GPtraining(self.data, hyperparameters)
        ########################################
        ###init prior instance [tier 3]#########
        ########################################
        self.prior = GPprior(self.data,
                             self.trainer,
                             kernel=kernel_function,
                             prior_mean_function=prior_mean_function,
                             kernel_grad=kernel_function_grad,
                             prior_mean_function_grad=prior_mean_function_grad,
                             gp2Scale_batch_size=gp2Scale_batch_size,
                             )
        ########################################
        ###init likelihood instance [tier 3]####
        ########################################
        self.likelihood = GPlikelihood(self.data,
                                       self.trainer,
                                       noise_function=noise_function,
                                       noise_function_grad=noise_function_grad,
                                       )

        ##########################################
        #######prepare KV object [tier 3]#########
        ##########################################
        self.kv = GPkv(
            self.data,
            self.prior,
            self.likelihood,
            linalg_mode=linalg_mode,
        )
        ##########################################
        #######prepare marg. likelih. [tier 4]####
        ##########################################
        self.marginal_likelihood = GPMarginalLikelihood(
            self.data,
            self.prior,
            self.likelihood,
            self.trainer,
            self.kv)

        ##########################################
        #######prepare posterior [tier 4]#########
        ##########################################
        self.posterior = GPposterior(self.data,
                                     self.prior,
                                     self.trainer,
                                     self.kv,
                                     self.likelihood)

        # Register this instance for the cross-instance race-condition guard above.
        # Entry is removed automatically when self is garbage-collected.
        if gp2Scale and dask_client is not None:
            _GP_INSTANCES_PER_CLIENT[dask_client.id] = self

    #########PROPERTIES#########################################
    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def noise_variances(self):
        return self.data.noise_variances

    @property
    def index_set_dim(self):
        return self.data.index_set_dim

    @property
    def input_set_dim(self):
        return self.data.input_set_dim

    @property
    def mcmc_info(self):
        return self.trainer.mcmc_info

    @property
    def args(self):
        return self.data.args

    @args.setter
    def args(self, args):
        self.data.args = args

    @property
    def K(self):
        return self.prior.K

    @property
    def m(self):
        return self.prior.m

    @property
    def V(self):
        return self.likelihood.V

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters

    @property
    def gp2Scale(self):
        return self.data.gp2Scale
    
    @property
    def dask_client(self):
        return self.data.dask_client

    ###############################################################
    def set_args(self, new_args):
        """
        Use this function to change the arguments for the GP.

        Note
        ----
        New ``args`` do not invalidate cached state (``K``, ``m``, ``V``, factorizations,
        ``KVinvY``). If your ``kernel``, ``prior_mean_function``, or ``noise_function``
        consumes ``args``, the new values will only be picked up the next time those
        callables are invoked: a call to :py:meth:`set_hyperparameters`,
        :py:meth:`update_gp_data` with ``append=False``, a fresh :py:meth:`train`,
        or a posterior call with an explicit ``hyperparameters`` argument.
        For an explicit flush, call ``set_hyperparameters(self.hyperparameters)``.

        Parameters
        ----------
        new_args : dict
            The new advanced settings.
        """
        self.args = new_args

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
        self.trainer.hyperparameters = hps
        self.prior.update_state_hyperparameters()
        self.likelihood.update_state()
        self.kv.update_state_hyperparameters()

    def update_gp_data(
        self,
        x_new,
        y_new,
        noise_variances_new=None,
        append=True,
        rank_n_update=None
    ):
        """
        This function updates the data in the gp object instance.
        The data will only be overwritten if ``append=False``, otherwise
        the data will be appended. This is a change from earlier versions.
        Now, the default is not to overwrite the existing data.


        Parameters
        ----------
        x_new : np.ndarray or list
            The point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.index_set_dim`.
            If dealing with non-Euclidean inputs
            ``x_new`` should be a list, not a numpy array.
        y_new : np.ndarray
            The values of the data points. Shape (V).
        noise_variances_new : np.ndarray, optional
            A numpy array defining the uncertainties in the data ``y_data`` in form of a point-wise variance.
            Shape(y_new)==Shape(noise_variances_new).
            Note: if no variances are provided here, the noise_covariance
            callable will be used; if the callable is not provided the noise variances
            will be set to ``abs(np.mean(y_data)) / 100.0``. If you provided a noise function at initialization,
            the ``noise_variances_new`` will be ignored.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default=True.
            In the default case, data will be appended.
        rank_n_update : bool, optional
            Indicates whether the GP marginal likelihood should be rank-n updated or recomputed. The default
            is ``rank_n_update=append``, meaning if data is only appended, the rank_n_update will
            be performed.
        """
        assert isinstance(x_new, list) or isinstance(x_new, np.ndarray), "wrong format in x_new"
        assert isinstance(y_new, np.ndarray) and (np.ndim(y_new) == 1 or np.ndim(y_new) == 2), "wrong format in y_new"
        assert isinstance(noise_variances_new, np.ndarray) or noise_variances_new is None, \
            "wrong format in noise_variances_new"
        assert len(x_new) == len(y_new), "updated x and y do not have the same lengths."
        if rank_n_update is None: rank_n_update = append
        if not append and rank_n_update:
            warnings.warn("`rank_n_update=True` is invalid when `append=False` "
                          "(the previous factorization belongs to data that no longer "
                          "exists). Forcing `rank_n_update=False`.")
            rank_n_update = False
        # update data
        self.data.update(x_new, y_new, noise_variances_new, append=append)

        # update prior
        if append: self.prior.augment_state_data()
        else:self.prior.update_state_data()

        # update likelihood
        self.likelihood.update_state()

        # update kv state
        self.kv.update_state_data(rank_n_update)
        ##########################################

    def _get_default_hyperparameter_bounds(self):
        """
        This function will create hyperparameter bounds for the default kernel based
        on the data only.


        Returns
        -------
        hyperparameter bounds for the default kernel : np.ndarray
        """
        if not self.data.Euclidean: raise Exception("Please provide custom hyperparameter bounds to "
                                                    "the training in the non-Euclidean setting")
        if len(self.hyperparameters) != self.index_set_dim + 1:
            raise Exception("Please provide custom hyperparameter_bounds when kernel, mean or noise"
                            " functions are customized")
        hyperparameter_bounds = np.zeros((self.index_set_dim + 1, 2))
        hyperparameter_bounds[0] = np.array([np.var(self.y_data) / 100., np.var(self.y_data) * 10.])
        for i in range(self.index_set_dim):
            range_xi = np.max(self.x_data[:, i]) - np.min(self.x_data[:, i])
            hyperparameter_bounds[i + 1] = np.array([range_xi / 100., range_xi * 10.])
        assert isinstance(hyperparameter_bounds, np.ndarray) and np.ndim(hyperparameter_bounds) == 2, \
            "hyperparameter_bounds must be a 2-d np.ndarray"
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
              max_iter=10000,
              mcmc_prior=None,
              mcmc_prop_distrs="normal",
              mcmc_args={},
              local_optimizer="L-BFGS-B",
              global_optimizer="genetic",
              constraints=(),
              dask_client=None,
              info=False,
              asynchronous=False):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be ``hgdl`` and
        providing a dask client. Methods ``hgdl``, ``mcmc``, and ``adam`` can also be run asynchronously.
        The GP prior will automatically be updated with the new hyperparameters after the training or when
        the :py:meth:`update_hyperparameters` method is called.


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
            The default is a random draw from a uniform distribution within the ``hyperparameter_bounds``.
        method : str or Callable, optional
            The method used to train the hyperparameters.
            The options are ``global``, ``local``, ``hgdl``, ``mcmc``, ``adam``, and a callable.
            The callable gets a :py:class:`fvgp.GP` instance and has to return a 1d np.ndarray of hyperparameters.
            The default is ``mcmc``.
            If method = ``mcmc`` or default,
            the attribute :py:attr:`fvgp.GP.mcmc_info` is updated and contains convergence and distribution information.
            For ``hgdl``, please provide a :py:class:`distributed.Client`.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 10000.
        mcmc_prior : callable, optional
            A function that defines the prior probability distribution for the MCMC sampler.
            The form of the function is f(x, bounds, args) and returns a scalar.
            The default is a uniform distribution within the ``hyperparameter_bounds``. The ``args`` are the same as the ``args`` of the GP instance.
        mcmc_prop_distrs : list of callables, optional
            A list of functions that define the proposal distributions for the MCMC sampler. 
            Each function should have the form f(x, para, obj) and return a vector of the same shape as x.
            See :py:class:`~fvgp.gp_mcmc.ProposalDistribution` in the documentation for more information.
        mcmc_args : dict, optional
            A dictionary of additional arguments for the MCMC sampler. The default is an empty dictionary.
        local_optimizer : str, optional
            Defining the local optimizer. Default = ``L-BFGS-B``, most :py:func:`scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to ``hgdl``. Default = ``genetic``
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is ``hgdl``, see the hgdl documentation.
            If the optimizer is a ``scipy`` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for asynchronous training. This can also be provided at initialization, but
            this will be used if not provided.
        info : bool, optional
            Provides a way how to access information reports during training of the GP. The default is False.
            If other information is needed please utilize ``logger`` as described in the online
            documentation (separately for HGDL and fvgp if needed).
        asynchronous : bool, optional
            When True, submit the training job and return immediately with an optimizer
            proxy object. Supported for ``method='hgdl'``, ``'mcmc'``, and ``'adam'``.
            Call ``get_latest()`` on the returned object to poll intermediate results,
            or call :py:meth:`update_hyperparameters` directly to apply them.

        Returns
        -------
        optimized hyperparameters (only fyi, gp is already updated) : np.ndarray
        """
        #gp2Scale checks
        if self.gp2Scale and asynchronous:  # pragma: no cover
            asynchronous = False
            warnings.warn("gp2Scale does not allow asynchronous training! `asynchronous` set to False")
        if self.gp2Scale and method != 'mcmc':  # pragma: no cover
            warnings.warn("gp2Scale enabled. Method switched to MCMC!")
            method = 'mcmc'

        #async checks
        _async_methods = {"hgdl", "mcmc", "adam"}
        if asynchronous and method not in _async_methods:  # pragma: no cover
            warnings.warn(f"Asynchronous execution is not supported for method=`{method}`. "
                          f"Supported async methods: {sorted(_async_methods)}. `asynchronous` set to False.")
            asynchronous = False

        if method in _async_methods and asynchronous and dask_client is None:
            dask_client = self.dask_client
            if dask_client is None: raise Exception("Please provide a dask_client for asynchronous training")   # pragma: no cover
        
        #hyperparameter bounds and init checks
        if hyperparameter_bounds is None:
            hyperparameter_bounds = self._get_default_hyperparameter_bounds()
            warnings.warn("Default hyperparameter_bounds initialized because none were provided. "
                          "This will fail for custom kernel,"
                          " mean, or noise functions")
        if init_hyperparameters is None:
            if out_of_bounds(self.hyperparameters, hyperparameter_bounds):
                init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                         high=hyperparameter_bounds[:, 1],
                                                         size=len(hyperparameter_bounds))
            else:
                init_hyperparameters = self.hyperparameters
        else:
            if out_of_bounds(init_hyperparameters, hyperparameter_bounds):
                warnings.warn("Your init_hyperparameters are out of bounds. They will be over-written")
                init_hyperparameters = np.random.uniform(low=hyperparameter_bounds[:, 0],
                                                         high=hyperparameter_bounds[:, 1],
                                                         size=len(hyperparameter_bounds))

        #objective function checks
        user_provided_obj = objective_function is not None
        if method == 'mcmc':
            if user_provided_obj:
                warnings.warn("MCMC always optimizes the log marginal likelihood; "
                              "the user-defined objective_function is ignored.")
            objective_function = self.marginal_likelihood.log_likelihood
        elif objective_function is None:
            objective_function = self.marginal_likelihood.neg_log_likelihood
        if user_provided_obj and objective_function_gradient is None and method in ('local', 'hgdl'):
            raise Exception("A gradient (and Hessian) of the objective function must be provided "
                            "for method='local' or method='hgdl'.")
        if objective_function_gradient is None:
            objective_function_gradient = self.marginal_likelihood.neg_log_likelihood_gradient
        if objective_function_hessian is None:
            objective_function_hessian = self.marginal_likelihood.neg_log_likelihood_hessian

        logger.debug("objective function: {}", objective_function)
        logger.debug("method: {}", method)

        if not asynchronous:
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
                mcmc_prior=mcmc_prior,
                mcmc_prop_distrs=mcmc_prop_distrs,
                mcmc_args=mcmc_args,
                local_optimizer=local_optimizer,
                global_optimizer=global_optimizer,
                constraints=constraints,
                dask_client=dask_client,
                info=info)
            self.set_hyperparameters(hyperparameters)
            return hyperparameters
        else:
            opt_obj = self.trainer.train_async(
                dask_client,
                objective_function=objective_function,
                objective_function_gradient=objective_function_gradient,
                objective_function_hessian=objective_function_hessian,
                hyperparameter_bounds=hyperparameter_bounds,
                init_hyperparameters=init_hyperparameters,
                method=method,
                pop_size=pop_size,
                tolerance=tolerance,
                max_iter=max_iter,
                mcmc_prior=mcmc_prior,
                mcmc_prop_distrs=mcmc_prop_distrs,
                mcmc_args=mcmc_args,
                local_optimizer=local_optimizer,
                global_optimizer=global_optimizer,
                constraints=constraints,
                info=info,
            )
            return opt_obj

    ##################################################################################
    def stop_training(self, opt_obj):
        """
        Function to stop an asynchronous ``hgdl`` training.
        This leaves the :py:class:`distributed.client.Client` alive.

        Parameters
        ----------
        opt_obj : object instance
            Object returned by ``train(asynchronous=True)``.
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
            Object returned by ``train(asynchronous=True)``.
        """
        self.trainer.kill_client(opt_obj)

    ##################################################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters if an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train(asynchronous=True)`.

        Returns
        -------
        hyperparameters : np.ndarray
            The latest hyperparameter vector pulled from the running optimizer.
        """

        hps = self.trainer.update_hyperparameters(opt_obj)
        self.set_hyperparameters(hps)
        return hps

    ##################################################################################
    def get_hyperparameters(self):
        """
        Get the current hyperparameters.

        .. deprecated::
            Use the :py:attr:`hyperparameters` property instead.

        Returns
        -------
        hyperparameters : np.ndarray
        """
        warnings.warn('`get_hyperparameters()` is deprecated. Please use `hyperparameters`',
                      DeprecationWarning, stacklevel=2)
        return self.hyperparameters

    ##################################################################################
    def get_prior_pdf(self):
        """
        Return the current GP prior covariance matrix and mean vector.

        Returns
        -------
        prior : dict
            Keys: ``"prior covariance (K)"`` (ndarray) and ``"prior mean"`` (ndarray).
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


        Returns
        -------
        log_likelihood : float
            Log marginal likelihood of the data.
        """
        if hyperparameters is not None:
            assert isinstance(hyperparameters, np.ndarray), "wrong format in hyperparameters"
            assert np.ndim(hyperparameters) == 1, "wrong format in hyperparameters"
        return self.marginal_likelihood.log_likelihood(hyperparameters=hyperparameters)

    def neg_log_likelihood_gradient(self, hyperparameters=None, component=0):
        """
        Function that computes the gradient of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.
        component : int, optional
            In case many GPs are computed in parallel, this specifies which one is considered.

        Returns
        -------
        gradient : np.ndarray
            Gradient of the negative log marginal likelihood, shape (N,).
        """
        return self.marginal_likelihood.neg_log_likelihood_gradient(hyperparameters=hyperparameters, component=component)

    def test_log_likelihood_gradient(self, hyperparameters, epsilon=1e-6):
        """
        Function to test your gradient of the log-likelihood and therefore of the kernel function.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).

        Returns
        -------
        fd_gradient : np.ndarray
            Finite-difference gradient of the log-likelihood, shape (N,).
        analytical_gradient : np.ndarray
            Analytical gradient of the log-likelihood, shape (N,).
        """
        assert isinstance(hyperparameters, np.ndarray), "wrong format in hyperparameters"
        assert np.ndim(hyperparameters) == 1, "wrong format in hyperparameters"
        return self.marginal_likelihood.test_log_likelihood_gradient(hyperparameters, epsilon=epsilon)

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

        Returns
        -------
        Solution points and function values : dict
        """
        return self.posterior.posterior_mean(x_pred, hyperparameters=hyperparameters, x_out=x_out)

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None, component=0):
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
        component : int, optional
            In case ``y_data`` is multi-modal and no fvgp.GPOptimizer is used --- this means y_data.shape[1] independent
            GPs are being executed --- this indicates which GP's gradient is evaluated. The default is 0.

        Returns
        -------
        Solution : dict
        """
        return self.posterior.posterior_mean_grad(x_pred, hyperparameters=hyperparameters,
                                                  x_out=x_out, direction=direction, component=component)

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
            Default = False. This is only relevant if the inverse of the covariance matrix is stored (linalg_mode == 'CholInv' or linalg_mode == 'Inv').
        add_noise : bool, optional
            If True the noise variances will be added to the posterior variances. Default = False.

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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
        at once as in :py:meth:`gp_relative_information_entropy`.


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

        Returns
        -------
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
        comp_cov: np.ndarray
            Covariance matrix, in R^{len(x_pred) x len(x_pred)}
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N),
            where N is the number evaluation points in the output direction.
            Usually this is np.ndarray([0,1,2,...]).

        Returns
        -------
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

    def crps(self, x_test, y_test):
        """
        This function calculates the continuous rank probability score.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        CRPS, standard dev. of CRPS : (float, float)
        """

        mean = self.posterior_mean(x_test)["m(x)"]
        sigma = np.sqrt(self.posterior_covariance(x_test)["v(x)"])
        assert mean.shape == sigma.shape == y_test.shape, \
            f"crps: shape mismatch mean={mean.shape} sigma={sigma.shape} y_test={y_test.shape}"
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

        Returns
        -------
        RMSE : float
        """

        v1 = y_test
        v2 = self.posterior_mean(x_test)["m(x)"]
        assert v1.shape == v2.shape, f"rmse: y_test shape {v1.shape} != posterior mean shape {v2.shape}"
        return np.sqrt(np.sum((v1 - v2) ** 2) / v1.size)

    def nrmse(self, x_test, y_test):
        """
        This function calculates the normalized root mean squared error.
        Note that in the multi-task setting the user should perform their
        input point transformation beforehand.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        NRMSE : float
        """

        return self.rmse(x_test, y_test) / (np.max(y_test) - np.min(y_test))

    def nlpd(self, x_test, y_test):  # correct, tested
        """
        This function calculates the Negative log predictive density.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        NLPD : float
        """

        mean = self.posterior_mean(x_test)["m(x)"]
        v = self.posterior_covariance(x_test)["v(x)"]

        assert mean.shape == v.shape == y_test.shape, \
            f"nlpd: shape mismatch mean={mean.shape} v={v.shape} y_test={y_test.shape}"

        term1 = 0.5 * np.log(2 * np.pi * v)
        term2 = 0.5 * ((y_test - mean) ** 2) / v
        nlpd = np.mean(term1 + term2)
        return nlpd

    def r2(self, x_test, y_test):
        """
        This function calculates the R2 prediction score.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        R2 : float
        """
        y_pred_mean = self.posterior_mean(x_test)["m(x)"]
        assert y_pred_mean.shape == y_test.shape, \
            f"r2: y_test shape {y_test.shape} != posterior mean shape {y_pred_mean.shape}"
        ss_res = np.sum((y_test - y_pred_mean) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1. - ss_res / ss_tot

    def picp(self, x_test, y_true, interval=0.95):
        """
        Computes the Prediction Interval Coverage Probability (PICP)
        for a Gaussian Process posterior.

        Parameters
        ----------
        x_test : array-like, shape (N,dim)
        y_true : array-like, shape (N,)
            True values of the target variable.
        interval : float, optional
            Confidence interval (default 0.95 for 95% intervals).

        Returns
        -------
        picp : float
            Prediction Interval Coverage Probability
        lower_bounds : ndarray
            Lower bounds of prediction intervals
        upper_bounds : ndarray
            Upper bounds of prediction intervals
        """
        mu = self.posterior_mean(x_test)["m(x)"]
        sigma = np.sqrt(self.posterior_covariance(x_test, add_noise=True)["v(x)"])

        z = norm.ppf(1 - (1 - interval) / 2)
        lower_bounds = mu - z * sigma
        upper_bounds = mu + z * sigma

        inside = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        picp = np.mean(inside)

        return picp

    def coverage_curve(self, x_test, y_test, intervals=None):
        """
        This function computes the coverage curve (calibration curve) of the GP posterior
        by evaluating :py:meth:`picp` across a range of target coverage levels.
        Plotting ``target_coverage`` against ``measured_coverage`` reveals whether the
        posterior is well-calibrated (diagonal), overconfident (below diagonal),
        or underconfident (above diagonal).

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.
        intervals : np.ndarray, optional
            A 1d array of target coverage levels in (0, 1). Default is np.linspace(0.05, 0.95, 19).

        Returns
        -------
        dict with keys ``target_coverage`` and ``measured_coverage``, each a list of floats.
        """
        if intervals is None:
            intervals = np.linspace(0.05, 0.95, 19)
        target_coverage = list(intervals)
        measured_coverage = [self.picp(x_test, y_test, interval=q) for q in intervals]
        return {"target_coverage": target_coverage, "measured_coverage": measured_coverage}

    def mpiw(self, x_test, interval=0.95):
        """
        This function calculates the Mean Prediction Interval Width (MPIW).
        It measures the average width of the posterior credible intervals and
        is best interpreted alongside :py:meth:`picp`: a narrow interval with
        high coverage indicates a well-calibrated, sharp model.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as an array of input point positions.
        interval : float, optional
            Credible interval level. Default = 0.95.

        Returns
        -------
        MPIW : float
        """
        v = self.posterior_covariance(x_test, add_noise=True)["v(x)"]
        sigma = np.sqrt(np.clip(v, 0.0, None))
        z = norm.ppf(1 - (1 - interval) / 2)
        return np.mean(2 * z * sigma)

    def interval_score(self, x_test, y_test, interval=0.95):
        """
        This function calculates the Interval Score (also known as the Winkler Score).
        It penalizes both missed coverage and unnecessarily wide prediction intervals,
        combining the concerns of :py:meth:`picp` and :py:meth:`mpiw` into a single
        scalar. Lower is better.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.
        interval : float, optional
            Credible interval level. Default = 0.95.

        Returns
        -------
        Interval Score : float
        """
        mean = self.posterior_mean(x_test)["m(x)"]
        sigma = np.sqrt(self.posterior_covariance(x_test, add_noise=True)["v(x)"])
        assert mean.shape == sigma.shape == y_test.shape, \
            f"interval_score: shape mismatch mean={mean.shape} sigma={sigma.shape} y_test={y_test.shape}"

        alpha = 1 - interval
        z = norm.ppf(1 - alpha / 2)
        lower = mean - z * sigma
        upper = mean + z * sigma
        width = upper - lower
        penalty_low = (2 / alpha) * np.maximum(lower - y_test, 0)
        penalty_high = (2 / alpha) * np.maximum(y_test - upper, 0)
        return np.mean(width + penalty_low + penalty_high)

    def mae(self, x_test, y_test):
        """
        This function calculates the Mean Absolute Error (MAE).
        Note that in the multi-task setting the user should perform their
        input point transformation beforehand.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        MAE : float
        """
        v1 = y_test
        v2 = self.posterior_mean(x_test)["m(x)"]
        assert v1.shape == v2.shape, f"mae: y_test shape {v1.shape} != posterior mean shape {v2.shape}"
        return np.mean(np.abs(v1 - v2))

    def mape(self, x_test, y_test):
        """
        This function calculates the Mean Absolute Percentage Error (MAPE).
        Note that in the multi-task setting the user should perform their
        input point transformation beforehand.
        Avoid using this metric when ``y_test`` contains values close to zero,
        as the percentage error becomes unstable.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        MAPE : float
        """
        v1 = y_test
        v2 = self.posterior_mean(x_test)["m(x)"]
        assert v1.shape == v2.shape, f"mape: y_test shape {v1.shape} != posterior mean shape {v2.shape}"
        return np.mean(np.abs((v1 - v2) / v1))

    def msll(self, x_test, y_test):
        """
        This function calculates the Mean Standardized Log Loss (MSLL).
        It is the :py:meth:`nlpd` of the GP posterior minus the NLPD of a
        trivial baseline model (a Gaussian with the empirical mean and variance
        of the training targets). Negative values indicate that the GP predicts
        better than the baseline; zero means it matches it.

        Parameters
        ----------
        x_test : np.ndarray
            A numpy array of shape (V x D), interpreted as an array of input point positions.
        y_test : np.ndarray
            A numpy array of shape V or (V x No) in the multi-output case. These are the y data to compare against.

        Returns
        -------
        MSLL : float
        """
        mean = self.posterior_mean(x_test)["m(x)"]
        v = self.posterior_covariance(x_test)["v(x)"]
        assert mean.shape == v.shape == y_test.shape, \
            f"msll: shape mismatch mean={mean.shape} v={v.shape} y_test={y_test.shape}"

        nlpd_gp = np.mean(0.5 * np.log(2 * np.pi * v) + 0.5 * ((y_test - mean) ** 2) / v)

        baseline_mean = np.mean(self.y_data)
        baseline_var = np.var(self.y_data)
        nlpd_baseline = np.mean(0.5 * np.log(2 * np.pi * baseline_var)
                                + 0.5 * ((y_test - baseline_mean) ** 2) / baseline_var)

        return nlpd_gp - nlpd_baseline

    def plot_observed_vs_predicted(self, x_test, y_test, title=None, ax=None):
        """
        Scatter plot of observed vs. predicted values with a reference diagonal
        and 1-sigma predictive error bars (noise-inflated posterior variance).
        Useful for a quick visual check of model fit on a held-out test set.

        Parameters
        ----------
        x_test : np.ndarray
            Test input positions, shape (V, D).
        y_test : np.ndarray
            Observed test values, shape (V,) or (V, No) for multi-output.
        title : str, optional
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on; if ``None``, a fresh figure + axes is created.

        Returns
        -------
        None
            If matplotlib is not installed a ``UserWarning`` is emitted; otherwise
            the plot is drawn on the supplied or freshly-created axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn(
                "matplotlib is not installed; cannot create observed-vs-predicted plot. "
                "Install with `pip install matplotlib` (or `pip install -e .[plotting]`) "
                "to enable plotting."
            )
            return

        y_pred = self.posterior_mean(x_test)["m(x)"]
        y_var  = self.posterior_covariance(x_test, add_noise=True)["v(x)"]
        y_obs_flat  = np.asarray(y_test).reshape(-1)
        y_pred_flat = np.asarray(y_pred).reshape(-1)
        y_sigma_flat = np.sqrt(np.clip(np.asarray(y_var).reshape(-1), 0.0, None))

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(y_obs_flat, y_pred_flat, yerr=y_sigma_flat,
                    fmt="o", alpha=0.6, markersize=4, capsize=2,
                    elinewidth=0.8, label="prediction ± 1σ")

        lo = float(min(y_obs_flat.min(), (y_pred_flat - y_sigma_flat).min()))
        hi = float(max(y_obs_flat.max(), (y_pred_flat + y_sigma_flat).max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")

        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        if title is not None:
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")

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

        Returns
        -------
        Evaluations of the Gaussian : np.ndarray
        """
        # Gaussian function formula
        coefficient = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
        return coefficient * np.exp(exponent)

    @staticmethod
    def make_2d_x_pred(bx, by, resx=100, resy=100):
        """
        This is a purely convenience-driven function calculating prediction points
        on a grid.
        Parameters
        ----------
        bx : iterable
            A numpy array or list of shape (2) defining lower and upper bounds in x direction.
        by : iterable
            A numpy array of shape (2) defining lower and upper bounds in y direction.
        resx : int, optional
            Resolution in x direction. Default = 100.
        resy : int, optional
            Resolution in y direction. Default = 100.
        Returns
        -------
        prediction points : np.ndarray
        """

        x = np.linspace(bx[0], bx[1], resx)
        y = np.linspace(by[0], by[1], resy)
        from itertools import product
        x_pred = np.array(list(product(x, y)))
        return x_pred

    @staticmethod
    def make_1d_x_pred(b, res=100):
        """
        This is a purely convenience-driven function calculating prediction points
        on a 1d grid.

        Parameters
        ----------
        b : iterable
            A numpy array or list of shape (2) defining lower and upper bounds
        res : int, optional
            Resolution. Default = 100

        Returns
        -------
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

        Returns
        -------
        estimated execution time : float
        """
        b = self.prior.batch_size
        D = len(self.x_data)
        tb = time_per_worker_execution
        n = number_of_workers
        return (D ** 2 * tb) / (2. * n * b ** 2)

    def initialize_gp2Scale_dask_client(self, gp2Scale, dask_client):
        """
        Ensure a Dask client exists when ``gp2Scale=True``, creating a local one if needed.

        Parameters
        ----------
        gp2Scale : bool
            Whether the sparse gp2Scale mode is active.
        dask_client : distributed.Client or None
            An existing Dask client, or None to auto-create a local one.

        Returns
        -------
        client : distributed.Client or None
            A valid Dask client, or None when ``gp2Scale=False``.
        """
        if gp2Scale:
            try:
                from imate import logdet as imate_logdet
            except:
                raise Exception(
                    "You have activated `gp2Scale`. You need to install imate"
                    " manually for this to work.")
            if dask_client is None:
                logger.debug("Creating my own local client.")
                try:
                    dask_client = Client()
                except:
                    logger.debug("no client available")
        return dask_client

    def __getstate__(self):
        state = dict(
            data=self.data,
            prior=self.prior,
            likelihood=self.likelihood,
            kv=self.kv,
            marginal_likelihood=self.marginal_likelihood,
            trainer=self.trainer,
            posterior=self.posterior,
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
def out_of_bounds(x, bounds):
    assert isinstance(x, np.ndarray), "x must be np.ndarray for bounds check"
    assert isinstance(bounds, np.ndarray), "bounds must be np.ndarray"
    assert np.ndim(bounds) == 2, "bounds must be 2-d (n_params × 2)"
    for i in range(len(x)):
        if x[i] < bounds[i, 0] or x[i] > bounds[i, 1]:
            return True
    return False
