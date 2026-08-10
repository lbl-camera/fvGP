import numpy as np
from .gp import GP


class fvGP(GP):
    """
    This class provides all the tools for a multi-task Gaussian Process (GP).
    After initialization, this class provides all the methods described for the
    :py:class:`fvgp.GP` class, including full HPC support via the ``hgdl`` package
    and large-scale sparse GPs via ``gp2Scale``.

    V ... number of input points

    Di... input space dimensionality

    No... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The main logic of fvGP is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best
    way to give the user optimal control and power. At various instances, for example
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input ``x`` is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [0,1], the input to the mean, kernel, and noise functions might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]

    This has to be understood and taken into account when customizing fvGP for multi-task
    use. The examples will provide deeper insights.

    Parameters
    ----------
    x_data : np.ndarray | list
        The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_set_dim`.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        In this case, both the index set and the input space dim are set to 1.
    y_data : np.ndarray
        The values of the data points. Shape (V,No).
        It is possible that not every entry in ``x_data``
        has all corresponding tasks available. In that case ``y_data`` may have np.nan as the corresponding entries.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). The task direction is
        simply considered a separate dimension. If ``gp2Scale`` is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
        The full hyperparameter vector is passed to the kernel, mean, and noise callables,
        but the index ranges used by each callable are **disjoint and user-defined**.
        Each callable must only read the indices reserved for it. The gradient
        computation relies on this: when a hyperparameter index belongs to the mean
        function its kernel derivative is assumed zero, and vice versa.
    noise_variances : np.ndarray, optional
        A numpy array defining the uncertainties/noise in the
        ``y_data`` in form of a point-wise variance. Shape (V, No).
        If ``y_data`` has np.nan entries, the corresponding
        ``noise_variances`` have to be np.nan.
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to ``abs(np.mean(y_data)) / 100.0``. If
        noise covariances are required (correlated noise), make use of the ``noise_function``.
        Only provide a noise function OR ``noise_variances``, not both.
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
        The input ``x1`` is a N1 x Di+1 array of positions, ``x2`` is a N2 x Di+1
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperparameters are initialized.
        The default is a stationary anisotropic Matern kernel of first-order
        differentiability, which performs automatic relevance determination (ARD)
        through one length scale per input dimension. The task
        direction is simply considered an additional dimension. This kernel should only be used for tests and in the
        simplest of cases.
        If ``gp2Scale`` is enabled, the default changes to a compactly supported anisotropic
        Wendland kernel instead -- compact support is what makes the covariance matrix sparse.
        Which one is used depends on ``compute_device``:
        :py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_cpu` or
        :py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_gpu`.
        See :py:mod:`fvgp.kernels` for those, their support-aware sparse variants, and the
        other built-in kernels.
        The output is a matrix, an N1 x N2 numpy array.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the kernel (disjoint from mean and noise indices).
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``kernel_function`` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input ``x1`` (a N1 x Di + 1 array of positions),
        ``x2`` (a N2 x Di + 1 array of positions) and
        ``hyperparameters`` (a 1d array of length Di+2 for the default kernel).
        The default is an analytical gradient for the default kernel or a finite difference calculation otherwise.
        If ``ram_economy`` is True, the function's input is x1, x2, hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If ``ram_economy`` is ``False``, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See ``ram_economy``.
    prior_mean_function : Callable, optional
        A function f(x, hyperparameters, [args]) that evaluates the prior mean at a set of input position.
        It accepts as input
        an array of positions (of shape N1 x Di+1) and
        hyperparameters (a 1d array of length Di+2 for the default kernel).
        Optionally, the third argument ``args`` can be defined.
        The return value is a 1d array of length N1. If None is provided,
        :py:meth:`fvgp.GP._default_mean_function` is used, which is the average of the ``y_data``.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the mean function (disjoint from kernel and noise indices).
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``prior_mean_function`` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x Di+1) and hyperparameters
        (a 1d array of length Di+2 for the default kernel).
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
        The input ``x`` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the noise function (disjoint from kernel and mean indices).
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``noise_function``
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1) and
        hyperparameters (a 1d array of length Di+1 for the default kernel).
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
        How finely the covariance is cut up for distributed computing in gp2Scale.
        The default is 10000. It is a **maximum, not a target**: every chunk is exactly
        this size except the last along each axis, which carries the remainder -- 100000
        points at 15000 give six chunks of 15000 and one of 10000.

        What it sizes depends on ``gp2Scale_distribution``. For ``"blockwise"`` it is the
        side of a square block, so a worker evaluates
        ``gp2Scale_batch_size x gp2Scale_batch_size``. For ``"rowwise"`` it is the
        **strip width**: a worker evaluates ``gp2Scale_batch_size x N``, the whole row at
        once. With a dense kernel that allocation is what bounds worker memory, so pick
        it against the memory a worker has -- at N=100000 a strip width of 10000 is 8 GB
        in float64. Compactly supported kernels that return a sparse block
        (:py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_cpu_sparse`) never
        materialize it.

        Row-wise also produces far fewer tasks -- ``N / gp2Scale_batch_size`` against
        roughly ``(N / gp2Scale_batch_size)^2 / 2`` for block-wise -- so on a large
        cluster it usually wants a smaller value than block-wise would.

        These two meanings are **not comparable numbers**. At ``gp2Scale_batch_size=10000``
        and N=1e6, block-wise sanctions 0.8 GB per task and row-wise 80 GB, so the same
        setting can leave a computation local under one distribution and distribute it
        under the other. That is deliberate -- each mode gets the memory its own user
        actually declared.

        The budget also decides **whether to distribute at all**: anything that fits in one
        task is computed in a single kernel call instead, because scheduling costs more
        than the work. A 10 000 x 2 cross-covariance takes 0.83 ms directly against 1.8 s
        spread over 1000 dask tasks. That local call happens on the client, so it can hold
        up to one task's budget in client memory -- worth remembering under ``"rowwise"``,
        where the budget grows with N.

        Cross-covariances -- the posterior's ``k(x_data, x_pred)`` and the ``k(x_old,
        x_new)`` block of an append -- are **always cut into strips**, whatever
        ``gp2Scale_distribution`` says, with the strip width taken from the budget above.
        Blocking a tall thin matrix multiplies the task count without reducing per-task
        memory, since the long axis has to be traversed either way.
    gp2Scale_distribution : str, optional
        How the covariance computation is cut across the workers in gp2Scale.
        ``"blockwise"`` (default) sends (row block, column block) pairs and, for the
        symmetric prior covariance, schedules only the upper triangle, so the cluster
        performs half the kernel evaluations and the host mirrors the result.
        ``"rowwise"`` sends whole row strips -- one kernel call per strip against every
        column, ``gp2Scale_batch_size`` rows wide -- and has each worker return a
        finished sparse strip, so the assembly is a concatenation rather than a global
        re-sort on the host. Row-wise cannot exploit symmetry and so doubles the kernel evaluations, but
        it removes the host as a bottleneck and lowers its peak memory considerably; it is
        the better choice when assembly, not kernel evaluation, dominates the run time.
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
        * ``"Inv"`` — computes and stores the explicit inverse directly (no Cholesky). Only suitable for
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

        **Preconditioner aliases (incomplete Cholesky):**

        The IC family has a compiled fast path and a pure-Python legacy path. The
        names in this list are the canonical backends that ``sparse_preconditioner_type``
        resolves to internally; the items after "aliases:" all map to the same backend.

        * ``"ichol"`` — thresholded incomplete Cholesky via the optional
          ``ilupp`` package (fast). **Aliases:** ``"ic"``, ``"incomplete_cholesky"``.
          Tunable through ``sparse_preconditioner_ichol_fill_in`` and
          ``sparse_preconditioner_ichol_threshold``.
        * ``"ichol0"`` — zero-fill incomplete Cholesky via ``ilupp`` (cheap to build,
          weaker as a preconditioner). No aliases.
        * ``"native_incomplete_cholesky"`` — pure-Python IC(0); slower, intended as
          a legacy/debugging fallback that always works without optional deps.
          **Aliases:** ``"native_ic"``, ``"native_ichol"``,
          ``"legacy_ic"``, ``"legacy_ichol"``,
          ``"legacy_incomplete_cholesky"``. (``"native_"`` and ``"legacy_"`` are
          interchangeable — "native" means built-in to fvGP, "legacy" reminds you
          this is the original pre-``ilupp`` path.)

        The two ``ilupp`` backends (``"ichol"``/``"ichol0"``) raise a clear
        ``ImportError`` with install instructions if ``ilupp`` is missing
        (``pip install ilupp``). All three IC backends honor the shared
        ``sparse_preconditioner_shift`` / ``_shift_growth`` / ``_shift_attempts``
        diagonal-shift-retry knobs (see the ``args`` section below).

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
        - "random_logdet_min_num_samples" : int; default = 10 — fewest Hutchinson
          probes to draw
        - "random_logdet_max_num_samples" : int; default = 5000 — most to draw.
          The probe count is the fidelity dial of this estimator: its variance falls
          as 1/t while its cost grows as t, so it trades accuracy against compute.
          The variance actually achieved is reported by
          :py:meth:`fvgp.gp_marginal_likelihood.GPmarginalLikelihood.log_likelihood_variance`.

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
          previous training iteration's ``KVinvY`` as ``x0`` to the next solve.
          Honored only for ``train(method='mcmc')``, whose steps are small enough for
          the previous solution to be a good guess, and only while K+V has not drifted
          (see ``sparse_preconditioner_max_matrix_drift``): a warm start from distant
          hyperparameters is worse than a cold one, and on a truncated solve it leaves
          an error that depends on which hyperparameters ran before it.
        - "sparse_preconditioner_type" : str; default = "ilu". One of "ilu",
          "ichol"/"ic"/"incomplete_cholesky", "ichol0", "native_ic"/"native_ichol",
          "block_jacobi", "schwarz"/"additive_schwarz", "amg" (requires pyamg)
        - "sparse_preconditioner_max_matrix_drift" : float; default = 0.1 — relative
          change in K+V, measured by trace and Frobenius norm, beyond which the cached
          preconditioner and any warm start are stale and rebuilt. This, rather than a
          count of reuses, is what decides reuse: a fixed count cannot tell many tiny
          MCMC steps from one jump across the domain.
        - "sparse_preconditioner_refresh_interval" : int; default = None (no cap) —
          optional hard cap on consecutive reuses, applied on top of the drift test
          above. ``set_KV`` always force-refreshes.
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
        - "sparse_preconditioner_shift" / "sparse_preconditioner_shift_growth" /
          "sparse_preconditioner_shift_attempts" — diagonal shift retry knobs
          (defaults: 0.0 / 10.0 / 5). Applies to "ichol" and "ichol0" (compiled),
          "native_ic"/"native_ichol" (legacy), "block_jacobi", and
          "additive_schwarz" when a local factorization encounters a non-PD pivot

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
        - Preconditioner reuse across nearby K+V updates is automatic and needs no tuning:
          the cached factor is kept until K+V has actually drifted past
          ``sparse_preconditioner_max_matrix_drift``. Reach for
          ``sparse_preconditioner_refresh_interval`` only to impose an extra hard cap.
          ``sparse_krylov_warm_start`` is a separate opt-in and is honored only for
          ``train(method='mcmc')``, whose steps are small enough for the previous solution
          to be a good starting guess; the other methods sample non-locally, where a warm
          start is worse than a cold one.

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


    This class inherits all capabilities from :py:class:`fvgp.GP`.
    Check there for a full list of capabilities. Here are the most important.

    Base-GP Methods:

    :py:meth:`fvgp.GP.train`

    :py:meth:`fvgp.GP.stop_training`

    :py:meth:`fvgp.GP.kill_client`

    :py:meth:`fvgp.GP.update_hyperparameters`

    :py:meth:`fvgp.GP.set_hyperparameters`

    :py:meth:`fvgp.GP.hyperparameters`

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

    :py:meth:`fvgp.GP.gp_mutual_information`

    :py:meth:`fvgp.GP.gp_total_correlation`

    :py:meth:`fvgp.GP.gp_relative_information_entropy`

    :py:meth:`fvgp.GP.gp_relative_information_entropy_set`

    :py:meth:`fvgp.GP.posterior_probability`

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
        gp2Scale_distribution="blockwise",
        linalg_mode=None,
        ram_economy=False,
        args=None
    ):
        if isinstance(y_data, np.ndarray):
            if np.ndim(y_data) == 1:
                raise ValueError("The output number is 1, you can use the GP class for single-task GPs")
        self.output_num = y_data.shape[1]

        assert len(x_data) == len(y_data), "x_data and y_data have different lengths"
        ####transform the space
        fvgp_x_data = x_data
        fvgp_y_data = y_data
        fvgp_noise_variances = noise_variances
        x_data, y_data, noise_variances = self._transform_index_set2(x_data, y_data, noise_variances)

        ####init GP
        super().__init__(
            x_data,
            y_data,
            init_hyperparameters=init_hyperparameters,
            noise_variances=noise_variances,
            compute_device=compute_device,
            kernel_function=kernel_function,
            kernel_function_grad=kernel_function_grad,
            prior_mean_function=prior_mean_function,
            prior_mean_function_grad=prior_mean_function_grad,
            noise_function=noise_function,
            noise_function_grad=noise_function_grad,
            gp2Scale=gp2Scale,
            dask_client=dask_client,
            gp2Scale_batch_size=gp2Scale_batch_size,
            gp2Scale_distribution=gp2Scale_distribution,
            linalg_mode=linalg_mode,
            ram_economy=ram_economy,
            args=args)

        self.data.set_fvgp_data(fvgp_x_data, fvgp_y_data, fvgp_noise_variances, np.arange(0, self.output_num))

    @property
    def fvgp_x_data(self):
        """Multi-task input data including the output-index column, shape (N, D+1)."""
        return self.data.fvgp_x_data

    @property
    def fvgp_y_data(self):
        """Observed values in the multi-task (output-index-augmented) space, shape (N,)."""
        return self.data.fvgp_y_data

    @property
    def fvgp_noise_variances(self):
        """Point-wise noise variances in the multi-task space, shape (N,), or None."""
        return self.data.fvgp_noise_variances

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
            The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_set_dim`.
            For multi-task GPs, the index set dimension = input space dimension + 1.
            If dealing with non-Euclidean inputs
            ``x_new`` should be a list, not a numpy array.
        y_new : np.ndarray
            The values of the data points. Shape (V,No).
            It is possible that not every entry in ``x_new``
            has all corresponding tasks available. In that case ``y_new`` may contain np.nan entries.
        noise_variances_new : np.ndarray, optional
            A numpy array defining the uncertainties/noise in the
            ``y_data`` in form of a point-wise variance. Shape (V, No).
            If ``y_data`` has np.nan entries, the corresponding
            ``noise_variances`` have to be np.nan.
            Note: if no noise_variances are provided here, the noise_function
            callable will be used; if the callable is not provided, the noise variances
            will be set to ``abs(np.mean(y_data)) / 100.0``. If
            noise covariances are required (correlated noise), make use of the ``noise_function``.
            Only provide a noise function OR ``noise_variances``, not both.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default = True.
            In the default case, data will be appended.
        rank_n_update : bool, optional
            Indicates whether the GP marginal likelihood should be rank-n updated or recomputed. The default
            is ``rank_n_update=append``, meaning if data is only appended, the rank_n_update will
            be performed.
        """
        assert isinstance(x_new, np.ndarray) or isinstance(x_new, list), "Wrong format in x_new."
        assert isinstance(y_new, np.ndarray), "Wrong format in y_new."
        assert len(x_new) == len(y_new), "updated x and y do not have the same lengths."
        if append:
            if noise_variances_new is not None:
                assert isinstance(noise_variances_new, np.ndarray), \
                    "noise_variances_new must be np.ndarray"
                fvgp_noise_variances = np.vstack([self.fvgp_noise_variances, noise_variances_new])
            else: fvgp_noise_variances = None

            if isinstance(x_new, np.ndarray): fvgp_x_data = np.vstack([self.fvgp_x_data, x_new])
            # a list x_new is a non-Euclidean multi-task input space
            elif isinstance(x_new, list): fvgp_x_data = self.fvgp_x_data + x_new
            # unreachable: the assert above already requires an np.ndarray or a list
            else: raise Exception(  # pragma: no cover - defensive
                "x_new not given in an allowed format")

            fvgp_y_data = np.vstack([self.fvgp_y_data, y_new])
        else:
            fvgp_noise_variances = noise_variances_new
            fvgp_x_data = x_new
            fvgp_y_data = y_new
        self.data.set_fvgp_data(fvgp_x_data, fvgp_y_data, fvgp_noise_variances, np.arange(0, self.output_num))

        ######################################
        #####transform to index set###########
        ######################################
        x_data, y_data, noise_variances = self._transform_index_set2(x_new, y_new, noise_variances_new)
        super().update_gp_data(x_data, y_data, noise_variances, append=append, rank_n_update=rank_n_update)

    ################################################################################################
    def _transform_index_set2(self, x_data, y_data, noise_variances):
        assert isinstance(x_data, np.ndarray) or isinstance(x_data, list), \
            "x_data must be np.ndarray or list"
        assert isinstance(y_data, np.ndarray), "y_data must be np.ndarray"
        assert len(x_data) == len(y_data), "x_data and y_data have different lengths"
        if noise_variances is not None: assert len(noise_variances) == len(y_data), \
            "noise_variances and y_data have different lengths"
        new_x_data = []
        new_y_data = []
        output_indices = np.arange(0, self.output_num)
        if noise_variances is not None: new_variances = []
        else: new_variances = None
        for i in range(self.output_num):
            for j in range(len(x_data)):
                if noise_variances is not None: assert len(noise_variances[j]) == self.output_num, \
                    f"noise_variances row {j} length must equal output_num={self.output_num}"
                assert len(y_data[j]) == self.output_num, \
                    f"y_data row {j} length must equal output_num={self.output_num}"
                if np.isnan(y_data[j, i]): continue
                if isinstance(x_data, np.ndarray): new_x_data.append(np.append(x_data[j], output_indices[i]))
                else: new_x_data.append([x_data[j], output_indices[i]])
                new_y_data.append(y_data[j, i])
                if new_variances is not None:
                    new_variances.append(noise_variances[j, i])
        if isinstance(x_data, np.ndarray): new_x_data = np.asarray(new_x_data)
        new_y_data = np.asarray(new_y_data)
        if new_variances is not None: new_variances = np.asarray(new_variances)

        assert isinstance(new_x_data, list) or isinstance(new_x_data, np.ndarray), \
            "transformed x_data must be list or np.ndarray"
        assert isinstance(new_y_data, np.ndarray) and np.ndim(new_y_data) == 1, \
            "transformed y_data must be a 1-d np.ndarray"
        if new_variances is not None: assert isinstance(new_variances, np.ndarray) and np.ndim(new_variances) == 1, \
            "transformed noise_variances must be a 1-d np.ndarray"
        return new_x_data, new_y_data, new_variances

    ################################################################################################
    def __getstate__(self):
        state = dict(
            output_num=self.output_num
            )
        state.update(super().__getstate__())
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
