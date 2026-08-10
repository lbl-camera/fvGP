=======
History
=======

This changelog was restarted; for releases before the first entry below, see the
git history and the release tags.

Unreleased
----------

New features
~~~~~~~~~~~~

* ``train(method='bo')`` trains the hyperparameters by Bayesian optimization, joining
  ``global``, ``local``, ``hgdl``, ``mcmc`` and ``adam``. It is the method for an
  outer marginal likelihood that is expensive, noisy, and effectively gradient-free --
  a distributed mBCG solve, where the log-determinant comes from stochastic Lanczos
  quadrature and the solve from truncated CG, so repeated evaluations at the same
  hyperparameters disagree.

  The surrogate is itself a small :py:class:`fvgp.GP`, deliberately fvGP rather than
  gpCAM: gpCAM depends on fvGP, so reaching for it here would make the dependency
  circular. The recursion bottoms out immediately -- the inner GP sees only the few
  dozen points the search evaluates, uses an ARD Matern-5/2 kernel, and is trained
  with ``method='local'``.

  Search happens in log space for the dimensions whose bounds are strictly positive
  and linearly elsewhere, since a hyperparameter such as a prior-mean coefficient may
  legitimately be negative. A least-squares quadratic mean absorbs the global bowl of
  the likelihood surface, the initial design is Sobol and includes the incoming
  hyperparameters as a warm start, and the acquisition is noisy expected improvement
  (the incumbent is itself noisy, so plain EI is ill-defined) maximized by multi-start
  L-BFGS-B. Note that ``max_iter`` here counts *objective evaluations*, since that is
  the expensive quantity.

  Configure through the new ``bo_args``. The most valuable key is ``noise_function``
  or ``noise_variance``: if the objective is a stochastic estimator whose own spread
  you can compute, supplying it beats spending scarce evaluations fitting a noise
  term. Results land in the new :py:attr:`fvgp.GP.bo_info`, which includes a
  curvature-based sensitivity ranking of the hyperparameters and a Laplace
  approximation of the theta-posterior -- both obtained with no further likelihood
  evaluations.

* The stochastic-Lanczos log-determinant now reports its own precision.
  :py:func:`fvgp.gp_lin_alg.calculate_random_logdet` takes an optional ``info_out``
  dict, filled with the variance of the estimate, the probe count, and the error
  bounds; the return type is unchanged for existing callers. The new
  ``log_likelihood_variance()`` on the marginal likelihood turns that into the variance
  of the (negative) log marginal likelihood, and ``train(method='bo')`` feeds it to the
  optimizer automatically. Users of the sparse modes therefore get a properly
  noise-aware search without having to characterize the estimator themselves. The probe
  count -- the fidelity dial, since its noise falls as 1/sqrt(t) while its cost grows as
  t -- is now settable through ``random_logdet_min_num_samples`` and
  ``random_logdet_max_num_samples``.

* Cached linear-algebra state is now invalidated by measuring the matrix rather than by
  counting steps. A preconditioner used to be reusable for a fixed
  ``sparse_preconditioner_refresh_interval`` number of calls, a control that knows
  nothing about whether K+V actually moved -- k tiny MCMC steps and one jump across the
  domain counted the same. Reuse, and the Krylov warm start alongside it, are now
  conditioned on the relative drift of a cheap O(nnz) fingerprint of K+V (trace and
  Frobenius norm), with the threshold ``sparse_preconditioner_max_matrix_drift``
  defaulting to 0.1. The refresh interval survives only as an optional hard cap and no
  longer has a default.

  The threshold is calibrated against how much benefit a preconditioner actually retains
  as the matrix moves: on a truncated CG solve it holds ~100% of the speed-up out to a
  drift of a few percent and fades past ~0.15. In hyperparameter terms that admits steps
  of several percent -- covering MCMC proposals and local optimizer steps -- while a
  doubling of the hyperparameters drifts 0.58 and a tenfold change 0.92, both refused.

* Krylov warm starts and preconditioner reuse are additionally restricted to
  ``method='mcmc'``; every other method runs with them off and the user's settings
  restored afterwards, with a warning if an explicit choice had to be overridden. Both
  carry state between likelihood evaluations, which is sound only when successive
  evaluations are close. Measured on a truncated CG solve, a warm start from nearby
  hyperparameters cuts the error 25x while one from distant hyperparameters is *worse
  than a cold start*, and a reused preconditioner is then worth no more than none at all.
  The cost is not the lost speed but the leftover residual, which makes the objective
  depend on the order the points were evaluated in -- a bias, and so exactly what a
  Bayesian optimizer's zero-mean noise model cannot absorb. Both settings already
  defaulted to the safe value, so this matters for users who tuned them for MCMC and then
  switched method.

* ``method='bo'`` is roughly 4x faster. The dominant cost was not the inner GP's
  training, which is only a few percent of a run, but the acquisition: it asks for
  ``variance_only=True`` at hundreds of candidates per iteration, and fvGP can only take
  that shortcut when the inverse is stored. In the default ``Chol`` mode it instead built
  the full (V x V) posterior covariance and kept only the diagonal -- a 512x512 matmul
  per call. The surrogate now uses ``linalg_mode='CholInv'``, which for a covariance over
  a few dozen points costs nothing and was worth 3.7x on its own. The acquisition's
  multi-start count also drops from 5 to 3, measured across two problems and eight seeds
  to match 5 and 8 exactly on solution quality while removing another 15-24%.

* The BO surrogate now uses analytic derivatives throughout. The acquisition is
  maximized with an exact gradient rather than scipy's finite differences, and the
  surrogate's own hyperparameter training gets an analytic ``kernel_function_grad`` and
  ``noise_function_grad`` instead of fvGP's finite-difference fallback (``_dkernel_dh``,
  a fixed 1e-8 step that rebuilds the covariance twice per hyperparameter). The
  hyperparameter gradient is 20-60x more accurate and 1.5-2x cheaper; on a real marginal
  likelihood the whole run is ~2.6x faster at indistinguishable quality.

  The expected-improvement gradient collapses to ``Phi(z) dmu/dx + phi(z) dsigma/dx`` --
  the chain-rule terms through ``z`` cancel identically -- and the Monte-Carlo incumbent
  enters only as two scalars, so nothing has to be differentiated through the sampling.

  One caveat worth stating: exact gradients maximize the acquisition *better*, which is
  not always better BO. On a deliberately rugged surface where the surrogate collapses
  (its learned noise absorbing the entire residual variance), the sharper search exploits
  that bad surrogate harder and does worse than the sloppier finite-difference one, whose
  failure to converge acted as accidental exploration. Marginal-likelihood surfaces, which
  is what this method exists for, are far smoother and show no such effect.

* ``method='bo'`` stops on its own once the answer stops moving, so ``max_iter`` is a
  cap rather than a target. Neither the best value found nor its location changing
  materially for ``patience`` consecutive iterations (default 10) ends the run; both
  conditions are required, since a flat stretch of values while the recommendation is
  still traveling means the search is exploring rather than converged. Improvement is
  judged against the observed spread, not the value, so the test stays meaningful for a
  marginal likelihood that may be large, negative or near zero. Tunable through
  ``patience``, ``f_rtol`` and ``x_tol``, and ``patience=0`` restores the old behavior of
  spending the whole budget. ``bo_info['stopping reason']`` reports which criterion
  ended the run.

  This matters because ``max_iter`` inherits ``GP.train``'s default of 10000, which for
  this method counts evaluations of an expensive objective. A smooth two-hyperparameter
  problem left at that default now converges in about 21 evaluations rather than running
  to the cap.

* ``train(info=True)`` now reports progress for every method, not just ``mcmc``. It was
  effectively a no-op elsewhere: ``mcmc`` printed, while the other methods reported
  through ``logger.debug`` and fvGP disables the loguru logger at import, so nothing
  reached the user. ``bo`` prints one line per objective evaluation -- the value reached,
  the best so far, and the expected improvement that selected the point -- plus the size
  of the initial design and the reason the run ended; ``local`` prints the objective each
  iteration; ``adam`` prints the objective and gradient norm every ten. ``global`` already
  worked, since ``disp=info`` reaches ``scipy``; ``hgdl`` still reports through its own
  logger.

  The cadence differs on purpose: a ``bo`` iteration is one full evaluation of an
  expensive objective, whereas ``adam``'s ``max_iter`` counts cheap optimizer steps and
  runs to thousands. The ``local`` report costs nothing extra -- naming the callback
  parameter ``intermediate_result`` makes ``scipy`` hand over an ``OptimizeResult`` whose
  objective value has already been computed.

* ``bo_args['log_scale']`` controls which hyperparameters are searched logarithmically.
  The default still guesses from the bounds -- log wherever both are strictly positive --
  because length scales, variances and noise are scale-like and a log makes the
  likelihood far more stationary in them. Positivity is only a proxy for that, though: a
  hyperparameter that is positive but enters the likelihood *additively*, such as a
  position in a non-stationary or Gibbs kernel, is hurt by it. On an objective quadratic
  in two such parameters over bounds [0.1, 100], the log turned a clean quadratic into a
  flat-then-explosive surface -- the worst case for a stationary surrogate -- and cost a
  median error of 0.42 against 0.00 searching linearly. Pass ``False``, ``True``, or a
  per-dimension sequence of booleans.

  There is deliberately no automatic warning for this. Whether a positive hyperparameter
  is scale-like cannot be read off its bounds, and the obvious proxy -- positive bounds
  spanning several decades -- fires on ``[1e-2, 1e1]`` length scales, the commonest
  correct usage.

* ``train(method='bo')`` now warns when it is being misapplied, since it degrades
  quietly rather than raising. Bayesian optimization is built for a handful of
  hyperparameters, which is what the default kernel gives; a user-supplied kernel, prior
  mean or noise function can produce a far longer vector -- a deep kernel runs to
  hundreds -- and the method falls apart there. On a smooth quadratic with a 60-evaluation
  budget, the friendliest surface there is, the distance to the known optimum goes 0.00 at
  2 hyperparameters, 0.67 at 5, 3.22 at 10, 5.20 at 20 and 15.03 at 40.

  Two checks, both before the expensive run starts: above roughly 20 hyperparameters
  (with a softer note past 10), and when the space-filling initial design would consume
  the whole budget, which leaves the run doing no Bayesian optimization at all and is
  otherwise silent. The design-size rule is shared with the optimizer through the new
  ``fvgp.gp_bo.default_initial_design_size`` so the warning cannot drift out of step with
  it.

* When no noise is known -- an exact linalg mode, or a user objective that cannot report
  its precision -- ``method='bo'`` learns a single homoscedastic noise level as an extra
  surrogate hyperparameter, with its lower bound acting as a nugget. A deterministic
  objective drives it to that bound and the surrogate interpolates. This is what makes
  ``bo`` usable with any objective rather than only with stochastic ones.

* ``method='bo'`` also runs asynchronously via ``train(asynchronous=True)``, like
  ``hgdl``, ``mcmc`` and ``adam``. Polling with ``get_latest()`` returns the best
  hyperparameters found so far, so the GP stays usable while the remaining expensive
  evaluations continue on a worker. Note that on completion the reported point
  switches from the best *observed* to the noise-aware recommendation, so the reported
  objective can tick up on the final poll.

* gp2Scale may now train with ``method='bo'`` instead of being forced to ``mcmc``,
  which is the point: its stochastic log-determinant is the regime BO is built for.
  gp2Scale training remains synchronous whatever the method, since it already owns the
  Dask client for the covariance.

* gp2Scale assembles every distributed covariance through one primitive, and lets you
  choose how the work is cut across the cluster. ``distributed_covariance`` in the new
  ``fvgp/gp2Scale_covariance.py`` serves the prior covariance, the blocks an append adds,
  and the posterior's cross-covariance; they differ only in whether the two point sets
  are the same. Previously the first two were near-duplicate functions and the posterior
  had no distributed path at all -- it evaluated ``k(x_data, x_pred)`` on the client as a
  single dense (N x n_pred) array, which is the one thing gp2Scale exists to avoid. That
  cross-covariance is now distributed and stays sparse, so ``posterior_mean`` never
  materializes it. ``posterior_covariance`` cannot escape a dense solve, since KV^-1 is
  dense whatever KV is, and instead chunks over prediction points to cap the intermediate
  at (N x chunk).

  The new ``gp2Scale_distribution`` chooses between ``"blockwise"`` -- the default and the
  historical behavior, mapping (row block, column block) pairs and scheduling only the
  upper triangle when symmetric, so the cluster does half the kernel evaluations -- and
  ``"rowwise"``, where each worker returns a finished CSR row strip. Row-wise moves the
  COO-to-CSR sort onto the workers and reduces host assembly to a concatenation with no
  global COO and no mirroring, at the cost of doubling the kernel evaluations, since
  symmetry cannot then be exploited. It is the choice when host assembly rather than
  kernel evaluation is the bottleneck. Measured at N=6000 on two workers with a
  dense-evaluated kernel, block-wise is still ahead (0.84 s against 1.28 s): row-wise
  earns its keep on host memory and assembly at scale, not on a small problem.

  Assembly is faster either way. The per-block mask that keeps the upper triangle of a
  diagonal block was a Python list comprehension over every nonzero -- 97.6 ms against
  0.34 ms vectorized, on 1.1M nonzeros, once per diagonal block per likelihood
  evaluation. Empty blocks are skipped instead of shipped, indices are int32 wherever the
  matrix allows, and a block and its mirror are written into a single preallocation
  rather than two ``np.hstack`` passes over everything.

* gp2Scale no longer shreds a thin covariance into near-empty tasks. Both distributions
  chunked the *row* axis by ``gp2Scale_batch_size`` however narrow the other axis was, so a
  posterior at 2 prediction points with a batch size of 10 became 1000 dask tasks of 20
  entries: 1.8 s for a 10 000 x 2 result that takes 0.83 ms in a single kernel call, a
  factor of 2171. Every append had the same shape -- ``k(x_old, x_new)`` with 5 new points
  is N x 5. Two things let it through: the posterior gated on ``len(x_data) > batch_size``,
  a *row count* that is true for any real dataset whatever the column count, and the append
  path had no gate at all.

  ``gp2Scale_batch_size`` is now read as the per-task RAM budget it already implies --
  ``B x B`` entries block-wise, ``N x B`` row-wise -- and that budget decides both whether
  to distribute and how to cut. Anything that fits in one task is computed in a single
  kernel call instead of being scheduled; ``k(x_new, x_new)`` and ``k(x_pred, x_pred)`` fall
  out of this automatically.

  Note that the two budgets are **not comparable numbers**: at ``B = 10000`` and N = 1e6
  they are 0.8 GB and 80 GB. The same setting can therefore leave a computation local under
  one distribution and distribute it under the other, which is deliberate -- each mode gets
  the memory its own user declared, and no single budget can serve both without
  over-committing one of them.

* Cross-covariances are always computed in strips, whatever ``gp2Scale_distribution`` says.
  The posterior's ``k(x_data, x_pred)`` and the ``k(x_old, x_new)`` block of an append are
  tall and thin, and blocking a tall thin matrix multiplies the task count without reducing
  per-task memory -- the long axis has to be traversed either way. The short axis is split
  into strips as wide as the budget allows, each task spanning the long axis: at N = 1e8
  with 8 prediction points and a strip width of 4, two tasks of 1e8 x 4.
  ``gp2Scale_distribution`` now governs only the symmetric case, the prior and
  ``k(x_new, x_new)``.

* ``gp2Scale_batch_size`` is a maximum with a remainder, not a target to divide around.
  It used to cut ``n`` into ``n // batch_size`` *equal* chunks, which made the setting a
  lower bound: 19 999 points at a batch size of 15 000 came out as one 19 999-wide chunk,
  a 3.2 GB dense block from a number that reads like a promise of 15 000. Every chunk is
  now exactly ``batch_size`` except the last along each axis -- 100 000 points at 15 000
  give six chunks of 15 000 and one of 10 000 -- so peak memory per worker follows the
  number the user set.

* ``gp2Scale_distribution="rowwise"`` is now genuinely row-wise: one kernel call per
  strip against *every* column, where it previously walked the strip in column chunks and
  so was block-wise work under a different task shape. ``gp2Scale_batch_size`` is the
  **strip width** in this mode, and that is documented now -- it was not before.

  Handing the kernel a whole row is the point of the mode. A support-aware kernel prunes
  an entire row in one neighbor search, and a kernel that vectorizes or offloads gets one
  large call instead of many small ones. Measured at N=6000 on two workers, this flips the
  comparison for the kernels the mode exists for: with the support-aware sparse Wendland,
  row-wise now beats block-wise (0.171 s against 0.193 s) despite doing twice the nominal
  work, where with a dense kernel block-wise still leads (0.85 s against 1.39 s).

  The cost is that a dense kernel materializes ``strip_width x N`` on the worker, so the
  strip width is what bounds worker memory -- 8 GB at N=100 000 and a 10 000-wide strip.
  Row-wise also produces ``N / strip_width`` tasks against block-wise's
  ``(N / batch_size)^2 / 2``, so it usually wants a smaller batch size on a large cluster.
  Both are now in the ``gp2Scale_batch_size`` docstring.

* A requested GPU engine is honored wherever that engine is used, and a request that
  cannot be met now says why instead of quietly computing on the CPU.
  ``args["GPU_engine"]`` accepts ``"torch"`` (or ``"pytorch"``) and ``"cupy"``, and
  ``gp_lin_alg.get_gpu_engine`` -- which every GPU path resolves through -- distinguishes
  the two failure modes a user has to fix differently: "cupy is not installed" against
  "pytorch is installed but exposes no usable CUDA or MPS device". ``kernels.py`` kept a
  second, private backend detector that took no ``args`` and so ignored the request
  entirely; it resolves through the same helper now, and the GPU Wendland kernels take an
  optional ``args`` so the choice reaches the dask workers.

* Preconditioner construction is logged and timed like the solvers it feeds. Every type
  reports ``"<type> preconditioner construction in progress ..."`` and a compute time
  carrying the problem size (``n``, ``K+V nnz``), so the cost can be judged against the
  solve it buys -- which is the actual question when tuning
  ``sparse_preconditioner_max_matrix_drift``. A *reused* preconditioner says so too,
  phrased with the same leading type name: without that line its missing construction
  time reads as a preconditioner that never ran, and one grep now shows the whole story
  of builds and reuses. Application cost is not broken out separately; it happens inside
  the CG/MINRES iterations and is already inside their reported time.

* fvGP supports Python 3.10 through 3.14, and CI tests all five. The obstacle was never
  the code but the ``~=`` dependency pins: ``scipy ~= 1.16.0`` requires Python >= 3.11
  and had no cp314 wheel, so no single pin can span the range. The dependencies are
  ranges now and pip resolves per interpreter -- 3.10 lands on scipy 1.13 / dask 2024.1
  through hgdl 2.2.3, while 3.11 and up get the 2025-era stack. ``pyamg`` and ``ilupp``
  moved to a ``preconditioners`` extra: ilupp publishes no Linux wheels at all and pyamg
  none for 3.14, so both build from source, and the suite skips the tests that need them
  rather than failing.

Bug fixes
~~~~~~~~~

* Fixed the BO surrogate provoking fvGP's "Negative variances encountered" warning.
  The cause was the design, not the noise level as such: once the search converges it
  proposes points a whisker apart, and near-duplicate rows make K+V numerically
  singular, so the posterior variance -- a difference of nearly equal numbers -- tips
  below zero. Over 340 surrogate fits, 198 contained a pair of points closer than 1e-6
  in the unit cube, with a median closest separation of 5.6e-8. The nugget rises from
  1e-10 to 1e-7 of the residual scale, which clears the warning in every case tested;
  it is far below any noise a real objective would carry, and the recovered optimum is
  unchanged at every level tried.

  The worst offender was the path with a *declared* noise level, which applied no floor
  at all and so was less protected than the learned one. Reported noise is now floored
  by the same nugget: a stochastic estimator can legitimately report a variance far
  below what the conditioning of a duplicated design tolerates.

  This also interacts with the ``CholInv`` mode adopted for speed -- an explicit inverse
  is less accurate than a Cholesky solve on an ill-conditioned K+V, so part of the
  nugget is buying back the headroom that traded away.

* Multi-task ``posterior_covariance`` returned a scrambled ``"S"``. The flat
  product-space index is task-major (``k = point + Npts*task``), which is what
  ``cartesian_product`` builds and what the ``order='F'`` reshape of ``v`` assumes,
  but ``S`` was reshaped straight from ``(Npts*No, Npts*No)`` to
  ``(Npts, Npts, No, No)``. Those place values do not line up, so the point and task
  axes were interleaved. ``S[i, j, t, u]`` is now
  ``Cov(f(x_i, task_t), f(x_j, task_u))`` as intended, and the returned shape is
  unchanged, so only the previously-wrong values move.

  Nothing inside fvGP or gpCAM read the reshaped ``S`` -- ``gp_kl_div``,
  ``gp_relative_information_entropy`` and ``posterior_probability`` all use
  ``"S_flat"``, ``gp_mutual_information`` and ``gp_total_correlation`` build their own
  covariance, and gpCAM's acquisition functions use ``"S_flat"`` too. No information
  measure or acquisition function was affected; the scrambled array only ever reached
  a caller who asked for ``"S"`` directly. (The ``"S"`` returned by
  ``joint_gp_prior`` is a separate, flat, and correct array.)

* Every data update on a non-Euclidean GP raised ``AttributeError``. ``GPdata.update``
  asserted ``x_data_new.shape[1]`` on what is by definition a list, so evaluating the
  assertion's own condition threw before any update logic ran -- and only because
  assertions were enabled, which makes ``python -O`` the one way the feature worked. The
  same assertion also demanded a 2-d list of width ``index_set_dim``, contradicting
  ``__init__``, which accepts any list and sets ``index_set_dim`` to 1 regardless. Both
  documented forms were rejected: the flat single-task list the examples use, and the
  ``[point, task]`` pairs the fvGP index-set transform builds itself. A non-Euclidean
  point is an arbitrary object, so there is no dimensionality to check, and the assertion
  now checks only that it is a list.

  ``GPprior._update_prior`` separately rebuilt the full input set with ``np.vstack``,
  which cannot concatenate lists. It uses ``self.x_data``, already the appended set at
  that point, which is also one fewer array build on every Euclidean append.

* gp2Scale never worked on a non-Euclidean input space. ``client.scatter`` on a list
  scatters it *element-wise* and hands back a list of futures rather than one future for
  the point set, so the workers received something other than the data they meant to
  slice. Point sets are now scattered as a single object. Object-typed points are
  regression-tested across strings, lists, lists of differing length, tuples, dicts and
  custom classes -- a point that is itself a sequence is where numpy-based introspection
  of the input set goes wrong, silently for a uniform list and by raising for a ragged
  one.

* A four-argument, ``args``-taking kernel raised ``TypeError`` on the gp2Scale workers.
  The distributed workers called ``kernel(x1, x2, hps)`` unconditionally, while the
  signature is supported everywhere else in fvGP. The worker now dispatches on arity the
  same way ``GPprior.compute_covariances`` does.

* imate's GPU was gated on pytorch and cupy. imate ships its own CUDA backend and reaches
  a GPU through neither, so the check disabled a perfectly usable GPU on any machine
  where those two packages happen not to see one -- which is every machine with a
  CPU-only pytorch build. The stochastic log-determinant now asks imate itself, and says
  so when a GPU was requested and imate reports none.

* Importing imate silenced every subsequent warning, process-wide. imate calls
  ``logging.captureWarnings(True)`` on import, which redirects warnings to the
  ``py.warnings`` logger; that logger gets a ``NullHandler``, so they are discarded --
  not only fvGP's, but numpy's and scipy's too. fvGP imports imate on the first
  stochastic log-determinant, so every gp2Scale training run went quiet from that point
  on. The import now restores whatever ``showwarning`` was in place, and only if imate
  actually changed it, so a caller's own handler survives.

* ``fvGP.update_gp_data`` no longer accepts list-valued noise variances, a relic from
  before missing tasks were signalled with ``np.nan``. The index-set transform reads
  noise as ``noise_variances[j, i]``, which a list cannot do, so that branch could never
  have worked; it now fails at the argument check with a clear message instead.

Documentation
~~~~~~~~~~~~~

* ``posterior_covariance`` now documents its return keys -- the task-major flat
  indexing and the shapes of ``v(x)``/``S`` against ``v_flat``/``S_flat``. Previously
  the docstring said only ``Solution : dict``, which is much of why the layout above
  could be wrong without anyone noticing.

* The ``kernel_function`` docstrings in ``GP`` and ``fvGP`` now state what the default
  is under gp2Scale -- a compactly supported anisotropic Wendland kernel, chosen by
  ``compute_device`` -- and point at ``fvgp.kernels`` for those, their support-aware
  sparse variants, and the rest. They also no longer reference ``fvgp.GP.default_kernel``,
  which does not exist; the default is described for what it is, a stationary anisotropic
  Matern kernel of first-order differentiability with one length scale per input
  dimension.

Internal
~~~~~~~~

* ``sequential_linalg_state`` and its two setting constants moved from ``gp_bo`` to
  ``gp_kv``. They govern which linear-algebra state may persist between likelihood
  evaluations for *every* training method, so they never belonged with Bayesian
  optimization; ``gp_kv`` already owns the per-evaluation staleness checks they pair
  with. Import path only -- no behavior change.

* Test coverage of the package is 100% (224 tests), up from 80%. ``ggmp.py`` is omitted
  through a new ``[tool.coverage]`` section in ``pyproject.toml``, which is what
  ``CLAUDE.md`` already claimed was happening. Three groups of code had never been
  measured for structural reasons rather than neglect: the ``gp_actor`` classes and the
  gp2Scale worker functions only ever run on dask workers, where coverage is not
  collected, and are now driven in-process, which is a sharper test as well as a
  measurable one; ``gp_lin_alg``'s CPU-side error paths and message builders had no tests
  at all; and nothing had ever updated a non-Euclidean GP, which is how the bugs above
  survived. Seventeen ``# pragma: no cover`` markers cover what genuinely cannot run on a
  CPU-only runner -- GPU backends, branches unreachable behind an earlier assertion, and
  numerical breakdowns that cannot be forced deterministically -- each carrying its
  reason.
