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

Bug fixes
~~~~~~~~~

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

Documentation
~~~~~~~~~~~~~

* ``posterior_covariance`` now documents its return keys -- the task-major flat
  indexing and the shapes of ``v(x)``/``S`` against ``v_flat``/``S_flat``. Previously
  the docstring said only ``Solution : dict``, which is much of why the layout above
  could be wrong without anyone noticing.
