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
