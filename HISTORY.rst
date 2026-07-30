=======
History
=======

This changelog was restarted; for releases before the first entry below, see the
git history and the release tags.

Unreleased
----------

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
