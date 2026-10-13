"""Async optimizer actors for MCMC and Adam training on a Dask worker."""

import threading
import numpy as np


class _MCMCActor:
    """Runs gpMCMC in a background thread; lives on a Dask worker."""

    def __init__(self, 
                 log_likelihood_function,
                 bounds,
                 prior_function, 
                 proposal_distributions,
                 args, 
                 x0, 
                 n_updates, 
                 info,
                 burn_in=0.5,
                 break_condition="default",
                 rng=None,
                 log_scale=None):
        from .gp_mcmc import gpMCMC
        self._mcmc = gpMCMC(log_likelihood_function, prior_function=prior_function, proposal_distributions=proposal_distributions,
                            bounds=bounds, args=args, log_scale=log_scale)
        self._x0 = x0
        self._n_updates = n_updates
        self._info = info
        self._burn_in = burn_in
        self._break_condition = break_condition
        self._rng = rng
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        if self._break_condition == "default": condition = self._mcmc._default_break_condition
        elif self._break_condition is None: condition = lambda mcmc_obj: False
        else: condition = self._break_condition

        def _break(mcmc_obj):
            return not self._running or condition(mcmc_obj)

        def _run():
            self._mcmc.run_mcmc(
                x0=self._x0,
                n_updates=self._n_updates,
                info=self._info,
                break_condition=_break,
                burn_in=self._burn_in,
                rng=self._rng,
            )
            self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get_latest(self):
        """The summary of the chain so far (see :py:func:`fvgp.gp_mcmc.summarize`).

        Built here, when polled, rather than in every iteration, which made a long run
        quadratic. The run thread only appends to the trace lists, so copying them is a
        consistent snapshot once cut to their common length (summarize does that).
        """
        from .gp_mcmc import summarize
        trace = self._mcmc.trace
        if not trace:
            return {}
        with self._lock:
            snapshot = {key: list(values) for key, values in trace.items()}
        return summarize(snapshot, self._burn_in)

    def stop(self):
        self._running = False


class _AdamActor:
    """Runs Adam in a background thread; lives on a Dask worker."""

    def __init__(self, nlml, grad_nlml, theta0, lr, beta1, beta2, eps, max_iter, tol):
        self._nlml = nlml
        self._grad_nlml = grad_nlml
        self._theta0 = theta0.copy()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._max_iter = max_iter
        self._tol = tol
        self._lock = threading.Lock()
        self._latest = {"x": theta0.copy(), "iteration": 0, "nlml": None, "grad_norm": None}
        self._running = False
        self._thread = None

    def start(self):
        self._running = True

        def _callback(theta, fval, grad, iteration):
            with self._lock:
                self._latest = {
                    "x": theta.copy(),
                    "iteration": iteration,
                    "nlml": float(fval),
                    "grad_norm": float(np.linalg.norm(grad)),
                }

        def _run():
            from .gp_training import GPtraining
            GPtraining.adam_optimize(
                self._nlml,
                self._grad_nlml,
                self._theta0,
                lr=self._lr,
                beta1=self._beta1,
                beta2=self._beta2,
                eps=self._eps,
                max_iter=self._max_iter,
                tol=self._tol,
                callback=_callback,
                early_stop=lambda: not self._running,
            )
            self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get_latest(self):
        with self._lock:
            return dict(self._latest)

    def stop(self):
        self._running = False


class _BOActor:
    """Runs Bayesian optimization in a background thread; lives on a Dask worker."""

    def __init__(self, objective_function, hyperparameter_bounds, init_hyperparameters,
                 max_iter, bo_args, info):
        self._objective_function = objective_function
        self._bounds = hyperparameter_bounds
        self._x0 = np.asarray(init_hyperparameters).copy()
        self._max_iter = max_iter
        self._bo_args = bo_args
        self._info = info
        self._lock = threading.Lock()
        self._latest = {"x": self._x0.copy(), "iteration": 0,
                        "objective": None, "n_evaluations": 0, "status": "queued"}
        self._running = False
        self._thread = None

    def start(self):
        self._running = True

        def _callback(theta, fval, iteration, state):
            # while running, `x` is the best point *observed* so far
            with self._lock:
                self._latest = {
                    "x": np.asarray(theta).copy(),
                    "iteration": iteration,
                    "objective": float(fval),
                    "n_evaluations": state.get("n_evaluations", 0),
                    "status": "running",
                }

        def _run():
            from .gp_bo import bayesian_optimize
            try:
                theta, bo_info = bayesian_optimize(
                    self._objective_function,
                    self._bounds,
                    self._x0,
                    max_iter=self._max_iter,
                    bo_args=self._bo_args,
                    info=self._info,
                    callback=_callback,
                    early_stop=lambda: not self._running,
                )
                with self._lock:
                    self._latest = dict(self._latest)
                    # On completion `x` switches from the best point observed to the
                    # noise-aware recommendation (the evaluated point with the best
                    # surrogate posterior mean). Those differ on purpose: the smallest
                    # observed value is partly a lucky draw of the estimator noise. So
                    # the reported `objective` can tick *up* on the final poll -- that
                    # is the recommendation being less lucky, not the search regressing.
                    self._latest["x"] = np.asarray(theta).copy()
                    self._latest["objective"] = float(bo_info["f(x)"])
                    self._latest["n_evaluations"] = bo_info["n_evaluations"]
                    self._latest["status"] = "finished"
                    # the surrogate itself is not shipped back across the wire
                    self._latest["sensitivity"] = bo_info["sensitivity"]
                    self._latest["posterior covariance"] = bo_info["posterior covariance"]
                    self._latest["ard length scales"] = bo_info["ard length scales"]
            finally:
                self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get_latest(self):
        with self._lock:
            return dict(self._latest)

    def stop(self):
        self._running = False


class AsyncOptimizer:
    """
    Proxy returned by ``train(asynchronous=True)`` for ``method='mcmc'`` or ``method='adam'``.

    Wraps a Dask Actor so the caller works with a plain synchronous interface,
    mirroring the pattern used by the HGDL optimizer.

    Parameters
    ----------
    actor : Dask Actor
        Actor instance returned by ``dask_client.submit(..., actor=True).result()``.
    """

    def __init__(self, actor):
        self._actor = actor

    def get_latest(self):
        """
        Return the latest optimizer state.

        Returns
        -------
        state : dict
            For MCMC: the full trace summary dict (same keys as the synchronous result).
            For Adam: ``{"x", "iteration", "nlml", "grad_norm"}``.
            For BO: ``{"x", "iteration", "objective", "n_evaluations", "status"}``, plus
            ``sensitivity``, ``posterior covariance`` and ``ard length scales`` once
            ``status == "finished"``. While ``status == "running"``, ``x`` is the best
            point observed so far; on completion it becomes the noise-aware
            recommendation, so ``objective`` may tick up on the final poll.
        """
        return self._actor.get_latest().result()

    def stop(self):
        """Signal the optimizer to stop after the current iteration."""
        self._actor.stop().result()

    def cancel_tasks(self):
        """Alias for :py:meth:`stop`; matches the HGDL interface."""
        self.stop()

    def kill_client(self):
        """Stop the optimizer. The Dask client is managed externally by the user."""
        self.stop()
