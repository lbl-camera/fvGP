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
                 info):
        from .gp_mcmc import gpMCMC
        self._mcmc = gpMCMC(log_likelihood_function, prior_function=prior_function, proposal_distributions=proposal_distributions , bounds=bounds, args=args)
        self._x0 = x0
        self._n_updates = n_updates
        self._info = info
        self._lock = threading.Lock()
        self._latest = {}
        self._running = False
        self._thread = None

    def start(self):
        self._running = True

        def _callback(mcmc_obj):
            trace = mcmc_obj.trace
            if not trace["f(x)"]:
                return
            xs = np.asarray(trace["x"])
            fxs = np.asarray(trace["f(x)"])
            arg_max = int(np.argmax(fxs))
            dist_index = max(0, int(len(xs) - len(xs) / 100))
            with self._lock:
                self._latest = {
                    "f(x)": fxs,
                    "max f(x)": fxs[arg_max],
                    "MAP": fxs[arg_max],
                    "max x": xs[arg_max],
                    "time stamps": list(trace["time stamp"]),
                    "x": xs,
                    "mean(x)": np.mean(xs[dist_index:], axis=0),
                    "median(x)": np.median(xs[dist_index:], axis=0),
                    "var(x)": np.var(xs[dist_index:], axis=0),
                }

        def _break(mcmc_obj):
            if not self._running:
                return True
            return self._mcmc._default_break_condition(mcmc_obj)

        def _run():
            self._mcmc.run_mcmc(
                x0=self._x0,
                n_updates=self._n_updates,
                info=self._info,
                break_condition=_break,
                run_in_every_iteration=_callback,
            )
            self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get_latest(self):
        with self._lock:
            return dict(self._latest)

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
