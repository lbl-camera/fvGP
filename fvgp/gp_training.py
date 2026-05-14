import warnings
import numpy as np
from loguru import logger
from scipy.optimize import differential_evolution
from hgdl.hgdl import HGDL
from scipy.optimize import minimize
from .gp_mcmc import *
from .gp_actor import _MCMCActor, _AdamActor, AsyncOptimizer
warnings.simplefilter("once", UserWarning)


class GPtraining:
    def __init__(self, data, hyperparameters):
        self.mcmc_info = None
        self.data = data
        self.hyperparameters = hyperparameters

    @property
    def args(self):
        return self.data.args

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    def train(self,
              objective_function=None,
              objective_function_gradient=None,
              objective_function_hessian=None,
              hyperparameter_bounds=None,
              init_hyperparameters=None,
              method="global",
              pop_size=20,
              tolerance=0.0001,
              max_iter=120,
              local_optimizer="L-BFGS-B",
              global_optimizer="genetic",
              constraints=(),
              mcmc_prior=None,
              mcmc_prop_distrs="normal",
              mcmc_args={},
              dask_client=None,
              info=False):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be 'hgdl' and
        providing a dask client. However, in that case fvgp.GP.hgdl_async() is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.
        """
        if not self._in_bounds(init_hyperparameters, hyperparameter_bounds):
            raise Exception("Starting positions outside of optimization bounds.", init_hyperparameters, hyperparameter_bounds)

        ############################
        ####global optimization:##
        ############################
        if method == "global":
            logger.debug(
                "fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hyperparameter_bounds)
            res = differential_evolution(
                objective_function,
                hyperparameter_bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=tolerance,
                disp=info,
                polish=False,
                x0=init_hyperparameters.reshape(1, -1),
                constraints=constraints,
                workers=1,
            )
            hyperparameters = np.array(res["x"])
            logger.debug(f"fvGP found hyperparameters {hyperparameters} with objective function eval {res['fun']} \
            via global optimization")
        ############################
        ####local optimization:#####
        ############################
        elif method == "local":
            logger.debug("fvGP is performing a local update of the hyper parameters.")
            logger.debug("starting hyperparameters: {}", init_hyperparameters)
            logger.debug("Attempting a BFGS optimization.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hyperparameter_bounds)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                OptimumEvaluation = minimize(
                    objective_function,
                    init_hyperparameters,
                    method=local_optimizer,
                    jac=objective_function_gradient,
                    hess=objective_function_hessian,
                    bounds=hyperparameter_bounds,
                    tol=tolerance,
                    callback=None,
                    constraints=constraints,
                    options={"maxiter": max_iter})

            if OptimumEvaluation["success"]:
                logger.debug(f"fvGP local optimization successfully concluded with result: "
                             f"{OptimumEvaluation['fun']} at {OptimumEvaluation['x']}")
            else:
                logger.debug("fvGP local optimization not successful.")
            hyperparameters = OptimumEvaluation["x"]
        ############################
        ####hybrid optimization:####
        ############################
        elif method == "hgdl":
            logger.debug("fvGP submitted HGDL optimization")
            logger.debug("starting hyperparameters: {}", init_hyperparameters)
            logger.debug('bounds are {}', hyperparameter_bounds)

            opt_obj = HGDL(objective_function,
                           objective_function_gradient,
                           hyperparameter_bounds,
                           hess=objective_function_hessian,
                           local_optimizer=local_optimizer,
                           global_optimizer=global_optimizer,
                           num_epochs=max_iter,
                           constraints=constraints)

            opt_obj.optimize(dask_client=dask_client, x0=init_hyperparameters.reshape(1, -1))
            try:
                hyperparameters = opt_obj.get_final()[0]["x"]
            except Exception as ex:
                raise Exception("Something has gone wrong with the objective function evaluation.") from ex

        elif method == "mcmc":
            logger.debug("MCMC started in fvGP")
            logger.debug('bounds are {}', hyperparameter_bounds)

            def prior_function(theta, bounds, args):
                if self._in_bounds(theta, bounds): return 0.
                else: return -np.inf

            def likelihood_func(hps, args):
                return objective_function(hps)
    
            if mcmc_prior is not None: prior_function = mcmc_prior

            myMCMC = gpMCMC(likelihood_func, prior_function=prior_function, proposal_distributions=mcmc_prop_distrs , bounds=hyperparameter_bounds, args=mcmc_args)
            res = myMCMC.run_mcmc(x0=init_hyperparameters, n_updates=max_iter, info=info, break_condition="default")
            hyperparameters = res["median(x)"]
            self.mcmc_info = res
        elif method == "adam":
            hyperparameters, history = self.adam_optimize(objective_function,
                                                          objective_function_gradient,
                                                          init_hyperparameters, max_iter=max_iter)
        elif callable(method): hyperparameters = method(self)
        else: raise ValueError("No optimization mode specified in fvGP")
        assert isinstance(hyperparameters, np.ndarray) and np.ndim(hyperparameters) == 1, \
            "Optimizer returned invalid hyperparameters: " + str(hyperparameters)
        return hyperparameters
    
    def train_async(self,
              dask_client,
              objective_function=None,
              objective_function_gradient=None,
              objective_function_hessian=None,
              hyperparameter_bounds=None,
              init_hyperparameters=None,
              method="global",
              pop_size=20,
              tolerance=0.0001,
              max_iter=120,
              local_optimizer="L-BFGS-B",
              global_optimizer="genetic",
              constraints=(),
              mcmc_prior=None,
              mcmc_prop_distrs="normal",
              mcmc_args={},
              info=False):

        """
        Submit an asynchronous training run and return an optimizer proxy.

        Supports ``method='hgdl'``, ``'mcmc'``, and ``'adam'``. The returned object
        can be polled with ``get_latest()`` and stopped with ``stop()``.
        Pass the returned object to ``GP.update_hyperparameters()`` to pull the
        latest result into the GP.
        """
        assert method == "hgdl" or method == "mcmc" or method == "adam", \
            "Asynchronous training only supported for hgdl, mcmc, adam; got method=" + str(method)
        if method == 'hgdl':
            opt_obj = self.hgdl_async(
                objective_function=objective_function,
                objective_function_gradient=objective_function_gradient,
                objective_function_hessian=objective_function_hessian,
                hyperparameter_bounds=hyperparameter_bounds,
                init_hyperparameters=init_hyperparameters,
                max_iter=max_iter,
                local_optimizer=local_optimizer,
                global_optimizer=global_optimizer,
                constraints=constraints,
                dask_client=dask_client,
            )
        elif method == 'mcmc':
            opt_obj = self.mcmc_async(
                objective_function=objective_function,
                hyperparameter_bounds=hyperparameter_bounds,
                mcmc_prior=mcmc_prior,
                mcmc_prop_distrs=mcmc_prop_distrs,
                init_hyperparameters=init_hyperparameters,
                max_iter=max_iter,
                info=info,
                mcmc_args=mcmc_args,
                dask_client=dask_client,
            )
        elif method == 'adam':
            opt_obj = self.adam_async(
                objective_function=objective_function,
                objective_function_gradient=objective_function_gradient,
                init_hyperparameters=init_hyperparameters,
                max_iter=max_iter,
                dask_client=dask_client,
            )
        return opt_obj

    ##################################################################################
    def hgdl_async(self,
                   objective_function=None,
                   objective_function_gradient=None,
                   objective_function_hessian=None,
                   hyperparameter_bounds=None,
                   init_hyperparameters=None,
                   max_iter=10000,
                   local_optimizer="L-BFGS-B",
                   global_optimizer="genetic",
                   constraints=(),
                   dask_client=None):
        """
        This function asynchronously finds the maximum of the log marginal likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to :py:meth:`fvgp.GP.update_hyperparameters`,
        which will automatically update the GP prior with the new hyperparameters.
        """

        opt_obj = self._optimize_log_likelihood_async(
            objective_function,
            objective_function_gradient,
            objective_function_hessian,
            init_hyperparameters,
            hyperparameter_bounds,
            max_iter,
            constraints,
            local_optimizer,
            global_optimizer,
            dask_client)
        return opt_obj

    ##################################################################################
    def mcmc_async(self,
                   objective_function,
                   hyperparameter_bounds,
                   mcmc_prior,
                   mcmc_prop_distrs,
                   init_hyperparameters,
                   max_iter,
                   info,
                   mcmc_args,
                   dask_client=None):
        """
        Submit an asynchronous MCMC run to a Dask worker and return an :py:class:`AsyncOptimizer` proxy.

        Parameters
        ----------
        objective_function : callable
            Log-likelihood function ``f(hps) -> float``.
        hyperparameter_bounds : np.ndarray
            Bounds of shape (N, 2).
        mcmc_prior : callable or None
            Optional prior function ``p(hps) -> float``. If None, a uniform prior.
        mcmc_prop_distrs : list of callables or str
            A list of functions that define the proposal distributions for the MCMC sampler. 
            Each function should have the form f(x, para, obj) and return a vector of the same shape as x.
            See :py:class:`fvgp.gp_mcmc.ProposalDistribution` in the documentation.
        init_hyperparameters : np.ndarray
            Starting hyperparameters of shape (N,).
        max_iter : int
            Maximum number of MCMC steps. Default is 10000.
        info : bool
            Print progress every 10 iterations. Default is False.
        mcmc_args : dict
            A dictionary of additional arguments for the MCMC sampler. The default is an empty dictionary.
        dask_client : distributed.Client
            Dask client used to host the actor on a worker.

        Returns
        -------
        opt_obj : AsyncOptimizer
            Proxy with ``get_latest()`` and ``stop()`` methods.
        """

        def prior_function(theta, bounds, args):
            if self._in_bounds(theta, bounds): return 0.
            else: return -np.inf

        def likelihood_func(hps, args):
            return objective_function(hps)

        if mcmc_prior is not None: prior_function = mcmc_prior

        actor_future = dask_client.submit(
            _MCMCActor,
            likelihood_func,
            hyperparameter_bounds,
            prior_function,
            mcmc_prop_distrs,
            mcmc_args,
            init_hyperparameters,
            max_iter,
            info,
            actor=True,
        )
        actor = actor_future.result()
        actor.start()
        return AsyncOptimizer(actor)

    ##################################################################################
    def adam_async(self,
                   objective_function,
                   objective_function_gradient,
                   init_hyperparameters,
                   max_iter=1000,
                   dask_client=None):
        """
        Submit an asynchronous Adam run to a Dask worker and return an :py:class:`AsyncOptimizer` proxy.

        Parameters
        ----------
        objective_function : callable
            Negative log-likelihood ``f(hps) -> float``.
        objective_function_gradient : callable
            Gradient ``g(hps) -> np.ndarray`` of shape (N,).
        init_hyperparameters : np.ndarray
            Starting hyperparameters of shape (N,).
        max_iter : int, optional
            Maximum number of Adam steps. Default is 1000.
        dask_client : distributed.Client
            Dask client used to host the actor on a worker.

        Returns
        -------
        opt_obj : AsyncOptimizer
            Proxy with ``get_latest()`` and ``stop()`` methods.
        """
        actor_future = dask_client.submit(
            _AdamActor,
            objective_function,
            objective_function_gradient,
            init_hyperparameters,
            1e-2,    # lr
            0.9,     # beta1
            0.999,   # beta2
            1e-8,    # eps
            max_iter,
            1e-6,    # tol
            actor=True,
        )
        actor = actor_future.result()
        actor.start()
        return AsyncOptimizer(actor)

    ##################################################################################
    @staticmethod
    def stop_training(opt_obj):
        """
        Stop an asynchronous training run, leaving the Dask client alive.

        Parameters
        ----------
        opt_obj : object
            Object returned by :py:meth:`train_async`.
        """
        try:
            opt_obj.cancel_tasks()
            logger.debug("fvGP successfully cancelled the current training.")
        except Exception:
            warnings.warn("No asynchronous training to be cancelled in fvGP, \
            no training is running.", UserWarning, stacklevel=2)

    ###################################################################################
    @staticmethod
    def kill_client(opt_obj):
        """
        Stop an asynchronous training run and shut down its Dask client.

        Parameters
        ----------
        opt_obj : object
            Object returned by :py:meth:`train_async`.
        """

        try:
            opt_obj.kill_client()
            logger.debug("fvGP successfully killed the training.")
        except Exception:
            warnings.warn("No asynchronous training to be killed, no training is running.", UserWarning, stacklevel=2)

    def update_hyperparameters(self, opt_obj):
        """
        Pull the latest hyperparameters from a running asynchronous optimizer.

        Parameters
        ----------
        opt_obj : object
            Object returned by :py:meth:`train_async` (HGDL, MCMC, or Adam).

        Returns
        -------
        hyperparameters : np.ndarray
            The latest hyperparameter vector from the running optimizer.
        """
        try:
            opt_list = opt_obj.get_latest()
        except Exception as err:   # pragma: no cover
            logger.debug("      The optimizer object could not be queried")
            logger.debug("      That probably means you are not optimizing the hyperparameters asynchronously")
            logger.info("       Hyperparameter update failed with ERROR: " + str(err))
            return self.hyperparameters
        if len(opt_list) == 0:   # pragma: no cover
            logger.debug("      The list of optima had len=0., No update.")
            warnings.warn("Hyperparameter update not successful len(optima list) = 0", UserWarning, stacklevel=2)
            return self.hyperparameters
        else:   # pragma: no cover
            if isinstance(opt_list, list):
                updated_hyperparameters = opt_list[0]["x"]
            elif isinstance(opt_list, dict):
                if "median(x)" in opt_list: updated_hyperparameters = opt_list["median(x)"]
                elif "x" in opt_list: updated_hyperparameters = opt_list["x"]
                else: raise Exception("Reading the `updated_hyperparameters` was not successful", opt_list)
            else: raise Exception("Reading the `updated_hyperparameters` was not successful", opt_list)
            assert isinstance(updated_hyperparameters, np.ndarray) and np.ndim(updated_hyperparameters) == 1, \
                "async optimizer returned invalid hyperparameters: " + str(updated_hyperparameters)
            #self.hyperparameters = updated_hyperparameters #not needed I believe.
            return updated_hyperparameters

    def _optimize_log_likelihood_async(self,
                                       objective_function,
                                       objective_function_gradient,
                                       objective_function_hessian,
                                       starting_hps,
                                       hp_bounds,
                                       max_iter,
                                       constraints,
                                       local_optimizer,
                                       global_optimizer,
                                       dask_client):

        logger.debug("fvGP hyperparameter tuning in progress. Old hyperparameters: {}", starting_hps)
        if not self._in_bounds(starting_hps, hp_bounds):
            raise Exception("Starting positions outside of optimization bounds.")

        opt_obj = HGDL(objective_function,
                       objective_function_gradient,
                       hp_bounds,
                       hess=objective_function_hessian,
                       local_optimizer=local_optimizer,
                       global_optimizer=global_optimizer,
                       num_epochs=max_iter,
                       constraints=constraints)

        logger.debug("HGDL successfully initialized. Calling optimize()")
        opt_obj.optimize(dask_client=dask_client, x0=np.array(starting_hps).reshape(1, -1))
        logger.debug("optimize() called")
        return opt_obj

    @staticmethod
    def adam_optimize(
        nlml,
        grad_nlml,
        theta0,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        max_iter=1000,
        tol=1e-6,
        callback=None,
        early_stop=None,
    ):
        """
        Adam optimizer for GP hyperparameters.

        Parameters
        ----------
        nlml : callable
            Negative log marginal likelihood: f(theta) -> scalar
        grad_nlml : callable
            Gradient of nlml: g(theta) -> ndarray (d,)
        theta0 : ndarray
            Initial parameter vector (d,)
        lr : float
            Learning rate
        beta1 : float
            Exponential decay for first moment
        beta2 : float
            Exponential decay for second moment
        eps : float
            Numerical stability constant
        max_iter : int
            Maximum iterations
        tol : float
            Stopping tolerance on parameter update norm
        callback : callable or None
            Optional: callback(theta, fval, grad, iteration)

        Returns
        -------
        theta : ndarray
            Optimized parameters
        history : dict
            Optimization trace
        """

        theta = theta0.copy()
        d = theta.size

        m = np.zeros(d)  # first moment
        v = np.zeros(d)  # second moment

        history = {
            "theta": [],
            "nlml": [],
            "grad_norm": [],
        }

        for t in range(1, max_iter + 1):
            fval = nlml(theta)
            g = grad_nlml(theta)

            # Adam moments
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g ** 2)

            # Bias correction
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)

            # Parameter update
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            theta_new = theta - step

            # Bookkeeping
            history["theta"].append(theta.copy())
            history["nlml"].append(fval)
            history["grad_norm"].append(np.linalg.norm(g))

            if callback is not None:
                callback(theta, fval, g, t)

            # Convergence check or external stop signal
            if np.linalg.norm(theta_new - theta) < tol or (early_stop is not None and early_stop()):
                theta = theta_new
                break

            theta = theta_new

        return theta, history

    @staticmethod
    def _in_bounds(v, bounds):
        assert isinstance(bounds, np.ndarray), "bounds must be np.ndarray"
        if np.any(v < bounds[:, 0]) or np.any(v > bounds[:, 1]): return False
        return True

    def __getstate__(self):
        state = dict(
            data=self.data,
            mcmc_info=self.mcmc_info,
            hyperparameters=self.hyperparameters
            )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

