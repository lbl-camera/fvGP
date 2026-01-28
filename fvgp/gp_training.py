import warnings
warnings.simplefilter("once", UserWarning)

from loguru import logger
import numpy as np
from scipy.optimize import differential_evolution
from hgdl.hgdl import HGDL
from scipy.optimize import minimize
from .gp_mcmc import *


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
              dask_client=None,
              info=False):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be 'hgdl' and
        providing a dask client. However, in that case fvgp.GP.train_async() is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.
        """
        hyperparameters = self._optimize_log_likelihood(
            objective_function,
            objective_function_gradient,
            objective_function_hessian,
            init_hyperparameters,
            hyperparameter_bounds,
            method,
            max_iter,
            pop_size,
            tolerance,
            constraints,
            local_optimizer,
            global_optimizer,
            dask_client,
            info
        )
        assert isinstance(hyperparameters, np.ndarray) and np.ndim(hyperparameters) == 1, "hps="+str(hyperparameters)
        self.hyperparameters = hyperparameters
        return hyperparameters

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
        an object which can be given to `fvgp.GP.update_hyperparameters()`,
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
    @staticmethod
    def stop_training(opt_obj):
        """
        This function stops the training if HGDL is used. It leaves the dask client alive.

        Parameters
        ----------
        opt_obj : HGDL object instance
            HGDL object instance returned by `fvgp.GP.train_async()`
        """
        try:
            opt_obj.cancel_tasks()
            logger.debug("fvGP successfully cancelled the current training.")
        except:
            warnings.warn("No asynchronous training to be cancelled in fvGP, \
            no training is running.", UserWarning, stacklevel=2)

    ###################################################################################
    @staticmethod
    def kill_client(opt_obj):
        """
        This function stops the training if HGDL is used, and kills the dask client.

        Parameters
        ----------
        opt_obj : HGDL object instance
            HGDL object instance returned by `fvgp.GP.train_async()`
        """

        try:
            opt_obj.kill_client()
            logger.debug("fvGP successfully killed the training.")
        except:
            warnings.warn("No asynchronous training to be killed, no training is running.", UserWarning, stacklevel=2)

    def update_hyperparameters(self, opt_obj):
        """
        This function asynchronously finds the maximum of the marginal log_likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function just submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`, which will automatically
        update the GP prior with the new hyperparameters.

        Parameters
        ----------
        opt_obj : HGDL object instance
            HGDL object instance returned by `fvgp.GP.train_async()`

        Return
        ------
        The current hyperparameters : np.ndarray
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
            if isinstance(opt_list, list): updated_hyperparameters = opt_obj.get_latest()[0]["x"]
            elif isinstance(opt_list, dict): updated_hyperparameters = opt_obj.get_latest().result()["median(x)"]
            else: raise Exception("Reading the `updated_hyperparameters` was not successful", opt_list)
            assert isinstance(updated_hyperparameters, np.ndarray) and np.ndim(updated_hyperparameters) == 1
            self.hyperparameters = updated_hyperparameters
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

    ##################################################################################
    def _optimize_log_likelihood(self,
                                 objective_function,
                                 objective_function_gradient,
                                 objective_function_hessian,
                                 starting_hps,
                                 hp_bounds,
                                 method,
                                 max_iter,
                                 pop_size,
                                 tolerance,
                                 constraints,
                                 local_optimizer,
                                 global_optimizer,
                                 dask_client,
                                 info):

        if not self._in_bounds(starting_hps, hp_bounds):
            raise Exception("Starting positions outside of optimization bounds.", starting_hps, hp_bounds)

        ############################
        ####global optimization:##
        ############################
        if method == "global":
            logger.debug(
                "fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hp_bounds)
            res = differential_evolution(
                objective_function,
                hp_bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=tolerance,
                disp=info,
                polish=False,
                x0=starting_hps.reshape(1, -1),
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
            logger.debug("starting hyperparameters: {}", starting_hps)
            logger.debug("Attempting a BFGS optimization.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.debug("bounds: {}", hp_bounds)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                OptimumEvaluation = minimize(
                    objective_function,
                    starting_hps,
                    method=local_optimizer,
                    jac=objective_function_gradient,
                    hess=objective_function_hessian,
                    bounds=hp_bounds,
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
            logger.debug("starting hyperparameters: {}", starting_hps)
            logger.debug('bounds are', hp_bounds)

            opt_obj = HGDL(objective_function,
                           objective_function_gradient,
                           hp_bounds,
                           hess=objective_function_hessian,
                           local_optimizer=local_optimizer,
                           global_optimizer=global_optimizer,
                           num_epochs=max_iter,
                           constraints=constraints)

            opt_obj.optimize(dask_client=dask_client, x0=starting_hps.reshape(1, -1))
            try:
                hyperparameters = opt_obj.get_final()[0]["x"]
            except Exception as ex:
                raise Exception("Something has gone wrong with the objective function evaluation.") from ex

        elif method == "mcmc":
            logger.debug("MCMC started in fvGP")
            logger.debug('bounds are {}', hp_bounds)

            def prior_function(theta, args):
                bounds = args["bounds"]
                if self._in_bounds(theta, bounds): return 0.
                else: return -np.inf

            def likelihood_func(hps, args):
                return objective_function(hps)

            myMCMC = gpMCMC(likelihood_func, prior_function, args={"bounds": hp_bounds})
            res = myMCMC.run_mcmc(x0=starting_hps, n_updates=max_iter, info=info, break_condition="default")
            hyperparameters = res["median(x)"]
            self.mcmc_info = res
        elif method == "adam":
            hyperparameters, history = self.adam_optimize(objective_function,
                                                          objective_function_gradient,
                                                          starting_hps, max_iter=max_iter)
        elif callable(method): hyperparameters = method(self)
        else: raise ValueError("No optimization mode specified in fvGP")
        return hyperparameters

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

            # Convergence check
            if np.linalg.norm(theta_new - theta) < tol:
                theta = theta_new
                break

            theta = theta_new

        return theta, history

    @staticmethod
    def _in_bounds(v, bounds):
        assert isinstance(bounds, np.ndarray)
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

