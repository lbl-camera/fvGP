import warnings
from loguru import logger
import numpy as np
from scipy.optimize import differential_evolution
from hgdl.hgdl import HGDL
from .mcmc import mcmc
from scipy.optimize import minimize


class GPtraining:
    def __init__(self, gp2Scale=False):
        self.mcmc_info = None
        self.gp2Scale = gp2Scale

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
        return hyperparameters

    ##################################################################################
    def train_async(self,
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
    def stop_training(self, opt_obj):
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
            no training is running.", stacklevel=2)

    ###################################################################################
    def kill_training(self, opt_obj):
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
            warnings.warn("No asynchronous training to be killed, no training is running.", stacklevel=2)

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
            updated_hyperparameters = opt_obj.get_latest()[0]["x"]
        except Exception as err:
            logger.debug("      The optimizer object could not be queried")
            logger.debug("      That probably means you are not optimizing the hyperparameters asynchronously")
            warnings.warn("     Hyperparameter update failed with ERROR: "+str(err))
            updated_hyperparameters = None

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
            opt_obj.kill_client()
        elif method == "mcmc":
            logger.debug("MCMC started in fvGP")
            logger.debug('bounds are {}', hp_bounds)
            res = mcmc(objective_function, hp_bounds, x0=starting_hps, n_updates=max_iter, info=info)
            hyperparameters = np.array(res["distribution mean"])
            self.mcmc_info = res
        elif callable(method): hyperparameters = method(self)
        else: raise ValueError("No optimization mode specified in fvGP")
        return hyperparameters

    def _in_bounds(self, v, bounds):
        assert isinstance(bounds, np.ndarray)
        if any(v < bounds[:, 0]) or any(v > bounds[:, 1]): return False
        return True


if __name__ == "__main__":
    a = GPtraining()
