import warnings
from loguru import logger
import numpy as np
from scipy.optimize import differential_evolution
from hgdl.hgdl import HGDL
from .mcmc import mcmc
from scipy.optimize import minimize


class GPtraining:
    def __init__(self, info=False, gp2Scale=False):
        self.mcmc_info = None
        self.gp2Scale = gp2Scale
        self.info = info

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
              dask_client=None):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be 'hgdl' and
        providing a dask client. However, in that case fvgp.GP.train_async() is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.

        Parameters
        ----------
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the hessian of the negative log marginal likelihood.
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization.
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default is None, in which case the hyperparameter_bounds are estimated from the domain size
            and the y_data. If the data set changes significantly,
            the hyperparameters and the bounds should be changed/retrained.
            The default only works for the default kernels.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the bounds.
        method : str or Callable, optional
            The method used to train the hyperparameters.
            The options are `global`, `local`, `hgdl`, `mcmc`, and a callable.
            The callable gets a `gp.GP` instance and has to return a 1d np.ndarray of hyperparameters.
            The default is `global` (scipy's differential evolution).
            If method = "mcmc",
            the attribute fvgp.GP.mcmc_info is updated and contains convergence and distribution information.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize` functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a scipy optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.
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
            dask_client
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

        Parameters
        ----------
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the hessian of the negative log marginal likelihood.
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization.
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default is None, in which case the hyperparameter_bounds are estimated from the domain size
            and the y_data. If the data set changes significantly,
            the hyperparameters and the bounds should be changed/retrained.
            The default only works for the default kernels.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the bounds.
        max_iter : int, optional
            Maximum number of epochs for HGDL. Default = 10000.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of hgdl.NonLinearConstraint instances, optional
            Equality and inequality constraints for the optimization. See `hgdl.readthedocs.io`
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.

        Return
        ------
        Optimization object that can be given to `fvgp.GP.update_hyperparameters()`
        to update the prior GP : object instance
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
        except:
            logger.debug("      The optimizer object could not be queried")
            logger.debug("      That probably means you are not optimizing the hyperparameters asynchronously")
            warnings.warn("     Hyperparameter update failed")
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

        logger.info("fvGP hyperparameter tuning in progress. Old hyperparameters: {}", starting_hps)
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
                                 dask_client):

        if not self._in_bounds(starting_hps, hp_bounds):
            raise Exception("Starting positions outside of optimization bounds.", starting_hps, hp_bounds)

        ############################
        ####global optimization:##
        ############################
        if method == "global":
            logger.info(
                "fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.info("bounds: {}", hp_bounds)
            res = differential_evolution(
                objective_function,
                hp_bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=tolerance,
                disp=self.info,
                polish=False,
                x0=starting_hps.reshape(1, -1),
                constraints=constraints,
                workers=1,
            )
            hyperparameters = np.array(res["x"])
            logger.info(f"fvGP found hyperparameters {hyperparameters} with objective function eval {res['fun']} \
            via global optimization")
        ############################
        ####local optimization:#####
        ############################
        elif method == "local":
            logger.info("fvGP is performing a local update of the hyper parameters.")
            logger.debug("starting hyperparameters: {}", starting_hps)
            logger.debug("Attempting a BFGS optimization.")
            logger.debug("maximum number of iterations: {}", max_iter)
            logger.debug("termination tolerance: {}", tolerance)
            logger.info("bounds: {}", hp_bounds)
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
            logger.info("fvGP submitted HGDL optimization")
            logger.info("starting hyperparameters: {}", starting_hps)
            logger.info('bounds are', hp_bounds)

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
            logger.info("MCMC started in fvGP")
            logger.info('bounds are {}', hp_bounds)
            res = mcmc(objective_function, hp_bounds, x0=starting_hps, n_updates=max_iter, info=self.info)
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
