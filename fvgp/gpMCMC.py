import numpy as np
import time
import warnings

## --------------------------------------------------------------------- ##
#  A generic Metropolis sampler.  You have to supply the log likelihood   #
#  function, which need not really be a likelihood function at all.       #
#
#  Translated from Shaby's R code.
#
#  Uppercase K is the size of the blocks of iterations used for
#  adapting the proposal.
#  Lowercase k is the offset to get rid of wild swings in adaptation
#  process that otherwise happen the early
#  iterations.
# Dr. Likun Zhang: https://stat.missouri.edu/people/zhang


# x0 ........................................... initial values
# likelihood_fn ............................................ log likelihood
# prior_fn ....................................... prior function for theta
# prior_args .............................................. for prior functions
# n_updates .................................. number of Metropolis updates
# prop_Sigma ................................... proposal covariance matrix

# adapt_cov ......................... whether to update proposal covariance
# return_prop_Sigma_trace........ save proposal covariance from each update
# r_opt ........................................... optimal acceptance rate
# c_0, c_1 .. two coefficients for updating sigma_m and proposal covariance
# K .............................................. adapt every K iterations
#                                                                         #

# TODO:
# *


class gpMCMC:
    """
    This class provides allows the user to customize an MCMC via user-defined proposal distributions and a prior.


    Parameters
    ----------
    log_likelihood_function : Callable
        The log of the likelihood to be sampled.
    dim : int
        The dimensionality of the space.
    prior_function : Callable
        Function to query for the prior probability of form: func(x, obj), where
        x is the current vector and obj is this gpMCMC object instance.
    proposal_distributions : iterable
        A list of object instances of ProposalDistribution.
    args : Any, optional
        Arguments that will be communicated to all Callables.


    Attributes
    ----------
    trace : dict
        Solution after run_mcmc is executed.

    """

    def __init__(self,
                 log_likelihood_function,
                 dim,
                 prior_function,
                 proposal_distributions,
                 args=None
                 ):  # pragma: no cover
        self.log_likelihood_function = log_likelihood_function
        self.dim = dim
        self.prior_function = prior_function
        self.proposal_distributions = proposal_distributions
        self.args = args
        self.trace = None
        self.mcmc_info = None
        warnings.warn("Thank you for trying our new MCMC. While we tested it quite a lot it's still in\n \
         its experimental stage. Use caution.")

    def run_mcmc(self, n_updates=10000,
                 x0=None,
                 info=False,
                 break_condition=None,
                 run_in_every_iteration=None):  # pragma: no cover
        """
        This function runs the mcmc.


        Parameters
        ----------
        n_updates: int, optional
            The log of the likelihood to be sampled.
        x0 : np.ndarray
            Starting point of the mcmc.
        info : bool
            Whether to print information is the mcmc runs.
        break_condition : callable or string or None
            A break condition that specified when the mcmc is terminated. If None,
            mcmc will run until `n_updates` is reached. If callable will get the mcmc object instance as
            input: def break(obj). The only allowed string is `default` and in that case the mcmc
            will be terminated if the mean of the position has not changed significantly in the lsat 200 iterations.

        Return
        ------
        trace information : dict
            Mean and variances are presented of the last 1% of x. All other returns consider all x.
            x here are all the accepted positions in the MCMC.

        """
        start_time = time.time()
        n_updates = max(n_updates, 2)
        if x0 is None: x0 = np.ones(self.dim)
        if not isinstance(x0, np.ndarray): raise Exception("x0 is not a numpy array")
        if np.ndim(x0) != 1: raise Exception("x0 is not a vector in MCMC")
        if break_condition is None: break_condition = lambda a: False
        elif break_condition == "default": break_condition = self._default_break_condition
        if run_in_every_iteration is None: run_in_every_iteration = lambda a: False

        self.trace = {"f(x)": [], "x": [], "time stamp": []}
        # Set up and initialize trace objects
        self.trace["x"].append(x0)

        # Initialize Metropolis
        x = x0.copy()
        likelihood = self.log_likelihood_function(x, self.args)
        if info:
            print("Starting likelihood. f(x)=", likelihood)
        prior = self.prior_function(x, self.args)
        #########################################################
        # Begin main loop
        st = time.time()
        for i in np.arange(1, n_updates):
            for obj in self.proposal_distributions:
                x, prior, likelihood, jt = self._jump(x, obj, prior, likelihood)
                obj.jump_trace.append(jt)
                obj.adapt(i, self)

            # Update the trace objects
            self.trace["x"].append(x)
            self.trace["f(x)"].append(likelihood)
            self.trace["time stamp"].append(time.time() - start_time)
            run_in_every_iteration(self)

            if info and (i % 100) == 0:
                print("Finished " + str(i) + " out of " + str(n_updates), " iterations. f(x)=", likelihood)
            if break_condition(self): break
        # End main loop

        # Collect trace objects to return
        arg_max = np.argmax(self.trace["f(x)"])
        x = np.asarray(self.trace["x"])
        dist_index = int(len(x) - (len(x) / 100))
        self.mcmc_info = {"f(x)": self.trace["f(x)"],
                          "max f(x)": self.trace["f(x)"][arg_max],
                          "max x": x[arg_max],
                          "time stamps": self.trace["time stamp"],
                          "x": x,
                          "mean(x)": np.mean(x[dist_index:], axis=0),
                          "var(x)": np.var(x[dist_index:], axis=0)}

        return self.mcmc_info
    ###############################################################
    def _default_break_condition(self,obj):
        x = obj.trace["x"]

        if len(x) > 201:
            latest_mean = np.mean(x[-100:], axis=0)
            earlier_mean = np.mean(x[-200:-100], axis=0)
            abs_diff = abs(latest_mean - earlier_mean)
            max_index = np.argmax(abs_diff)
            ratio = (abs_diff[max_index] / abs(latest_mean[max_index])) * 100.
            if ratio < 0.1: return True
            else: return False
        else: return False
    ###############################################################
    def _jump(self, x_old, obj, prior_eval, likelihood):  # pragma: no cover
        x_star = x_old.copy()
        if callable(obj.prop_dist):
            x_star[obj.indices] = obj.prop_dist(x_old[obj.indices].copy(), obj)
        else:
            raise Exception("A proposal distribution is not callable.")

        prior_evaluation_x_star = self.prior_function(x_star, self.args)
        jump_trace = 0.
        if prior_evaluation_x_star != -np.inf:
            likelihood_star = self.log_likelihood_function(x_star, self.args)
            if np.isnan(likelihood_star): raise Exception("Likelihood evaluated to NaN in gpMCMC")
            metr_ratio = np.exp(prior_evaluation_x_star + likelihood_star -
                                prior_eval - likelihood)
            if np.isnan(metr_ratio):  metr_ratio = 0.
            if metr_ratio > np.random.uniform(0, 1, 1):
                x = x_star
                prior_eval = prior_evaluation_x_star
                likelihood = likelihood_star
                jump_trace = 1.
            else:
                x = x_old
        else:
            x = x_old


        return x, prior_eval, likelihood, jump_trace

    ###############################################################


###############################################################
class ProposalDistribution:  # pragma: no cover
    def __init__(self, prop_dist,
                 indices,
                 init_prop_Sigma=None,
                 adapt_callable=None,
                 r_opt=.234,
                 c_0=10,
                 c_1=.8,
                 K=100,
                 adapt_cov=True,
                 prop_args=None):  # pragma: no cover
        """
        Class to define a proposal distribution.

        Parameters
        ----------
        prop_dist : Callable
            A callable to calculate the proposal distribution evaluation.
            It is defined as `def name(x, obj)`, where `obj` is a `proposal_distribution`
            object instance. The function should return a new proposal for `x`.
        indices : iterable of int
            The indices that should be drawn from this proposal distribution.
        init_prop_Sigma : np.ndarray, optional
            If the proposal distribution is normal this is the covariance of the initial proposal distribution.
            It will be updated if adapt_callable = `normal` or a callable.
            While it is optional to provide it, it is highly recommended to do so.
            A warning will be printed in that case. A good rule of thumb
            is to orient yourself on the size of your domain.
        adapt_callable : Callable or None or string, optional
            A callable to adapt the distribution. The default is None which means
            the proposal distribution will not be adapted.
            Use `normal` for the default adaption procedure for normal distributions.
            The callable should be defined as `def adapt(index, mcmc_obj)` and not return anything
            but update the `mcmc_obj.prop_args`. Note, any adapt function will have to be well thought through.
            Most adapt functions will not lead to a stationary final distributions. Use with caution.
        prop_args : Any, optional
            Arguments that will be available as obj attribute in `prop_dist`and `adapt_callable`.

        """

        self.prop_dist = prop_dist
        self.indices = indices
        self.r_opt = r_opt
        self.c_0 = c_0
        self.c_1 = c_1
        self.K = K
        self.adapt_cov = adapt_cov
        dim = len(indices)
        self.jump_trace = []
        if adapt_callable == "normal" and init_prop_Sigma is None:
            init_prop_Sigma = np.identity(dim)
            warnings.warn("You are using the normal adaption mechanism for normal distributions\n \
                           but did not provide `init_prop_sigma`. This can lead to slow convergence")

        if callable(adapt_callable): self.adapt = adapt_callable
        elif adapt_callable == "normal": self.adapt = self._adapt
        else: self.adapt = self._no_adapt

        if prop_args is None:
            self.prop_args = {"prop_Sigma": init_prop_Sigma, "sigma_m": 2.4 ** 2 / dim}
        else:
            self.prop_args = prop_args
            if adapt_callable == "normal":
                self.prop_args["prop_Sigma"] = init_prop_Sigma
                self.prop_args["sigma_m"] = 2.4 ** 2 / dim

            #########################################################
    def _adapt(self, end, mcmc_obj):  # pragma: no cover
        K = self.K
        if (end % K) == 0:
            k = 3
            c_0 = self.c_0
            c_1 = self.c_1
            r_opt = self.r_opt
            prop_Sigma = self.prop_args["prop_Sigma"]
            sigma_m = self.prop_args["sigma_m"]
            jump_trace = self.jump_trace
            trace = np.asarray(mcmc_obj.trace["x"]).T
            start = (end - K + 1)
            gamma2 = 1. / ((end / K) + k) ** c_1
            gamma1 = c_0 * gamma2
            r_hat = np.mean(jump_trace[start: end])
            sigma_m = np.exp(np.log(sigma_m) + gamma1 * (r_hat - r_opt))
            if self.adapt_cov: prop_Sigma = prop_Sigma + gamma2 * (np.cov(trace[self.indices, start: end]) - prop_Sigma)
            self.prop_args["prop_Sigma"] = prop_Sigma
            self.prop_args["sigma_m"] = sigma_m

    def _no_adapt(self, end, mcmc_obj):  # pragma: no cover
        return


###############################################################
###############################################################
###############################################################
###############################################################
def out_of_bounds(x, bounds):  # pragma: no cover
    for i in range(len(x)):
        if x[i] < bounds[i, 0] or x[i] > bounds[i, 1]:
            return True
    return False


def project_onto_bounds(x, bounds):  # pragma: no cover
    for i in range(len(x)):
        if x[i] < bounds[i, 0]: x[i] = bounds[i, 0]
        if x[i] > bounds[i, 1]: x[i] = bounds[i, 1]
    return x


def in_bounds(v, bounds):  # pragma: no cover
    for i in range(len(v)):
        if v[i] < bounds[i, 0] or v[i] > bounds[i, 1]: return False
    return True


def prior_func(theta, bounds):  # pragma: no cover
    if in_bounds(theta, bounds):
        return 0.
    else:
        return -np.inf
