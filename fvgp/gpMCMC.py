import numpy as np
import time


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
# *the proposal distribution has to receive and return parameters without knowing what they are


class gpMCMC:  # pragma: no cover
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

    def run_mcmc(self, n_updates=10000,
                 x0=None,
                 info=False,
                 break_condition=None):  # pragma: no cover
        start_time = time.time()
        n_updates = max(n_updates, 2)
        if x0 is None: x0 = np.ones((self.dim))
        if not isinstance(x0, np.ndarray): raise Exception("x0 is not a numpy array")
        if np.ndim(x0) != 1: raise Exception("x0 is not a vector in MCMC")
        if break_condition is None: break_condition = lambda a: False

        self.trace = {"f(x)": [], "x": [], "time stamp": []}
        # Set up and initialize trace objects
        self.trace["x"].append(x0)

        # Initialize Metropolis
        x = x0.copy()
        likelihood = self.log_likelihood_function(x, self.args)
        prior = self.prior_function(x, self.args)
        #########################################################
        # Begin main loop
        for i in np.arange(1, n_updates):
            for obj in self.proposal_distributions:
                x, prior, likelihood, jt = self._jump(x, obj, prior, likelihood)
                obj.jump_trace.append(jt)
                obj.adapt(i, self)

            # Update the trace objects
            self.trace["x"].append(x)
            self.trace["f(x)"].append(likelihood)
            self.trace["time stamp"].append(time.time() - start_time)

            if info and (i % 100) == 0:
                print("Finished " + str(i) + " out of " + str(n_updates), " iterations. f(x)=", likelihood)
            if break_condition(self): break
        # End main loop

        # Collect trace objects to return
        arg_max = np.argmax(self.trace["f(x)"])
        x = np.asarray(self.trace["x"]).T

        return {"max f(x)": self.trace["f(x)"][arg_max],
                "max x": x[arg_max],
                "trace": self.trace,
                "stripped distribution": x[int(len(x) - (len(x) / 10)):],
                "full distribution": x,
                "distribution mean": np.mean(x[int(len(x) - (len(x) / 10)):], axis=0),
                "distribution var": np.var(x[int(len(x) - (len(x) / 10)):], axis=0)}

    ###############################################################
    def _jump(self, x_old, obj, prior_eval, likelihood):  # pragma: no cover
        x_star = x_old.copy()
        if callable(obj.prop_dist):
            print("obj indices: ", obj.indices)
            x_star[obj.indices] = obj.prop_dist(x_old[obj.indices], obj)
        else:
            raise Exception("A proposal distribution is not callable.")

        prior_evaluation_x_star = self.prior_function(x_star, self.args)
        jump_trace = 0.
        if prior_evaluation_x_star != -np.inf:
            likelihood_star = self.log_likelihood_function(x_star, self.args)
            if np.isnan(likelihood_star): likelihood_star = -np.inf
            metr_ratio = np.exp(prior_evaluation_x_star + likelihood_star -
                                prior_eval - likelihood)
            if np.isnan(metr_ratio):  metr_ratio = 0.
            if metr_ratio > np.random.uniform(0, 1, 1):
                x = x_star.copy()
                prior_eval = prior_evaluation_x_star
                likelihood = likelihood_star
                jump_trace = 1.
                print("accepted")
            else:
                x = x_old.copy()
                print("NOT accepted")
        else:
            print("prior probability 0")

        print("old x  :", x_old)
        print("new x  :", x)
        input()
        return x, prior_eval, likelihood, jump_trace

    ###############################################################


###############################################################
class ProposalDistribution:  # pragma: no cover
    def __init__(self, prop_dist,
                 indices,
                 should_be_adapted=False,
                 adapt_callable=None,
                 r_opt=.234,
                 c_0=10,
                 c_1=.8,
                 K=10,
                 init_prop_Sigma=None,
                 adapt_cov=False,
                 args=None):  # pragma: no cover
        """
        Function to define a set of proposal distributions.

        Parameters
        ----------
        prop_dist : Callable
            A callable to calculate the proposal distribution evaluation.
        indices : iterable of int
            Which indices should be drawn from this proposal distribution.
        should_be_adapted : bool
            Whether ot not to update this proposal distribution.
            If True either a callable for the update should be provided
            or the proposal is assumed to be normal and the default adaption procedure will be performed.
        adapt_callable : Callable, option
            A callable to adapt the distribution. The default is an adaption procedure for normal distributions.
            The callable should not return anything but update the `args`.
        args : Any, optional
            Arguments that will be communicated to the user-provided `adapt_callable`.
            Leave blank if the default adaption procedure is used.

        """

        self.prop_dist = prop_dist
        self.indices = indices
        self.should_be_adapted = should_be_adapted
        self.r_opt = r_opt
        self.c_0 = c_0
        self.c_1 = c_1
        self.K = K
        self.adapt_cov = adapt_cov
        dim = len(indices)
        self.jump_trace = []
        if not callable(adapt_callable) and args:
            raise Exception("The args should only be provided for a user-defined `adapt_callable`")
        if not callable(adapt_callable) and init_prop_Sigma is None and should_be_adapted:
            raise Exception("You are using the default adaption mechanism for normal distributions.\n \
                            Please provide an initial_prop_Sigma")

        if init_prop_Sigma is None and not callable(adapt_callable):
            init_prop_Sigma = np.identity(dim)
        if callable(adapt_callable):
            self.adapt = adapt_callable
        else:
            self.adapt = self._adapt

        if args is None:
            self.args = {"prop_Sigma": init_prop_Sigma, "sigma_m": 2.4 ** 2 / dim}
        else:
            self.args = args

    #########################################################
    def _adapt(
        self,
        end,
        obj
        ):  # pragma: no cover
        k = 3
        c_0 = self.c_0
        c_1 = self.c_1
        K = self.K
        r_opt = self.r_opt
        prop_Sigma = self.args["prop_Sigma"]
        sigma_m = self.args["sigma_m"]
        jump_trace = self.jump_trace
        trace = obj.trace["x"]
        if (end % K) == 0:
            start = (end - K + 1)
            gamma2 = 1. / ((end / K) + k) ** c_1
            gamma1 = c_0 * gamma2
            r_hat = np.mean(jump_trace[start: end])
            sigma_m = np.exp(np.log(sigma_m) + gamma1 * (r_hat - r_opt))
            if self.adapt_cov: prop_Sigma = prop_Sigma + gamma2 * (np.cov(trace[:, start: end]) - prop_Sigma)
        self.args["prop_Sigma"] = prop_Sigma
        self.args["sigma_m"] = sigma_m


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
