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

#TODO:
# *the proposal distribution has to receive and return parameters without knowing what they are


class gpMCMC():  # pragma: no cover
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
                 args = None
                 ):  # pragma: no cover
        self.log_likelihood_function = log_likelihood_function
        self.dim = dim
        self.prior_function = prior_function
        self.proposal_distributions=proposal_distributions
        self.args=args



    def run_mcmc(self,n_updates=10000,
                 x0=None,
                 info=False,
                 adapt_cov=True,
                 break_condition = None):  # pragma: no cover
        start_time = time.time()
        n_updates = max(n_updates, 2)

        if x0 is None: x0 = np.ones((self.dim))
        if isinstance(x0) != np.ndarray: raise Exception("x0 is not a numpy array")
        if np.ndim(x0) != 1: raise Exception("x0 is not a vector in MCMC")

        eps = .001
        k = 3  # the iteration offset
        trace = {"f(x)": [], "x":[], "time stamp":[]}
        # If the supplied proposal covariance matrix is either not given or invalid,
        # just use the identity.

        # Initialize sigma_m to the rule of thumb
        sigma_m = 2.4 ** 2 / self.dim
        r_hat = 0
        # Set up and initialize trace objects
        jump_trace = np.zeros(n_updates)
        trace["x"].append(x0)

        # Initialize Metropolis
        x = x0.copy()
        likelihood = self.log_likelihood_function(x, self.args)
        prior = self.prior_function(x, self.args)
        #########################################################
        # Begin main loop
        for i in np.arange(1, n_updates):
            for obj in self.proposal_distributions:
                x, prior, likelihood, jump_trace[i] = self._jump(x,obj,prior,likelihood)
                prop_Sigma, sigma_m = obj._adapt(i, trace["x"], k, jump_trace, sigma_m, prop_Sigma, adapt_cov)

            # Update the trace objects
            trace["x"].append(x)
            trace["f(x)"].append(likelihood
            trace["time stamp"].append(time.time() - start_time)
            self.trace = trace

            if info and (i % 100) == 0:
                print("Finished " + str(i) + " out of " + str(n_updates), " iterations. f(x)=",likelihood)
            if break_condition(self): break
        # End main loop

        # Collect trace objects to return
        arg_max = np.argmax(trace["f(x)"])
        x = np.asarray(trace["x"].T)


        return {"max f(x)": f[arg_max],
                "max x": x[arg_max],
                "trace": trace,
                "stripped distribution": x[int(len(x) - (len(x) / 10)):],
                "full distribution": x,
                "distribution mean": np.mean(x[int(len(x) - (len(x) / 10)):], axis=0),
                "distribution var": np.var(x[int(len(x) - (len(x) / 10)):], axis=0)}

    ###############################################################
    def _jump(self,x,obj,prior,likelihood):  # pragma: no cover
        x_star = np.zeros((self.dim))
        if callable(obj.prop_dist):
            x_star[obj.indices] = obj.prop_dist(x[obj.indices], self)
        else:
            raise Exception("A proposal distribution is not callable.")

        prior_evaluation_x_star = self.prior_function(x_star, self.args)
        jump_trace = 0.
        if prior_evaluation_x_star != -np.inf:
            likelihood_star = self.log_likelihood_function(hyperparameters=x_star)
            if np.isnan(likelihood_star): likelihood_star = -np.inf
            metr_ratio = np.exp(prior_evaluation_x_star + likelihood_star -
                                prior - likelihood)
            if np.isnan(metr_ratio):  metr_ratio = 0.
            if metr_ratio > np.random.uniform(0, 1, 1):
                x = x_star
                prior = prior_evaluation_x_star
                likelihood = likelihood_star
                jump_trace = 1.
        return x, prior, likelihood, jump_trace

    ###############################################################


###############################################################
class ProposalDistribution():  # pragma: no cover
    def __init__(self, prop_dist,
                 indices,
                 should_be_adapted = False,
                 adapt_callable=None,
                 r_opt=.234,
                 c_0=10,
                 c_1=.8,
                 K=10,
                 args = None):  # pragma: no cover
        """
        Function to define a set of proposal distributions.

        Parameters
        ----------
        prop_dist : Callable
            A callable to calculate the proposal distribution evaluation.
        indices : iterable of int
            Which indices should be drawn from this proposal distribution.
        should_be_adapted : bool
            Whether ot not to update this proposal distribution
        adapt_callable : Callable, option
            A callable to adapt the distribution.
        """
        self.prop_dist = prop_dist
        self.indices = indices
        self.should_be_adapted = should_be_adapted
        self.args = args
        if callable(adapt_callable):
            self.adapt = adapt_callable
        else:
            self.adapt = self._adapt

        self.r_opt = r_opt
        self.c_0 = c_0
        self.c_1 = c_1
        self.K = K
    #########################################################
    def _adapt(self,i,trace, k, jump_trace, sigma_m, prop_Sigma, adapt_cov):  # pragma: no cover
        if (i % self.K) == 0:
            gamma2 = 1. / ((i / self.K) + k) ** self.c_1
            gamma1 = self.c_0 * gamma2
            r_hat = np.mean(jump_trace[(i - self.K + 1): i])
            sigma_m = np.exp(np.log(sigma_m) + gamma1 * (r_hat - self.r_opt))
            if adapt_cov: prop_Sigma = prop_Sigma + gamma2 * (np.cov(trace[:, (i - self.K + 1): i]) - prop_Sigma)
        return prop_Sigma, sigma_m




###############################################################
###############################################################
###############################################################
###############################################################
def out_of_bounds(x,bounds):  # pragma: no cover
    for i in range(len(x)):
        if x[i] < bounds[i,0] or x[i] > bounds[i,1]:
            return True
    return False


def project_onto_bounds(x,bounds):  # pragma: no cover
    for i in range(len(x)):
        if x[i] < bounds[i,0]: x[i] = bounds[i,0]
        if x[i] > bounds[i,1]: x[i] = bounds[i,1]
    return x

def in_bounds(v,bounds):  # pragma: no cover
    for i in range(len(v)):
        if v[i] < bounds[i,0] or v[i] > bounds[i,1]: return False
    return True


def prior_func(theta,bounds):  # pragma: no cover
    if in_bounds(theta, bounds): return 0.
    else: return -np.inf


