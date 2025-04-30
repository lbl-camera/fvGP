import numpy as np
import time
from loguru import logger


def project_onto_bounds(x, bounds):
    for i in range(len(x)):
        if x[i] < bounds[i, 0]: x[i] = bounds[i, 0]
        if x[i] > bounds[i, 1]: x[i] = bounds[i, 1]
    return x


def in_bounds(v, bounds):
    for i in range(len(v)):
        if v[i] < bounds[i, 0] or v[i] > bounds[i, 1]: return False
    return True


def prior_func(theta, bounds):
    if in_bounds(theta, bounds):
        return 0.
    else:
        return -np.inf


# --------------------------------------------------------------------- #
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


def mcmc(likelihood_fn, bounds, x0=None, n_updates=10000,
         prior_args=None,
         info=False, prior_fn=None,
         prop_Sigma=np.nan, adapt_cov=True,
         r_opt=.234, c_0=10, c_1=.8,
         K=10):
    start_time = time.time()
    n_updates = max(n_updates, 2)
    if np.ndim(x0) != 1: raise Exception("x0 is not a vector in MCMC")
    if x0 is None: x0 = np.ones((len(bounds)))
    if prior_args is None: prior_args = bounds
    if prior_fn is None: prior_fn = prior_func
    eps = .001
    k = 3  # the iteration offset
    f = []
    ctime = []
    if type(x0).__module__ != 'numpy' or isinstance(x0, np.float64):
        x0 = np.array(x0)
    p = len(x0)
    # If the supplied proposal covariance matrix is either not given or invalid,
    # just use the identity.
    if np.any(np.isnan(prop_Sigma)) or prop_Sigma.size != p ** 2:
        axis_std = (bounds[:, 1] - bounds[:, 0]) / 10.
        prop_Sigma = np.diag(axis_std ** 2)
        prop_C = np.linalg.cholesky(prop_Sigma)
    else:
        try:
            # Initialize prop_C
            prop_C = np.linalg.cholesky(prop_Sigma)
        except np.linalg.LinAlgError:
            prop_Sigma = np.eye(p)
            prop_C = np.eye(p)
    # Initialize sigma_m to the rule of thumb
    sigma_m = 2.4 ** 2 / p
    # Set up and initialize trace objects
    trace = np.zeros((p, n_updates))
    jump_trace = np.zeros(n_updates)

    trace[:, 0] = x0
    theta = x0
    likelihood = likelihood_fn(hyperparameters=theta)
    prior = prior_fn(theta, prior_args)
    #########################################################
    # Begin main loop
    for i in np.arange(1, n_updates):
        theta_star = theta + sigma_m * np.random.standard_normal(p) @ prop_C
        prior_star = prior_fn(theta_star, prior_args)
        if prior_star != -np.inf:
            likelihood_star = likelihood_fn(hyperparameters=theta_star)
            if np.isnan(likelihood_star): likelihood_star = -np.inf
            metr_ratio = np.exp(prior_star + likelihood_star - prior - likelihood)
            if np.isnan(metr_ratio):  metr_ratio = 0.
            if metr_ratio > np.random.uniform(0, 1, 1):
                theta = theta_star
                prior = prior_star
                likelihood = likelihood_star
                jump_trace[i] = 1

        # Adapt via my method
        if (i % K) == 0:
            gamma2 = 1 / ((i / K) + k) ** c_1
            gamma1 = c_0 * gamma2
            r_hat = jump_trace[(i - K + 1): i].mean()

            sigma_m = np.exp(np.log(sigma_m) + gamma1 * (r_hat - r_opt))

            if adapt_cov:
                prop_Sigma = prop_Sigma + gamma2 * (np.cov(trace[:, (i - K + 1): i]) - prop_Sigma)
                check_chol_cont = True
                while check_chol_cont:
                    try:
                        # Initialize prop_C
                        prop_C = np.linalg.cholesky(prop_Sigma)
                        check_chol_cont = False
                    except np.linalg.LinAlgError:
                        prop_Sigma = prop_Sigma + eps * np.eye(p)
        # Update the trace objects
        trace[:, i] = theta
        trace_as_array = np.asarray(trace[:, 0:i].T)
        f.append(likelihood)
        ctime.append(time.time() - start_time)
        # Echo every 100 iterations
        if info:
            if (i % 10) == 0: print("Finished ", i, " out of ", n_updates, " MCMC iterations. f(x)= ", likelihood)
        if i > 400 and np.sum(abs(np.mean(trace_as_array[-10:], axis=0) -
                                  np.mean(trace_as_array[-20:-10], axis=0))) \
                                  < 0.01 * np.sum(abs(np.mean(trace_as_array[-10:], axis=0))):
            break
    # End main loop
    #########################################################

    # Collect trace objects to return

    arg_max = np.argmax(f)

    return {"f(x)": f[arg_max],
            "x": trace_as_array[arg_max],
            "F": f,
            "stripped distribution": trace_as_array[int(len(trace_as_array) - (len(trace_as_array) / 10)):],
            "full distribution": trace_as_array,
            "distribution mean": np.mean(trace_as_array[int(len(trace_as_array) - (len(trace_as_array) / 10)):], axis=0),
            "distribution var": np.var(trace_as_array[int(len(trace_as_array) - (len(trace_as_array) / 10)):], axis=0),
            "compute time": ctime}
