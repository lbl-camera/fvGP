#!/usr/bin/env python

import numpy as np
import time
import warnings
from dask.distributed import get_worker


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


#: The ``"default"`` break condition of :py:meth:`gpMCMC.run_mcmc`, a single-chain heuristic.
#: It is checked every ``_STOP_CHECK_EVERY`` iterations once the chain has
#: ``_STOP_MIN_LENGTH`` states, and stops the chain after ``_STOP_CONSECUTIVE`` passing
#: checks in a row. A check passes when, on the post-burn-in chain, every parameter's
#: effective sample size is at least ``_STOP_MIN_ESS`` and a Geweke z-score (first 10%
#: against last 50%) is below ``_STOP_MAX_Z``, and when some proposal was accepted since the
#: previous check -- a stuck chain looks perfectly stationary and must never count.
_STOP_MIN_LENGTH = 2000
_STOP_CHECK_EVERY = 500
_STOP_CONSECUTIVE = 2
_STOP_MIN_ESS = 100
_STOP_MAX_Z = 2.0


def _as_chains(chains):
    """``(n,)``, ``(n, d)`` or ``(m, n, d)`` (or a list of ``(n, d)`` traces) as ``(m, n, d)``."""
    c = np.asarray(chains, dtype=float)
    if c.ndim == 1: c = c[None, :, None]
    elif c.ndim == 2: c = c[None]
    assert c.ndim == 3, "chains must be (n,), (n, d) or (m, n, d)"
    return c


def ess(chains):
    """Effective sample size of each parameter of one or several MCMC chains.

    Uses the chain-averaged autocorrelation with Geyer's initial monotone sequence
    truncation. Autocorrelations are computed by FFT, so the cost is O(n log n).

    Parameters
    ----------
    chains : np.ndarray or list
        One chain, ``(n,)`` or ``(n, d)``, or several of equal length, ``(m, n, d)``.

    Returns
    -------
    ess : np.ndarray
        Shape ``(d,)``; ``nan`` for a parameter that never moved.
    """
    c = _as_chains(chains)
    m, n, d = c.shape
    x = c - c.mean(axis=1, keepdims=True)
    var = x.var(axis=1)
    f = np.fft.rfft(x, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = np.where(var[:, None, :] > 0, acov / var[:, None, :], 0.).mean(axis=0)
    out = np.full(d, np.nan)
    for j in range(d):
        if not np.any(var[:, j] > 0): continue
        pairs = acf[1:n - 1:2, j] + acf[2:n:2, j]
        negative = np.flatnonzero(pairs < 0)
        if len(negative): pairs = pairs[:negative[0]]
        out[j] = m * n / (1. + 2. * np.minimum.accumulate(pairs).sum())
    return out


def split_rhat(chains):
    """Split-R̂ of each parameter: every chain is halved and the halves compared.

    Values near 1 (below about 1.01) indicate the chains sample the same distribution.

    Parameters
    ----------
    chains : np.ndarray or list
        Several chains of equal length, ``(m, n, d)``, or a list of ``(n, d)`` traces; a
        single chain works too, its two halves are compared.

    Returns
    -------
    rhat : np.ndarray
        Shape ``(d,)``; ``nan`` for a parameter that never moved.
    """
    c = _as_chains(chains)
    half = c.shape[1] // 2
    assert half >= 2, "split_rhat needs chains of at least 4 states"
    halves = np.concatenate([c[:, :half], c[:, half:2 * half]])
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = half * halves.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(W > 0, np.sqrt(((half - 1) / half * W + B / half) / W), np.nan)


def summarize(trace, burn_in=0.5):
    """The summary :py:meth:`gpMCMC.run_mcmc` returns, built from a trace.

    ``mean(x)``, ``median(x)``, ``var(x)`` and ``acceptance rate`` cover the chain after
    the first ``burn_in`` fraction of its states; everything else covers the whole chain.
    ``max f(x)`` / ``max x`` are the highest log-likelihood and its state; ``MAP`` / ``MAP x``
    the highest log posterior (likelihood plus prior) and its state, identical under the
    default flat prior.
    """
    n = min(len(trace["x"]), len(trace["f(x)"]))
    x = np.asarray(trace["x"][:n])
    f = np.asarray(trace["f(x)"][:n], dtype=float)
    log_posterior = np.asarray(trace.get("log posterior", f)[:n], dtype=float)
    start = min(int(burn_in * n), n - 1)
    post = x[start:]
    accepted = np.asarray(trace.get("accepted", [])[start:n], dtype=float)
    accepted = accepted[np.isfinite(accepted)]
    arg_max = int(np.argmax(f))
    arg_map = int(np.argmax(log_posterior))
    return {"f(x)": list(trace["f(x)"][:n]),
            "max f(x)": f[arg_max],
            "max x": x[arg_max],
            "MAP": log_posterior[arg_map],
            "MAP x": x[arg_map],
            "time stamps": list(trace["time stamp"][:n]),
            "x": x,
            "burn in": start,
            "mean(x)": np.mean(post, axis=0),
            "median(x)": np.median(post, axis=0),
            "var(x)": np.var(post, axis=0),
            "acceptance rate": float(np.mean(accepted)) if len(accepted) else np.nan}


class gpMCMC:
    """
    This class allows the user to customize an MCMC via user-defined proposal distributions and a prior.


    Parameters
    ----------
    log_likelihood_function : callable
        The log of the likelihood to be sampled. Function of the form def likelihood(x,args) that returns a scalar.
    bounds : np.ndarray, optional
        The bounds of the domain. Only used if the default uniform prior is used.
        The default is None, which means no bounds and thus a uniform prior over the whole space.
    prior_function : callable, optional
        Function to query for the prior probability of form: func(x, bounds, args), where
        x is the current vector and bounds is the bounds array. The default is a uniform prior within ``bounds``.
        In that case, bounds have to be provided.
    proposal_distributions : iterable, optional
        A list of object instances of ProposalDistribution. The default is a single normal distribution with the default ``init_prop_Sigma``.
    args : Any, optional
        Arguments that will be communicated to all Callables.


    Attributes
    ----------
    trace : dict
        Solution after run_mcmc is executed.

    """

    def __init__(self,
                 log_likelihood_function,
                 bounds=None,
                 prior_function=None,
                 proposal_distributions="normal",
                 args=None
                 ):
        self.log_likelihood_function = log_likelihood_function
        assert bounds is not None or prior_function is not None, \
            "Provide either bounds (for the default uniform prior) or a prior_function."
        if prior_function is None:
            self.prior_function = lambda theta, b, _: 0. if np.all(
                (theta >= b[:, 0]) & (theta <= b[:, 1])) else -np.inf
        else:
            self.prior_function = prior_function
        if proposal_distributions == "normal":
            assert bounds is not None, \
                "bounds must be provided to initialize the default normal proposal distribution."
            domain_size = bounds[:, 1] - bounds[:, 0]
            std_diag = domain_size * 0.2 / np.sqrt(12)
            proposal_distributions = [ProposalDistribution(np.arange(len(bounds)),
                                                           init_prop_Sigma=np.diag(std_diag**2))]
        self.proposal_distributions = proposal_distributions

        self.args = args
        self.bounds = bounds
        self.trace = None
        self.mcmc_info = {}

    def run_mcmc(self, *, x0,
                 n_updates=10000,
                 info=False,
                 break_condition=None,
                 run_in_every_iteration=None,
                 burn_in=0.5,
                 rng=None):
        """
        This function runs the mcmc.


        Parameters
        ----------
        x0 : np.ndarray
            Starting point of the mcmc.
        n_updates: int, optional
            The length of the chain, including ``x0``, unless a break condition ends it
            earlier. Default 10000.
        info : bool
            Whether to print information about the mcmc iterations.
        break_condition : callable or string or None
            When to end the chain before ``n_updates``. ``None`` runs all ``n_updates``.
            A callable gets the mcmc instance, ``def break(obj)``, after every iteration.
            ``"default"`` is a single-chain convergence heuristic: checked every 500
            iterations from 2000 on, it ends the chain after two passing checks in a row,
            a check passing when, on the post-burn-in chain, every parameter has an
            effective sample size of at least 100 and a Geweke z-score (first 10% against
            last 50%) below 2, and some proposal was accepted since the previous check. It
            is a heuristic; checking convergence properly takes several chains and
            :py:func:`split_rhat`.
        run_in_every_iteration : callable, optional
            A callable that is executed in every iteration. Form: func(obj). Default no-op.
        burn_in : float, optional
            Fraction of the chain, from its start, left out of ``mean(x)``, ``median(x)``,
            ``var(x)``, ``acceptance rate`` and the ``"default"`` break condition. Default 0.5.
        rng : int or np.random.Generator, optional
            Seed or generator for the proposals and the accept step, available to custom
            proposal callables as ``obj.rng``. The default, None, uses the global ``np.random``.


        Returns
        -------
        trace information : dict
            ``x`` holds the state at every iteration, rejections (repeats) included, which is
            the MCMC chain; ``f(x)``, ``time stamps`` and the rest are aligned with it, so the
            same index is the same state. See :py:func:`summarize` for the summary keys.
        """
        start_time = time.time()
        n_updates = max(n_updates, 2)
        assert isinstance(x0, np.ndarray) and np.ndim(x0) == 1, "x0 must be a 1-d np.ndarray"
        assert 0. <= burn_in < 1., "burn_in must be a fraction in [0, 1)"

        #break condition
        if break_condition is None: break_condition = lambda a: False
        elif break_condition == "default": break_condition = self._default_break_condition
        elif callable(break_condition): pass
        else: raise Exception("No valid input for break condition provided!")
        if run_in_every_iteration is None: run_in_every_iteration = lambda a: False

        self.rng = np.random if rng is None else (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng))
        for obj in self.proposal_distributions:
            obj.rng = self.rng
        self._burn_in = burn_in
        self._stop_checks_passed = 0

        # Initialize Metropolis
        x = x0.copy()
        likelihood = self.log_likelihood_function(x, self.args)
        if info: print("Starting likelihood. f(x)= ", likelihood)
        prior = self.prior_function(x, self.bounds, self.args)
        # every trace starts with x0, so the same index is the same state in all of them
        self.trace = {"x": [x0], "f(x)": [likelihood], "log posterior": [likelihood + prior],
                      "accepted": [np.nan], "time stamp": [time.time() - start_time]}

        #########################################################
        # Begin main loop
        for i in np.arange(1, n_updates):
            accepted = 0.
            for obj in self.proposal_distributions:
                x, prior, likelihood, jt = self._jump(x, obj, prior, likelihood)
                obj.jump_trace.append(jt)
                accepted += jt

            # Update the trace objects
            self.trace["x"].append(x)
            self.trace["f(x)"].append(likelihood)
            self.trace["log posterior"].append(likelihood + prior)
            self.trace["accepted"].append(accepted / len(self.proposal_distributions))
            self.trace["time stamp"].append(time.time() - start_time)
            # adapt only now, so the adaptation window includes the current state
            for obj in self.proposal_distributions:
                obj.adapt(i, self)
            run_in_every_iteration(self)

            if info and (i % 10) == 0: print("Finished ", i, " out of ", n_updates, " iterations. f(x)= ", likelihood)
            if break_condition(self): break
        # End main loop
        # once, from the final trace: rebuilt every iteration it made the run time quadratic
        self.mcmc_info = summarize(self.trace, burn_in)
        return self.mcmc_info

    @staticmethod
    def _default_break_condition(obj):
        """The ``"default"`` break condition; see :py:meth:`run_mcmc` and ``_STOP_*``."""
        n = len(obj.trace["x"])
        if n < _STOP_MIN_LENGTH or n % _STOP_CHECK_EVERY: return False
        passed = gpMCMC._looks_converged(obj)
        obj._stop_checks_passed = getattr(obj, "_stop_checks_passed", 0) + 1 if passed else 0
        return obj._stop_checks_passed >= _STOP_CONSECUTIVE

    @staticmethod
    def _looks_converged(obj):
        # a chain that accepted nothing since the last check is stuck, not converged
        if not any(np.any(np.asarray(p.jump_trace[-_STOP_CHECK_EVERY:]) > 0)
                   for p in obj.proposal_distributions):
            return False
        x = np.asarray(obj.trace["x"])
        post = x[int(getattr(obj, "_burn_in", 0.5) * len(x)):]
        n_eff = ess(post)
        if not np.all(n_eff >= _STOP_MIN_ESS): return False   # also False for nan (never moved)
        early, late = post[:len(post) // 10], post[len(post) // 2:]
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(early.var(axis=0) / ess(early) + late.var(axis=0) / ess(late))
            z = np.abs(early.mean(axis=0) - late.mean(axis=0)) / se
        return bool(np.all(z < _STOP_MAX_Z))   # nan fails too

    ###############################################################
    def _jump(self, x_old, obj, prior_eval, likelihood):
        x_star = x_old.copy()
        assert callable(obj.proposal_dist), "proposal_dist must be callable"

        # get proposed x (x_star)
        x_star[obj.indices] = obj.proposal_dist(x_old[obj.indices].copy(), x_old, obj)
        # evaluate prior(x_star)
        prior_evaluation_x_star = self.prior_function(x_star, self.bounds, self.args)
        jump_trace = 0.

        # if prior(x_start) is not -inf, get likelihood
        if prior_evaluation_x_star != -np.inf:
            likelihood_star = self.log_likelihood_function(x_star, self.args)
            if np.isnan(likelihood_star): raise Exception("Likelihood evaluation = NaN in gpMCMC")
            expo = prior_evaluation_x_star + likelihood_star - prior_eval - likelihood
            if expo < 50: metr_ratio = np.exp(expo)
            else: metr_ratio = 1.1
            if np.isnan(metr_ratio):  metr_ratio = 0.
            if metr_ratio > getattr(self, "rng", np.random).uniform(0, 1) or obj.auto_accept:
                x = x_star
                prior_eval = prior_evaluation_x_star
                likelihood = likelihood_star
                jump_trace = 1.
            else:
                x = x_old
        else:
            x = x_old

        return x, prior_eval, likelihood, jump_trace

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


###############################################################
class ProposalDistribution:
    def __init__(self,
                 indices,
                 proposal_dist="normal",
                 init_prop_Sigma=None,
                 adapt_callable=None,
                 r_opt=.234,
                 c_0=10,
                 c_1=.8,
                 K=10,
                 auto_accept=False,
                 adapt_cov=True,
                 prop_args=None,
                 ID=None):
        """
        Class to define a proposal distribution.

        Parameters
        ----------
        indices : iterable of int
            The indices of the parameters that should be drawn from this proposal distribution.
        proposal_dist : callable, optional
            A callable to calculate the proposal distribution evaluation.
            It is defined as ``def name(x, para, obj)``, where ``obj`` is a :py:class:`ProposalDistribution`
            instance. The function should return a new proposal for ``x``.
            ``para`` are all other parameters. Default is a normal distribution with the default
            ``init_prop_Sigma``.
        init_prop_Sigma : np.ndarray, optional
            If the proposal distribution is normal this is the covariance of the initial proposal distribution.
            It will be updated if adapt_callable = ``normal`` or a callable.
            While it is optional to provide it, it is highly recommended to do so.
            A warning will be printed in that case. A good rule of thumb
            is to orient yourself on the size of your domain. The default is the identity matrix.
        adapt_callable : Callable or None or string, optional
            A callable to adapt the distribution. The default is None which means
            the proposal distribution will not be adapted.
            Use ``normal`` (default) for the default adaption procedure for normal distributions.
            The callable should be defined as ``def adapt(index, mcmc_obj)`` and not return anything
            but update the :py:attr:`ProposalDistribution.prop_args` attribute. Note, any
            adapt function will have to be well thought through.
            Most adapt functions will not lead to a stationary final distributions. Use with caution.
        auto_accept : bool, optional
            Indicates whether to auto-accept the jump.
        prop_args : Any, optional
            Arguments that will be available as obj attribute in ``proposal_dist`` and ``adapt_callable``.
        """

        self.indices = indices
        self.r_opt = r_opt
        self.c_0 = c_0
        self.c_1 = c_1
        self.K = K
        self.auto_accept = auto_accept
        self.adapt_cov = adapt_cov
        self.ID = ID
        dim = len(indices)
        self.jump_trace = []
        # set by gpMCMC.run_mcmc: the run's random generator, for custom proposals too
        self.rng = None
        if proposal_dist == "normal":
            self.proposal_dist = self.normal_proposal_dist
        elif callable(proposal_dist):
            self.proposal_dist = proposal_dist
        else:
            raise Exception("No proposal distribution specified!")

        if proposal_dist == "normal" and init_prop_Sigma is None:
            init_prop_Sigma = np.identity(dim)
            warnings.warn("You are using the normal proposal distribution for normal distributions\n \
                           but did not provide `init_prop_sigma`. This can lead to slow convergence")

        if callable(adapt_callable):
            self.adapt = adapt_callable
        elif adapt_callable == "normal" or proposal_dist == "normal":
            self.adapt = self._adapt
        else:
            if isinstance(adapt_callable, str): raise Exception("Invalid string provided for adapt callable.")
            self.adapt = self._no_adapt

        if prop_args is None:
            self.prop_args = {"prop_Sigma": init_prop_Sigma, "sigma_m": 2.4 ** 2 / dim}
        else:
            self.prop_args = prop_args
            if adapt_callable == "normal":
                self.prop_args["prop_Sigma"] = init_prop_Sigma
                self.prop_args["sigma_m"] = 2.4 ** 2 / dim

    #########################################################
    def normal_proposal_dist(self, x, hps, obj):
        """
        Default multivariate-normal proposal: draw from N(x, sigma_m Σ), with Σ =
        ``obj.prop_args["prop_Sigma"]`` and the step scale sigma_m =
        ``obj.prop_args["sigma_m"]`` (1 if absent), which adaptation tunes toward the
        acceptance rate ``r_opt``.

        Parameters
        ----------
        x : np.ndarray
            Current position in hyperparameter space, shape (N,).
        hps : np.ndarray
            Current hyperparameters (unused here; present for interface compatibility).
        obj : ProposalDistribution
            The enclosing proposal-distribution object, which carries ``prop_args``.

        Returns
        -------
        proposal : np.ndarray
            Proposed hyperparameter vector, shape (N,).
        """
        # sigma_m is what adaptation tunes; without it the step size never reacts to the
        # acceptance rate (Shaby & Wells propose from N(x, sigma_m * prop_Sigma))
        prop_Sigma = obj.prop_args["prop_Sigma"]
        sigma_m = obj.prop_args.get("sigma_m", 1.0)
        rng = np.random if getattr(obj, "rng", None) is None else obj.rng
        # factor once per change of prop_Sigma or sigma_m, not on every draw; the cache holds
        # prop_Sigma itself, so `is` cannot be fooled by a recycled object id
        cache = getattr(obj, "_proposal_factor", None)
        if cache is None or cache[0] is not prop_Sigma or cache[1] != sigma_m:
            try:
                factor = np.linalg.cholesky(sigma_m * np.atleast_2d(prop_Sigma))
            except np.linalg.LinAlgError:
                factor = None   # positive semi-definite only: sample through the SVD instead
            obj._proposal_factor = cache = (prop_Sigma, sigma_m, factor)
        if cache[2] is None:
            return rng.multivariate_normal(mean=x, cov=sigma_m * np.atleast_2d(prop_Sigma))
        return x + cache[2] @ rng.standard_normal(len(x))

    def _adapt(self, end, mcmc_obj):
        K = self.K
        if (end % K) == 0:
            k = 3
            c_0 = self.c_0
            c_1 = self.c_1
            r_opt = self.r_opt
            prop_Sigma = self.prop_args["prop_Sigma"]
            sigma_m = self.prop_args["sigma_m"]
            gamma2 = 1. / ((end / K) + k) ** c_1
            gamma1 = c_0 * gamma2
            # the last K iterations, the current one included (run_mcmc adapts after
            # appending it); slicing the lists keeps this O(K) rather than O(chain length)
            r_hat = np.mean(self.jump_trace[-K:])
            sigma_m = np.exp(np.log(sigma_m) + gamma1 * (r_hat - r_opt))
            if self.adapt_cov:
                window = np.asarray(mcmc_obj.trace["x"][-K:])[:, self.indices]
                prop_Sigma = prop_Sigma + gamma2 * (np.atleast_2d(np.cov(window.T)) - prop_Sigma)
            self.prop_args["prop_Sigma"] = prop_Sigma
            self.prop_args["sigma_m"] = sigma_m

    def _no_adapt(self, end, mcmc_obj):
        return

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
