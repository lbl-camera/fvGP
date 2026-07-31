"""Bayesian optimization of GP hyperparameters (``method='bo'``).

This is the inner optimizer used when the outer marginal likelihood is expensive,
noisy, and effectively gradient-free -- the regime a distributed mBCG solve puts
you in, where the log-determinant comes from stochastic Lanczos quadrature and the
solve from truncated CG, so ``L(theta)`` is only ever observed with noise.

The surrogate is itself an :py:class:`fvgp.GP`, so this module deliberately uses
fvGP and not gpCAM: gpCAM depends on fvGP, and importing it here would make the
dependency circular. Everything needed for the inner loop -- an ARD Matern-5/2
kernel, fixed per-point observation noise, and exact type-II ML training with
analytic gradients -- already exists in fvGP.

The recursion bottoms out immediately. The inner GP only ever sees the
tens-to-low-hundreds of theta points the BO evaluates, so it is tiny, uses a
standard analytic kernel, and is trained with ``method='local'`` (L-BFGS-B with
analytic gradients). There is no infinite regress and no BO-tuning of the BO.
"""

import warnings
import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from .kernels import matern_kernel_diff2, get_anisotropic_distance_matrix


# ----------------------------------------------------------------------------- #
# search-space transform
# ----------------------------------------------------------------------------- #
class _LogAffineTransform:
    """Maps hyperparameters to the space the BO actually searches.

    Length scales, signal variances and noise terms are positive and act
    multiplicatively, so a log transform makes the likelihood surface far more
    stationary and better conditioned. But fvGP hyperparameters are not all of that
    kind -- a coefficient of a user-supplied prior mean is free to be negative -- so
    the transform is decided per dimension: log where the bounds are strictly
    positive, identity everywhere else. The searched box is then rescaled to the
    unit cube, which is what the Sobol design and the acquisition optimizer expect.
    """

    def __init__(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        self.log_mask = (bounds[:, 0] > 0.0) & (bounds[:, 1] > 0.0)
        lo = np.where(self.log_mask, np.log(np.where(self.log_mask, bounds[:, 0], 1.0)), bounds[:, 0])
        hi = np.where(self.log_mask, np.log(np.where(self.log_mask, bounds[:, 1], 1.0)), bounds[:, 1])
        self.lo, self.hi = lo, hi
        self.span = np.where(hi > lo, hi - lo, 1.0)
        self.dim = len(bounds)

    def to_unit(self, theta):
        """Hyperparameters -> unit cube."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        w = np.where(self.log_mask, np.log(np.clip(theta, 1e-300, None)), theta)
        return np.clip((w - self.lo) / self.span, 0.0, 1.0)

    def from_unit(self, u):
        """Unit cube -> hyperparameters."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        w = self.lo + np.clip(u, 0.0, 1.0) * self.span
        return np.where(self.log_mask, np.exp(w), w)


# ----------------------------------------------------------------------------- #
# surrogate
# ----------------------------------------------------------------------------- #
def _surrogate_kernel(x1, x2, hps):
    """Matern-5/2 with ARD: hps[0] is the signal variance, hps[1:] the length scales.

    Matern-5/2 rather than a squared exponential because the latter's infinite
    smoothness gives overconfident extrapolation and an ill-conditioned covariance
    when there are only a few dozen points. ARD matters here specifically: different
    hyperparameters move the marginal likelihood on very different scales and some
    directions are nearly flat (the classic signal-variance/length-scale ridge), so
    the learned length scales double as a sensitivity readout.
    """
    d = get_anisotropic_distance_matrix(x1, x2, hps[1:])
    return hps[0] * matern_kernel_diff2(d, 1.0)


_SQRT5 = np.sqrt(5.0)


def _surrogate_kernel_dx(x_query, x_data, hps):
    """d/dx_query of the Matern-5/2 ARD kernel between one query point and the data.

    With r the anisotropic distance and k = s * (1 + sqrt5 r + 5r^2/3) exp(-sqrt5 r),

        dk/dr    = -(5/3) s r (1 + sqrt5 r) exp(-sqrt5 r)
        dr/dx_i  = (x_i - x'_i) / (l_i^2 r)

    so the r cancels and

        dk/dx_i  = -(5/3) s (1 + sqrt5 r) exp(-sqrt5 r) (x_i - x'_i) / l_i^2

    which is finite at r = 0, where it correctly vanishes -- no special case needed.
    Written out rather than taken from fvGP's ``d_kernel_dx``, which is a finite
    difference with a fixed 1e-8 step and so would defeat the point of doing this.

    Parameters
    ----------
    x_query : np.ndarray, shape (D,)
    x_data : np.ndarray, shape (N, D)
    hps : np.ndarray
        ``[signal_variance, length_scales...]``.

    Returns
    -------
    np.ndarray, shape (D, N)
    """
    signal, lengths = hps[0], np.asarray(hps[1:1 + x_data.shape[1]], dtype=float)
    delta = (np.asarray(x_query, dtype=float)[None, :] - x_data) / lengths[None, :]   # (N, D)
    r = np.sqrt(np.sum(delta ** 2, axis=1))                                           # (N,)
    radial = -(5.0 / 3.0) * signal * (1.0 + _SQRT5 * r) * np.exp(-_SQRT5 * r)          # (N,)
    # delta / lengths converts d/d(scaled) back to d/d(x_i)
    return (radial[:, None] * delta / lengths[None, :]).T                              # (D, N)


def _surrogate_kernel_grad(x1, x2, hps):
    """dk/d(hyperparameters) for the Matern-5/2 ARD kernel, shape (H, N1, N2).

    Supplied so the surrogate's own type-II ML training runs on real gradients. Without
    it fvGP falls back to ``_dkernel_dh``, a central difference with a fixed 1e-8 step
    that rebuilds the full covariance matrix twice per hyperparameter -- 2(D+2) matrix
    builds for every gradient evaluation, each one only accurate to a few digits.

    With s the signal variance, l the length scales and r the anisotropic distance,

        dk/ds   = (1 + sqrt5 r + 5r^2/3) exp(-sqrt5 r)
        dk/dl_i = (5/3) s (1 + sqrt5 r) exp(-sqrt5 r) (x1_i - x2_i)^2 / l_i^3

    The second follows from dk/dr = -(5/3) s r (1 + sqrt5 r) exp(-sqrt5 r) and
    dr/dl_i = -(x1_i - x2_i)^2 / (l_i^3 r); the r cancels, so it is finite at r = 0.
    Any trailing hyperparameters (the learned noise level) do not enter the kernel and
    get a zero row.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    hps = np.asarray(hps, dtype=float)
    dim = x1.shape[1]
    lengths = hps[1:1 + dim]

    diff = x1[:, None, :] - x2[None, :, :]                       # (N1, N2, D)
    scaled = diff / lengths[None, None, :]
    r = np.sqrt(np.sum(scaled ** 2, axis=2))                     # (N1, N2)
    decay = np.exp(-_SQRT5 * r)

    grad = np.zeros((len(hps), x1.shape[0], x2.shape[0]))
    grad[0] = (1.0 + _SQRT5 * r + (5.0 / 3.0) * r ** 2) * decay
    common = (5.0 / 3.0) * hps[0] * (1.0 + _SQRT5 * r) * decay
    for i in range(dim):
        grad[1 + i] = common * diff[:, :, i] ** 2 / lengths[i] ** 3
    return grad


def _homoscedastic_noise_grad(dim):
    """d(noise)/d(hyperparameters) for the learned homoscedastic level, shape (H, N)."""
    def noise_grad(x, hps):
        out = np.zeros((len(hps), len(x)))
        out[dim + 1] = 1.0
        return out
    return noise_grad


def _polynomial_mean_dx(coefficients, u, dim):
    """d/du of the fitted quadratic mean ``c0 + sum c1_i u_i + sum c2_i u_i^2``."""
    if coefficients is None:
        return np.zeros(dim)
    linear = np.asarray(coefficients[1:1 + dim], dtype=float)
    quadratic = np.asarray(coefficients[1 + dim:1 + 2 * dim], dtype=float)
    return linear + 2.0 * quadratic * np.asarray(u, dtype=float)


def _posterior_mean_var_and_grad(u, gp, dim):
    """Posterior mean, variance and both of their gradients at a single point.

    Everything is assembled from quantities the surrogate already holds -- KVinvY and
    the stored inverse -- so the gradient costs one (D, N) derivative matrix and two
    matvecs on top of the value, rather than the D+1 full acquisition evaluations a
    finite-difference gradient needs.
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    x_data = np.asarray(gp.x_data, dtype=float)
    hps = np.asarray(gp.hyperparameters, dtype=float)

    k = np.asarray(_surrogate_kernel(x_data, u[None, :], hps)).reshape(-1)   # (N,)
    dk = _surrogate_kernel_dx(u, x_data, hps)                               # (D, N)
    kvinv_y = np.asarray(gp.kv.KVinvY, dtype=float)[:, 0]                   # (N,)
    alpha = np.asarray(gp.kv.solve(k.reshape(-1, 1)), dtype=float).reshape(-1)  # KV^-1 k

    mean_f = getattr(gp, "_bo_mean_function", None)
    coefficients = getattr(mean_f, "coefficients", None) if mean_f is not None else None
    prior = 0.0 if mean_f is None else float(np.asarray(mean_f(u[None, :], None)).reshape(-1)[0])

    mean = prior + float(k @ kvinv_y)
    d_mean = _polynomial_mean_dx(coefficients, u, dim) + dk @ kvinv_y

    # k(u, u) is the signal variance for a stationary kernel, so its gradient is zero
    var = float(hps[0] - k @ alpha)
    d_var = -2.0 * (dk @ alpha)

    # Below a small fraction of the signal variance the posterior variance is a
    # difference of two nearly equal numbers and has lost all of its significant
    # digits -- the GP is interpolating there. Its derivative is then noise, and
    # dividing by the vanishing std to get d(std) would amplify that noise by 1/std.
    # Report the floor with a zero derivative instead: a variance indistinguishable
    # from zero carries no directional information. This costs nothing where it
    # matters, since the acquisition is driven by the mean once the variance is gone.
    floor = 1e-10 * max(float(hps[0]), 1e-300)
    if var <= floor:
        return mean, floor, d_mean, np.zeros_like(d_var)
    return mean, var, d_mean, d_var


def _polynomial_mean(u_data, y_data, dim):
    """Least-squares quadratic (no cross terms) in the searched coordinates.

    Marginal-likelihood surfaces are roughly bowl-shaped away from the optimum.
    Letting the mean absorb that global trend leaves the kernel to model only the
    near-optimum residual, which improves extrapolation and cuts the number of
    expensive evaluations.

    The fit is ordinary least squares on the data observed so far rather than extra
    GP hyperparameters: a full quadratic would add 1 + d + d(d+1)/2 parameters, which
    at d = 20 is 231 -- hopeless to train on the few dozen points BO can afford. The
    diagonal-only form needs 1 + 2d, and is refit each iteration. Returns None when
    there are too few points to fit it stably, in which case the caller falls back to
    fvGP's default constant mean.
    """
    n_coef = 1 + 2 * dim
    if len(u_data) < 2 * n_coef:
        return None
    design = np.hstack([np.ones((len(u_data), 1)), u_data, u_data ** 2])
    try:
        coef, *_ = np.linalg.lstsq(design, y_data, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None
    if not np.all(np.isfinite(coef)):  # pragma: no cover
        return None

    def mean_f(x, hps):
        x = np.atleast_2d(x)
        return (np.hstack([np.ones((len(x), 1)), x, x ** 2]) @ coef).reshape(len(x))

    mean_f.coefficients = coef
    return mean_f


def _homoscedastic_noise(dim):
    """Noise function exposing a single learned noise variance as hyperparameter dim+1."""
    def noise_f(x, hps):
        return np.full(len(x), max(float(hps[dim + 1]), 1e-14))
    return noise_f


def _fit_surrogate(u_data, y_data, v_data, dim, train_max_iter):
    """Fit the inner GP on the points evaluated so far.

    N is tiny and the kernel is standard and differentiable, so this is an exact GP
    trained by type-II maximum likelihood with L-BFGS-B and analytic gradients.

    ``v_data`` is the known per-point observation variance, or None when the objective
    does not report its own precision. In that case a single homoscedastic noise level
    is learned as an extra surrogate hyperparameter, which is what lets ``method='bo'``
    be used with any user objective: a deterministic one simply drives the learned
    noise to its lower bound and the surrogate interpolates, while a noisy one has its
    noise estimated rather than assumed.
    """
    from .gp import GP

    mean_f = _polynomial_mean(u_data, y_data, dim)
    # Scale the signal-variance bounds to what the kernel actually has to explain.
    # Using the raw spread when a polynomial mean is already absorbing the bowl lets
    # the optimizer run off along the variance/length-scale ridge to a degenerate fit
    # (huge variance, maximal length scale) that carries no usable ARD information.
    if mean_f is not None:
        residual = y_data - mean_f(u_data, None)
    else:
        residual = y_data - np.mean(y_data)
    scale = float(np.var(residual))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.var(y_data)) or 1.0

    init = np.concatenate([[scale], np.full(dim, 0.3)])
    # A length scale beyond ~1 already means near-total correlation across the unit
    # cube, so the upper bound is kept close to the box: it stops the optimizer from
    # wandering into the flat, unidentifiable region and keeps the ARD values
    # meaningful as a relative ranking.
    bounds = np.vstack([[1e-4 * scale + 1e-12, 1e2 * scale + 1e-9],
                        np.tile([1e-2, 2.0], (dim, 1))])

    # analytic derivatives so the surrogate's own training uses real gradients rather
    # than fvGP's finite-difference fallback
    kwargs = dict(kernel_function=_surrogate_kernel,
                  kernel_function_grad=_surrogate_kernel_grad)
    if v_data is not None:
        kwargs["noise_variances"] = v_data
    else:
        # learn one noise level; the lower bound doubles as a nugget so a perfectly
        # deterministic objective still yields a well-conditioned covariance
        nugget = max(1e-10 * scale, 1e-14)
        init = np.concatenate([init, [max(1e-4 * scale, nugget)]])
        bounds = np.vstack([bounds, [nugget, max(scale, 10.0 * nugget)]])
        kwargs["noise_function"] = _homoscedastic_noise(dim)
        kwargs["noise_function_grad"] = _homoscedastic_noise_grad(dim)
    kwargs["init_hyperparameters"] = init
    if mean_f is not None:
        kwargs["prior_mean_function"] = mean_f

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # linalg_mode="CholInv" is a large win here, not a stylistic choice. The
        # acquisition asks for variance_only=True at hundreds of candidates per
        # iteration, but fvGP can only take that shortcut when the inverse is stored;
        # in the default "Chol" mode it builds the full (V x V) posterior covariance
        # and then throws away everything but the diagonal. At V = n_raw = 512 that is
        # a 512x512 matmul per call. Storing the inverse of a covariance over a few
        # dozen points costs nothing and made a 40-evaluation run 3.7x faster with
        # identical results.
        gp = GP(u_data, y_data, linalg_mode="CholInv", **kwargs)
        # method='local', never 'bo' -- this is where the recursion stops
        gp.train(hyperparameter_bounds=bounds, method="local", max_iter=train_max_iter)
    gp._bo_mean_function = mean_f
    gp._bo_learned_noise = None if v_data is not None else float(gp.hyperparameters[dim + 1])
    return gp


def _laplace_posterior(gp, u_best, tf):
    """Laplace approximation of the theta-posterior from the surrogate's curvature.

    A single BO run therefore yields not just the MLE-II point estimate but the
    hyperparameter uncertainty that would otherwise cost an MCMC run. The surrogate
    is analytic and cheap, so the Hessian is taken by finite differences on its
    posterior mean. Reported in the searched coordinates -- log-theta for the
    dimensions that were log-transformed -- since that is where the quadratic
    approximation is actually reasonable.

    Returns (covariance, curvature) or (None, None) if the mode is not a clean
    maximum, which happens when the budget was too small to resolve it.
    """
    dim = len(u_best)
    h = 1e-3

    def neg_mean(u):
        # surrogate models the negated objective, so this is the objective itself
        return -float(np.asarray(gp.posterior_mean(np.atleast_2d(u))["m(x)"]).reshape(-1)[0])

    hess = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            ei, ej = np.zeros(dim), np.zeros(dim)
            ei[i], ej[j] = h, h
            val = (neg_mean(u_best + ei + ej) - neg_mean(u_best + ei - ej)
                   - neg_mean(u_best - ei + ej) + neg_mean(u_best - ei - ej)) / (4.0 * h * h)
            hess[i, j] = hess[j, i] = val
    # map from unit-cube coordinates to the transformed (log) coordinates
    jac = np.diag(1.0 / tf.span)
    hess_w = jac @ hess @ jac
    hess_w = 0.5 * (hess_w + hess_w.T)
    eigenvalues = np.linalg.eigvalsh(hess_w)
    if not np.all(np.isfinite(hess_w)) or np.any(eigenvalues <= 0.0):
        return None, hess_w
    try:
        return np.linalg.inv(hess_w), hess_w
    except np.linalg.LinAlgError:  # pragma: no cover
        return None, hess_w


# ----------------------------------------------------------------------------- #
# acquisition
# ----------------------------------------------------------------------------- #
def _noisy_expected_improvement(u, gp, y_best_samples, rng):
    """Noisy expected improvement, for a surrogate of the *negated* objective.

    Plain EI is ill-defined here because the incumbent is itself observed with noise.
    Noisy-EI (Letham et al. 2019) averages EI over the posterior of the incumbent,
    represented by ``y_best_samples`` drawn once per BO iteration so that common
    random numbers keep the surface smooth for the acquisition optimizer.
    """
    u = np.atleast_2d(u)
    m = np.asarray(gp.posterior_mean(u)["m(x)"]).reshape(len(u))
    var = np.asarray(gp.posterior_covariance(u, variance_only=True)["v(x)"]).reshape(len(u))
    std = np.sqrt(np.maximum(var, 1e-12))
    imp = m[:, None] - y_best_samples[None, :]
    z = imp / std[:, None]
    ei = imp * norm.cdf(z) + std[:, None] * norm.pdf(z)
    return np.maximum(np.mean(ei, axis=1), 0.0)


def _nei_value_and_grad(u, gp, y_best_samples, dim):
    """Noisy expected improvement and its exact gradient at a single point.

    The gradient of expected improvement collapses. With imp = mu - y*, z = imp/sigma,

        dEI/dx = (dimp/dx) Phi(z) + imp phi(z) dz/dx + (dsigma/dx) phi(z)
                 + sigma phi'(z) dz/dx

    and phi'(z) = -z phi(z) with sigma z = imp, so the second and fourth terms cancel
    identically and

        dEI/dx = Phi(z) dmu/dx + phi(z) dsigma/dx.

    Averaging over the sampled incumbents leaves that structure untouched: the Monte
    Carlo enters only through the two scalars mean(Phi(z_k)) and mean(phi(z_k)), so
    nothing has to be differentiated through the sampling.
    """
    mean, var, d_mean, d_var = _posterior_mean_var_and_grad(u, gp, dim)
    std = np.sqrt(var)
    d_std = d_var / (2.0 * std)

    imp = mean - y_best_samples                       # (K,)
    z = imp / std
    cdf, pdf = norm.cdf(z), norm.pdf(z)
    value = float(np.mean(imp * cdf + std * pdf))
    grad = np.mean(cdf) * d_mean + np.mean(pdf) * d_std
    if value <= 0.0:
        # the acquisition is clipped at zero, so its gradient is too
        return 0.0, np.zeros(dim)
    return value, grad


def _maximize_acquisition(acq, dim, rng, n_restarts, n_raw, acq_grad=None):
    """Multi-start L-BFGS-B on the surrogate.

    The outer likelihood having no usable gradient says nothing about the acquisition,
    which is an analytic function of a cheap surrogate -- the two are routinely
    conflated.

    ``acq_grad`` supplies the exact gradient, which is both faster and better
    conditioned than letting scipy difference the acquisition: a finite-difference
    gradient costs D+1 evaluations per step, and its error dominates precisely where
    the true gradient is small, which is the flat region near the optimum where the
    search spends most of its time. The vectorized ``acq`` is still used for the random
    pre-screen, where only values are needed.
    """
    raw = rng.random((n_raw, dim))
    vals = acq(raw)
    starts = raw[np.argsort(-vals)[:n_restarts]]
    best_u, best_v = starts[0], float(vals.max())
    if acq_grad is None:
        objective, jac = lambda z: -float(acq(z.reshape(1, -1))[0]), None
    else:
        def objective(z):
            value, gradient = acq_grad(z)
            return -value, -gradient
        jac = True
    for u0 in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(objective, u0, jac=jac,
                           method="L-BFGS-B", bounds=[(0.0, 1.0)] * dim)
        if res.success and -res.fun > best_v:
            best_v, best_u = -float(res.fun), np.clip(res.x, 0.0, 1.0)
    return best_u, best_v


# ----------------------------------------------------------------------------- #
# the optimizer
# ----------------------------------------------------------------------------- #
def bayesian_optimize(objective_function,
                      hyperparameter_bounds,
                      init_hyperparameters,
                      max_iter=50,
                      bo_args=None,
                      info=False,
                      callback=None,
                      early_stop=None):
    """Minimize ``objective_function`` over ``hyperparameter_bounds`` by Bayesian optimization.

    Parameters
    ----------
    objective_function : callable
        ``f(hps) -> float``, MINIMIZED. Expensive and noisy by assumption.
    hyperparameter_bounds : np.ndarray
        Array of shape (N, 2).
    init_hyperparameters : np.ndarray
        Warm start of shape (N,); included as one of the initial design points. In an
        autonomous-experimentation loop theta usually moves very little between data
        acquisitions, so seeding from the previous optimum often converges in a
        handful of evaluations.
    max_iter : int, optional
        Budget in *objective evaluations*, including the initial design. Each one is
        an expensive likelihood, so this is the quantity worth capping. Default 50.
    bo_args : dict, optional
        ``n_init``, ``n_restarts``, ``n_raw``, ``n_incumbent_samples``, ``seed``,
        ``noise_function``, ``noise_variance``, ``ei_tolerance``,
        ``surrogate_train_max_iter``, ``refit_every``.
    info : bool, optional
        Log progress each iteration.
    callback : callable, optional
        ``callback(theta, fval, iteration, state)`` after every evaluation.
    early_stop : callable, optional
        Polled each iteration; return True to stop. Used for asynchronous runs.

    Returns
    -------
    theta : np.ndarray
        Recommended hyperparameters.
    bo_info : dict
        Trace, surrogate diagnostics, ARD sensitivity, and the approximate posterior.
    """
    a = dict(bo_args or {})
    bounds = np.asarray(hyperparameter_bounds, dtype=float)
    dim = len(bounds)
    tf = _LogAffineTransform(bounds)
    rng = np.random.default_rng(a.get("seed", 0))

    n_init = int(a.get("n_init", min(max(2 * (dim + 1), 5), max(10 * dim, 5))))
    n_init = max(2, min(n_init, max_iter))
    # 3 rather than 5: measured across two problems and eight seeds, 3 restarts match
    # 5 and 8 exactly on solution quality while cutting 15-24% of the runtime. Dropping
    # to 2 does start to cost quality on the harder surface.
    n_restarts = int(a.get("n_restarts", 3))
    n_raw = int(a.get("n_raw", 512))
    n_inc = int(a.get("n_incumbent_samples", 64))
    ei_tol = float(a.get("ei_tolerance", 0.0))
    refit_every = max(1, int(a.get("refit_every", 1)))
    train_max_iter = int(a.get("surrogate_train_max_iter", 100))
    noise_function = a.get("noise_function", None)
    fixed_noise = a.get("noise_variance", None)

    # --- observation noise ---------------------------------------------------
    # The estimator's own spread is the right per-point noise -- with SLQ/Hutchinson
    # probes it is directly available, and feeding it in beats spending scarce data
    # fitting a homoscedastic noise term. fvGP's objective_function returns a bare
    # scalar though, so there is nowhere for that variance to come from unless the
    # caller supplies it. Hence: use `noise_function` if given, else a fixed
    # `noise_variance`, else fall back to a small jitter scaled to the observed
    # spread and let the surrogate absorb the rest.
    def _noise_for(theta, y_obs):
        """Known observation variance at ``theta``, or None if the objective cannot say.

        Called immediately after the objective has been evaluated at ``theta``, which is
        what lets a ``noise_function`` simply report the precision of the evaluation that
        just happened -- the way the stochastic-Lanczos estimator does.
        """
        if callable(noise_function):
            v = noise_function(theta)
            if v is not None and np.isfinite(v) and float(v) > 0.0:
                return float(v)
            return None
        if fixed_noise is not None:
            return max(float(fixed_noise), 1e-12)
        return None

    # --- initial design ------------------------------------------------------
    # Sobol rather than uniform: a space-filling design is markedly more efficient in
    # the low dimensions theta lives in.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_init = qmc.Sobol(d=dim, scramble=True, seed=int(a.get("seed", 0))).random(n_init)
    u_init[0] = tf.to_unit(init_hyperparameters)[0]          # warm start

    u_list, y_list, v_list, theta_list = [], [], [], []
    stopped_early = False

    def _evaluate(u_row, iteration):
        theta = tf.from_unit(u_row)[0]
        val = float(objective_function(theta))
        if not np.isfinite(val):
            val = np.finfo(float).max / 1e6
        u_list.append(np.asarray(u_row, dtype=float))
        theta_list.append(theta)
        y_list.append(val)
        v_list.append(_noise_for(theta, val))
        if callable(callback):
            best = int(np.argmin(y_list))
            callback(theta_list[best], y_list[best], iteration,
                     {"n_evaluations": len(y_list)})
        return val

    for i in range(len(u_init)):
        if callable(early_stop) and early_stop():
            stopped_early = True
            break
        _evaluate(u_init[i], i)

    # --- BO loop -------------------------------------------------------------
    gp = None
    ei_history = []
    n_eval = len(y_list)
    while n_eval < max_iter and not stopped_early:
        if callable(early_stop) and early_stop():
            stopped_early = True
            break
        u_arr = np.asarray(u_list)
        # the surrogate models the NEGATED objective, so BO maximizes throughout
        y_arr = -np.asarray(y_list)
        spread = float(np.std(y_arr))
        if not np.isfinite(spread) or spread <= 0.0:
            spread = 1.0
        # Known per-point variances are used as they are; if the objective reported none
        # at all, hand the surrogate None so it learns a single noise level instead.
        # A partial report is filled in with the mean of what is known.
        if all(v is None for v in v_list):
            v_arr = None
        else:
            known = [v for v in v_list if v is not None]
            filler = float(np.mean(known))
            v_arr = np.array([filler if v is None else v for v in v_list])

        if gp is None or (n_eval % refit_every) == 0:
            try:
                gp = _fit_surrogate(u_arr, y_arr, v_arr, dim, train_max_iter)
            except Exception as ex:  # pragma: no cover
                warnings.warn(f"Inner BO surrogate fit failed ({ex}); stopping BO early.")
                break

        # incumbent posterior, sampled once per iteration (common random numbers)
        m_obs = np.asarray(gp.posterior_mean(u_arr)["m(x)"]).reshape(len(u_arr))
        s_obs = np.sqrt(np.maximum(
            np.asarray(gp.posterior_covariance(u_arr, variance_only=True)["v(x)"]).reshape(len(u_arr)), 1e-12))
        y_best_samples = np.max(m_obs[:, None] + s_obs[:, None] * rng.standard_normal((len(u_arr), n_inc)), axis=0)

        u_next, ei = _maximize_acquisition(
            lambda z: _noisy_expected_improvement(z, gp, y_best_samples, rng),
            dim, rng, n_restarts, n_raw,
            acq_grad=lambda z: _nei_value_and_grad(z, gp, y_best_samples, dim))
        ei_history.append(float(ei))
        if info:
            logger.debug("BO iteration {}: best={:.6g}, EI={:.3g}", n_eval, min(y_list), ei)

        # Stop when the expected gain no longer matters. Calibrate `ei_tolerance` to
        # what actually changes the outer GP's predictions, not to likelihood digits
        # that make no downstream difference.
        if ei_tol > 0.0 and ei < ei_tol:
            logger.debug("BO terminated: EI {} below tolerance {}", ei, ei_tol)
            break

        _evaluate(u_next, n_eval)
        n_eval = len(y_list)

    # --- recommendation ------------------------------------------------------
    # Under noise the smallest observed value is a biased pick -- it is partly a lucky
    # draw of the estimator -- so the right recommendation is the evaluated point with
    # the best surrogate posterior mean, which averages that luck out.
    #
    # That reasoning only holds to the extent the observations really are noisy. With a
    # deterministic objective the observation is exact and deferring to the surrogate
    # only imports its fitting error -- which can return a point worse than the one the
    # search started from. The scale of the correction is therefore tied to the actual
    # noise level, whether it was reported by the objective or learned by the surrogate,
    # and the smoothed pick is rejected if it is more than a few noise standard
    # deviations worse than the best observation. A deterministic objective drives the
    # learned noise to its nugget, which collapses this back to the best observation.
    y_arr = np.asarray(y_list)
    u_arr = np.asarray(u_list)
    best_idx = int(np.argmin(y_arr))
    known = [v for v in v_list if v is not None]
    noise_learned = False
    if known:
        noise_var = float(np.mean(known))
    elif gp is not None and getattr(gp, "_bo_learned_noise", None) is not None:
        noise_var = float(gp._bo_learned_noise)
        noise_learned = True
    else:
        noise_var = 0.0
    # Only a *reported* noise level earns the right to override the observations. A
    # learned one cannot separate estimator noise from surrogate misfit: on a hard
    # deterministic surface the surrogate explains its own misfit as noise, and this was
    # measured inflating the level to the order of the spread of the data itself. Tying
    # the tolerance to robust spread statistics does not rescue it either, because a
    # marginal-likelihood surface is heavy-tailed enough to inflate those too. Since the
    # upside of smoothing is small and returning a point worse than one already measured
    # is a visible failure, a learned level is used for conditioning and acquisition but
    # not for the final pick.
    if gp is not None and known and noise_var > 0.0:
        try:
            m_obs = np.asarray(gp.posterior_mean(u_arr)["m(x)"]).reshape(len(u_arr))
            cand = int(np.argmax(m_obs))
            if y_arr[cand] <= y_arr[best_idx] + 3.0 * np.sqrt(noise_var):
                best_idx = cand
        except Exception:  # pragma: no cover
            pass
    theta_best = np.asarray(theta_list[best_idx], dtype=float)

    # The surrogate carries two things worth handing back for free, with no further
    # likelihood evaluations: a sensitivity ranking and an approximate theta-posterior.
    #
    # The ARD length scales are the usual sensitivity readout, but here they describe
    # only the residual left *after* the polynomial mean, so on a near-quadratic
    # likelihood -- the common case -- the mean absorbs the structure and the ARD
    # values saturate and rank nothing. The curvature of the fitted surface is the
    # honest ranking, so it is what `sensitivity` reports; the raw ARD values are
    # still exposed separately.
    hps_surrogate, ard = None, None
    sensitivity, posterior_cov, curvature = None, None, None
    if gp is not None:
        hps_surrogate = np.asarray(gp.hyperparameters, dtype=float)
        ard = hps_surrogate[1:1 + dim]
        u_best = u_arr[best_idx]
        try:
            posterior_cov, curvature = _laplace_posterior(gp, u_best, tf)
            if curvature is not None:
                sensitivity = np.abs(np.diag(curvature))
        except Exception:  # pragma: no cover
            pass
        if sensitivity is None:
            sensitivity = 1.0 / np.maximum(ard, 1e-12)

    bo_info = {
        "x": theta_best,
        "f(x)": float(y_arr[best_idx]),
        "trace x": np.asarray(theta_list),
        "trace f(x)": y_arr,
        "trace u": u_arr,
        "n_evaluations": len(y_list),
        "ei history": np.asarray(ei_history),
        "surrogate hyperparameters": hps_surrogate,
        "ard length scales": ard,
        "sensitivity": sensitivity,
        "posterior covariance": posterior_cov,
        "curvature": curvature,
        "log-transformed dimensions": tf.log_mask,
        "stopped early": stopped_early,
        "observation noise variance": noise_var if noise_var > 0.0 else None,
        "noise was learned": noise_learned,
        "surrogate": gp,
    }
    return theta_best, bo_info
