import numpy as np
import time
from .gp_lin_alg import *
from loguru import logger


class GPMarginalLikelihood:
    def __init__(self,
                 data,
                 prior,
                 likelihood,
                 trainer,
                 kv):

        self.data = data
        self.prior = prior
        self.likelihood = likelihood
        self.trainer = trainer
        self.kv = kv
        # Warm-start cache for iterative training solves; written by
        # compute_new_KVinvY / compute_new_KVlogdet_KVinvY when
        # args["sparse_krylov_warm_start"] is True.
        self._warm_start_KVinvY = None


    ##################################################################
    @property
    def args(self):
        return self.data.args

    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def K(self):
        return self.prior.K

    @property
    def m(self):
        return self.prior.m

    @property
    def V(self):
        return self.likelihood.V

    @property
    def ram_economy(self):
        return self.data.ram_economy

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    @property
    def compute_device(self):
        return self.data.compute_device

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters
    
    def _get_KVm(self):
        return self.K, self.V, self.m

    ##################################################################
    def compute_prior_covariance_matrix(self, x_data, hyperparameters):
        return self.prior.compute_prior_covariance_matrix(x_data, hyperparameters)

    def compute_mean(self, x_data, hyperparameters):
        return self.prior.compute_mean(x_data, hyperparameters)

    def dk_dh(self, x1, x2, hyperparameters, direction=None):
        return self.prior.dk_dh(x1, x2, hyperparameters, direction=direction)

    def dm_dh(self, x, hyperparameters):
        return self.prior.dm_dh(x, hyperparameters)

    def calculate_V(self, x, hyperparameters):
        return self.likelihood.calculate_V(x, hyperparameters)

    def calculate_V_grad(self, x, hyperparameters, direction=None):
        return self.likelihood.calculate_V_grad(x, hyperparameters, direction=direction)
    ###################################################################
    def addKV(self, K, V):
        return self.kv.addKV(K, V)

    def _warm_start_enabled(self):
        return bool(self.args.get("sparse_krylov_warm_start", False))

    def _iterative_initial_guess(self, target_shape):
        """Return a warm-start ``x0`` from the cached training KVinvY if shapes match.

        Source priority: ``self._warm_start_KVinvY`` (previous training iteration's
        solution) when available, else ``self.kv.KVinvY`` (the committed solution).
        Returns ``None`` when warm-start is disabled or no compatible cache exists.
        """
        if not self._warm_start_enabled():
            return None
        for candidate in (self._warm_start_KVinvY, self.kv.KVinvY):
            if candidate is None:
                continue
            guess = np.asarray(candidate)
            if guess.ndim == 1:
                guess = guess.reshape(-1, 1)
            if guess.shape[0] != target_shape[0]:
                continue
            if guess.shape[1] == target_shape[1]:
                return guess
            if guess.shape[1] == 1 and target_shape[1] > 1:
                return np.repeat(guess, target_shape[1], axis=1)
        return None

    def _update_warm_start(self, KVinvY):
        if self._warm_start_enabled():
            self._warm_start_KVinvY = np.asarray(KVinvY, copy=True)

    def compute_new_KVinvY(self, KV, m, K=None, V=None):
        y_mean_shape = (self.y_data.shape[0], self.y_data.shape[1])
        x0 = self._iterative_initial_guess(y_mean_shape)
        KVinvY = self.kv.compute_new_KVinvY(KV, m, x0=x0, K=K, V=V)
        self._update_warm_start(KVinvY)
        return KVinvY

    def compute_new_KVlogdet_KVinvY(self, K, V, m):
        y_mean_shape = (self.y_data.shape[0], self.y_data.shape[1])
        x0 = self._iterative_initial_guess(y_mean_shape)
        KVinvY, KVlogdet = self.kv.compute_new_KVlogdet_KVinvY(K, V, m, x0=x0)
        self._update_warm_start(KVinvY)
        return KVinvY, KVlogdet

    ##################################################################################
    def log_likelihood(self, hyperparameters=None):
        """
        Function that computes the marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Returns
        -------
        log marginal likelihood of the data : float
        """
        logger.debug("log marginal likelihood is being evaluated")
        if hyperparameters is None:
            K, V, m = self._get_KVm()
            KVinvY = self.kv.KVinvY
            KVlogdet = self.kv.logdet_KV
        else:
            st = time.time()
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            logger.debug("   Prior covariance matrix computed after {} seconds.", time.time() - st)
            V = self.calculate_V(self.x_data, hyperparameters)
            logger.debug("   V computed after {} seconds.", time.time() - st)
            m = self.compute_mean(self.x_data, hyperparameters)
            logger.debug("   Prior mean computed after {} seconds.", time.time() - st)
            try:
                KVinvY, KVlogdet = self.compute_new_KVlogdet_KVinvY(K, V, m)
            except Exception as e:
                raise Exception(
                    f"Linear algebra failed for hyperparameters {hyperparameters}: {e}") from e
            logger.debug("   KVinvY and logdet computed after {} seconds.", time.time() - st)

        # y_data is 2-d: one column per right-hand side, each a problem of its own sharing K,
        # whose likelihoods are averaged -- or, in Kronecker mode, one (n_x, n_out) tensor
        problems = 1 if self.data.kronecker else self.y_data.shape[1]
        y_mean = self.y_data - self.prior.mean_columns(m)
        assert y_mean.shape == KVinvY.shape, "(y-m).shape != KVinvY.shape in log_likelihood"+str(y_mean.shape)+" ,"+str(KVinvY.shape)
        n = y_mean.size // problems
        L = -0.5 * (np.sum(y_mean * KVinvY) / problems + KVlogdet + n * np.log(2.0 * np.pi))
        return L

    ##################################################################################
    def log_likelihood_variance(self):
        """Variance of the most recent (log) marginal-likelihood evaluation.

        In the sparse modes the log-determinant is estimated by stochastic Lanczos
        quadrature, so the marginal likelihood is a random variable rather than a
        number: evaluating it twice at the same hyperparameters gives two different
        answers. This reports how much it wobbles.

        ``L = -0.5 * (l1 + logdet + n log 2pi)``, so the log-determinant enters with a
        coefficient of -0.5 and its variance is scaled by 0.25. ``neg_log_likelihood``
        is a sign flip and has the same variance. The truncated-CG solve contributes
        additional noise through ``l1`` that is not captured here, so this is a lower
        bound on the total.

        Returns
        -------
        variance : float or None
            None for the exact modes, where the likelihood is deterministic.
        """
        v = getattr(self.kv, "last_logdet_variance", None)
        if v is None:
            return None
        return 0.25 * float(v)

    ##################################################################################
    def neg_log_likelihood(self, hyperparameters=None):
        """
        Function that computes the negative marginal log-likelihood

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N)
            If not provided, the covariance will not be recomputed.

        Returns
        -------
        negative log marginal likelihood of the data : float
        """
        return -self.log_likelihood(hyperparameters=hyperparameters)

    ##################################################################################
    def neg_log_likelihood_gradient(self, hyperparameters=None, component=0):
        """
        Function that computes the gradient of the marginal log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray, optional
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.
        component : int, optional
            In case many GPs are computed in parallel, this specifies which one is considered.

        Returns
        -------
        Gradient of the negative log marginal likelihood : np.ndarray
        """
        if self.data.kronecker: return self._kronecker_neg_log_likelihood_gradient(hyperparameters)
        if self.gp2Scale: raise Exception("Can't compute neg_log_likelihood_gradient for gp2Scale")
        dK_dH = None
        # "low_rank": every solve below goes through the Woodbury factorization, O(N^2 r) per
        # hyperparameter instead of the O(N^3) dense solve, and K + V is never formed
        low_rank = self.kv.mode == "low_rank"

        if hyperparameters is None:
            KVinvY = self.kv.KVinvY
            K = self.K
            V = self.V
            if low_rank: factor = self.kv.low_rank_factor
            else: KV = self.addKV(K, V)
            hyperparameters = self.hyperparameters
        else:
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            V = self.calculate_V(self.x_data, hyperparameters)
            m = self.compute_mean(self.x_data, hyperparameters)
            if low_rank:
                factor = self.kv.low_rank_factorization(K, V)
                KVinvY = factor.solve(self.y_data - self.prior.mean_columns(m)).reshape(self.y_data.shape)
            else:
                KV = self.addKV(K, V)
                KVinvY = self.compute_new_KVinvY(KV, m)

        b = KVinvY[:, component]
        a = None
        if self.ram_economy is False:
            try:
                noise_der = self.calculate_V_grad(self.x_data, hyperparameters)
                assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 3, \
                    "noise gradient must be 2-d (diagonal per hyperparameter) or 3-d (full matrix per hyperparameter)"
                if np.ndim(noise_der) == 2:
                    noise_der_V = np.zeros((len(hyperparameters), len(self.x_data), len(self.x_data)))
                    for i in range(len(hyperparameters)): np.fill_diagonal(noise_der_V[i], noise_der[i])
                else: noise_der_V = noise_der
                dK_dH = self.dk_dh(self.x_data, self.x_data, hyperparameters) + noise_der_V
            except Exception as e:
                raise Exception(
                    "The gradient evaluation dK/dh + dNoise/dh was not successful. "
                    "That normally means the combination of gradient_per_direction and definition "
                    "of the gradient function is wrong.") from e
            if low_rank:
                a = np.array([factor.solve(dK_dH[i]) for i in range(len(hyperparameters))])
            else:
                KV = np.array([KV, ] * len(hyperparameters))
                a = solve(KV, dK_dH, compute_device=self.compute_device)
        bbT = np.outer(b, b.T)
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.dm_dh(self.x_data, hyperparameters)

        for i in range(len(hyperparameters)):
            dL_dHm[i] = -self.prior.mean_columns(dm_dh[i])[:, component] @ b
            if self.ram_economy is False:
                matr = a[i]
            else:
                try:
                    noise_der = self.calculate_V_grad(self.x_data, hyperparameters, direction=i)
                    assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 1, \
                        "noise gradient in gradient_per_direction mode must be 1-d (diagonal) or 2-d (full matrix)"
                    if np.ndim(noise_der) == 1:
                        dK_dH = self.dk_dh(
                            self.x_data, self.x_data, hyperparameters, direction=i) + np.diag(noise_der)
                    else:
                        dK_dH = self.dk_dh(
                            self.x_data, self.x_data, hyperparameters, direction=i) + noise_der
                except Exception as e:
                    raise Exception(
                        "The gradient evaluation dK/dh + dNoise/dh was not successful. "
                        "That normally means the combination of gradient_per_direction and definition of "
                        "the gradient function is wrong.") from e
                matr = factor.solve(dK_dH) if low_rank else solve(KV, dK_dH, compute_device=self.compute_device)
            if dL_dHm[i] == 0.0:
                if self.ram_economy is False:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH[i])
                else:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH)
                dL_dH[i] = - 0.5 * (mtrace - np.trace(matr))
            else:
                dL_dH[i] = 0.0
        return dL_dH + dL_dHm

    def _kronecker_neg_log_likelihood_gradient(self, hyperparameters):
        """The gradient for fvGP(kronecker=True), without ever forming K or dK/dh.

        By the product rule dK/dh is a sum of Kronecker products, one per factor that depends
        on h, with that factor replaced by its derivative. Its data-fit term alpha^T dK alpha
        is a mode product, and its trace term tr((K+V)^-1 dK) needs only the diagonal of the
        derivative in the whitened eigenbasis. The noise term uses diag((K+V)^-1).
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
            factors, factor, KVinvY = self.K, self.kv.kronecker_factor, self.kv.KVinvY
        else:
            factors = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            V = self.calculate_V(self.x_data, hyperparameters)
            m = self.compute_mean(self.x_data, hyperparameters)
            factor = self.kv.kronecker_factorization(factors, V)
            KVinvY = factor.solve(self.y_data - self.prior.mean_columns(m))
        n_x = len(KVinvY)
        alpha = KVinvY.reshape(factor.sizes)
        dm_dh = self.dm_dh(self.x_data, hyperparameters)
        if not self.ram_economy:
            noise_der = self.calculate_V_grad(self.x_data, hyperparameters)
            factor_der = self.prior.kronecker_dk_dh(self.x_data, hyperparameters)
        diag_inverse = None
        gradient = np.zeros(len(hyperparameters))
        for i in range(len(hyperparameters)):
            gradient[i] = -np.sum(self.prior.mean_columns(dm_dh[i]) * KVinvY)
            # as in the dense gradient: mean and kernel/noise hyperparameters are disjoint
            if gradient[i] != 0.0: continue
            fit, trace = 0.0, 0.0
            derivatives = self.prior.kronecker_dk_dh(self.x_data, hyperparameters, i) if self.ram_economy \
                else factor_der[i]
            for f, dF in enumerate(derivatives):
                if dF is None or not np.any(dF): continue
                matrices = list(factors)
                matrices[f] = dF
                fit += np.sum(alpha * kronecker_mode_product(alpha, matrices))
                trace += factor.trace_with(f, dF)
            dV = self.calculate_V_grad(self.x_data, hyperparameters, direction=i) if self.ram_economy \
                else noise_der[i]
            dV = np.reshape(dV, (n_x, 1))   # one derivative per input, the same for every output
            if np.any(dV):
                if diag_inverse is None: diag_inverse = factor.diag_inverse().reshape(n_x, -1)
                fit += np.sum(dV * KVinvY ** 2)
                trace += np.sum(dV * diag_inverse)
            gradient[i] = -0.5 * (fit - trace)
        return gradient

    ##################################################################################
    def neg_log_likelihood_hessian(self, hyperparameters=None):
        """
        Function that computes the Hessian of the marginal log-likelihood.
        It does so by a first-order approximation of the exact gradient.

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (N).
            If not provided, the covariance will not be recomputed.

        Returns
        -------
        Hessian of the negative log marginal likelihood : np.ndarray
        """
        ##implemented as first-order approximation
        len_hyperparameters = len(hyperparameters)
        d2L_dmdh = np.zeros((len_hyperparameters, len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.neg_log_likelihood_gradient(hyperparameters=hyperparameters)
        for i in range(len_hyperparameters):
            hps_temp = np.array(hyperparameters)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i, i:] = ((self.neg_log_likelihood_gradient(hyperparameters=hps_temp) - grad_at_hps) / epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))

    def test_log_likelihood_gradient(self, hyperparameters, epsilon=1e-6):
        """
        Compare the finite-difference and analytical gradients of the log-likelihood.

        Parameters
        ----------
        hyperparameters : np.ndarray
            Vector of hyperparameters of shape (N).
        epsilon : float, optional
            Step size for the finite-difference approximation. Default is 1e-6.

        Returns
        -------
        fd_gradient : np.ndarray
            Finite-difference gradient of shape (N,).
        analytical_gradient : np.ndarray
            Analytical gradient of shape (N,).
        """
        thps = np.array(hyperparameters)
        grad = np.empty((len(thps)))
        eps = epsilon
        for i in range(len(thps)):
            thps_aux = np.array(thps)
            thps_aux[i] = thps_aux[i] + eps
            grad[i] = (self.log_likelihood(hyperparameters=thps_aux) - self.log_likelihood(hyperparameters=thps)) / eps
        analytical = -self.neg_log_likelihood_gradient(hyperparameters=thps)
        return grad, analytical

    #########################################################################################################
    ##################LVM####################################################################################
    #########################################################################################################
    #########################################################################################################
    def __getstate__(self):
        state = dict(
            data=self.data,
            prior=self.prior,
            likelihood=self.likelihood,
            trainer=self.trainer,
            kv=self.kv,
            _warm_start_KVinvY=self._warm_start_KVinvY,
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_warm_start_KVinvY" not in self.__dict__:
            self._warm_start_KVinvY = None


