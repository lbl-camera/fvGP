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

    def compute_new_KVinvY(self, KV, m):
        return self.kv.compute_new_KVinvY(KV, m)


    def compute_new_KVlogdet_KVinvY(self, K, V, m):
        return self.kv.compute_new_KVlogdet_KVinvY(K, V, m)

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
            KVlogdet = self.kv.logdet()
        else:
            st = time.time()
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            logger.debug("   Prior covariance matrix computed after {} seconds.", time.time() - st)
            V = self.calculate_V(self.x_data, hyperparameters)
            logger.debug("   V computed after {} seconds.", time.time() - st)
            m = self.compute_mean(self.x_data, hyperparameters)
            logger.debug("   Prior mean computed after {} seconds.", time.time() - st)
            KVinvY, KVlogdet = self.compute_new_KVlogdet_KVinvY(K, V, m)
            logger.debug("   KVinvY and logdet computed after {} seconds.", time.time() - st)

        n = len(self.y_data)
        y_mean = self.y_data - m[:, None]
        assert np.ndim(y_mean) == 2, "y minus mean must be 2-d"
        assert y_mean.shape == KVinvY.shape, "(y-m).shape != KVinvY.shape in log_likelihood"+str(y_mean.shape)+" ,"+str(KVinvY.shape)
        if np.ndim(y_mean) == 2: l1 = np.sum(y_mean * KVinvY)/y_mean.shape[1]
        else: l1 = np.sum(y_mean * KVinvY)
        L = -0.5 * (l1 + KVlogdet + n * np.log(2.0 * np.pi))
        return L

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
        if self.gp2Scale: raise Exception("Can't compute neg_log_likelihood_gradient for gp2Scale")
        dK_dH = None

        if hyperparameters is None:
            KVinvY = self.kv.KVinvY
            K = self.K
            V = self.V
            KV = self.addKV(K, V)
            hyperparameters = self.hyperparameters
        else:
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            V = self.calculate_V(self.x_data, hyperparameters)
            m = self.compute_mean(self.x_data, hyperparameters)
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
                    "That normally means the combination of ram_economy and definition "
                    "of the gradient function is wrong.") from e
            KV = np.array([KV, ] * len(hyperparameters))
            a = solve(KV, dK_dH, compute_device=self.compute_device)
        bbT = np.outer(b, b.T)
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.dm_dh(self.x_data, hyperparameters)

        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if self.ram_economy is False:
                matr = a[i]
            else:
                try:
                    noise_der = self.calculate_V_grad(self.x_data, hyperparameters, direction=i)
                    assert np.ndim(noise_der) == 2 or np.ndim(noise_der) == 1, \
                        "noise gradient in ram_economy mode must be 1-d (diagonal) or 2-d (full matrix)"
                    if np.ndim(noise_der) == 1:
                        dK_dH = self.dk_dh(
                            self.x_data, self.x_data, hyperparameters, direction=i) + np.diag(noise_der)
                    else:
                        dK_dH = self.dk_dh(
                            self.x_data, self.x_data, hyperparameters, direction=i) + noise_der
                except Exception as e:
                    raise Exception(
                        "The gradient evaluation dK/dh + dNoise/dh was not successful. "
                        "That normally means the combination of ram_economy and definition of "
                        "the gradient function is wrong.") from e
                matr = solve(KV, dK_dH, compute_device=self.compute_device)
            if dL_dHm[i] == 0.0:
                if self.ram_economy is False:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH[i])
                else:
                    mtrace = np.einsum('ij,ji->', bbT, dK_dH)
                dL_dH[i] = - 0.5 * (mtrace - np.trace(matr))
            else:
                dL_dH[i] = 0.0
        return dL_dH + dL_dHm

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
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


