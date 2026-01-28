import numpy as np
from loguru import logger
from .gp_lin_alg import *


class GPposterior:
    def __init__(self,
                 data,
                 prior,
                 trainer,
                 marginal_density,
                 likelihood):

        self.marginal_density = marginal_density
        self.prior = prior
        self.likelihood = likelihood
        self.data = data
        self.trainer = trainer
        self.noise_function_available = callable(self.likelihood.noise_function)

    def compute_covariances(self, x1, x2, hps):
        return self.prior.compute_covariances(x1, x2, hps)

    def compute_mean(self, x, hps):
        return self.prior.compute_mean(x, hps)

    def d_kernel_dx(self, x_pred, x_data, direction, hyperparameters):
        return self.prior.d_kernel_dx(x_pred, x_data, direction, hyperparameters)

    def KVsolve(self, v):
        return self.marginal_density.KVlinalg.solve(v)

    def compute_new_KVinvY(self, KV, m):
        return self.marginal_density.compute_new_KVinvY(KV, m)

    def compute_prior_covariance_matrix(self, x_data, hyperparameters):
        return self.prior.compute_prior_covariance_matrix(x_data, hyperparameters)

    def calculate_V(self, x_data, hyperparameters):
        return self.likelihood.calculate_V(x_data, hyperparameters)

    def noise_function(self, x_pred, hyperparameters):
        return self.likelihood.noise_function(x_pred, hyperparameters)

    def addKV(self, K, V):
        return self.marginal_density.addKV(K, V)

    #####################################################
    @property
    def args(self):
        return self.data.args

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters

    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def x_out(self):
        return self.data.x_out

    @property
    def KVinvY(self):
        return self.marginal_density.KVinvY

    @property
    def KVinv(self):
        return self.marginal_density.KVlinalg.KVinv

    @property
    def input_set_dim(self):
        return self.data.input_set_dim

    @property
    def K(self):
        return self.prior.K

    @property
    def m(self):
        return self.prior.m
    ##########################################################

    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        x_data, KVinvY = self.x_data.copy(), self.KVinvY.copy()
        if hyperparameters is not None:
            K = self.compute_prior_covariance_matrix(x_data, hyperparameters)
            V = self.calculate_V(x_data, hyperparameters)
            m = self.compute_mean(x_data, hyperparameters)
            if np.ndim(V) == 1: V = np.diag(V)
            KVinvY = self.compute_new_KVinvY(self.addKV(K, V), m)
        else:
            hyperparameters = self.hyperparameters

        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, hyperparameters)
        A = k.T @ KVinvY
        prior_mean = self.compute_mean(x_pred, hyperparameters)
        #assert A.shape == prior_mean.shape, str(KVinvY.shape)+ ", " + str(A.shape) + ", " + str(x_pred.shape)
        posterior_mean = prior_mean[:, None] + A
        if isinstance(x_out, np.ndarray): posterior_mean_re = posterior_mean.reshape(len(x_orig), len(x_out), order='F')
        else: posterior_mean_re = posterior_mean

        if KVinvY.shape[1] == 1 and not isinstance(x_out, np.ndarray):
            return {"x": x_orig,
                    "m(x)": np.squeeze(posterior_mean_re),
                    "m(x)_flat": np.squeeze(posterior_mean),
                    "x_pred": x_pred}
        elif KVinvY.shape[1] == 1 and isinstance(x_out, np.ndarray):
            return {"x": x_orig,
                    "m(x)": posterior_mean_re,
                    "m(x)_flat": np.squeeze(posterior_mean),
                    "x_pred": x_pred}
        elif KVinvY.shape[1] > 1 and isinstance(x_out, np.ndarray):
            raise Exception("KVinvY.shape[1] > 1 and isinstance(x_out, np.ndarray)=True")
        else:
            return {"x": x_orig,
                    "m(x)": posterior_mean_re,
                    "m(x)_flat": posterior_mean,
                    "x_pred": x_pred}

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None, component=0):
        x_data, KVinvY = self.x_data.copy(), self.KVinvY.copy()[:, component]

        if hyperparameters is not None:
            K = self.compute_prior_covariance_matrix(x_data, hyperparameters)
            V = self.calculate_V(x_data, hyperparameters)
            m = self.compute_mean(x_data, hyperparameters)
            if np.ndim(V) == 1: V = np.diag(V)
            KVinvY = self.compute_new_KVinvY(self.addKV(K, V), m)[:, component]
            assert np.ndim(KVinvY) == 1
        else:
            hyperparameters = self.hyperparameters

        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        f = self.compute_mean(x_pred, hyperparameters)
        eps = 1e-6
        if direction is not None:
            x1 = np.array(x_pred)
            x1[:, direction] = x1[:, direction] + eps
            mean_der = (self.compute_mean(x1, hyperparameters) - f) / eps
            k_g = self.d_kernel_dx(x_pred, x_data, direction, hyperparameters)
            posterior_mean_grad = mean_der + (k_g @ KVinvY)
            if isinstance(x_out, np.ndarray):
                posterior_mean_grad = posterior_mean_grad.reshape(len(x_orig), len(x_out), order='F')
        else:
            posterior_mean_grad = np.zeros((len(x_pred), x_orig.shape[1]))
            for direction in range(len(x_orig[0])):
                x1 = np.array(x_pred)
                x1[:, direction] = x1[:, direction] + eps
                mean_der = (self.compute_mean(x1, hyperparameters) - f) / eps
                k_g = self.d_kernel_dx(x_pred, x_data, direction, hyperparameters)
                posterior_mean_grad[:, direction] = mean_der + (k_g @ KVinvY)
            direction = "ALL"
            if isinstance(x_out, np.ndarray):
                posterior_mean_grad = posterior_mean_grad.reshape(len(x_orig), len(x_orig[0]), len(x_out), order='F')

        return {"x": x_orig,
                "direction": direction,
                "dm/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        x_data = self.x_data.copy()
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters)

        if self.KVinv is not None:
            if variance_only and self.y_data.shape[1] == 1:
                S = None
                v = np.diag(kk) - np.einsum('ij,jk,ki->i', k.T,
                                            self.KVinv, k, optimize=True)
            else:
                S = kk - (k.T @ self.KVsolve(k))
                v = np.array(np.diag(S))
        else:
            k_cov_prod = self.KVsolve(k)
            S = kk - (k_cov_prod.T @ k)
            v = np.array(np.diag(S))
        if np.any(v < -0.0001):
            warnings.warn(
                "Negative variances encountered. That normally means that the model is unstable. "
                "Rethink the kernel definition, add more noise to the data, "
                "or double check the hyperparameter optimization bounds. This will not "
                "terminate the algorithm, but expect anomalies.")
            logger.debug("Negative variances encountered.")
            v[v < 0.0] = 0.0
            if not variance_only: np.fill_diagonal(S, v)

        if add_noise: v, S = self.add_noise(x_pred, v, S)

        if isinstance(x_out, np.ndarray):
            v_re = v.reshape(len(x_orig), len(x_out), order='F')
            if S is not None: S_re = S.reshape(len(x_orig), len(x_orig), len(x_out), len(x_out), order='F')
            else: S_re = None
        else:
            v_re = v
            S_re = S
            if self.y_data.shape[1] > 1:
                v = np.tile(v[:, None], (1, self.y_data.shape[1]))
                v_re = np.tile(v_re[:, None], (1, self.y_data.shape[1]))

        return {"x": x_orig,
                "x_pred": x_pred,
                "v(x)": v_re,
                "S": S_re,
                "S_flat": S,
                "v_flat": v}

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        x_data = self.x_data.copy()
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        k_covariance_prod = self.KVsolve(k)
        if direction is not None:
            k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
            x1 = np.array(x_pred)
            x2 = np.array(x_pred)
            eps = 1e-6
            x1[:, direction] = x1[:, direction] + eps
            kk_g = (self.compute_covariances(x1, x1, self.hyperparameters) -
                    self.compute_covariances(x2, x2, self.hyperparameters)) / eps
            dSdx = kk_g - (2.0 * k_g.T @ k_covariance_prod)
            a = np.diag(dSdx)
            if isinstance(x_out, np.ndarray):
                a = a.reshape(len(x_orig), len(x_out), order='F')
                dSdx = dSdx.reshape(len(x_orig), len(x_orig), len(x_out), len(x_out), order='F')
            return {"x": x_orig,
                    "dv/dx": a,
                    "dS/dx": dSdx}
        else:
            grad_v = np.zeros((len(x_pred), len(x_orig[0])))
            for direction in range(len(x_orig[0])):
                k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
                x1 = np.array(x_pred)
                x2 = np.array(x_pred)
                eps = 1e-6
                x1[:, direction] = x1[:, direction] + eps
                kk_g = (self.compute_covariances(x1, x1, self.hyperparameters) -
                        self.compute_covariances(x2, x2, self.hyperparameters)) / eps
                grad_v[:, direction] = np.diag(kk_g - (2.0 * k_g.T @ k_covariance_prod))

            if isinstance(x_out, np.ndarray):
                grad_v = grad_v.reshape(len(x_orig), len(x_orig[0]), len(x_out), order='F')

            return {"x": x_orig,
                    "dv/dx": grad_v}

    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out=None):
        x_data, K, prior_mean_vec = (self.x_data.copy(),
                                     self.K.copy() + (np.identity(len(self.K)) * 1e-9),
                                     self.m.copy())
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters)
        post_mean = self.compute_mean(x_pred, self.hyperparameters)
        joint_gp_prior_mean = np.append(prior_mean_vec, post_mean)
        joint_gp_prior_cov = np.block([[K, k], [k.T, kk]])

        return {"x": x_pred,
                "K": K + np.identity(len(K)) * 1e-9,
                "k": k,
                "kappa": kk,
                "prior mean": joint_gp_prior_mean,
                "S": joint_gp_prior_cov + np.identity(len(joint_gp_prior_cov)) * 1e-9}

    ###########################################################################
    def joint_gp_prior_grad(self, x_pred, direction, x_out=None):
        x_data, K, prior_mean_vec = (self.x_data.copy(),
                                     self.K.copy() + (np.identity(len(self.K)) * 1e-9),
                                     self.m.copy())
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        eps = 1e-6
        x1[:, direction] = x1[:, direction] + eps
        x2[:, direction] = x2[:, direction] - eps
        kk_g = (self.compute_covariances(x1, x1, self.hyperparameters) -
                self.compute_covariances(x2, x2, self.hyperparameters)) / (2.0 * eps)

        mean_der = ((self.compute_mean(x1, self.hyperparameters) -
                     self.compute_mean(x2, self.hyperparameters)) /
                    (2.0 * eps))
        full_gp_prior_mean_grad = np.append(np.zeros(prior_mean_vec.shape), mean_der)
        prior_cov_grad = np.zeros(K.shape)
        return {"x": x_pred,
                "K": K,
                "dk/dx": k_g,
                "d kappa/dx": kk_g,
                "d prior mean/x": full_gp_prior_mean_grad,
                "dS/dx": np.block([[prior_cov_grad, k_g], [k_g.T, kk_g]])}

    ###########################################################################
    @staticmethod
    def entropy(S):
        dim = len(S[0])
        ldet = calculate_logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * ldet)

    ###########################################################################
    def gp_entropy(self, x_pred, x_out=None):
        """
        Function to compute the entropy of the gp prior probability distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Entropy : float
        """

        priors = self.joint_gp_prior(x_pred, x_out=x_out)
        S = priors["S"]
        dim = len(S[0])
        ldet = calculate_logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * ldet)

    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out=None):
        priors1 = self.joint_gp_prior(x_pred, x_out=x_out)
        priors2 = self.joint_gp_prior_grad(x_pred, direction, x_out=x_out)
        S1 = priors1["S"]
        S2 = priors2["dS/dx"]
        return 0.5 * np.trace(calculate_inv(S1) @ S2)

    ###########################################################################
    @staticmethod
    def kl_div(mu1, mu2, S1, S2):
        logdet1 = calculate_logdet(S1)
        logdet2 = calculate_logdet(S2)
        x1 = solve(S2, S1)
        mu = np.subtract(mu2, mu1)
        x2 = solve(S2, mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu)[0] - float(dim) + (logdet2 - logdet1))
        if kld < -1e-4:
            warnings.warn("Negative KL divergence encountered. That happens when "
                          "one of the covariance matrices is close to positive semi definite "
                          "and therefore the logdet() calculation becomes unstable. "
                          "Returning abs(KLD)")
            logger.debug("Negative KL divergence encountered")
        return abs(kld)

    ###########################################################################
    def gp_kl_div(self, x_pred, comp_mean, comp_cov, x_out=None):
        if x_out is None: x_out = self.x_out

        res = self.posterior_mean(x_pred, x_out=x_out)
        gp_mean = res["m(x)_flat"]
        gp_cov = self.posterior_covariance(x_pred, x_out=x_out)["S_flat"]
        gp_cov = gp_cov + np.identity(len(gp_cov)) * 1e-9
        comp_cov = comp_cov + np.identity(len(comp_cov)) * 1e-9
        return {"x": x_pred,
                "gp posterior mean": gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}

    ###########################################################################
    def mutual_information(self, joint, m1, m2):
        return self.entropy(m1) + self.entropy(m2) - self.entropy(joint)

    ###########################################################################
    def gp_mutual_information(self, x_pred, x_out=None, add_noise=False):
        x_data, K = self.x_data.copy(), self.K.copy() + (np.identity(len(self.K)) * 1e-9)
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters) + (np.identity(len(x_pred)) * 1e-9)
        if add_noise: v, kk = self.add_noise(x_pred, np.diag(kk), kk)

        joint_covariance = np.block([[K, k], [k.T, kk]])
        return {"x": x_orig,
                "mutual information": self.mutual_information(joint_covariance, kk, K)}

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None, add_noise=False):
        x_data, K = self.x_data.copy(), self.K.copy() + (np.identity(len(self.K)) * 1e-9)
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters) + (np.identity(len(x_pred)) * 1e-9)
        if add_noise: v, kk = self.add_noise(x_pred, np.diag(kk), kk)
        joint_covariance = np.block([[K, k], [k.T, kk]])

        prod_covariance = np.block([[K, k * 0.], [k.T * 0., kk * np.identity(len(kk))]])

        return {"x": x_orig,
                "total correlation": self.kl_div(np.zeros((len(joint_covariance))), np.zeros((len(joint_covariance))),
                                                 joint_covariance, prod_covariance)}

    ###########################################################################
    def gp_relative_information_entropy(self, x_pred, x_out=None, add_noise=False):
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred_aux = self.cartesian_product(x_pred, x_out)
        else: x_pred_aux = x_pred
        kk = self.compute_covariances(x_pred_aux, x_pred_aux, self.hyperparameters) + (np.identity(len(x_pred_aux)) * 1e-9)
        post_cov = self.posterior_covariance(x_pred, x_out=x_out, add_noise=add_noise)["S_flat"]
        post_cov = post_cov + (np.identity(len(post_cov)) * 1e-9)
        hyperparameters = self.hyperparameters
        post_mean = self.posterior_mean(x_pred, x_out=x_out)["m(x)_flat"]
        prio_mean = self.compute_mean(x_pred_aux, hyperparameters)
        return {"x": x_orig,
                "RIE": self.kl_div(prio_mean, post_mean, kk, post_cov)}

    ###########################################################################
    def gp_relative_information_entropy_set(self, x_pred, x_out=None, add_noise=False):
        if x_out is None: x_out = self.x_out
        #self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        #if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)
        RIE = np.zeros((len(x_pred)))
        for i in range(len(x_pred)):
            RIE[i] = self.gp_relative_information_entropy(
                x_pred[i].reshape(1, len(x_pred[i])), x_out=x_out, add_noise=add_noise)["RIE"]

        return {"x": x_orig,
                "RIE": RIE}

    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out=None):
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        #if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        gp_mean = self.posterior_mean(x_pred, x_out=x_out)["m(x)_flat"]
        gp_cov = self.posterior_covariance(x_pred, x_out=x_out, add_noise=True)["S_flat"]
        gp_cov_inv = calculate_inv(gp_cov)
        comp_cov_inv = calculate_inv(comp_cov)
        cov = calculate_inv(gp_cov_inv + comp_cov_inv)
        mu = cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        logdet1 = calculate_logdet(cov)
        logdet2 = calculate_logdet(gp_cov)
        logdet3 = calculate_logdet(comp_cov)
        dim = len(mu)
        C = 0.5 * (((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T
                    @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean))
                   - (gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0 * np.pi) ** (dim / 2.0)) + (0.5 * (logdet2 + logdet3)))
        return {"mu": mu,
                "covariance": cov,
                "probability":
                    np.exp(ln_p)
                }

    def add_noise(self, x_pred, v, S):
        if self.noise_function_available:
            noise = self.noise_function(x_pred, self.hyperparameters)
            assert isinstance(noise, np.ndarray)
            try:
                if np.ndim(noise) == 1:
                    v = v + noise
                    if S is not None: S = S + np.diag(noise)
                elif np.ndim(noise) == 2:
                    v = v + np.diag(noise)
                    if S is not None: S = S + noise
                else:
                    raise Exception("Wrong noise format")
            except:
                warnings.warn("Noise could not be added, you did not provide a noise callable at initialization")
        return v, S

    ###########################################################################
    @staticmethod
    def _int_gauss(S):
        return ((2.0 * np.pi) ** (len(S) / 2.0)) * np.sqrt(np.linalg.det(S))

    def _perform_input_checks(self, x_pred, x_out):
        assert isinstance(x_pred, np.ndarray) or isinstance(x_pred, list), "wrong format in x_pred"
        if isinstance(x_pred, np.ndarray):
            assert np.ndim(x_pred) == 2, "wrong dim in x_pred, has to be 2-d"
            assert x_pred.shape[1] == self.input_set_dim, "wrong number of columns in x_pred"

        assert isinstance(x_out, np.ndarray) or x_out is None or isinstance(x_out, list), "wrong format in x_out"
        if isinstance(x_out, np.ndarray): assert np.ndim(x_out) == 1, "wrong dim in x_out, has to be 1-d"

    @staticmethod
    def cartesian_product(x, y):
        """
        Input x,y have to be 2d numpy arrays
        The return is the cartesian product of the two sets
        """
        assert isinstance(y, np.ndarray)
        assert np.ndim(y) == 1

        res = []
        if isinstance(x, list):
            for j in range(len(y)):
                for i in range(len(x)):
                    res.append([x[i], y[j]])
            return res
        elif isinstance(x, np.ndarray):
            for j in range(len(y)):
                for i in range(len(x)):
                    res.append(np.append(x[i], y[j]))
            return np.asarray(res)
        else:
            raise Exception("Cartesian product out of options")

    def __getstate__(self):
        state = dict(
            marginal_density=self.marginal_density,
            prior=self.prior,
            likelihood=self.likelihood,
            data=self.data,
            trainer=self.trainer,
            noise_function_available=self.noise_function_available
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
