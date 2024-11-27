import numpy as np
from loguru import logger
from .gp_lin_alg import *


class GPposterior:
    def __init__(self,
                 data_obj,
                 prior_obj,
                 marginal_density_obj,
                 likelihood_obj):

        self.marginal_density_obj = marginal_density_obj
        self.prior_obj = prior_obj
        self.likelihood_obj = likelihood_obj
        self.data_obj = data_obj
        self.kernel = self.prior_obj.kernel
        self.mean_function = self.prior_obj.mean_function
        self.d_kernel_dx = self.prior_obj.d_kernel_dx
        self.x_out = None

    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        x_data, y_data, KVinvY = \
            self.data_obj.x_data.copy(), self.data_obj.y_data.copy(), self.marginal_density_obj.KVinvY.copy()
        if hyperparameters is not None:
            K = self.prior_obj.compute_prior_covariance_matrix(self.data_obj.x_data, hyperparameters=hyperparameters)
            V = self.likelihood_obj.calculate_V(hyperparameters)
            m = self.prior_obj.compute_mean(self.data_obj.x_data, hyperparameters=hyperparameters)
            KVinvY = self.compute_new_KVinvY(K + V, m)
        else:
            hyperparameters = self.prior_obj.hyperparameters

        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()

        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.kernel(x_data, x_pred, hyperparameters)
        A = k.T @ KVinvY
        posterior_mean = self.mean_function(x_pred, hyperparameters) + A
        if isinstance(x_out, np.ndarray): posterior_mean_re = posterior_mean.reshape(len(x_orig), len(x_out), order='F')
        else: posterior_mean_re = posterior_mean

        return {"x": x_orig,
                "f(x)": posterior_mean_re,
                "f(x)_flat": posterior_mean,
                "x_pred": x_pred}

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None):
        x_data, y_data, KVinvY = \
            self.data_obj.x_data.copy(), self.data_obj.y_data.copy(), self.marginal_density_obj.KVinvY.copy()

        if hyperparameters is not None:
            K = self.prior_obj.compute_prior_covariance_matrix(self.data_obj.x_data, hyperparameters=hyperparameters)
            V = self.likelihood_obj.calculate_V(hyperparameters)
            m = self.prior_obj.compute_mean(self.data_obj.x_data, hyperparameters=hyperparameters)
            KVinvY = self.compute_new_KVinvY(K + V, m)
        else:
            hyperparameters = self.prior_obj.hyperparameters

        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        f = self.mean_function(x_pred, hyperparameters)
        eps = 1e-6
        if direction is not None:
            x1 = np.array(x_pred)
            x1[:, direction] = x1[:, direction] + eps
            mean_der = (self.mean_function(x1, hyperparameters) - f) / eps
            k_g = self.d_kernel_dx(x_pred, x_data, direction, hyperparameters)
            posterior_mean_grad = mean_der + (k_g @ KVinvY)
            if isinstance(x_out, np.ndarray):
                posterior_mean_grad = posterior_mean_grad.reshape(len(x_orig), len(x_out), order='F')
        else:
            posterior_mean_grad = np.zeros((len(x_pred), x_orig.shape[1]))
            for direction in range(len(x_orig[0])):
                x1 = np.array(x_pred)
                x1[:, direction] = x1[:, direction] + eps
                mean_der = (self.mean_function(x1, hyperparameters) - f) / eps
                k_g = self.d_kernel_dx(x_pred, x_data, direction, hyperparameters)
                posterior_mean_grad[:, direction] = mean_der + (k_g @ KVinvY)
            direction = "ALL"
            if isinstance(x_out, np.ndarray):
                posterior_mean_grad = posterior_mean_grad.reshape(len(x_orig), len(x_orig[0]), len(x_out), order='F')

        return {"x": x_orig,
                "direction": direction,
                "df/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        x_data = self.data_obj.x_data.copy()
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.kernel(x_data, x_pred, self.prior_obj.hyperparameters)
        kk = self.kernel(x_pred, x_pred, self.prior_obj.hyperparameters)

        if self.marginal_density_obj.KVlinalg.KVinv is not None:
            if variance_only:
                S = None
                v = np.diag(kk) - np.einsum('ij,jk,ki->i', k.T,
                                            self.marginal_density_obj.KVlinalg.KVinv, k, optimize=True)
            else:
                S = kk - (k.T @ self.marginal_density_obj.KVlinalg.KVinv @ k)
                v = np.array(np.diag(S))
        else:
            k_cov_prod = self.marginal_density_obj.KVlinalg.solve(k)
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

        return {"x": x_orig,
                "x_pred": x_pred,
                "v(x)": v_re,
                "S": S_re,
                "S_flat": S,
                "v_flat": v}

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        x_data = self.data_obj.x_data.copy()
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.kernel(x_data, x_pred, self.prior_obj.hyperparameters)
        k_covariance_prod = self.marginal_density_obj.KVlinalg.solve(k)
        if direction is not None:
            k_g = self.d_kernel_dx(x_pred, x_data, direction, self.prior_obj.hyperparameters).T
            x1 = np.array(x_pred)
            x2 = np.array(x_pred)
            eps = 1e-6
            x1[:, direction] = x1[:, direction] + eps
            kk_g = (self.kernel(x1, x1, self.prior_obj.hyperparameters) -
                    self.kernel(x2, x2, self.prior_obj.hyperparameters)) / eps
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
                k_g = self.d_kernel_dx(x_pred, x_data, direction, self.prior_obj.hyperparameters).T
                x1 = np.array(x_pred)
                x2 = np.array(x_pred)
                eps = 1e-6
                x1[:, direction] = x1[:, direction] + eps
                kk_g = (self.kernel(x1, x1, self.prior_obj.hyperparameters) -
                        self.kernel(x2, x2, self.prior_obj.hyperparameters)) / eps
                grad_v[:, direction] = np.diag(kk_g - (2.0 * k_g.T @ k_covariance_prod))

            if isinstance(x_out, np.ndarray):
                grad_v = grad_v.reshape(len(x_orig), len(x_orig[0]), len(x_out), order='F')

            return {"x": x_orig,
                    "dv/dx": grad_v}

    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out=None):
        x_data, K, prior_mean_vec = (self.data_obj.x_data.copy(),
                                     self.prior_obj.K.copy() + (np.identity(len(self.prior_obj.K)) * 1e-9),
                                     self.prior_obj.m.copy())
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.kernel(x_data, x_pred, self.prior_obj.hyperparameters)
        kk = self.kernel(x_pred, x_pred, self.prior_obj.hyperparameters)
        post_mean = self.mean_function(x_pred, self.prior_obj.hyperparameters)
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
        x_data, K, prior_mean_vec = (self.data_obj.x_data.copy(),
                                     self.prior_obj.K.copy() + (np.identity(len(self.prior_obj.K)) * 1e-9),
                                     self.prior_obj.m.copy())
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k_g = self.d_kernel_dx(x_pred, x_data, direction, self.prior_obj.hyperparameters).T
        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        eps = 1e-6
        x1[:, direction] = x1[:, direction] + eps
        x2[:, direction] = x2[:, direction] - eps
        kk_g = (self.kernel(x1, x1, self.prior_obj.hyperparameters) -
                self.kernel(x2, x2, self.prior_obj.hyperparameters)) / (2.0 * eps)

        mean_der = ((self.mean_function(x1, self.prior_obj.hyperparameters) -
                     self.mean_function(x2, self.prior_obj.hyperparameters)) /
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
    def kl_div_grad(self, mu1, dmu1dx, mu2, S1, dS1dx, S2):
        x1 = solve(S2, dS1dx)
        mu = np.subtract(mu2, mu1)
        x2 = solve(S2, mu)
        x3 = solve(S2, -dmu1dx)
        kld = 0.5 * (np.trace(x1) + ((x3.T @ mu) + (x2.T @ -dmu1dx)) - np.trace(calculate_inv(S1) @ dS1dx))
        return kld

    ###########################################################################
    def kl_div(self, mu1, mu2, S1, S2):
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
        gp_mean = res["f(x)_flat"]
        gp_cov = self.posterior_covariance(x_pred, x_out=x_out)["S_flat"]
        gp_cov = gp_cov + + np.identity(len(gp_cov)) * 1e-9
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
        x_data, K = self.data_obj.x_data.copy(), self.prior_obj.K.copy() + (np.identity(len(self.prior_obj.K)) * 1e-9)
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.kernel(x_data, x_pred, self.prior_obj.hyperparameters)
        kk = self.kernel(x_pred, x_pred, self.prior_obj.hyperparameters) + (np.identity(len(x_pred)) * 1e-9)
        if add_noise: v, kk = self.add_noise(x_pred, np.diag(kk), kk)

        joint_covariance = np.block([[K, k], [k.T, kk]])
        return {"x": x_orig,
                "mutual information": self.mutual_information(joint_covariance, kk, K)}

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None, add_noise=False):
        x_data, K = self.data_obj.x_data.copy(), self.prior_obj.K.copy() + (np.identity(len(self.prior_obj.K)) * 1e-9)
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.kernel(x_data, x_pred, self.prior_obj.hyperparameters)
        kk = self.kernel(x_pred, x_pred, self.prior_obj.hyperparameters) + (np.identity(len(x_pred)) * 1e-9)
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
        kk = self.kernel(x_pred_aux, x_pred_aux, self.prior_obj.hyperparameters) + (np.identity(len(x_pred_aux)) * 1e-9)
        post_cov = self.posterior_covariance(x_pred, x_out=x_out, add_noise=add_noise)["S_flat"]
        post_cov = post_cov + (np.identity(len(post_cov)) * 1e-9)
        hyperparameters = self.prior_obj.hyperparameters
        post_mean = self.posterior_mean(x_pred, x_out=x_out)["f(x)_flat"]
        prio_mean = self.mean_function(x_pred_aux, hyperparameters)
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

        gp_mean = self.posterior_mean(x_pred, x_out=x_out)["f(x)_flat"]
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
        if callable(self.likelihood_obj.noise_function):
            noise = self.likelihood_obj.noise_function(x_pred, self.prior_obj.hyperparameters)
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
        assert isinstance(x_pred, np.ndarray) or isinstance(x_pred, list)
        if isinstance(x_pred, np.ndarray):
            assert np.ndim(x_pred) == 2
            if isinstance(x_out, np.ndarray) or isinstance(x_out, list):
                assert x_pred.shape[1] == self.data_obj.index_set_dim - 1
            else:
                assert x_pred.shape[1] == self.data_obj.index_set_dim

        assert isinstance(x_out, np.ndarray) or x_out is None or isinstance(x_out, list)
        if isinstance(x_out, np.ndarray): assert np.ndim(x_out) == 1

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
