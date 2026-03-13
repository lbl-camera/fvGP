from fvgp.gp import GP
from hgdl.hgdl import HGDL
import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.special import logsumexp

def integral(f, domain):
    Int = np.sum(f * np.gradient(domain))
    return Int


def gaussian(mean, std, x):
    g = (1. / (np.sqrt(2. * np.pi) * std)) * np.exp(-np.power(x - mean, 2.) / (2. * np.power(std, 2.)))
    if np.all(g < 1e-6): g[:] = 1e-6
    inte = integral(g, x)
    gn = g / inte
    if np.any(np.isnan(gn)): print("NaN in Gaussian normalized")
    return gn


def gaussian_mixture_pdf(x, means, sigmas):
    """
    Equal-weight Gaussian mixture PDF.
    """
    N = len(means)
    pdf = np.zeros_like(x, dtype=float)
    for m, s in zip(means, sigmas):
        pdf += norm.pdf(x, m, s)
    return pdf / N


def residuals(params, x, y, N):
    """
    Residuals between histogram density and GMM PDF.
    """
    means = params[:N]
    log_sigmas = params[N:]        # optimize log(sigma) to enforce positivity
    sigmas = np.exp(log_sigmas)

    model = gaussian_mixture_pdf(x, means, sigmas)
    return model - y


def fit_fixed_weight_gmm(x, y, N, means_init=None, sigmas_init=None):
    """
    Fit a fixed-weight (1/N) Gaussian mixture to histogram data.

    Parameters
    ----------
    x : array
        Bin centers
    y : array
        Histogram density (should integrate to ~1)
    N : int
        Number of Gaussian components
    means_init : array, optional
    sigmas_init : array, optional

    Returns
    -------
    means, sigmas
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # --- initialization ---
    if means_init is None:
        means_init = np.linspace(x.min(), x.max(), N)

    if sigmas_init is None:
        sigmas_init = np.full(N, 0.5 * (x.max() - x.min()) / N)

    params0 = np.concatenate([means_init, np.log(sigmas_init)])

    # --- optimization ---
    result = least_squares(
        residuals,
        params0,
        args=(x, y, N),
        method="trf"
    )

    means = result.x[:N]
    sigmas = np.exp(result.x[N:])

    idx = np.argsort(means)
    means = means[idx]
    sigmas = sigmas[idx]

    return means, sigmas, result


class GGMP:
    def __init__(
            self,
            x_data,
            y_data,
            *,
            init_hyperparameters,
            hps_bounds,
            kernel_function=None,
            likelihood_terms=100
    ):
        """
        The constructor for the GGMP class.
        type help(GGMP) for more information about attributes, methods and their parameters
        """
        assert len(x_data) == len(y_data)
        number_of_GPs = 1
        self.len_data = len(x_data)

        self.hyperparameters = [
            np.append(init_hyperparameters.copy(), 0.)
            for _ in range(likelihood_terms)
        ]
        self.number_kernel_hyperparameters = len(init_hyperparameters)

        #lower = -0.001 #np.min(np.asarray([np.min(y_data[i][0]) for i in range(len(y_data))]))
        #upper = 0.001 #np.max(np.asarray([np.max(y_data[i][0]) for i in range(len(y_data))]))
        lower = np.min(np.asarray([np.min(y_data[i][0]) for i in range(len(y_data))]))
        upper = np.max(np.asarray([np.max(y_data[i][0]) for i in range(len(y_data))]))
        self.hps_bounds = np.vstack([np.vstack([hps_bounds, np.array([[lower, upper]])])] * likelihood_terms)
        self.x_data = x_data  # points as usual
        self.y_data = y_data  # each x gets a probability distribution as ordered pairs (domain, image)
        self.likelihood_terms = likelihood_terms
        self.kernel_function = kernel_function
        self.prior_mean_functions = [
            lambda x, hps: np.zeros(len(x)) + hps[-1]
            for _ in range(likelihood_terms)
        ]
        #if prior_mean_functions is None: self.prior_mean_functions = [None] * likelihood_terms
        self.GPs = None
        self.likelihoods = None

    def __getattr__(self, name):
        def not_implemented(*args, **kwargs):
            print(f"method {name} is not implemented in GGMPs")
        return not_implemented

    #def gp_hyperparameters(self, hps, index):
    #    return np.append(hps[0:self.number_kernel_hyperparameters],
    #                     hps[self.number_kernel_hyperparameters+index])

    #def gp_hyperparameter_bounds(self, index):
    #    return np.vstack([self.hps_bounds[0:self.number_kernel_hyperparameters],
    #                      self.hps_bounds[self.number_kernel_hyperparameters+index].reshape(1,-1)])

    def unravel_hps(self, hps: list) -> np.ndarray:
        return np.concatenate(hps)

    def ravel_hps(self, hps: np.ndarray) -> list:
        return list(hps.reshape(-1, self.number_kernel_hyperparameters + 1))

    def _compute_Gaussian_mixture_outputs(self):
        """
        This function takes the histograms in the y_data and the domains and
        creates Gaussian mixture for further calculations. The resulting Gaussian mixtures
        have a common weight vector of length(likelihood_terms). The output is a weight vector,
        and len(dataset) mean and variance vectors also of length(likelihood_terms).

        Returns
        -------
        weight vector : np.ndarray
        mean : list
        variances : list
        """
        weight_vector = np.zeros(self.likelihood_terms) + 1./self.likelihood_terms
        mean_list = []
        vars_list = []
        for dist in self.y_data:
            means, sigmas, result = fit_fixed_weight_gmm(dist[0], dist[1], self.likelihood_terms)
            mean_list.append(means)
            vars_list.append(sigmas)

        return weight_vector, mean_list, vars_list

    def initLikelihoods(self, likelihoods=None):
        if likelihoods is not None:
            self.likelihoods = likelihoods
            return self.likelihoods
        weights, means, sigmas = self._compute_Gaussian_mixture_outputs()
        means = np.asarray(means).reshape(len(means), self.likelihood_terms)
        sigmas= np.asarray(sigmas).reshape(len(sigmas), self.likelihood_terms)

        self.likelihoods = []
        for i in range(self.likelihood_terms):
            self.likelihoods.append(NormalLikelihood(means[:, i], sigmas[:, i]**2, weights[i]))

        return self.likelihoods

    def initGPs(self):
        self.GPs = [GP(self.x_data, self.likelihoods[j].mean, self.hyperparameters[j],
                    noise_variances=self.likelihoods[j].variance,
                    kernel_function=self.kernel_function,
                    prior_mean_function=self.prior_mean_functions[j])
                    for j in range(self.likelihood_terms)]

    def calculate_data_generation(self, plot=False):
        """
        y0 is a list of initial y means for the likelihoods len(y0) = likelihood terms
        len(y0[i]) = number of data points

        sigma0 is a list of initial variances, same structure as y0
        """
        score = 0
        for i in range(self.len_data):
            domain = self.y_data[i][0]
            data_distr = self.y_data[i][1]
            density = np.zeros(len(data_distr))
            for j in range(self.likelihood_terms):
                g = gaussian(self.likelihoods[j].mean[i], np.sqrt(self.likelihoods[j].variance[i]), domain)
                density += self.likelihoods[j].weight * g
            if plot:
                plt.plot(domain, data_distr, label="data")
                plt.plot(domain, density, label="likelihood")
                plt.legend()
                plt.show()
            score += integral(np.sqrt(data_distr * density), domain)
        print("SCORE: ", score / float(self.len_data))
        return score / float(self.len_data)

    def train(self,
              pop_size=20,
              tolerance=0.001,
              max_iter=120,
              dask_client=None
              ):
        #if hps_obj is None: hps_obj = self.hps_obj
        #(weights,
        hps = self.optimize_log_likelihood(
            max_iter,
            pop_size,
            tolerance
        )
        self.hyperparameters = hps
        print("new hps     after training: ", hps)

        for i in range(self.likelihood_terms): self.GPs[i].set_hyperparameters(self.hyperparameters[i])

    def optimize_log_likelihood(self,
                                max_iter,
                                pop_size,
                                tolerance,
                                workers=1,
                                ):
        print("Ensemble fvGP submitted to global optimization")
        print('bounds are', self.hps_bounds)
        print("maximum number of iterations: ", max_iter)
        print("termination tolerance: ", tolerance)
        x0 = self.unravel_hps(self.hyperparameters)
        Eval = self.GGMP_log_likelihood(x0)
        print("Start @: ", x0)
        print(" Old likelihood: ", Eval)

        res = differential_evolution(
            self.neg_GGMP_log_likelihood,
            self.hps_bounds,
            disp=True,
            maxiter=max_iter,
            popsize=pop_size,
            tol=tolerance,
            workers=workers,
            x0=x0,
            polish=False
        )
        r = np.array(res["x"])

        Eval = self.GGMP_log_likelihood(r)
        print(" New likelihood: ", Eval)
        return self.ravel_hps(r)

    def GGMP_log_likelihood(self, hps):
        """
        computes the marginal log-likelihood
        input:
            hyperparameters
        output:
            marginal log-likelihood (scalar)
        """
        weights_y = [self.likelihoods[i].weight for i in range(self.likelihood_terms)]

        B = np.empty(self.likelihood_terms)
        if isinstance(hps, np.ndarray): hps = self.ravel_hps(hps)
        for j in range(self.likelihood_terms):
            B[j] = np.log(weights_y[j]) + self.GPs[j].log_likelihood(hps[j])
        res1 = logsumexp(B)
        return res1

    #def GGMP_log_likelihood(self, v):
    #    """
    #    Local mixture-of-experts log-likelihood.
    #    No global GP collapse.
    #    """
    #
    #    # unpack hyperparameters
    #    weights_f, hps = self.hps_obj.devectorize_hps(v)
    #    weights_y = np.array([self.likelihoods[k].weight
    #                          for k in range(self.likelihood_terms)])

    #    logL = 0.0
    #    eps = 1e-12  # numerical safety

    #    # loop over datapoints
    #    for n in range(self.len_data):

            # accumulate local mixture terms
    #        local_terms = []

    #        for j in range(self.number_of_GPs):
    #            for k in range(self.likelihood_terms):
    #                # GP j likelihood term k evaluated at datapoint n
    #                sigma_k = 1e-3
    #                self.GPs[k][j].set_hyperparameters(hps[j])
    #                domain = self.y_data[n][0]
    #                y_density = self.y_data[n][1]
    #                ll = self.local_predictive_log_likelihood(self.GPs[k][j],
    #                                                          self.x_data[n].reshape(1, -1),
    #                                                          domain,
    #                                                          y_density,
    #                                                          sigma_k)
    #                local_terms.append(
    #                    np.log(weights_f[j] + eps)
    #                    + np.log(weights_y[k] + eps)
    #                    + ll
    #                )
    #        # log-sum-exp over local mixture
    #        local_terms = np.array(local_terms)
    #        m = np.max(local_terms)
    #        logL += m + np.log(np.sum(np.exp(local_terms - m)))

    #    return logL

    #@staticmethod
    #def local_predictive_log_likelihood(GP,
    #                                    x_n,
    #                                    y_domain,
    #                                    y_density,
    #                                    sigma_k2):
    #    """
    #    Local predictive log-likelihood for GGMP.

    #    Parameters
    #    ----------
    #    GP : trained GP object
    #        One GP expert (already conditioned on all data)
    #    x_n : array-like, shape (1, d)
    #        Input location
    #    y_domain : array
    #        Domain of the output distribution
    #    y_density : array
    #        Observed histogram (normalized)
    #    sigma_k2 : float
    #        Variance of likelihood component k

    #    Returns
    #    -------
    #    log_likelihood : float
    #    """
    #    # GP posterior moments at x_n
    #    mu = GP.posterior_mean(x_n)["m(x)"]
    #    var = GP.posterior_covariance(x_n)["v(x)"]

    #    total_var = var + sigma_k2
    #    total_std = np.sqrt(total_var)

    #    # model predictive density
    #    model_pdf = norm.pdf(y_domain, mu, total_std)
    #    model_pdf /= np.trapz(model_pdf, y_domain)

    #    # Bhattacharyya coefficient
    #    bc = np.trapezoid(np.sqrt(y_density * model_pdf), y_domain)

    #    # numerical safety
    #    bc = np.clip(bc, 1e-12, 1.0)

    #    return np.log(bc)

    def neg_GGMP_log_likelihood(self, v):
        return -self.GGMP_log_likelihood(v)

    ##########################################################
    def posterior(self, x_iset, res=100, lb=None, ub=None):

        #print([[self.GPs[j][i].prior.hyperparameters for i in range(self.number_of_GPs)] for j in
        #       range(self.likelihood_terms)])
        if np.ndim(x_iset) == 1: x_iset = x_iset.reshape(1, len(x_iset))
        means = [self.GPs[j].posterior_mean(x_iset)["m(x)"].reshape(len(x_iset)) for j in range(self.likelihood_terms)]
        covs = [self.GPs[j].posterior_covariance(x_iset)["v(x)"].reshape(len(x_iset)) for j in range(self.likelihood_terms)]

        means = np.array(means)
        covs = np.array(covs)

        print("posterior shape: ", means.shape)
        if lb is None: lb = np.min(means - 3.0 * np.sqrt(covs))
        if ub is None: ub = np.max(means + 3.0 * np.sqrt(covs))

        pdfs = []
        for i in range(len(x_iset)):
            pdf = np.zeros((res))
            for k in range(self.likelihood_terms):
                pdf += self.likelihoods[k].weight * gaussian(means[k, i],
                                                             np.sqrt(covs[k, i]),
                                                             np.linspace(lb, ub, res))
            pdfs.append(pdf)
        return {"m(x)": means, "v(x)": covs, "pdf": pdfs, "lb": lb, "ub": ub, "domain": np.linspace(lb, ub, res)}

    ##########################################################

    ############################################################################
    ############################################################################
    ############################################################################
    ########################GRADIENTS###########################################
    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
    def GGMP_log_likelihood_grad(self, v):
        weights, hps = self.hps_obj.devectorize_hps(v)
        w_grad = np.zeros(self.number_of_GPs)
        h_grad = []
        A = np.zeros(self.number_of_GPs)
        dA_dw = np.zeros(self.number_of_GPs)
        dA_dP = np.zeros(self.number_of_GPs)

        def kronecker(k, l):
            if int(k) == int(l):
                return 1.0
            else:
                return 0.0

        for i in range(self.number_of_GPs):
            A[i] = np.log(weights[i]) - self.GPs[i].log_likelihood(hps[i])

        k = np.argmax(A)
        A = A - A[k]
        indices = np.arange(self.number_of_GPs) != k
        s1 = np.sum(np.exp(A[indices]))

        for p in range(self.number_of_GPs):
            for i in range(self.number_of_GPs):
                dA_dw[i] = (kronecker(i, p) - kronecker(k, p)) / weights[p]
                dA_dP[i] = kronecker(i, p) - kronecker(k, p)

            s2 = np.exp(A[indices]).T @ dA_dw[indices]
            s3 = np.exp(A[indices]).T @ dA_dP[indices]

            w_grad[p] = -(kronecker(k, p) / weights[p] + (s2 / (1. + s1)))
            h_grad.append(
                (kronecker(k, p) + s3 / (1. + s1)) * self.GPs[p].marginal_density.neg_log_likelihood_gradient(hps[p]))
        return self.hps_obj.vectorize_hps(w_grad, h_grad)

    def GGMP_log_likelihood_hess(self, v):
        len_hyperparameters = len(v)
        d2L_dmdh = np.zeros((len_hyperparameters, len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.GGMP_log_likelihood_grad(v)
        for i in range(len_hyperparameters):
            hps_temp = np.array(v)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i, i:] = ((self.GGMP_log_likelihood_grad(hps_temp) - grad_at_hps) / epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))


#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
class hyperparameters:
    """
    Parameters:
        * weights: 1d numpy array
        * weights_bounds: 2d numpy array
        * hps: list of 1d numpy arrays
        * hps_bounds: list of 2d numpy arrays
    """

    def __init__(self, hps, hps_bounds):
        self.hps_bounds = hps_bounds
        #self.weights_bounds = weights_bounds
        #self.weights = weights
        self.hps = hps
        self.number_of_weights = len(weights)
        self.number_of_hps_sets = len(hps)
        self.number_of_hps = [len(hps[i]) for i in range(len(hps))]
        if len(hps) != len(hps_bounds): raise Exception("hps and hps_bounds have to be lists of equal length")
        if len(weights) != len(weights_bounds):
            raise Exception("weights (1d) and weights_bounds (2d) have to be numpy arrays of equal length")

        self.vectorized_hps = self.vectorize_hps(weights, hps)
        self.vectorized_bounds = self.vectorize_bounds(weights_bounds, hps_bounds)

    def set(self, weights, hps):
        if len(hps) != len(self.hps_bounds): raise Exception("hps and hps_bounds have to be lists of equal length")
        if len(weights) != len(self.weights_bounds):
            raise Exception("weights (1d) and weights_bounds (2d) have to be numpy arrays of equal length")

        self.weights = weights
        self.hps = hps
        self.vectorized_hps = self.vectorize_hps(weights, hps)

    def vectorize_hps(self, weights, hps):
        v = [weights[i] for i in range(self.number_of_weights)]
        for i in range(self.number_of_hps_sets):
            for j in range(self.number_of_hps[i]):
                v.append(hps[i][j])
        return np.asarray(v)

    def devectorize_hps(self, v):
        weights = v[0:self.number_of_weights]
        index = self.number_of_weights
        hps = []
        for i in range(self.number_of_hps_sets):
            hps.append(v[index:index + self.number_of_hps[i]])
            index += self.number_of_hps[i]
        return weights, hps

    def vectorize_bounds(self, weights_bounds, hps_bounds):
        b = [weights_bounds[i] for i in range(self.number_of_weights)]
        for i in range(self.number_of_hps_sets):
            for j in range(self.number_of_hps[i]):
                b.append(hps_bounds[i][j])
        return np.asarray(b)

    def devectorize_bounds(self, b):
        weights_bounds = b[0:self.number_of_weights]
        index = self.number_of_weights
        hps_bounds = []
        for i in range(self.number_of_hps_sets):
            hps_bounds.append(b[index:index + self.number_of_hps[i]])
            index += self.number_of_hps[i]
        return weights_bounds, hps_bounds


class NormalLikelihood:
    def __init__(self, mean, variance, weight):
        self.mean = mean
        self.variance = variance
        self.dim = len(mean)
        self.weight = weight
        self.weight_bounds = np.array([0, 1])

    def set_moments(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def set_weight(self, weight):
        self.weight = weight

    def unravel(self):
        return np.concatenate([self.mean, self.variance])

    def ravel(self, vec):
        return vec[0:self.dim], vec[self.dim:]

    def marginalize(self, domain, direction):
        return gaussian(self.mean[direction], np.sqrt(self.variances), domain)