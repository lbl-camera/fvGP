from fvgp.gp import GP
from hgdl.hgdl import HGDL
import numpy as np
from scipy.optimize import NonlinearConstraint,differential_evolution
import matplotlib.pyplot as plt
from scipy.optimize import minimize





def integral(f, domain):
    Int = np.sum(f * np.gradient(domain))
    return Int

def gaussian(mean, std, x):
    g = (1./ (np.sqrt(2. * np.pi) * std)) * np.exp(-np.power(x - mean, 2.) / (2. * np.power(std, 2.)))
    if np.all(g < 1e-6): g[:] = 1e-6
    inte = integral(g,x)
    gn =  g / inte
    if np.any(np.isnan(gn)): print("NaN in Gaussian normalized")
    return gn

class GGMP():
    def __init__(
        self,
        x_data,
        y_data,
        number_of_GPs,
        *,
        hps_obj,
        gp_kernel_functions = None,
        gp_mean_functions = None,
        likelihood_terms = 100
        ):
        """
        The constructor for the GGMP class.
        type help(GGMP) for more information about attributes, methods and their parameters
        """
        assert len(x_data) == len(y_data)
        self.number_of_GPs = number_of_GPs
        self.len_data = len(x_data)
        self.hps_obj = hps_obj
        self.init_weights = np.ones((number_of_GPs)) / float(number_of_GPs)
        self.x_data = x_data ##points as usual
        self.y_data = y_data ##each x gets a probabaility distribution as ordered pairs (domain, image)
        self.likelihood_terms = likelihood_terms
        self.gp_kernel_functions = gp_kernel_functions
        self.gp_mean_functions = gp_mean_functions
        if gp_kernel_functions is None: self.gp_kernel_functions = [None] * number_of_GPs
        if gp_mean_functions is None: self.gp_mean_functions = [None] * number_of_GPs



    def initLikelihoods(self, init_mean = None, init_std = None, weights = None):
        assert init_mean is None or isinstance(init_mean, list)
        assert init_std is None or isinstance(init_std, list)
        if isinstance(init_mean, list): assert len(init_mean) == self.likelihood_terms
        if isinstance(init_std, list): assert len(init_std) == self.likelihood_terms

        if init_mean is None: init_mean = [self.y_data.copy() for i in range(self.likelihood_terms)] ##for each likeihood
        if init_std is None: init_std   = [np.sqrt(np.var(self.y_data)) / 10. for i in range(self.likelihood_terms)]

        #var_bounds = std_bounds**2
        if weights is None:
            weights = np.ones((self.likelihood_terms))
            weights = weights/np.sum(weights)

        self.likelihoods = []
        for i in range(self.likelihood_terms):
            self.likelihoods.append(NormalLikelihood(init_mean[i], init_std[i]**2, weights[i]))

        return self.likelihoods


    def initGPs(self):
        self.GPs = [[GP(self.x_data,self.likelihoods[j].mean, self.hps_obj.hps[i],
                               noise_variances = self.likelihoods[j].variance,
                               gp_kernel_function = self.gp_kernel_functions[i],
                               gp_mean_function = self.gp_mean_functions[i])
                               for i in range(self.number_of_GPs)] for j in range(self.likelihood_terms)]


    def find_data_generating_likelihood(self, maxiter = 1000, popsize = 10, tol = 0.001, workers = 8):
        """
        y0 is a list of initial y means for the likleihoods len(y0) = likelihood terms
        len(y0[i]) = number of data points

        sigma0 is a list of initial variances, same structure as y0
        """
        weights0 = np.asarray([1./self.likelihood_terms] * self.likelihood_terms)
        #mean0 = np.asarray([likelihood.mean for likelihood in self.likelihoods]).flatten()
        #sigma0 = np.asarray([likelihood.variance for likelihood in self.likelihoods]).flatten()
        #x0 = np.concatenate([weights0, mean0, sigma0], axis = 0)
        x0 = weights0


        weight_bounds = np.vstack([np.array([0,1])] * self.likelihood_terms)
        #mean_bounds   = np.vstack([likelihood.mean_bounds for likelihood in self.likelihoods])
        #sigma_bounds  = np.vstack([likelihood.variance_bounds for likelihood in self.likelihoods])
        #bounds = np.row_stack([weight_bounds, mean_bounds, sigma_bounds])
        bounds = weight_bounds

        def constraint(v):
            return np.array(np.sum(v))
        nlc = NonlinearConstraint(constraint,0.99,1.0)


        print("initial score: ", self._get_data_generation(x0))


        res = differential_evolution(
            self._get_data_generation,
            bounds,
            disp = True,
            maxiter = maxiter,
            popsize = popsize,
            tol = tol,
            x0 = x0,
            constraints = (nlc),
            workers = workers, polish = False)

        #res = minimize(
        #        self.data_generation,
        #        method= "L-BFGS-B",
        #        x0 = np.random.uniform(low=bounds[:,0], high = bounds[:,1], size = len(bounds)),
        #        bounds = bounds,
        #        tol = tol,
        #        constraints = (nlc,),
        #        options = {"maxiter": maxiter, "disp": True})

        vec = res["x"]
        weights = vec
        weights = weights / np.sum(weights)

        #y = vec[self.likelihood_terms : self.likelihood_terms + self.likelihood_terms * self.len_data]
        #y = y.reshape(self.len_data, self.likelihood_terms, order = "F")

        #sigma = vec[self.likelihood_terms + self.likelihood_terms * self.len_data : ]
        #sigma = sigma.reshape(self.len_data, self.likelihood_terms, order = "F")



        for i in range(self.likelihood_terms):
            #self.likelihoods[i].set_moments(y[:,i], sigma[:,i])
            self.likelihoods[i].set_weight(weights[i])

        return self.likelihoods

    def _get_data_generation(self, vec):
        """
        y0 is a list of initial y means for the likleihoods len(y0) = likelihood terms
        len(y0[i]) = number of data points

        sigma0 is a list of initial variances, same structure as y0
        """

        weights = vec

        #y = vec[self.likelihood_terms : self.likelihood_terms + self.likelihood_terms * self.len_data]
        #y = y.reshape(self.len_data, self.likelihood_terms, order = "F")

        #sigma = vec[self.likelihood_terms + self.likelihood_terms * self.len_data : ]
        #sigma = sigma.reshape(self.len_data, self.likelihood_terms, order = "F")


        score = 0
        for i in range(self.len_data):
            domain =     self.y_data[i][0]
            data_distr = self.y_data[i][1]
            density = np.zeros(len(data_distr))
            for j in range(self.likelihood_terms):
                g = gaussian(self.likelihoods[j].mean[i], np.sqrt(self.likelihoods[j].variance[i]), domain)
                #g = gaussian(y[i,j], np.sqrt(sigma[i,j]), domain)
                density += weights[j] * g
            score += integral(np.sqrt(data_distr*density), domain) #self.BBdist(domain, y_data[i][1]), d)
        print("DATA GENERATION: ", score/float(self.len_data))
        return -score/float(self.len_data)



    def set_data_generating_likelihood(self, likelihoods):
        self.likelihoods = likelihoods
        return


    def calculate_data_generation(self, plot = False):
        """
        y0 is a list of initial y means for the likleihoods len(y0) = likelihood terms
        len(y0[i]) = number of data points

        sigma0 is a list of initial variances, same structure as y0
        """
        score = 0
        for i in range(self.len_data):
            domain =     self.y_data[i][0]
            data_distr = self.y_data[i][1]
            density = np.zeros(len(data_distr))
            for j in range(self.likelihood_terms):
                g = gaussian(self.likelihoods[j].mean[i], np.sqrt(self.likelihoods[j].variance[i]), domain)
                density += self.likelihoods[j].weight * g
            if plot == True:
                plt.plot(domain, data_distr, label = "data")
                plt.plot(domain, density, label = "likelihood")
                plt.legend()
                plt.show()
            score += integral(np.sqrt(data_distr*density), domain)
        print("SCORE: ", score/float(self.len_data))
        return -score/float(self.len_data)

    def train(self,
        hps_obj = None,
        pop_size = 20,
        tolerance = 0.001,
        max_iter = 120,
        dask_client = None
        ):
        if hps_obj is None: hps_obj = self.hps_obj
        weights, hps = self.optimize_log_likelihood(
            hps_obj,
            max_iter,
            pop_size,
            tolerance
            )
        self.hps_obj.set(weights,hps)
        print("new weights after training: ", self.hps_obj.weights)
        print("new hps     after training: ", self.hps_obj.hps)
        for i in range(self.number_of_GPs):
            for j in range(self.likelihood_terms):
                self.GPs[j][i].set_hyperparameters(self.hps_obj.hps[i])
        print("GPs updated")


    def optimize_log_likelihood(self,
            hps_obj,
            max_iter,
            pop_size,
            tolerance,
            workers = 1,
            ):
        print("Ensemble fvGP submitted to global optimization")
        print('bounds are',hps_obj.vectorized_bounds)
        print("maximum number of iterations: ", max_iter)
        print("termination tolerance: ", tolerance)
        x0 =  self.hps_obj.vectorized_hps
        Eval = self.GGMP_log_likelihood(x0)
        print(x0)
        print(" Old likelihood: ", Eval)

        def constraint(v):
            return np.array(np.sum(v[0:self.number_of_GPs]))

        nlc = NonlinearConstraint(constraint,0.99,1.0)
        #nlc2 = NonlinearConstraint(constraint2,0.2,0.4)
        res = differential_evolution(
            self.neg_GGMP_log_likelihood,
            hps_obj.vectorized_bounds,
            disp=True,
            maxiter=max_iter,
            popsize = pop_size,
            tol = tolerance,
            workers = workers,
            constraints = (nlc),
            x0 = x0,
            polish = False
        )

        #r = np.array(res["x"])
        #r[0:self.number_of_GPs] = 1./self.number_of_GPs
        #nlc = NonlinearConstraint(constraint,0.9999,1.0)
        #res = minimize(
        #        self.GGMP_log_likelihood,r,
        #        method= "SLSQP",
        #        jac=self.GGMP_log_likelihood_grad,
        #        bounds = hps_obj.vectorized_bounds,
        #        tol = tolerance,
        #        callback = None,
        #        options = {"maxiter": max_iter},
        #        constraints = (nlc,nlc2))
        r = np.array(res["x"])

        Eval = self.GGMP_log_likelihood(r)
        weights,hps = self.hps_obj.devectorize_hps(r)
        print(" New likelihood: ", Eval)
        return weights, hps


    def GGMP_log_likelihood(self,v):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        weights_f, hps = self.hps_obj.devectorize_hps(v)
        weights_y = [self.likelihoods[i].weight for i in range(self.likelihood_terms)]

        B = np.empty((self.number_of_GPs, self.likelihood_terms))
        for i in range(self.number_of_GPs):
            for j in range(self.likelihood_terms):
                B[i, j] = np.log(weights_y[j]) + np.log(weights_f[i]) + self.GPs[j][i].log_likelihood(hps[i])


        B = B.flatten()
        k =  np.argmax(B)
        B_i0j0 = B[k]
        diffB = B - B_i0j0
        diffB = np.delete(diffB, k)
        S = np.sum(np.exp(diffB))

        return B_i0j0 + np.log(1.0 + S)


    def neg_GGMP_log_likelihood(self,v):
        return -self.GGMP_log_likelihood(v)


    ##########################################################
    def posterior(self,x_iset, res = 100, lb = None, ub = None):

        print([[self.GPs[j][i].prior.hyperparameters for i in range(self.number_of_GPs)] for j in range(self.likelihood_terms)])
        means = [[self.GPs[j][i].posterior_mean(x_iset)["f(x)"] for i in range(self.number_of_GPs)] for j in range(self.likelihood_terms)]
        covs  = [[self.GPs[j][i].posterior_covariance(x_iset)["v(x)"] for i in range(self.number_of_GPs)] for j in range(self.likelihood_terms)]
        
        m = list(np.concatenate(means))
        for i in range(len(m)):
            plt.plot(x_iset.flatten(), m[i])
        plt.show()


        means = np.array(means)
        covs = np.array(covs)


        print("posterior shape: ",means.shape)
        if lb == None: lb = np.min(means - 3.0 * np.sqrt(covs))
        if ub == None: ub = np.max(means + 3.0 * np.sqrt(covs))

        pdfs = []
        for i in range(len(x_iset)):
            pdf = np.zeros((res))
            for j in range(self.number_of_GPs):
                for k in range(self.likelihood_terms):
                    pdf += self.hps_obj.weights[j] * self.likelihoods[k].weight * gaussian(means[k,j,i],np.sqrt(covs[k,j,i]),np.linspace(lb,ub,res))
            pdfs.append(pdf)
        return {"f(x)": means, "v(x)":covs, "pdf": pdfs, "lb": lb, "ub": ub, "domain" : np.linspace(lb,ub,res)}

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
    def GGMP_log_likelihood_grad(self,v):
        weights, hps = self.hps_obj.devectorize_hps(v)
        w_grad = np.zeros((self.number_of_GPs))
        h_grad = []
        A = np.zeros((self.number_of_GPs))
        dA_dw = np.zeros((self.number_of_GPs))
        dA_dP = np.zeros((self.number_of_GPs))
        def kronecker(k,l):
            if int(k) == int(l): return 1.0
            else: return 0.0

        for i in range(self.number_of_GPs):
            A[i] = np.log(weights[i]) - self.GPs[i].log_likelihood(hps[i])

        k = np.argmax(A)
        A = A - A[k]
        indices = np.arange(self.number_of_GPs) != k
        s1 = np.sum(np.exp(A[indices]))

        for p in range(self.number_of_GPs):
            for i in range(self.number_of_GPs):
                dA_dw[i] = (kronecker(i,p) - kronecker(k,p))/weights[p]
                dA_dP[i] = kronecker(i,p) - kronecker(k,p)

            s2 = np.exp(A[indices]).T @ dA_dw[indices]
            s3 = np.exp(A[indices]).T @ dA_dP[indices]

            w_grad[p] = -(kronecker(k,p)/weights[p] + (s2/(1. + s1)))
            h_grad.append((kronecker(k,p) + s3/(1. + s1)) * self.GPs[p].marginal_density.neg_log_likelihood_gradient(hps[p]))
        return self.hps_obj.vectorize_hps(w_grad,h_grad)

    def GGMP_log_likelihood_hess(self,v):
        len_hyperparameters = len(v)
        d2L_dmdh = np.zeros((len_hyperparameters,len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.GGMP_log_likelihood_grad(v)
        for i in range(len_hyperparameters):
            hps_temp = np.array(v)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i,i:] = ((self.GGMP_log_likelihood_grad(hps_temp) - grad_at_hps)/epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))




#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
class hyperparameters():
    """
    Parameters:
        * weights: 1d numpy array
        * weights_bounds: 2d numpy array
        * hps: list of 1d numpy arrays
        * hps_bounds: list of 2d numpy arrays
    """
    def __init__(self, weights, weights_bounds,hps,hps_bounds):
        self.hps_bounds = hps_bounds
        self.weights_bounds = weights_bounds
        self.weights = weights
        self.hps = hps
        self.number_of_weights = len(weights)
        self.number_of_hps_sets = len(hps)
        self.number_of_hps = [len(hps[i]) for i in range(len(hps))]
        if len(hps) != len(hps_bounds): raise Exception("hps and hps_bounds have to be lists of equal length")
        if len(weights) != len(weights_bounds):
            raise Exception("weights (1d) and weights_bounds (2d) have to be numpy arrays of equal length")

        self.vectorized_hps = self.vectorize_hps(weights,hps)
        self.vectorized_bounds = self.vectorize_bounds(weights_bounds,hps_bounds)

    def set(self,weights,hps):
        if len(hps) != len(self.hps_bounds): raise Exception("hps and hps_bounds have to be lists of equal length")
        if len(weights) != len(self.weights_bounds):
            raise Exception("weights (1d) and weights_bounds (2d) have to be numpy arrays of equal length")

        self.weights = weights
        self.hps = hps
        self.vectorized_hps = self.vectorize_hps(weights,hps)

    def vectorize_hps(self, weights,hps):
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

    def vectorize_bounds(self,weights_bounds,hps_bounds):
        b = [weights_bounds[i] for i in range(self.number_of_weights)]
        for i in range(self.number_of_hps_sets):
            for j in range(self.number_of_hps[i]):
                b.append(hps_bounds[i][j])
        return np.asarray(b)


    def devectorize_bounds(self,b):
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
        self.weight_bounds = np.array([0,1])

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
        return gaussian(self.mean[direction], np.sqrt(self.variances, domain))
