"""
Software: FVGP, version: 2.3.4
File containing the gp class
use help() to find information about usage
Author: Marcus Noack
Institution: CAMERA, Lawrence Berkeley National Laboratory
email: MarcusNoack@lbl.gov
This file contains the FVGP class which trains a Gaussian process and predicts
function values.

License:
Copyright (C) 2020 Marcus Michael Noack

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact: MarcusNoack@lbl.gov
"""

from fvgp.gp import GP
from hgdl.hgdl import HGDL
import numpy as np
from scipy.optimize import NonlinearConstraint,differential_evolution
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EnsembleGP():
    """
    Ensemble GP class: Provides all tool for a single-task ensemble GP.

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space

    Attributes:
        input_space_dim (int):         dim1
        points (N x dim1 numpy array): 2d numpy array of points
        values (N x n numpy array):    2d numpy array of values
        hps_obj:                       instance of hyperparameter class

    Optional Attributes:
        variances (N x n numpy array):                  variances of the values, default = array of shape of points
                                                        with 1 % of the values
        compute_device:                                 cpu/gpu, default = cpu
        gp_kernel_function(func):                       None/list of functions defining the 
                                                        kernel def name(x1,x2,hyperparameters,self), default = None
        gp_mean_function(func):                         None/list of functions def name(x, self), default = None
        sparse (bool):                                  default = False
        normalize_y:                                    default = False, normalizes the values \in [0,1]

    Example:
        obj = fvGP(3,np.array([[1,2,3],[4,5,6]]),
                         np.array([2,4]),
                         [np.array([2,3,4,5])],
                         variances = np.array([0.01,0.02]),
                         gp_kernel_function = [kernel_function],
                         gp_mean_function = [some_mean_function]
        )
    """
    def __init__(
        self,
        input_space_dim,
        points,
        values,
        number_of_GPs,
        hps_obj,
        variances = None,
        compute_device = "cpu",
        gp_kernel_functions = None,
        gp_mean_functions = None,
        sparse = False,
        normalize_y = False
        ):
        """
        The constructor for the ensemblegp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        self.number_of_GPs = number_of_GPs
        self.hps_obj = hps_obj
        self.init_weights = np.ones((number_of_GPs)) / float(number_of_GPs)
        self.points = points
        self.values = values
        if gp_kernel_functions is None: gp_kernel_functions = [None] * number_of_GPs
        if gp_mean_functions is None: gp_mean_functions = [None] * number_of_GPs
        self.EnsembleGPs = [GP(input_space_dim,points,values,hps_obj.hps[i],
                               variances = variances,compute_device = compute_device,
                               gp_kernel_function = gp_kernel_functions[i],
                               gp_mean_function = gp_mean_functions[i],
                               sparse = sparse, normalize_y = normalize_y)
                               for i in range(number_of_GPs)]

    def update_EnsembleGP_data(self, points, values, variances = None):
        for i in range(len(self.number_of_GPs)):
            self.EnsembleGPs[i].update_gp_data(points,values, variances = variances)
    ############################################################################
    def stop_training(self):
        print("Ensemble fvGP is cancelling the asynchronous training...")
        try: self.opt.cancel_tasks(); print("Ensemble fvGP successfully cancelled the current training.")
        except Exception as err: print("No asynchronous training to be cancelled in Ensemble fvGP, no training is running.", err)

    def kill_training(self):
        print("fvGP is killing asynchronous training....")
        try: self.opt.kill_client(); print("fvGP successfully killed the training.")
        except Exception as err: print("No asynchronous training to be killed, no training is running.", err)

    def train_async(self,
        hps_obj = None,
        max_iter = 10000,
        local_optimizer = "SLSQP",
        global_optimizer = "genetic",
        dask_client = None,
        deflation_radius = None
        ):
        if hps_obj is None: hps_obj = self.hps_obj
        self.optimize_log_likelihood_async(
            hps_obj,
            max_iter,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client
            )

    def train(self,
        hps_obj = None,
        pop_size = 20,
        tolerance = 0.1,
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
        for i in range(self.number_of_GPs): self.EnsembleGPs[i].hyperparameters = self.hps_obj.hps[i]
        self.compute_prior_pdf()
        print("GPs updated")

    def update_hyperparameters(self, n = 1):
        try:
            r = self.opt.get_latest(n)['x'][0]
            weights,hps = self.hps_obj.devectorize_hps(r)
            self.hps_obj.set(weights,hps)
            print("new weights after training: ", self.hps_obj.weights)
            print("new hps     after training: ", self.hps_obj.hps)
            for i in range(self.number_of_GPs): self.EnsembleGPs[i].hyperparameters = self.hps_obj.hps[i]
            self.compute_prior_pdf()
            print("Ensemble fvGP async hyperparameter update successful")
        except Exception as e:
            print("Async Hyper-parameter update not successful in Ensemble fvGP. I am keeping the old ones.")
            print("That probably means you are not optimizing them asynchronously")
            print("error: ", e)
            print("weights: ", self.hps_obj.weights)
            print("hps    : ", self.hps_obj.hps)

        return self.hps_obj.weights, self.hps_obj.hps

    def optimize_log_likelihood_async(self, 
            hps_obj,
            max_iter,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client):
        print("Ensemble fvGP submitted to HGDL optimization")
        print('bounds are',hps_obj.vectorized_bounds)
        print("initial weights: ", hps_obj.vectorized_hps)
        def constraint(v):
            return np.array(np.sum(v[0:self.number_of_GPs]))

        nlc = NonlinearConstraint(constraint,0.99,1.0)

        self.opt = HGDL(self.ensemble_log_likelihood,
                self.ensemble_log_likelihood_grad,
                hps_obj.vectorized_bounds,
                hess = self.ensemble_log_likelihood_hess,
                local_optimizer = local_optimizer,
                global_optimizer = global_optimizer,
                radius = deflation_radius,
                num_epochs = max_iter,
                constr = (nlc))

        self.opt.optimize(dask_client = dask_client)

    def optimize_log_likelihood(self,
            hps_obj,
            max_iter,
            pop_size,
            tolerance,
            ):
        print("Ensemble fvGP submitted to global optimization")
        print('bounds are',hps_obj.vectorized_bounds)
        print("maximum number of iterations: ", max_iter)
        print("termination tolerance: ", tolerance)
        def constraint(v):
            return np.array(np.sum(v[0:self.number_of_GPs]))
        def constraint2(v):
            return np.array(v[0])

        nlc = NonlinearConstraint(constraint,0.8,1.0)
        nlc2 = NonlinearConstraint(constraint2,0.2,0.4)
        res = differential_evolution(
            self.ensemble_log_likelihood,
            hps_obj.vectorized_bounds,
            disp=True,
            maxiter=max_iter,
            popsize = pop_size,
            tol = tolerance,
            workers = 1,
            constraints = (nlc,nlc2),
            polish = False
        )

        r = np.array(res["x"])
        r[0:self.number_of_GPs] = 1./self.number_of_GPs
        nlc = NonlinearConstraint(constraint,0.9999,1.0)
        res = minimize(
                self.ensemble_log_likelihood,r,
                method= "SLSQP",
                jac=self.ensemble_log_likelihood_grad,
                bounds = hps_obj.vectorized_bounds,
                tol = tolerance,
                callback = None,
                options = {"maxiter": max_iter},
                constraints = (nlc,nlc2))
        r = np.array(res["x"])

        Eval = self.ensemble_log_likelihood(r)
        weights,hps = self.hps_obj.devectorize_hps(r)
        print("fvGP found weights ",weights)
        print("and hyperparameters: ",hps)
        print(" with likelihood: ",Eval," via global optimization")
        return weights,hps


    def ensemble_log_likelihood(self,v):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        weights, hps = self.hps_obj.devectorize_hps(v)
        Psi = np.empty((self.number_of_GPs))
        A = np.empty((self.number_of_GPs))
        for i in range(self.number_of_GPs):
            A[i] = np.log(weights[i]) - self.EnsembleGPs[i].log_likelihood(hps[i])
        k = np.argmax(A)
        A_largest = A[k]
        indices = np.arange(self.number_of_GPs) != k
        A = A - A_largest
        L = np.sum(np.exp(A[indices]))
        return -(A_largest + np.log(1.0 + L))

    def ensemble_log_likelihood_grad(self,v):
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
            A[i] = np.log(weights[i]) - self.EnsembleGPs[i].log_likelihood(hps[i])

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
            h_grad.append((kronecker(k,p) + s3/(1. + s1)) * self.EnsembleGPs[p].log_likelihood_gradient(hps[p]))
        return self.hps_obj.vectorize_hps(w_grad,h_grad)

    def ensemble_log_likelihood_hess(self,v):
        len_hyperparameters = len(v)
        d2L_dmdh = np.zeros((len_hyperparameters,len_hyperparameters))
        epsilon = 1e-6
        grad_at_hps = self.ensemble_log_likelihood_grad(v)
        for i in range(len_hyperparameters):
            hps_temp = np.array(v)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i,i:] = ((self.ensemble_log_likelihood_grad(hps_temp) - grad_at_hps)/epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))



        #def kronecker(k,l):
        #    if int(k) == int(l): return 1.0
        #    else: return 0.0
        #for k in range(self.number_of_GPs):
        #    like = np.log(weights[k]) - self.EnsembleGPs[k].log_likelihood(hps[k])
        #    for i in range(self.number_of_GPs):
        #        t = np.log(weights[i])-self.EnsembleGPs[i].log_likelihood(hps[i])-like
        #        if t > 100.0: exp_a[i] = 10e16
        #        else: exp_a[i] = np.exp(np.log(weights[i])-self.EnsembleGPs[i].log_likelihood(hps[i])-like)
        #
        #    for l in range(k,self.number_of_GPs):
        #
        #        d2 = np.empty((self.number_of_GPs, len(hps[l])))
        #        for i in range(self.number_of_GPs):
        #            d[i] = (kronecker(i,l)/weights[i] - kronecker(k,l)/weights[k])
        #            if i != l and k != l: d2[i]= 0.0
        #            else: d2[i] =  (self.EnsembleGPs[i].log_likelihood_gradient(hps[l]) * kronecker(i,l) - \
        #                            self.EnsembleGPs[k].log_likelihood_gradient(hps[l]) * kronecker(k,l))

        #       index = np.arange(self.number_of_GPs) != k
        #        s = np.sum(exp_a[index])
        #        if s > 1e16: term = 0.0
        #        else: term = 1./(1.+s)
        #        term2 = d[index].T @ exp_a[index]
        #        term3 =  exp_a[index].T @ d2[index]
        #        w_hess[k,l] = w_hess[l,k] = -((kronecker(k,l)/(weights[k]**2)) * term) \
        #                                    -((1.0/weights[k])*(term)*term2)
        #        h_hess.append(self.EnsembleGPs[k].log_likelihood_hessian(hps[k]) * kronecker(l,k) - term*s*self.EnsembleGPs[k].log_likelihood_hessian(hps[l]) * kronecker(l,k) \
        #                    -(np.outer(self.EnsembleGPs[k].log_likelihood_gradient(hps[k]),term3) * (term**2)))
                #print("cc: ",self.EnsembleGPs[k].log_likelihood_hessian(hps[k]), np.outer(self.EnsembleGPs[k].log_likelihood_gradient(hps[k]),term3))
                #print("cc ",self.EnsembleGPs[k].log_likelihood_hessian(hps[k]) * kronecker(l,k) - term*s*self.EnsembleGPs[k].log_likelihood_hessian(hps[k]) * kronecker(l,k),np.outer(self.EnsembleGPs[k].log_likelihood_gradient(hps[k]),term3) * (term**2))
        #return -w_hess, h_hess
    ##########################################################
    def compute_prior_pdf(self):
        for i in range(self.number_of_GPs):
            self.EnsembleGPs[i].compute_prior_fvGP_pdf()

    ##########################################################
    def posterior(self,x_iset, res = 100, lb = None, ub = None):
        means = [self.EnsembleGPs[i].posterior_mean(x_iset)["f(x)"] for i in range(self.number_of_GPs)]
        covs  = [self.EnsembleGPs[i].posterior_covariance(x_iset)["v(x)"] for i in range(self.number_of_GPs)]
        means = np.array(means)
        covs = np.array(covs)
        if lb == None: lb = np.min(means - 3.0 * np.sqrt(covs))
        if ub == None: ub = np.max(means + 3.0 * np.sqrt(covs))

        pdfs = []
        for i in range(len(x_iset)):
            pdf = np.zeros((res))
            for j in range(self.number_of_GPs):
                pdf += self.hps_obj.weights[j] * self._Gaussian(means[j,i],covs[j,i],lb,ub, res)
            pdfs.append(pdf)
        return {"f(x)": means, "v(x)":covs, "pdf": pdfs, "lb": lb, "ub": ub, "domain" : np.linspace(lb,ub,res)}
    ##########################################################
    def _Gaussian(self,mean,var,lower,upper,res):
        x = np.linspace(lower,upper,res)
        return np.exp(-np.power(x - mean, 2.) / (2. * np.power(var, 2.)))


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



