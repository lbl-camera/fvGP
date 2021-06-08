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
        except: print("No asynchronous training to be cancelled in Ensemble fvGP, no training is running.")

    def kill_training(self):
        print("fvGP is killing asynchronous training....")
        try: self.opt.kill(); print("fvGP successfully killed the training.")
        except: print("No asynchronous training to be killed, no training is running.")

    def train_async(self,
        hps_bounds,
        hps_obj = None,
        pop_size = 20,
        tolerance = 0.1,
        max_iter = 120,
        dask_client = None
        ):
        if hps_obj is None: hps_obj = self.hps_obj
        self.optimize_log_likelihood_async(
            hps_obj,
            hps_bounds,
            max_iter,
            pop_size,
            tolerance,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client
            )
    
    def train(self,
        init_hps_obj = None,
        pop_size = 20,
        tolerance = 0.1,
        max_iter = 120,
        dask_client = None
        ):
        if init_hps_obj is None: init_hps_obj = self.hps_obj
        weights, hps = self.optimize_log_likelihood(
            init_hps_obj,
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
            res = self.opt.get_latest(n)
            self.hyperparameters = res["x"][0]
            self.compute_prior_pdf()
            print("Ensemble fvGP async hyperparameter update successful")
            print("Latest hyperparameters: ", self.hyperparameters)
        except:
            print("Async Hyper-parameter update not successful in Ensemble fvGP. I am keeping the old ones.")
            print("That probbaly means you are not optimizing them asynchronously")
            print("hyperparameters: ", self.hyperparameters)
        return self.hyperparameters

    def optimize_log_likelihood_async(self, x0):
        print("Ensemble fvGP submitted to HGDL optimization")
        print('bounds are',hp_bounds)

        self.opt = HGDL(self.log_likelihood,
                self.log_likelihood_gradient,
                #hess = self.log_likelihood_hessian,
                bounds = hp_bounds,
                num_epochs = max_iter)

        self.opt.optimize(dask_client = dask_client, x0 = x0)

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
        nlc = NonlinearConstraint(constraint,0.90,1.0)

        res = differential_evolution(
            self.ensemble_log_likelihood,
            hps_obj.vectorized_bounds,
            disp=True,
            maxiter=max_iter,
            popsize = pop_size,
            tol = tolerance,
            workers = 1,
            constraints = (nlc),
            polish = False
        )
        v = np.array(res["x"])
        Eval = self.ensemble_log_likelihood(v)
        weights,hps = self.hps_obj.devectorize_hps(v)
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
        L = 0.0
        weights, hps = self.hps_obj.devectorize_hps(v)
        phi_0 = np.log(weights[0]) + self.EnsembleGPs[0].log_likelihood(hps[0])
        for i in range(1,self.number_of_GPs):
            phi_i = np.log(weights[i]) + self.EnsembleGPs[i].log_likelihood(hps[i])
            L += np.exp(phi_i - phi_0)
        l = np.log(1.0 + L)
        return phi_0 + l

    def ensemble_log_likelihood_gradient(self,hyperparameters):
        weights, hps = self.hps_obj.devectorize_hps(v)
        A = 0.0
        dA_dw0 = 0.0
        B = np.empty((self.number_og_GPs))
        B[0] = np.log(weights[0]) + self.EnsembleGPs[0].log_likelihood(hps[0])
        dA_dw[0] = B[0]/weights[0]
        dA_dPsi0 = 0.0
        for i in range(1,self.number_of_GPs):
            B[i] = np.log(weights[i]) + self.EnsembleGPs[i].log_likelihood(hps[i])
            A += np.exp(B[i] - B[0])
            dA_dw0 += -B[i]/weights[0]
            dA_dw[i] = B[i]/weights[i]
            dA_dPsi0 += -B[i]/self.EnsembleGPs[0].log_likelihood(hps[0])
            dA_dPsii[i] = B[i]/self.EnsembleGPs[0].log_likelihood(hps[0])
        

        dL_dw0 = 1./weights[0] * dA_dw0/A
        dL_dwi = dA_dw[i]/A
        dL_dh0 = 1./self.EnsembleGPs[0].log_likelihood(hps[0]) + (dA_dPsi0*self.EnsembleGPs[0].log_likelihood_gradient(hps[0]))/A
        dL_dhi = (dA_dPsii*self.EnsembleGPs[i].log_likelihood_gradient(hps[i]))/A

        return gr

    def ensemble_log_likelihood_hessian(self,hyperparameters):
        return 0
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



