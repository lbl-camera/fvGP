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
from scipy.optimize import LinearConstraint,differential_evolution
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from random import choice

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
        number_of_G_likelihoods,
        hps_obj,
        variances = None,
        weights_d = None,
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
        self.input_space_dim = input_space_dim
        self.number_of_GPs = number_of_GPs
        self.hps_obj = hps_obj
        self.number_of_G_likelihoods = number_of_G_likelihoods
        self.init_weights = np.ones((number_of_GPs)) / float(number_of_GPs)
        self.points = points
        self.sparse = sparse
        self.normalize_y = normalize_y
        self.compute_device = compute_device
        if values.ndim == 1: values = values.reshape(-1,1)
        if len(values[0]) == 1: values = np.tile(values,(1,number_of_GPs))
        self.values = values
        if variances is None: variances = np.zeros(values.shape) + abs(np.mean(values)/100.)
        if variances.ndim == 1: variances = variances.reshape(-1,1)
        if len(variances[0]) == 1: variances = np.tile(variances,(1,number_of_GPs))
        self.variances = variances

        if weights_d is None: weights_d = np.zeros(values.shape) + 1./float(number_of_GPs)
        if weights_d.ndim == 1: weights_d = weights_d.reshape(-1,1)
        if len(weights_d[0]) == 1: weights_d = np.zeros(values.shape) + 1./float(number_of_GPs)
        self.weights_d = weights_d

        if gp_kernel_functions is None: gp_kernel_functions = [None] * number_of_GPs
        self.gp_kernel_functions = gp_kernel_functions
        if gp_mean_functions is None: gp_mean_functions = [None] * number_of_GPs
        self.gp_mean_functions = gp_mean_functions


    ############################################################################
    ############################################################################
    ########################finding G###########################################
    ############################################################################
    ############################################################################
    ############################################################################
    def find_set_G(self):
        ###returns |G| mean vectors (y), covariance matrices and weights w^y
        ln_w = np.log(self.weights_d)
        max_ind = np.argmax(ln_w, axis = 1)   ###a 1 d array (len(weights)) of the largest index
        new = [np.zeros((len(max_ind)))] * self.number_of_G_likelihoods
        res_weights = []
        res_v = []
        res_y = []
        W = []
        for i in range(self.number_of_G_likelihoods):
            sgn, logdet = np.linalg.slogdet(np.diag(self.weights_d[np.arange(self.weights_d.shape[0])[:, None], max_ind.reshape(-1,1)][:,0]))
            res_weights.append(sgn * logdet)
            res_y.append(self.values[np.arange(self.values.shape[0])[:, None], max_ind.reshape(-1,1)][:,0])
            res_v.append(self.variances[np.arange(self.variances.shape[0])[:, None], max_ind.reshape(-1,1)][:,0])
            change_ind = np.random.randint(0, high=len(max_ind)-1, size = 1, dtype=int)
            n = [i for i in range(len(ln_w[0]))]
            n.remove(max_ind[change_ind])
            max_ind[change_ind] = choice(n)
        res_weights = np.asarray(res_weights)
        index = np.argmax(res_weights)
        i = np.arange(len(res_weights)) != index
        Z = res_weights[index] + np.log(1.+np.sum(np.exp(res_weights[i]-res_weights[index])))
        res_weights = np.exp(res_weights-Z)
        print("final res weights", res_weights)
        print("with sum: ", np.sum(res_weights))
        return np.asarray(res_y), res_weights, np.asarray(res_v)


    def data_generating_potential(self,vec, plot = False):
        """
        definition: data_generating_potential(self,vec)
        input: vec containing |D| * |G| means and covariances
        output:
        """
        D = len(self.points)
        G = self.number_of_G_likelihoods

        weights = vec[0:G]/np.sum(vec[0:G])
        means = vec[G:G + (G * D)].reshape(D,G)
        variances = vec[G + (G * D):].reshape(D,G)

        res = 1000
        norm = 0
        y_lim = np.array([np.min(means,axis = 1) - 3. * np.sqrt(np.max(variances,axis = 1)),np.max(means,axis = 1) + 3. * np.sqrt(np.max(variances,axis = 1))]).T
        y = np.linspace(y_lim[:,0],y_lim[:,1],res).T
        dx = (y_lim[:,1] - y_lim[:,0]) / float(res)
        for i in range(D):
            Psi1 = np.subtract.outer(y[i],means[i])
            Psi1 = np.exp(-0.5*(Psi1**2/variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * variances[np.newaxis,i,:])
            Psi1 = weights[np.newaxis,:] * Psi1
            g = np.sum(Psi1,axis = 1)

            Psi2 = np.subtract.outer(y[i],self.values[i])
            Psi2 = np.exp(-0.5*(Psi2**2/self.variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * self.variances[np.newaxis,i,:])
            Psi2 = self.weights_d[np.newaxis,i,:] * Psi2
            d = np.sum(Psi2,axis = 1)
            norm += np.linalg.norm(g-d) * dx[i]
            if plot  is True:
                plt.plot(y[i],g, label = "g")
                plt.plot(y[i],d, label = "data")
                plt.legend()
                plt.show()
            #input()

        return norm

    def data_generating_potential2(self,vec, plot = False):
        """
        definition: data_generating_potential(self,vec)
        input: vec containing |D| * |G| means and covariances
        output:
        """
        D = len(self.points)
        G = self.number_of_G_likelihoods

        weights = vec[0:G]/np.sum(vec[0:G])
        means = vec[G:G + (G * D)].reshape(D,G)
        variances = vec[G + (G * D):].reshape(D,G)

        res = 1000
        norm = 0
        y_lim = np.array([np.min(means,axis = 1) - 3. * np.sqrt(np.max(variances,axis = 1)),np.max(means,axis = 1) + 3. * np.sqrt(np.max(variances,axis = 1))]).T
        y = np.linspace(y_lim[:,0],y_lim[:,1],res).T
        dx = (y_lim[:,1] - y_lim[:,0]) / float(res)

        Psi1 = 1.
        g = np.exp(-0.5*(Psi1**2/variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * variances[np.newaxis,i,:])

        #for i in range(D):
        #    Psi1 = np.subtract.outer(y[i],means[i])
        #    Psi1 = np.exp(-0.5*(Psi1**2/variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * variances[np.newaxis,i,:])
        #    Psi1 = weights[np.newaxis,:] * Psi1
        #    g = np.sum(Psi1,axis = 1)

        #    Psi2 = np.subtract.outer(y[i],self.values[i])
        #    Psi2 = np.exp(-0.5*(Psi2**2/self.variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * self.variances[np.newaxis,i,:])
        #    Psi2 = self.weights_d[np.newaxis,i,:] * Psi2
        #    d = np.sum(Psi2,axis = 1)
        #    norm += np.linalg.norm(g-d) * dx[i]
        #    if plot  is True:
        #        plt.plot(y[i],g, label = "g")
        #        plt.plot(y[i],d, label = "data")
        #        plt.legend()
        #        plt.show()
        #    #input()

        return norm


    def data_generating_potential_gradient(self,vec):
        """
        definition: data_generating_potential(self,vec)
        input: vec containing |D| * |G| means and covariances
        output:
        """
        D = len(self.points)
        G = self.number_of_G_likelihoods

        weights = vec[0:G]/np.sum(vec[0:G])
        means = vec[G:G + (G * D)].reshape(D,G)
        variances = vec[G + (G * D):].reshape(D,G)

        res = 1000
        norm = 0
        y_lim = [np.min(means,axis = 1) - 3. * np.sqrt(np.max(variances,axis = 1)),np.max(means,axis = 1) + 3. * np.sqrt(np.max(variances,axis = 1))]
        y = np.linspace(y_lim[0],y_lim[1],res)
        dx = (y_lim[1] - y_lim[0]) / float(res)
        for i in range(D):
            Psi1 = np.subtract.outer(y[:,i],means[i])
            Psi1 = np.exp(-0.5*(Psi1**2/variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * variances[np.newaxis,i,:])
            Psi1 = weights[np.newaxis,:] * Psi1
            g = np.sum(Psi1,axis = 1)

            Psi2 = np.subtract.outer(y[:,i],self.values[i])
            Psi2 = np.exp(-0.5*(Psi2**2/self.variances[np.newaxis,i,:]))/np.sqrt(2.* np.pi * self.variances[np.newaxis,i,:])
            Psi2 = self.weights_d[np.newaxis,i,:] * Psi2
            d = np.sum(Psi2,axis = 1)
            norm += np.linalg.norm(g-d) * dx[i]

        return norm



    def optimize_set_G(self,vec, bounds, popsize = 10, maxiter = 10, tol = 0.001):
        def constraint(v):
            return np.array(np.sum(v[0:self.number_of_g_likelihoods]))

        x0 = np.array(vec)
        constraint = np.zeros((len(vec),len(vec)))
        constraint[0:self.number_of_G_likelihoods, 0:self.number_of_G_likelihoods] = np.identity(self.number_of_G_likelihoods)
        nlc = LinearConstraint(constraint,0.9,1.0)
        G = self.number_of_G_likelihoods
        D = len(self.points)
        #res = minimize(
        #        self.data_generating_potential,x0,
        #        method= "SLSQP",
        #        bounds = bounds,
        #        tol = 1e-6,
        #        callback = None,
        #        options = {"disp" : True},
        #        constraints = (nlc))
        res = differential_evolution(self.data_generating_potential, 
                bounds, args=(), strategy='best1bin', maxiter=maxiter,
                popsize=popsize, tol=tol, mutation=(0.5, 1), 
                recombination=0.7,
                seed=None, callback=None, disp=True, polish=True, 
                init='latinhypercube', atol=0,
                updating='immediate', workers=1, constraints=(), x0=None)
        r = np.array(res["x"])

        print("result")
        print(r)
        print("------------")

        weights = r[0:G]/np.sum(r[0:G])
        means = r[G:G + (G * D)].reshape(G,D).T
        variances = r[G + (G * D):].reshape(G,D).T



        return weights, means, variances



    ############################################################################
    ############################################################################
    ############################################################################
    ####################enGP init###############################################
    ############################################################################
    ############################################################################

    def init_GPs(self, weights, vals, vs):
        self.EnsembleGPs = np.empty((self.number_of_G_likelihoods,self.number_of_GPs), dtype = object)
        for i in range(self.number_of_G_likelihoods):
            for j in range(self.number_of_GPs):
                self.EnsembleGPs[i,j] = GP(self.input_space_dim,self.points,vals[:,i],self.hps_obj.hps[j],
                             variances = vs[:,i],compute_device = self.compute_device,
                               gp_kernel_function = self.gp_kernel_functions[j],
                               gp_mean_function = self.gp_mean_functions[j],
                               sparse = self.sparse, normalize_y = self.normalize_y)
        self.weights_y = weights
        self.means_y = vals
        self.variances_y = vs

        print("All GPs in the ensemble GP successfully initalized")
        print(self.EnsembleGPs)
        print("==================================================")
    
    #def update_EnsembleGP_data(self, points, values, variances = None, weights_d = None):
    #    if values.ndim == 1: values = values.reshape(-1,1)
    #    if len(values[0]) == 1: values = np.tile(values,(1,number_of_GPs))
    #    if variances is None: variances = np.zeros(values.shape) + abs(np.mean(values)/100.)
    #    if variances.ndim == 1: variances = variances.reshape(-1,1)
    #    if len(variances[0]) == 1: variances = np.tile(variances,(1,number_of_GPs))
    #    if weights_d is None: weights_d = np.zeros(values.shape) + 1./float(number_of_GPs)
    #    if weights_d.ndim == 1: weights_d = weights_d.reshape(-1,1)
    #    if len(weights_d[0]) == 1: weights_d = np.zeros(values.shape) + 1./float(number_of_GPs)
    #    self.weights_d = weights_d
        
    #    #####find new likelihood here
    #    raise Exception("not implemented yet")
    #    #####update GPs
    #    for i in range(self.number_of_G_likelihoods):
    #        for j in range(self.number_of_GPs):
    #            self.EnsembleGPs[i,j].update_gp_data(points,values[:,i], variances = variances[:,i])

    ############################################################################
    ############################################################################
    ############################################################################
    ##########################Training##########################################
    ############################################################################
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

        weights = weights/np.sum(weights)
        self.hps_obj.set(weights,hps)
        print("new weights after training: ", self.hps_obj.weights)
        print("new hps     after training: ", self.hps_obj.hps)
        for i in range(self.number_of_G_likelihoods):
            for j in range(self.number_of_GPs):
                self.EnsembleGPs[i,j].hyperparameters = self.hps_obj.hps[j]
        self.compute_prior_pdf()
        print("GPs updated")


    def update_hyperparameters(self, n = 1):
        try:
            r = self.opt.get_latest(n)['x'][0]
            weights,hps = self.hps_obj.devectorize_hps(r)
            self.hps_obj.set(weights,hps)
            print("new weights after training: ", self.hps_obj.weights)
            print("new hps     after training: ", self.hps_obj.hps)
            print("HAVE TO UPDATE GPs")
            #for i in range(self.number_of_GPs): self.EnsembleGPs[i].hyperparameters = self.hps_obj.hps[i]
            self.compute_prior_pdf()
            print("Ensemble fvGP async hyperparameter update successful")
        except Exception as e:
            print("Async Hyper-parameter update not successful in Ensemble fvGP. I am keeping the old ones.")
            print("That probably means you are not optimizing them asynchronously")
            print("Or there are simply no results yet.")
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
        print("initial hps: ", hps_obj.vectorized_hps)
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
                constr = ())

        self.opt.optimize(dask_client = dask_client, x0 = hps_obj.vectorized_hps)

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
        #def constraint(v):
        #    return np.array(np.sum(v[0:self.number_of_GPs]))
        #def constraint2(v):
        #    return np.array(v[0])

        #nlc = NonlinearConstraint(constraint,0.8,1.0)
        #nlc2 = NonlinearConstraint(constraint2,0.2,0.4)
        res = differential_evolution(
            self.ensemble_log_likelihood,
            hps_obj.vectorized_bounds,
            disp=True,
            maxiter=max_iter,
            popsize = pop_size,
            tol = tolerance,
            workers = 1,
            constraints = (),
            polish = False
        )

        r = np.array(res["x"])
        #r[0:self.number_of_GPs] = 1./self.number_of_GPs
        #nlc = NonlinearConstraint(constraint,0.9999,1.0)
        #res = minimize(
        #        self.ensemble_log_likelihood,r,
        #        method= "SLSQP",
        #        jac=self.ensemble_log_likelihood_grad,
        #        bounds = hps_obj.vectorized_bounds,
        #        tol = tolerance,
        #        callback = None,
        #        options = {"maxiter": max_iter},
        #        constraints = ())
        #r = np.array(res["x"])

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
        weights = weights/np.sum(weights)
        A = np.empty((self.number_of_G_likelihoods,self.number_of_GPs))
        #print("|G| :  ",self.number_of_G_likelihoods)
        #print("|GPs|: ",self.number_of_GPs)
        #print("len(weights GP):", len(weights))
        #print("len(w_y):       ", len(self.weights_y))
        #print("hps: ",hps)
        #print("enGPs: ", self.EnsembleGPs)
        for i in range(self.number_of_G_likelihoods):
            for j in range(self.number_of_GPs):
                A[i,j] = np.log(weights[j] * self.weights_y[i]) - self.EnsembleGPs[i,j].log_likelihood(hps[j])

        A = A.flatten()
        k = np.argmax(A)
        A_largest = A[k]
        indices = np.arange(len(A)) != k
        A = A - A_largest
        L = np.sum(np.exp(A[indices]))
        return -(np.log(1./np.sum(weights)) + A_largest + np.log(1.0 + L))

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

            w_grad[p] = -((-1./np.sum(weights)) + kronecker(k,p)/weights[p] + (s2/(1. + s1)))
            h_grad.append((kronecker(k,p) + s3/(1. + s1)) * self.EnsembleGPs[p].log_likelihood_gradient(hps[p]))
        return self.hps_obj.vectorize_hps(w_grad,h_grad)

    def ensemble_log_likelihood_hess(self,v):
        len_hyperparameters = len(v)
        d2L_dmdh = np.zeros((len_hyperparameters,len_hyperparameters))
        epsilon = 1e-5
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
        for i in range(self.number_of_G_likelihoods):
            for j in range(self.number_of_GPs):
                self.EnsembleGPs[i,j].compute_prior_fvGP_pdf()
    ############################################################################
    ############################################################################
    ############################################################################
    ##########################Posterior#########################################
    ############################################################################
    ############################################################################
    def posterior(self,x_iset, res = 100, lb = None, ub = None):
        means = np.zeros((self.number_of_G_likelihoods,self.number_of_GPs,len(x_iset)))
        covs = np.zeros((self.number_of_G_likelihoods,self.number_of_GPs,len(x_iset)))
        for i in range(self.number_of_G_likelihoods):
            for j in range(self.number_of_GPs):
                means[i,j,:] = self.EnsembleGPs[i,j].posterior_mean(x_iset)["f(x)"]
                covs[i,j,:]  = self.EnsembleGPs[i,j].posterior_covariance(x_iset)["v(x)"]
                #try: plt.plot(np.linspace(-3,3,res),means[i,j,:])
                #try: plt.errorbar(self.EnsembleGPs[i,j].data_x,self.EnsembleGPs[i,j].data_y, yerr=self.EnsembleGPs[i,j].variances, fmt="o", color = 'black')
        #try: plt.show()
        means = np.array(means)
        covs = np.array(covs)
        if lb == None: lb = np.min(means - 3.0 * np.sqrt(covs))
        if ub == None: ub = np.max(means + 3.0 * np.sqrt(covs))

        pdfs = []
        for i in range(len(x_iset)):
            pdf = np.zeros((res))
            for j in range(self.number_of_G_likelihoods):
                for k in range(self.number_of_GPs):
                    pdf += self.hps_obj.weights[k] * self._Gaussian(means[j,k,i],covs[j,k,i],lb,ub, res)
            pdfs.append(pdf/(np.sum(pdf)*((ub-lb)/float(res))))
        return {"f(x)": means, "v(x)":covs, "pdf": pdfs, "lb": lb, "ub": ub, "domain" : np.linspace(lb,ub,res)}
    ##########################################################

    def _Gaussian(self,mean,var,lower,upper,res):
        x = np.linspace(lower,upper,res)
        if var < 1e-6: print("CAUTION: var <= 0: ", var)
        return np.exp(-np.power(x - mean, 2.) / (2. * var))

    ############################################################################
    ############################################################################
    ############################################################################
    ##########################HP Class##########################################
    ############################################################################
    ############################################################################

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



