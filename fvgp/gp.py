#!/usr/bin/env python

import dask.distributed as distributed
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
along with this program. If not, see <https://www.gnu.org/licenses/>.

Contact: MarcusNoack@lbl.gov
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from .mcmc import mcmc

import itertools
import time
import torch
import numba as nb
from functools import partial
from hgdl.hgdl import HGDL


class GP():
    """
    GP class: Provides all tool for a single-task GP.

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space

    Attributes:
        input_space_dim (int):         dim1
        points (N x dim1 numpy array): 2d numpy array of points
        values (N dim numpy array):    2d numpy array of values
        init_hyperparameters:          1d numpy array (>0)

    Optional Attributes:
        variances (N dim numpy array):                  variances of the values, default = array of shape of points
                                                        with 1 % of the values
        compute_device:                                 cpu/gpu, default = cpu
        gp_kernel_function(callable):                   None/function defining the 
                                                        kernel def name(x1,x2,hyperparameters,self), 
                                                        make sure to return a 2d numpy array, default = None uses default kernel
        gp_mean_function(callable):                     None/function def name(gp_obj, x, hyperparameters), 
                                                        make sure to return a 1d numpy array, default = None
        sparse (bool):                                  default = False
        normalize_y:                                    default = False, normalizes the values \in [0,1]

    Example:
        obj = fvGP(3,np.array([[1,2,3],[4,5,6]]),
                         np.array([2,4]),
                         np.array([2,3,4,5]),
                         variances = np.array([0.01,0.02]),
                         gp_kernel_function = kernel_function,
                         gp_mean_function = some_mean_function
        )
    """

    def __init__(
        self,
        input_space_dim,
        points,
        values,
        init_hyperparameters,
        variances = None,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_mean_function = None,
        sparse = False,
        normalize_y = False
        ):
        """
        The constructor for the gp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        if input_space_dim != len(points[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(values) == 2: values = values[:,0]

        self.normalize_y = normalize_y
        self.input_dim = input_space_dim
        self.data_x = np.array(points)
        self.point_number = len(self.data_x)
        self.data_y = np.array(values)
        self.compute_device = compute_device
        self.sparse = sparse
        if self.normalize_y is True: self._normalize_y_data()
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.data_y.shape)) * abs(self.data_y / 100.0)
            print("CAUTION: you have not provided data variances in fvGP,")
            print("they will be set to 1 percent of the data values!")
        elif np.ndim(variances) == 2:
            self.variances = variances[:,0]
        elif np.ndim(variances) == 1:
            self.variances = np.array(variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        if (self.variances < 0.0).any(): raise Exception("Negative measurement variances communicated to fvgp.")
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if gp_kernel_function is None:
            self.kernel = self.default_kernel
        else:
            self.kernel = gp_kernel_function
        self.d_kernel_dx = self.d_gp_kernel_dx

        self.gp_mean_function = gp_mean_function
        if gp_mean_function is None:
            self.mean_function = self.default_mean_function
        else:
            self.mean_function = gp_mean_function
        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = np.array(init_hyperparameters)
        ##########################################
        #compute the prior########################
        ##########################################
        self.compute_prior_fvGP_pdf()

    def update_gp_data(
        self,
        points,
        values,
        variances = None,
        ):

        """
        This function updates the data in the gp_class.
        The data will NOT be appended but overwritten!
        Please provide the full updated data set

        Attributes:
            points (N x dim1 numpy array): A 2d  array of points.
            values (N)                   : A 1d  array of values.

        Optional Attributes:
            variances (N)                : variances for the values
        """
        if self.input_dim != len(points[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(values) == 2: values = values[:,0]

        self.data_x = np.array(points)
        self.point_number = len(self.data_x)
        self.data_y = np.array(values)
        if self.normalize_y is True: self._normalize_y_data()
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.data_y.shape)) * abs(self.data_y / 100.0)
        elif np.ndim(variances) == 2:
            self.variances = variances[:,0]
        elif np.ndim(variances) == 1:
            self.variances = np.array(variances)
        else:
            raise Exception("Variances are not given in an allowed format. Give variances as 1d numpy array")
        if (self.variances < 0.0).any(): raise Exception("Negative measurement variances communicated to fvgp.")
        ######################################
        #####transform to index set###########
        ######################################
        self.compute_prior_fvGP_pdf()
        print("fvGP data updated")
    ###################################################################################
    ###################################################################################
    ###################################################################################
    #################TRAINING##########################################################
    ###################################################################################
    def stop_training(self,opt_obj):
        print("fvGP is cancelling the asynchronous training...")
        try: opt_obj.cancel_tasks(); print("fvGP successfully cancelled the current training.")
        except: print("No asynchronous training to be cancelled in fvGP, no training is running.")
    ###################################################################################
    def kill_training(self,opt_obj):
        print("fvGP is killing asynchronous training....")
        try: opt_obj.kill_client(); print("fvGP successfully killed the training.")
        except Exception as err: print("No asynchronous training to be killed, no training is running.", str(err))
    ###################################################################################
    def train(self,
        hyperparameter_bounds,
        init_hyperparameters = None,
        method = "global",
        optimization_dict = None,
        pop_size = 20,
        tolerance = 0.0001,
        max_iter = 120,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        deflation_radius = None,
        dask_client = None):
        """
        This function finds the maximum of the log_likelihood and therefore trains the fvGP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be be 'hgdl' and 
        providing a dask client

        inputs:
            hyperparameter_bounds (2d numpy array)
        optional inputs:
            init_hyperparameters (1d numpy array):  default = None (= use earlier initialization)
            method = "global": "global"/"local"/"hgdl"/callable f(obj,optimization_dict)
            optimization_dict = None: if optimizer is callable, the this will be passed as dict
            pop_size = 20
            tolerance = 0.0001
            max_iter: default = 120
            local_optimizer = "L-BFGS-B"  important for local and hgdl optimization
            global_optimizer = "genetic"
            deflation_radius = None        for hgdl
            dask_client = None (will use local client, only for hgdl optimization)

        output:
            None, just updates the class with the new hyperparameters
        """
        ############################################
        if init_hyperparameters is None:
            init_hyperparameters = np.array(self.hyperparameters)
        print("fvGP training started with ",len(self.data_x)," data points")
        ######################
        #####TRAINING#########
        ######################
        self.hyperparameters = self.optimize_log_likelihood(
            init_hyperparameters,
            np.array(hyperparameter_bounds),
            method,
            optimization_dict,
            max_iter,
            pop_size,
            tolerance,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client
            )
        self.compute_prior_fvGP_pdf()
        ######################
        ######################
        ######################
    ##################################################################################
    def train_async(self,
        hyperparameter_bounds,
        init_hyperparameters = None,
        max_iter = 120,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        deflation_radius = None,
        dask_client = None):
        """
        This function finds the maximum of the log_likelihood and therefore trains the 
        GP (aynchronously) using 'hgdl'.
        This can be done on a remote cluster/computer by providing the right dask client

        inputs:
            hyperparameter_bounds (2d list)
        optional inputs:
            init_hyperparameters (list):  default = None
            max_iter: default = 120,
            local_optimizer = "L-BFGS-B"
            global_optimizer = "genetic"
            deflation_radius = None
            dask_client: True/False/dask client, default = None (will use a local client)

        output:
            returns an optimization object that can later be queried for solutions
            stopped and killed.
        """
        ############################################
        if dask_client is None: dask_client = distributed.Client()
        if init_hyperparameters is None:
            init_hyperparameters = np.array(self.hyperparameters)
        print("Async fvGP training started with ",len(self.data_x)," data points")
        ######################
        #####TRAINING#########
        ######################
        opt_obj = self.optimize_log_likelihood_async(
            init_hyperparameters,
            np.array(hyperparameter_bounds),
            max_iter,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client
            )
        return opt_obj
        ######################
        ######################
        ######################
    ##################################################################################
    def update_hyperparameters(self, opt_obj):
        print("Updating the hyperparameters in fvGP...")
        try:
            res = opt_obj.get_latest(1)["x"][0]
            l_n = self.log_likelihood(res)
            l_o = self.log_likelihood(self.hyperparameters)
            if l_n - l_o < 0.000001:
                self.hyperparameters = res
                self.compute_prior_fvGP_pdf()
                print("    fvGP async hyperparameter update successful")
                print("    Latest hyperparameters: ", self.hyperparameters)
            else:
                print("    The update was attempted but the new hyperparameters led to a lower likelihood, so I kept the old ones")
                print("Old likelihood: ", -l_o, " at ", self.hyperparameters)
                print("New likelihood: ", -l_n, " at ", res)
        except Exception as e:
            print("    Async Hyper-parameter update not successful in fvGP. I am keeping the old ones.")
            print("    That probably means you are not optimizing them asynchronously")
            print("    Here is the actual reason: ", str(e))
            print("    hyperparameters: ", self.hyperparameters)
        return self.hyperparameters
    ##################################################################################
    def optimize_log_likelihood_async(self,
        starting_hps,
        hp_bounds,
        max_iter,
        local_optimizer,
        global_optimizer,
        deflation_radius,
        dask_client):
        print("fvGP submitted HGDL optimization for asynchronous training")
        print("bounds:",hp_bounds)
        print("deflation radius: ",deflation_radius)
        opt_obj = HGDL(self.log_likelihood,
                    self.log_likelihood_gradient,
                    hp_bounds,
                    hess = self.log_likelihood_hessian,
                    local_optimizer = local_optimizer,
                    global_optimizer = global_optimizer,
                    radius = deflation_radius,
                    num_epochs = max_iter)
        opt_obj.optimize(dask_client = dask_client, x0 = np.array(starting_hps))
        return opt_obj
    ##################################################################################
    def optimize_log_likelihood(self,starting_hps,
        hp_bounds,method,optimization_dict,max_iter,
        pop_size,tolerance,
        local_optimizer,
        global_optimizer,
        deflation_radius,
        dask_client = None):

        start_log_likelihood = self.log_likelihood(starting_hps)

        print(
            "fvGP hyperparameter tuning in progress. Old hyperparameters: ",
            starting_hps, " with old log likelihood: ", start_log_likelihood)
        print("method: ", method)

        ############################
        ####global optimization:##
        ############################
        if method == "global":
            print("fvGP is performing a global differential evolution algorithm to find the optimal hyperparameters.")
            print("maximum number of iterations: ", max_iter)
            print("termination tolerance: ", tolerance)
            print("bounds: ", hp_bounds)
            res = differential_evolution(
                self.log_likelihood,
                hp_bounds,
                disp=True,
                maxiter=max_iter,
                popsize = pop_size,
                tol = tolerance,
                workers = 1,
            )
            hyperparameters = np.array(res["x"])
            Eval = self.log_likelihood(hyperparameters)
            print("fvGP found hyperparameters ",hyperparameters," with likelihood ",
                Eval," via global optimization")
        ############################
        ####local optimization:#####
        ############################
        elif method == "local":
            hyperparameters = np.array(starting_hps)
            print("fvGP is performing a local update of the hyper parameters.")
            print("starting hyperparameters: ", hyperparameters)
            print("Attempting a BFGS optimization.")
            print("maximum number of iterations: ", max_iter)
            print("termination tolerance: ", tolerance)
            print("bounds: ", hp_bounds)
            OptimumEvaluation = minimize(
                self.log_likelihood,
                hyperparameters,
                method= local_optimizer,
                jac=self.log_likelihood_gradient,
                bounds = hp_bounds,
                tol = tolerance,
                callback = None,
                options = {"maxiter": max_iter})

            if OptimumEvaluation["success"] == True:
                print(
                    "fvGP local optimization successfully concluded with result: ",
                    OptimumEvaluation["fun"]," at ", OptimumEvaluation["x"]
                )
                hyperparameters = OptimumEvaluation["x"]
            else:
                print("fvGP local optimization not successful.")
        ############################
        ####hybrid optimization:####
        ############################
        elif method == "hgdl":
            print("fvGP submitted HGDL optimization")
            print('bounds are',hp_bounds)
            opt = HGDL(self.log_likelihood,
                       self.log_likelihood_gradient,
                       hp_bounds,
                       hess = self.log_likelihood_hessian,
                       local_optimizer = local_optimizer,
                       global_optimizer = global_optimizer,
                       radius = deflation_radius,
                       num_epochs = max_iter)

            obj = opt.optimize(dask_client = dask_client, x0 = np.array(starting_hps))
            res = opt.get_final(2)
            hyperparameters = res["x"][0]
            opt.kill_client(obj)
        elif method == "mcmc":
            print("MCMC started in fvGP")
            print('bounds are',hp_bounds)
            res = mcmc(self.log_likelihood,hp_bounds)
            hyperparameters = np.array(res["x"])
        elif callable(method):
            hyperparameters = method(self,optimization_dict)
        else:
            raise ValueError("No optimization mode specified in fvGP")
        ###################################################
        if start_log_likelihood < self.log_likelihood(hyperparameters):
            hyperparameters = np.array(starting_hps)
            print("fvGP: Optimization returned smaller log likelihood; resetting to old hyperparameters.")
            print("New hyperparameters: ",
            hyperparameters,
            "with log likelihood: ",
            self.log_likelihood(hyperparameters))
        return hyperparameters
    ##################################################################################
    def log_likelihood(self,hyperparameters):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        mean = self.mean_function(self,self.data_x,hyperparameters)
        if mean.ndim > 1: raise Exception("Your mean function did not return a 1d numpy array!")
        x,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        y = self.data_y - mean
        sign, logdet = self.slogdet(K)
        n = len(y)
        if sign == 0.0: return (0.5 * (y.T @ x)) + (0.5 * n * np.log(2.0*np.pi))
        return (0.5 * (y.T @ x)) + (0.5 * sign * logdet) + (0.5 * n * np.log(2.0*np.pi))
    ##################################################################################
    @staticmethod
    @nb.njit
    def numba_dL_dH(y, a, b, length):
        dL_dH = np.empty((length))
        for i in range(length):
                dL_dH[i] = 0.5 * ((y.T @ (a[i] @ b)) - (np.trace(a[i])))
        return dL_dH
    ##################################################################################
    def log_likelihood_gradient(self, hyperparameters):
        """
        computes the gradient of the negative marginal log-likelihood
        input:
            hyper parameters
        output:
            gradient of the negative marginal log-likelihood (vector)
        """
        from numpy.core.umath_tests import inner1d
        mean = self.mean_function(self,self.data_x,hyperparameters)
        b,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        y = self.data_y - mean
        dK_dH = self.gradient_gp_kernel(self.data_x,self.data_x, hyperparameters)
        K = np.array([K,] * len(hyperparameters))
        a = self.solve(K,dK_dH)
        bbT = np.outer(b , b.T)
        #dL_dH = self.numba_dL_dH(y, a, b, len(hyperparameters))
        dL_dH = np.zeros((len(hyperparameters)))
        dL_dHm = np.zeros((len(hyperparameters)))
        dm_dh = self.dm_dh(hyperparameters)
        for i in range(len(hyperparameters)):
            dL_dHm[i] = -dm_dh[i].T @ b
            if dL_dHm[i] == 0.0:
                mtrace = np.sum(inner1d(bbT , dK_dH[i]))
                dL_dH[i] = - 0.5 * (mtrace - np.trace(a[i]))
            else:
                dL_dH[i] = 0.0
        return dL_dH + dL_dHm

    ##################################################################################
    @staticmethod
    @nb.njit
    def numba_d2L_dH2(x, y, s, ss):
        len_hyperparameters = s.shape[0]
        d2L_dH2 = np.empty((len_hyperparameters,len_hyperparameters))
        for i in range(len_hyperparameters):
            x1 = s[i]
            for j in range(i+1):
                x2 = s[j]
                x3 = ss[i,j]
                f = 0.5 * ((y.T @ (-x2 @ x1 @ x - x1 @ x2 @ x + x3 @ x)) - np.trace(-x2 @ x1 + x3))
                d2L_dH2[i,j] = d2L_dH2[j,i] = f
        return d2L_dH2
    ##################################################################################
    def log_likelihood_hessian(self, hyperparameters):
        """
        computes the hessian of the negative  marginal  log-likelihood
        input:
            hyper parameters
        output:
            hessian of the negative marginal log-likelihood (matrix)
        """
        #raise Exception("Hessian not correct, please use the gradient to approximate the Hessian")
        #mean = self.mean_function(self,self.data_x,hyperparameters)
        #x,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        #y = self.data_y - mean
        #dK_dH = self.gradient_gp_kernel(self.data_x,self.data_x, hyperparameters)
        #d2K_dH2 = self.hessian_gp_kernel(self.data_x,self.data_x, hyperparameters)
        #K = np.array([K,] * len(hyperparameters))
        #s = self.solve(K,dK_dH)
        #ss = self.solve(K,d2K_dH2)
        # make contiguous
        #K = np.ascontiguousarray(K, dtype=np.float64)
        #y = np.ascontiguousarray(y, dtype=np.float64)
        #s = np.ascontiguousarray(s, dtype=np.float64)
        #ss = np.ascontiguousarray(ss, dtype=np.float64)
        #d2L_dH2 = self.numba_d2L_dH2(x, y, s, ss)
        len_hyperparameters = len(hyperparameters)
        #d2L_dH2 =  np.empty((len_hyperparameters,len_hyperparameters))
        #d2L_dm2 =  np.empty((len_hyperparameters,len_hyperparameters))
        #d2L_dmdh = np.empty((len_hyperparameters,len_hyperparameters))
        d2L_dmdh = np.zeros((len_hyperparameters,len_hyperparameters))
        #d2m_dh2 = self.d2m_dh2(hyperparameters)
        #m1 = self.dm_dh(hyperparameters)
        epsilon = 1e-6
        grad_at_hps = self.log_likelihood_gradient(hyperparameters)
        for i in range(len_hyperparameters):
            #x1 = s[i]
            hps_temp = np.array(hyperparameters)
            hps_temp[i] = hps_temp[i] + epsilon
            d2L_dmdh[i,i:] = ((self.log_likelihood_gradient(hps_temp) - grad_at_hps)/epsilon)[i:]
            #print(d2L_dmdh)
        #i_lower = np.tril_indices(n, -1)
        #matrix[i_lower] = matrix.T[i_lower]  # make the matrix symmetric
            #for j in range(i+1):
                #x2 = s[j]
                #x3 = ss[i,j]
                #f = 0.5 * ((y.T @ (-x2 @ x1 @ x - x1 @ x2 @ x + x3 @ x)) - np.trace(-x2 @ x1 + x3))
                #d2L_dH2[i,j] = d2L_dH2[j,i] = f
                #d2L_dm2[i,j] = d2L_dm2[j,i] = (m1[i,:].T @ np.linalg.inv(K[0]) @ m1[j,:]) - (x.T @ d2m_dh2[i,j])
                #d2L_dmdh[i,j] = d2L_dmdh[j,i] = (x @ x2 @ m1[i,:])
                #hps_temp1 = np.array(hyperparameters)
                #hps_temp2 = np.array(hyperparameters)
                #hps_temp1[i] = hps_temp1[i] + epsilon
                #hps_temp2[j] = hps_temp2[j] + epsilon
                #d2L_dmdh[i,j] = d2L_dmdh[j,i] = self.log_likelihood_gradient(hps_temp)

        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))
        #return -d2L_dH2 - d2L_dm2 + d2L_dmdh
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
    ##################################################################################
    ##################################################################################
    def compute_prior_fvGP_pdf(self):
        """
        This function computes the important entities, namely the prior covariance and
        its product with the (values - prior_mean) and returns them and the prior mean
        input:
            none
        return:
            prior mean
            prior covariance
            covariance value product
        """
        self.prior_mean_vec = self.mean_function(self,self.data_x,self.hyperparameters)
        cov_y,K = self._compute_covariance_value_product(
                self.hyperparameters,
                self.data_y,
                self.variances,
                self.prior_mean_vec)
        self.prior_covariance = K
        self.covariance_value_prod = cov_y
    ##################################################################################
    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean):
        K = self.compute_covariance(hyperparameters, variances)
        y = values - mean
        x = self.solve(K, y)
        if x.ndim == 2: x = x[:,0]
        return x,K
    ##################################################################################
    def compute_covariance(self, hyperparameters, variances):
        """computes the covariance matrix from the kernel"""
        CoVariance = self.kernel(
            self.data_x, self.data_x, hyperparameters, self)
        self.add_to_diag(CoVariance, variances)
        return CoVariance

    def slogdet(self, A):
        """
        fvGPs slogdet method based on torch
        """
        #s,l = np.linalg.slogdet(A)
        #return s,l
        if self.compute_device == "cpu":
            A = torch.from_numpy(A)
            sign, logdet = torch.slogdet(A)
            sign = sign.numpy()
            logdet = logdet.numpy()
            logdet = np.nan_to_num(logdet)
            return sign, logdet
        elif self.compute_device == "gpu" or self.compute_device == "multi-gpu":
            A = torch.from_numpy(A).cuda()
            sign, logdet = torch.slogdet(A)
            sign = sign.cpu().numpy()
            logdet = logdet.cpu().numpy()
            logdet = np.nan_to_num(logdet)
            return sign, logdet

    def solve(self, A, b):
        """
        fvGPs slogdet method based on torch
        """
        #x = np.linalg.solve(A,b)
        #return x
        if b.ndim == 1: b = np.expand_dims(b,axis = 1)
        if self.compute_device == "cpu":
            #####for sparsity:
            if self.sparse == True:
                zero_indices = np.where(A < 1e-16)
                A[zero_indices] = 0.0
                if self.is_sparse(A):
                    try:
                        A = scipy.sparse.csr_matrix(A)
                        x = scipy.sparse.spsolve(A,b)
                        return x
                    except Exceprion as e:
                        print("fvGP: Sparse solve did not work out.")
                        print("reason: ", str(e))
            ##################
            A = torch.from_numpy(A)
            b = torch.from_numpy(b)
            try:
                x = torch.linalg.solve(A,b)
                return x.numpy()
            except Exception as e:
                try:
                    print("fvGP: except statement invoked: torch.solve() on cpu did not work")
                    print("reason: ", str(e))
                    #x, qr = torch.lstsq(b,A)
                    x, qr = torch.linalg.lstsq(A,b)
                except Exception as e:
                    print("fvGP: except statement 2 invoked: torch.solve() and torch.lstsq() on cpu did not work")
                    print("falling back to numpy.lstsq()")
                    print("reason: ", str(e))
                    x,res,rank,s = np.linalg.lstsq(A.numpy(),b.numpy())
                    return x
            return x.numpy()
        elif self.compute_device == "gpu" or A.ndim < 3:
            A = torch.from_numpy(A).cuda()
            b = torch.from_numpy(b).cuda()
            try:
                x = torch.linalg.solve(A, b)
            except Exception as e:
                print("fvGP: except statement invoked: torch.solve() on gpu did not work")
                print("reason: ", str(e))
                try:
                    #x, qr = torch.lstsq(b,A)
                    x = torch.linalg.lstsq(A,b)
                except Exception as e:
                    print("fvGP: except statement 2 invoked: torch.solve() and torch.lstsq() on gpu did not work")
                    print("falling back to numpy.lstsq()")
                    print("reason: ", str(e))
                    x,res,rank,s = np.linalg.lstsq(A.numpy(),b.numpy())
                    return x
            return x.cpu().numpy()
        elif self.compute_device == "multi-gpu":
            n = min(len(A), torch.cuda.device_count())
            split_A = np.array_split(A,n)
            split_b = np.array_split(b,n)
            results = []
            for i, (tmp_A,tmp_b) in enumerate(zip(split_A,split_b)):
                cur_device = torch.device("cuda:"+str(i))
                tmp_A = torch.from_numpy(tmp_A).cuda(cur_device)
                tmp_b = torch.from_numpy(tmp_b).cuda(cur_device)
                results.append(torch.linalg.solve(tmp_A,tmp_b)[0])
            total = results[0].cpu().numpy()
            for i in range(1,len(results)):
                total = np.append(total, results[i].cpu().numpy(), 0)
            return total
    ##################################################################################
    def add_to_diag(self,Matrix, Vector):
        d = np.einsum("ii->i", Matrix)
        d += Vector
        return Matrix
    def is_sparse(self,A):
        if float(np.count_nonzero(A))/float(len(A)**2) < 0.01: return True
        else: return False
    def how_sparse_is(self,A):
        return float(np.count_nonzero(A))/float(len(A)**2)
    def default_mean_function(self,gp_obj,x,hyperparameters):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)))
        mean[:] = np.mean(self.data_y)
        return mean
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###############################gp prediction###############################
    ###########################################################################
    ###########################################################################
    def posterior_mean(self, x_iset):
        """
        function to compute the posterior mean
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x":    the input points,
             "f(x)": the posterior mean vector (1d numpy array)}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        A = k.T @ self.covariance_value_prod
        posterior_mean = self.mean_function(self,p,self.hyperparameters) + A
        return {"x": p,
                "f(x)": posterior_mean}

    def posterior_mean_grad(self, x_iset, direction):
        """
        function to compute the gradient of the posterior mean in
        a specified direction
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            direction: direction in which to compute the gradient
        output:
        -------
            {"x":    the input points,
             "direction": the direction
             "df/dx": the gradient of the posterior mean vector (1d numpy array)}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        x1 = np.array(p)
        x2 = np.array(p)
        eps = 1e-6
        x1[:,direction] = x1[:,direction] + eps
        x2[:,direction] = x2[:,direction] - eps
        mean_der = (self.mean_function(self,x1,self.hyperparameters) - self.mean_function(self,x2,self.hyperparameters))/(2.0*eps)
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        k_g = self.d_kernel_dx(p,self.data_x, direction,self.hyperparameters)
        posterior_mean_grad = mean_der + (k_g @ self.covariance_value_prod)
        return {"x": p,
                "direction":direction,
                "df/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_iset):
        """
        function to compute the posterior covariance
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x":    the index set points,
             "v(x)": the posterior variances (1d numpy array) for each input point,
             "S":    covariance matrix, v(x) = diag(S)}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        k_cov_prod = self.solve(self.prior_covariance,k)
        a = kk - (k_cov_prod.T @ k)
        diag = np.diag(a)
        diag = np.where(diag<0.0,0.0,diag)
        if any([x < -0.001 for x in np.diag(a)]):
            print("In fvGP: CAUTION, negative variances encountered. That normally means that the model is unstable.")
            print("Rethink the kernel definitions, add more noise to the data,")
            print("or double check the hyperparameter optimization bounds. This will not ")
            print("terminate the algorithm, but expect anomalies.")
            print("diagonal of the posterior covariance: ",np.diag(a))
            p = np.block([[self.prior_covariance, k],[k.T, kk]])
            print("eigenvalues of the prior: ", np.linalg.eig(p)[0])

        np.fill_diagonal(a,diag)
        return {"x": p,
                "v(x)": np.diag(a),
                "S(x)": a}

    def posterior_covariance_grad(self, x_iset,direction):
        """
        function to compute the gradient of the posterior covariance
        in a specified direction
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            direction: direction in which the gradient to compute
        output:
        -------
            {"x":    the index set points,
             "dv/dx": the posterior variances (1d numpy array) for each input point,
             "dS/dx":    covariance matrix, v(x) = diag(S)}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        k_g = self.d_kernel_dx(p,self.data_x, direction,self.hyperparameters).T
        kk =  self.kernel(p, p,self.hyperparameters,self)
        x1 = np.array(p)
        x2 = np.array(p)
        eps = 1e-6
        x1[:,direction] = x1[:,direction] + eps
        x2[:,direction] = x2[:,direction] - eps
        kk_g = (self.kernel(x1, x1,self.hyperparameters,self)-self.kernel(x2, x2,self.hyperparameters,self)) /(2.0*eps)
        k_covariance_prod = self.solve(self.prior_covariance,k)
        k_g_covariance_prod = self.solve(self.prior_covariance,k_g)
        a = kk_g - ((k_covariance_prod.T @ k_g) + (k_g_covariance_prod.T @ k))
        return {"x": p,
                "dv/dx": np.diag(a),
                "dS/dx": a}

    ###########################################################################
    def gp_prior(self, x_iset):
        """
        function to compute the data-informed prior
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
        output:
        -------
            {"x": the index set points,
             "K": covariance matrix between data points
             "k": covariance between data and requested poins,
             "kappa": covariance matrix between requested points,
             "prior mean": the mean of the prior
             "S:": joint prior covariance}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        post_mean = self.mean_function(self,p, self.hyperparameters)
        full_gp_prior_mean = np.append(self.prior_mean_vec, post_mean)
        return  {"x": p,
                 "K": self.prior_covariance,
                 "k": k,
                 "kappa": kk,
                 "prior mean": full_gp_prior_mean,
                 "S(x)": np.block([[self.prior_covariance, k],[k.T, kk]])}
    ###########################################################################
    def gp_prior_grad(self, x_iset,direction):
        """
        function to compute the gradient of the data-informed prior
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            direction: direction in which to compute the gradient
        output:
        -------
            {"x": the index set points,
             "K": covariance matrix between data points
             "k": covariance between data and requested poins,
             "kappa": covariance matrix between requested points,
             "prior mean": the mean of the prior
             "dS/dx:": joint prior covariance}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        k_g = self.d_kernel_dx(p,self.data_x, direction,self.hyperparameters).T
        x1 = np.array(p)
        x2 = np.array(p)
        eps = 1e-6
        x1[:,direction] = x1[:,direction] + eps
        x2[:,direction] = x2[:,direction] - eps
        kk_g = (self.kernel(x1, x1,self.hyperparameters,self)-self.kernel(x2, x2,self.hyperparameters,self)) /(2.0*eps)
        post_mean = self.mean_function(self,p, self.hyperparameters)
        mean_der = (self.mean_function(self,x1,self.hyperparameters) - self.mean_function(self,x2,self.hyperparameters))/(2.0*eps)
        full_gp_prior_mean_grad = np.append(np.zeros((self.prior_mean_vec.shape)), mean_der)
        prior_cov_grad = np.zeros(self.prior_covariance.shape)
        return  {"x": p,
                 "K": self.prior_covariance,
                 "dk/dx": k_g,
                 "d kappa/dx": kk_g,
                 "d prior mean/x": full_gp_prior_mean_grad,
                 "dS/dx": np.block([[prior_cov_grad, k_g],[k_g.T, kk_g]])}

    ###########################################################################

    def entropy(self, S):
        """
        function comuting the entropy of a normal distribution
        res = entropy(S); S is a 2d numpy array, matrix has to be non-singular
        """
        dim  = len(S[0])
        s, logdet = self.slogdet(S)
        return (float(dim)/2.0) +  ((float(dim)/2.0) * np.log(2.0 * np.pi)) + (0.5 * s * logdet)
    ###########################################################################
    def gp_entropy(self, x_iset):
        """
        function to compute the entropy of the data-informed prior
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
        output:
        -------
            scalar: entropy
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        priors = self.gp_prior(p)
        S = priors["S(x)"]
        dim  = len(S[0])
        s, logdet = self.slogdet(S)
        return (float(dim)/2.0) +  ((float(dim)/2.0) * np.log(2.0 * np.pi)) + (0.5 * s * logdet)
    ###########################################################################
    def gp_entropy_grad(self, x_iset,direction):
        """
        function to compute the gradient of the entropy of the data-informed prior
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the
                    index set which results from a cartesian product of input and output space
            direction: direction in which to compute the gradient
        output:
        -------
            scalar: entropy gradient
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        priors1 = self.gp_prior(p)
        priors2 = self.gp_prior_grad(p,direction)
        S1 = priors1["S(x)"]
        S2 = priors2["dS/dx"]
        return 0.5 * np.trace(np.linalg.inv(S1) @ S2)
    ###########################################################################
    def kl_div(self,mu1, mu2, S1, S2):
        """
        function computing the KL divergence between two normal distributions
        a = kl_div(mu1, mu2, S1, S2); S1, S2 are a 2d numpy arrays, matrices has to be non-singular
        mu1, mu2 are mean vectors, given as 2d arrays
        returns a real scalar
        """
        s1, logdet1 = self.slogdet(S1)
        s2, logdet2 = self.slogdet(S2)
        x1 = self.solve(S2,S1)
        mu = np.subtract(mu2,mu1)
        x2 = self.solve(S2,mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu) - dim + ((s2*logdet2)-(s1*logdet1)))
        if kld < -1e-4: print("fvGP: Negative KL divergence encountered")
        return kld
    ###########################################################################
    def kl_div_grad(self,mu1,dmu1dx, mu2, S1, dS1dx, S2):
        """
        function comuting the gradient of the KL divergence between two normal distributions
        when the gradients of the mean and covariance are given
        a = kl_div(mu1, dmudx,mu2, S1, dS1dx, S2); S1, S2 are a 2d numpy arrays, matrices has to be non-singular
        mu1, mu2 are mean vectors, given as 2d arrays
        """
        s1, logdet1 = self.slogdet(S1)
        s2, logdet2 = self.slogdet(S2)
        x1 = self.solve(S2,dS1dx)
        mu = np.subtract(mu2,mu1)
        x2 = self.solve(S2,mu)
        x3 = self.solve(S2,-dmu1dx)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + ((x3.T @ mu) + (x2 @ -dmu1dx)) - np.trace(np.linalg.inv(S1) @ dS1dx))
        if kld < -1e-4: print("In fvGP: Negative KL divergence encountered")
        return kld
    ###########################################################################
    def gp_kl_div(self, x_iset, comp_mean, comp_cov):
        """
        function to compute the kl divergence of a posterior at given points
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
        output:
        -------
            {"x": the index set points,
             "gp posterior mean": ,
             "gp posterior covariance":  ,
             "given mean": the user-provided mean vector,
             "given covariance":  the use_provided covariance,
             "kl-div:": the kl div between gp pdf and given pdf}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        res = self.posterior_mean(p)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_iset)["S(x)"]

        return {"x": p,
                "gp posterior mean" : gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}


    ###########################################################################
    def gp_kl_div_grad(self, x_iset, comp_mean, comp_cov, direction):
        """
        function to compute the gradient of the kl divergence of a posterior at given points
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
            direction: direction in which the gradient will be computed
        output:
        -------
            {"x": the index set points,
             "gp posterior mean": ,
             "gp posterior mean grad": ,
             "gp posterior covariance":  ,
             "gp posterior covariance grad":  ,
             "given mean": the user-provided mean vector,
             "given covariance":  the use_provided covariance,
             "kl-div grad": the grad of the kl div between gp pdf and given pdf}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        gp_mean = self.posterior_mean(p)["f(x)"]
        gp_mean_grad = self.posterior_mean_grad(p,direction)["df/dx"]
        gp_cov  = self.posterior_covariance(p)["S(x)"]
        gp_cov_grad  = self.posterior_covariance_grad(p,direction)["dS/dx"]

        return {"x": p,
                "gp posterior mean" : gp_mean,
                "gp posterior mean grad" : gp_mean_grad,
                "gp posterior covariance": gp_cov,
                "gp posterior covariance grad": gp_cov_grad,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div grad": self.kl_div_grad(gp_mean, gp_mean_grad,comp_mean, gp_cov, gp_cov_grad, comp_cov)}
    ###########################################################################
    def shannon_information_gain(self, x_iset):
        """
        function to compute the shannon-information gain of data
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
        output:
        -------
            {"x": the index set points,
             "prior entropy": prior entropy
             "posterior entropy": posterior entropy
             "sig:" shannon_information gain}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)


        full_gp_covariances = \
                np.asarray(np.block([[self.prior_covariance,k],\
                            [k.T,kk]]))

        e1 = self.entropy(self.prior_covariance)
        e2 = self.entropy(full_gp_covariances)
        sig = (e2 - e1)
        return {"x": p,
                "prior entropy" : e1,
                "posterior entropy": e2,
                "sig":sig}
    ###########################################################################
    def shannon_information_gain_grad(self, x_iset, direction):
        """
        function to compute the gradient if the shannon-information gain of data
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
            direction: direction in which to compute the gradient
        output:
        -------
            {"x": the index set points,
             "sig_grad:" shannon_information gain gradient}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        e2 = self.gp_entropy_grad(p,direction)
        sig = e2
        return {"x": p,
                "sig grad":sig}
    ###########################################################################
    def posterior_probability(self, x_iset, comp_mean, comp_cov):
        """
        function to compute the probability of an uncertain feature given the gp posterior
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
            comp_mean: a vector of mean values, same length as x_iset
            comp_cov: covarianve matrix, \in R^{len(x_iset)xlen(x_iset)}

        output:
        -------
            {"mu":,
             "covariance": ,
             "probability":  ,
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        res = self.posterior_mean(p)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(p)["S(x)"]
        gp_cov_inv = np.linalg.inv(gp_cov)
        comp_cov_inv = np.linalg.inv(comp_cov)
        cov = np.linalg.inv(gp_cov_inv + comp_cov_inv)
        mu =  cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        s1, logdet1 = self.slogdet(cov)
        s2, logdet2 = self.slogdet(gp_cov)
        s3, logdet3 = self.slogdet(comp_cov)
        dim  = len(mu)
        C = 0.5*(((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T \
               @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean))\
               -(gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0*np.pi)**(dim/2.0)) + (0.5*(logdet2 + logdet3)))
        return {"mu": mu,
                "covariance": cov,
                "probability": 
                np.exp(ln_p)
                }
    def posterior_probability_grad(self, x_iset, comp_mean, comp_cov, direction):
        """
        function to compute the gradient of the probability of an uncertain feature given the gp posterior
        input:
        ------
            x_iset: 1d or 2d numpy array of points, note, these are elements of the 
                    index set which results from a cartesian product of input and output space
            comp_mean: a vector of mean values, same length as x_iset
            comp_cov: covarianve matrix, \in R^{len(x_iset)xlen(x_iset)}
            direction: direction in which to compute the gradient

        output:
        -------
            {"probability grad":  ,}
        """
        p = np.array(x_iset)
        if p.ndim == 1: p = np.array([p])
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        x1 = np.array(p)
        x2 = np.array(p)
        x1[:,direction] = x1[:,direction] + 1e-6
        x2[:,direction] = x2[:,direction] - 1e-6

        probability_grad = (posterior_probability(x1, comp_mean_comp_cov) - posterior_probability(x2, comp_mean_comp_cov))/2e-6
        return {"probability grad": probability_grad}

    ###########################################################################
    def _int_gauss(self,S):
        return ((2.0*np.pi)**(len(S)/2.0))*np.sqrt(np.linalg.det(S))

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Kernels#####################################################
    ##################################################################################
    ##################################################################################
    def squared_exponential_kernel(self, distance, length):
        """
        function for the squared exponential kernel

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """
        kernel = np.exp(-(distance ** 2) / (2.0 * (length ** 2)))
        return kernel

    def exponential_kernel(self, distance, length):
        """
        function for the exponential kernel

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = np.exp(-(distance) / (length))
        return kernel

    def matern_kernel_diff1(self, distance, length):
        """
        function for the matern kernel  1. order differentiability

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
            -(np.sqrt(3.0) * distance) / length
        )
        return kernel

    def matern_kernel_diff2(self, distance, length):
        """
        function for the matern kernel  2. order differentiability

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) / (length))
            + ((5.0 * distance ** 2) / (3.0 * length ** 2))
        ) * np.exp(-(np.sqrt(5.0) * distance) / length)
        return kernel

    def sparse_kernel(self, distance, radius):
        """
        function for the sparse kernel
        this kernel is compactly supported, which makes the covariance matrix sparse

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            radius (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d > radius] = radius
        kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
        ((3.0*(d/radius)**2*np.log((d/radius)/(1+np.sqrt(1.0 - (d/radius)**2))))+\
        ((2.0*(d/radius)**2+1.0)*np.sqrt(1.0-(d/radius)**2)))
        return kernel

    def periodic_kernel(self, distance, length, p):
        """periodic kernel
        Parameters:
        -----------
            distance: float or array containing distances
            length (float): the length scale
            p (float): period of the oscillation
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """
        kernel = np.exp(-(2.0/length**2)*(np.sin(np.pi*distance/p)**2))
        return kernel

    def linear_kernel(self, x1,x2, hp1,hp2,hp3):
        """
        function for the linear kernel in 1d

        Parameters:
        -----------
            x1 (float):  scalar position of point 1
            x2 (float):  scalar position of point 2
            hp1 (float): vertical offset of the linear kernel
            hp2 (float): slope of the linear kernel
            hp3 (float): horizontal offset of the linear kernel
        Return:
        -------
            scalar
        """
        kernel = hp1 + (hp2*(x1-hp3)*(x2-hp3))
        return kernel

    def dot_product_kernel(self, x1,x2,hp,matrix):
        """
        function for the dot-product kernel

        Parameters:
        -----------
            x1 (2d numpy array of points):  scalar position of point 1
            x2 (2d numpy array of points):  scalar position of point 2
            hp (float):                     vertical offset
            matrix (2d array of len(x1)):   a metric tensor to define the dot product
        Return:
        -------
            numpy array of shape len(x1) x len(x2)
        """
        kernel = hp + x1.T @ matrix @ x2
        return kernel

    def polynomial_kernel(self, x1, x2, p):
        """
        function for the polynomial kernel

        Parameters:
        -----------
            x1 (2d numpy array of points):  scalar position of point 1
            x2 (2d numpy array of points):  scalar position of point 2
            p (float):                      exponent
        Return:
        -------
            numpy array of shape len(x1) x len(x2)
        """
        kernel = (1.0+x1.T @ x2)**p
        return p
    def default_kernel(self,x1,x2,hyperparameters,obj):
        ################################################################
        ###standard anisotropic kernel in an input space with l2########
        ################################################################
        """
        x1: 2d numpy array of points
        x2: 2d numpy array of points
        obj: object containing kernel definition

        Return:
        -------
        Kernel Matrix
        """
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1),len(x2)))
        for i in range(len(hps)-1):
            distance_matrix += abs(np.subtract.outer(x1[:,i],x2[:,i])/hps[1+i])**2
        distance_matrix = np.sqrt(distance_matrix)
        return   hps[0] *  obj.exponential_kernel(distance_matrix,1)

    def _compute_distance_matrix_l2(self,points1,points2,hp_list):
        """computes the distance matrix for the l2 norm"""
        distance_matrix = np.zeros((len(points2), len(points1)))
        for i in range(len(points1[0])):
            distance_matrix += (
            np.abs(
            np.subtract.outer(points2[:, i], points1[:, i]) ** 2
            )/hp_list[i])
        return np.sqrt(distance_matrix)
    
    def _compute_distance_matrix_l1(self,points1,points2):
        """computes the distance matrix for the l1 norm"""
        distance_matrix = (
        np.abs(
        np.subtract.outer(points2, points1)
        ))
        return distance_matrix

    def d_gp_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-6
        new_points[:,direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters,self)
        b = self.kernel(points1,    points2, hyperparameters,self)
        derivative = ( a - b )/epsilon
        return derivative

    def d_gp_kernel_dh(self, points1, points2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-6
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyperparameters1,self)
        b = self.kernel(points1, points2, new_hyperparameters2,self)
        derivative = ( a - b )/(2.0*epsilon)
        return derivative

    def gradient_gp_kernel(self, points1, points2, hyperparameters):
        gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self.d_gp_kernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def d2_gp_kernel_dh2(self, points1, points2, direction1, direction2, hyperparameters):
        ###things to consider when things go south with the Hessian:
        ###make sure the epsilon is appropriate, not too large, not too small, 1e-3 seems alright
        epsilon = 1e-3
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        new_hyperparameters3 = np.array(hyperparameters)
        new_hyperparameters4 = np.array(hyperparameters)

        new_hyperparameters1[direction1] = new_hyperparameters1[direction1] + epsilon
        new_hyperparameters1[direction2] = new_hyperparameters1[direction2] + epsilon

        new_hyperparameters2[direction1] = new_hyperparameters2[direction1] + epsilon
        new_hyperparameters2[direction2] = new_hyperparameters2[direction2] - epsilon

        new_hyperparameters3[direction1] = new_hyperparameters3[direction1] - epsilon
        new_hyperparameters3[direction2] = new_hyperparameters3[direction2] + epsilon

        new_hyperparameters4[direction1] = new_hyperparameters4[direction1] - epsilon
        new_hyperparameters4[direction2] = new_hyperparameters4[direction2] - epsilon

        return (self.kernel(points1,points2,new_hyperparameters1,self) \
              - self.kernel(points1,points2,new_hyperparameters2,self) \
              - self.kernel(points1,points2,new_hyperparameters3,self) \
              + self.kernel(points1,points2,new_hyperparameters4,self))\
              / (4.0*(epsilon**2))
#    @profile
    def hessian_gp_kernel(self, points1, points2, hyperparameters):
        hessian = np.zeros((len(hyperparameters),len(hyperparameters), len(points1),len(points2)))
        for i in range(len(hyperparameters)):
            for j in range(i+1):
                hessian[i,j] = hessian[j,i] = self.d2_gp_kernel_dh2(points1, points2, i,j, hyperparameters)
        return hessian

    def dm_dh(self,hps):
        gr = np.empty((len(hps),len(self.data_x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.mean_function(self,self.data_x,temp_hps1)
            b = self.mean_function(self,self.data_x,temp_hps2)
            gr[i] = (a-b)/2e-6
        return gr
    ##########################
    def d2m_dh2(self,hps):
        hess = np.empty((len(hps),len(hps),len(self.data_x)))
        e = 1e-4
        for i in range(len(hps)):
            for j in range(i+1):
                temp_hps1 = np.array(hps)
                temp_hps2 = np.array(hps)
                temp_hps3 = np.array(hps)
                temp_hps4 = np.array(hps)
                temp_hps1[i] = temp_hps1[i] + e
                temp_hps1[j] = temp_hps1[j] + e

                temp_hps2[i] = temp_hps2[i] - e
                temp_hps2[j] = temp_hps2[j] - e

                temp_hps3[i] = temp_hps3[i] + e
                temp_hps3[j] = temp_hps3[j] - e

                temp_hps4[i] = temp_hps4[i] - e
                temp_hps4[j] = temp_hps4[j] + e


                a = self.mean_function(self,self.data_x,temp_hps1)
                b = self.mean_function(self,self.data_x,temp_hps2)
                c = self.mean_function(self,self.data_x,temp_hps3)
                d = self.mean_function(self,self.data_x,temp_hps4)
                hess[i,j] = hess[j,i] = (a - c - d + b)/(4.*e*e)
        return hess

    ################################################################
    def _normalize_y_data(self):
        mini = np.min(self.data_y)
        self.data_y = self.data_y - mini
        maxi = np.max(self.data_y)
        self.data_y = self.data_y / maxi

