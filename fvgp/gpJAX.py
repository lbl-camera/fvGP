#!/usr/bin/env python

import dask.distributed as distributed
"""
Software: FVGP, version: ALL
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

import jax.numpy as np
from jax import grad, vmap
from jax.config import config; config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from jax import jacfwd, jacrev



import math
import numpy
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from .mcmc import mcmc

import itertools
import time
import torch
from functools import partial
from hgdl.hgdl import HGDL
from jax.config import config
config.update("jax_enable_x64", True)

class GPJAX():
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
        gp_kernel_function_grad = None,
        gp_mean_function = None,
        gp_mean_function_grad = None,
        sparse = False,
        normalize_y = False,
        use_inv = False,
        ram_economy = True,
        ):
        """
        The constructor for the gp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        if input_space_dim != len(points[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        if np.ndim(values) == 2: values = values[:,0]

        points = np.array(points, dtype = np.float64)
        values = np.array(values, dtype = np.float64)
        if variances: variances = np.array(variances, dtype = np.float64)
        init_hyperparameters = np.array(init_hyperparameters, dtype = np.float64)

        self.normalize_y = normalize_y
        self.input_dim = input_space_dim
        self.data_x = np.array(points)
        self.point_number = len(self.data_x)
        self.data_y = np.array(values)
        self.compute_device = compute_device
        self.ram_economy = ram_economy
        #self.gp_kernel_function_grad = gp_kernel_function_grad
        #self.gp_mean_function_grad = gp_mean_function_grad

        self.sparse = sparse
        self.use_inv = use_inv
        self.K_inv = None
        if self.normalize_y is True: self._normalize_y_data()
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.data_y.shape), dtype = np.float64) * abs(self.data_y / 100.0)
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
        if callable(gp_kernel_function): self.kernel = gp_kernel_function
        else: self.kernel = self.default_kernel
        #self.d_kernel_dx = self.d_gp_kernel_dx

        self.gp_mean_function = gp_mean_function
        if  callable(gp_mean_function): self.mean_function = gp_mean_function
        else: self.mean_function = self.default_mean_function

        #if callable(gp_kernel_function_grad): self.dk_dh = gp_kernel_function_grad
        #else:
        #    if self.ram_economy is True: self.dk_dh = self.gp_kernel_derivative
        #    else: self.dk_dh = self.gp_kernel_gradient

        if callable(gp_mean_function_grad): self.dm_dh = gp_mean_function_grad
        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = np.array(init_hyperparameters)
        ##########################################
        #compute the prior########################
        ##########################################
        self.compute_prior_fvGP_pdf()
        print("fvGP successfully initiated")

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
        
        points = np.array(points, dtype = np.float64)
        values = np.array(points, dtype = np.float64)
        if variances: variances = np.array(variances, dtype = np.float64)


        self.data_x = np.array(points)
        self.point_number = len(self.data_x)
        self.data_y = np.array(values)
        if self.normalize_y is True: self._normalize_y_data()
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.data_y.shape), dtype = np.float64) * abs(self.data_y / 100.0)
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
        except: print("No asynchronous training to be killed, no training is running.")
    ###################################################################################
    def update_hyperparameters(self, opt_obj):
        """
        This function asynchronously finds the maximum of the marginal log_likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function just submits the training and returns
        an object which can be given to `fvgp.gp.update_hyperparameters`, which will automatically update the GP prior with the new hyperparameters.

        Parameters
        ----------
        object : HGDL class instance
            HGDL class instance returned by `fvgp.gp.train_async`

        Return
        ------
        The current hyperparameters : np.ndarray
        """
        success = False
        try:
            aux = opt_obj.get_latest()
            res = aux["x"]
            f = aux["f(x)"]
            success = True
        except:
            print("      The optimizer object could not be queried")
            print("      That probably means you are not optimizing the hyperparameters asynchronously")
        if success is True:
            try:
                res = res[0]
                l_n = self.log_likelihood_jax(res)
                l_o = self.log_likelihood_jax(self.hyperparameters)
                print("l new: ", l_n)
                print("l old: ", l_o)
                print("current f: ", f)
                if l_n - l_o < 0.000001:
                    self.hyperparameters = res
                    self.compute_prior_fvGP_pdf()
                    print("    fvGP async hyperparameter update successful")
                    print("    Latest hyperparameters: ", self.hyperparameters)
                else:
                    print("    The update was attempted but there was no improvement. I am keeping the old ones")
                    print("Old likelihood: ", -l_o, " at ", self.hyperparameters)
                    print("New likelihood: ", -l_n, " at ", res)
            except Exception as e:
                print("    Async Hyper-parameter update not successful in fvGP. I am keeping the old ones.")
                print("    hyperparameters: ", self.hyperparameters)
                print("    Reason: ", e)

        return self.hyperparameters
    ##################################################################################
    def train_async(self,
        hyperparameter_bounds,
        init_hyperparameters = None,
        max_iter = 10000,
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
        print("local optimizer: ",local_optimizer)
        print("global optimizer: ",global_optimizer)
        opt_obj = HGDL(self.log_likelihood_jax,
                    self.log_likelihood_gradient_jax,
                    hp_bounds,
                    hess = self.log_likelihood_hessian_jax,
                    local_optimizer = local_optimizer,
                    global_optimizer = global_optimizer,
                    radius = deflation_radius,
                    num_epochs = max_iter)
        opt_obj.optimize(dask_client = dask_client, x0 = np.array(starting_hps).reshape(1,-1))
        return opt_obj
    ##################################################################################
    def log_likelihood_jax(self,hyperparameters):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        hyperparameters = np.array(hyperparameters , dtype=np.float64)
        mean = self.mean_function(self.data_x,hyperparameters,self)
        if mean.ndim > 1: raise Exception("Your mean function did not return a 1d numpy array!")
        x,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        y = self.data_y - mean
        sign, logdet = np.linalg.slogdet(K)
        n = len(y)
        if sign == 0.0: return (0.5 * (y.T @ x)) + (0.5 * n * np.log(2.0*np.pi))
        return (0.5 * (y.T @ x)) + (0.5 * sign * logdet) + (0.5 * n * np.log(2.0*np.pi))

    def log_likelihood_gradient_jax(self, hyperparameters):
        hyperparameters = np.array(hyperparameters, dtype=np.float64)
        res = numpy.array(grad(self.log_likelihood_jax)(hyperparameters), dtype = numpy.float64).reshape(-1,1)
        return res[:,0]

    #def log_likelihood_hessian_jax(self, hyperparameters):
    #    hyperparameters = np.array(hyperparameters)
    #    return numpy.array(jacrev(jacrev(self.log_likelihood_jax))(hyperparameters))

    ##################################################################################
    def log_likelihood_hessian_jax(self, hyperparameters):
        """
        computes the hessian of the negative  marginal  log-likelihood
        input:
            hyper parameters
        output:
            hessian of the negative marginal log-likelihood (matrix)
        """
        ##implemented as first-order approximation
        len_hyperparameters = len(hyperparameters)
        d2L_dmdh = np.zeros((len_hyperparameters,len_hyperparameters), dtype = np.float64)
        epsilon = 1e-5
        grad_at_hps = self.log_likelihood_gradient_jax(hyperparameters)
        for i in range(len_hyperparameters):
            hps_temp = np.array(hyperparameters)
            #`x[idx] = y``, use ``x = x.at[idx].set(y)
            #hps_temp[i] = hps_temp[i] + epsilon
            hps_temp = hps_temp.at[i].set(hps_temp[i] + epsilon)
            d2L_dmdh = d2L_dmdh.at[i,i:].set(((self.log_likelihood_gradient_jax(hps_temp) - grad_at_hps)/epsilon)[i:])
            #d2L_dmdh[i,i:] = ((self.log_likelihood_gradient_jax(hps_temp) - grad_at_hps)/epsilon)[i:]
        return d2L_dmdh + d2L_dmdh.T - np.diag(np.diag(d2L_dmdh))
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
        self.prior_mean_vec = self.mean_function(self.data_x,self.hyperparameters,self)
        cov_y,K = self._compute_covariance_value_product(
                self.hyperparameters,
                self.data_y,
                self.variances,
                self.prior_mean_vec)
        self.prior_covariance = K
        if self.use_inv is True: self.K_inv = self.inv(K)
        self.covariance_value_prod = cov_y
    ##################################################################################
    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean):
        K = self.compute_covariance(hyperparameters, variances)
        y = values - mean
        x = np.linalg.solve(K, y)
        if x.ndim == 2: x = x[:,0]
        return x,K
    ##################################################################################
    def compute_covariance(self, hyperparameters, variances):
        """computes the covariance matrix from the kernel"""
        CoVariance = self.kernel(
            self.data_x, self.data_x, hyperparameters, self)
        self.add_to_diag(CoVariance, variances)
        return CoVariance

    ##################################################################################
    def add_to_diag(self,Matrix, Vector):
        Noise = np.diag(Vector)
        #d = np.einsum("ii->i", Matrix)
        #d += Vector
        Matrix = Matrix + Noise
        return Matrix
    def is_sparse(self,A):
        if float(np.count_nonzero(A))/float(len(A)**2) < 0.01: return True
        else: return False
    def how_sparse_is(self,A):
        return float(np.count_nonzero(A))/float(len(A)**2)
    def default_mean_function(self,x, hyperparameters,gp_obj):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)), dtype = np.float64)
        mean = np.mean(self.data_y)
        return mean


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


    def squared_exponential_kernel_robust(self, distance, phi):
        """
        function for the squared exponential kernel, This is the robust version, which means it is defined on [-infty,infty]
        instead of the usual (0,infty]
        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """
        kernel = np.exp(-(distance ** 2) * (phi ** 2))
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

    def exponential_kernel_robust(self, distance, phi):
        """
        function for the exponential kernel, This is the robust version, which means it is defined on [-infty,infty]
        instead of the usual (0,infty]
        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = np.exp(-(distance) * (phi**2))
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


    def matern_kernel_diff1_robust(self, distance, phi):
        """
        function for the matern kernel  1. order differentiability, This is the robust version, which means it is defined on [-infty,infty]
        instead of the usual (0,infty]
        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """
        ##1/l --> phi**2
        kernel = (1.0 + ((np.sqrt(3.0) * distance) * (phi**2))) * np.exp(
            -(np.sqrt(3.0) * distance) * (phi**2))
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


    def matern_kernel_diff2_robust(self, distance, length):
        """
        function for the matern kernel  2. order differentiability, This is the robust version, which means it is defined on [-infty,infty]
        instead of the usual (0,infty]
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
            + ((np.sqrt(5.0) * distance) * (phi**2))
            + ((5.0 * distance ** 2) * (3.0 * phi ** 4))
        ) * np.exp(-(np.sqrt(5.0) * distance) * (phi**2))
        return kernel
##################################################################






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
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)), dtype = np.float64)])
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        A = k.T @ self.covariance_value_prod
        posterior_mean = self.mean_function(p,self.hyperparameters,self) + A
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
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)), dtype = np.float64)])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        x1 = np.array(p)
        x2 = np.array(p)
        eps = 1e-6
        x1[:,direction] = x1[:,direction] + eps
        x2[:,direction] = x2[:,direction] - eps
        mean_der = (self.mean_function(x1,self.hyperparameters,self) - self.mean_function(x2,self.hyperparameters,self))/(2.0*eps)
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        k_g = self.d_kernel_dx(p,self.data_x, direction,self.hyperparameters)
        posterior_mean_grad = mean_der + (k_g @ self.covariance_value_prod)
        return {"x": p,
                "direction":direction,
                "df/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_iset, variance_only = False):
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
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)), dtype = np.float64)])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        if self.use_inv is True:
            if variance_only is True: v = np.diag(kk) - np.einsum('ij,jk,ki->i', k.T, self.K_inv, k); S = False
            if variance_only is False:  S = kk - (k.T @ self.K_inv @ k); v = np.array(np.diag(S))
        else:
            k_cov_prod = self.solve(self.prior_covariance,k)
            S = kk - (k_cov_prod.T @ k)
            v = np.array(np.diag(S))
        if np.any(v < -0.001):
            print("WARNING in fvGP: CAUTION, negative variances encountered. That normally means that the model is unstable.")
            print("Rethink the kernel definitions, add more noise to the data,")
            print("or double check the hyperparameter optimization bounds. This will not ")
            print("terminate the algorithm, but expect anomalies.")
            print("diagonal of the posterior covariance: ",v)
            p = np.block([[self.prior_covariance, k],[k.T, kk]])
            print("eigenvalues of the prior: ", np.linalg.eig(p)[0])
            i = np.where(v < 0.0)
            v[i] = 0.0
            if S is not False: S = np.fill_diagonal(S,v)

        return {"x": p,
                "v(x)": v,
                "S(x)": S}

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
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)), dtype = np.float64)])

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

