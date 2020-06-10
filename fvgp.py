#!/usr/bin/env python

"""
Software: FVGP
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


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import minres
import itertools
import time
import torch
from sys import exit

class FVGP:
    """
    ######################################################
    ######################################################
    ######################################################
    ######################################################
    ######################################################
    GP class: Finds hyper-parameters and therefore the mean
    and covariance of a (multi-output) Gaussian process

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space
        dim2: number of dimension of the output space

    Attributes:
        input_space_dim (int):         dim1
        output_space_dim (int):        dim2
        output_number (int):           n
        points (N x dim1 numpy array): array of points.
        values (N x n numpy array):    array of values.
    Optional Attributes:
        values_positions (N x dim1 x dim2 numpy array): the positions of the outputs in the output space
        variances (N x n numpy array):                  variances of the values
        gp_system_solve_method (str):                   inv/cg/minres; default = inv
        gp_kernel_function(func):                       None/function defining the kernel def name(x1,x2,hyper_parameters,self)
        gp_mean_function(func):                         None/a function def name(x, self), default = None
        input_hyper_parameters (1d list):               default: list of [1,1,...]
        output_hyper_parameters (1d list):              default: list of [1,1,...]

    Example:
        obj = FVGP(
            input_space_dim = 3,
            output_space_dim = 1,
            output_number = 2,
            points = np.array([[1,2,3],
                                4,5,6]),
            values = np.array([[2,3],
                            [13,27.2]]),
            value_positions = np.array([[[0]],[[1]]]),
            variances = np.array([[0.001,0.01],
                                [0.1,2]]),
            gp_system_solve_method="inv",
            gp_kernel_function = kernel_function,
            input_hyper_parameters = [2,3,4,5],
            output_hyper_parameters = [2],
            gp_mean_function = some_mean_function
        )
    ######################################################
    ######################################################
    ######################################################
    ######################################################
    ######################################################
    ######################################################
    """
    def __init__(
        self,
        input_space_dim,
        output_space_dim,
        output_number,
        points,
        values,
        value_positions = None,
        variances = None,
        compute_device = "cpu",
        gp_system_solve_method="solve",
        gp_kernel_function = None,
        gp_mean_function = None,
        init_input_hyper_parameters = None,
        init_output_hyper_parameters = None,
        ):

        """
        The constructor for the fvgp class.
        type help(FVGP) for more information about attributes, methods and their parameters
        """
        ##########################################
        #######check if dimensions match##########
        ##########################################
        if input_space_dim != len(points[0]):
            print("input space dimensions are not in agreement with the point positions given"); exit()
        ##########################################
        if output_number != len(values[0]):
            print("the output number is not in agreement with the data values given"); exit()
        ##########################################
        self.input_dim = input_space_dim
        self.output_dim = output_space_dim
        self.output_num = output_number
        self.points = points
        self.point_number = len(self.points)
        self.values = values
        self.compute_device = compute_device
        ##########################################
        #######prepare value positions############
        ##########################################
        if self.output_dim == 1 and isinstance(value_positions, np.ndarray) == False:
            self.value_positions = self.compute_standard_value_positions()
        elif self.output_dim > 1 and isinstance(value_positions, np.ndarray) == False:
            exit(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the FVGP class. EXIT"
            )
        else:
            self.value_positions = value_positions
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.zeros((values.shape))
        else:
            self.variances = variances

        ##########################################
        #######define solve method################
        ##########################################
        if gp_system_solve_method == "rank-n-update": 
            print("can't be initialized with gp_system_solve_method = 'rank-n update'. Changed to 'inv'")
            gp_system_solve_method = 'inv'
        self.gp_system_solve_method = gp_system_solve_method
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if gp_kernel_function is None:
            self.kernel = self.default_kernel
        else:
            self.kernel = gp_kernel_function
        self.d_kernel_dx = self.d_gp_kernel_dx
        self.gp_mean_function = gp_mean_function
        self.mean_vec = self.compute_mean()

        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        if init_input_hyper_parameters is None:
            init_input_hyper_parameters = [1.0] * (self.input_dim+1)
        if init_output_hyper_parameters is None:
            init_output_hyper_parameters = [1.0] * self.output_dim
        self.input_hyper_parameters = init_input_hyper_parameters
        self.output_hyper_parameters = init_output_hyper_parameters
        self.hyper_parameters = \
        self.input_hyper_parameters + self.output_hyper_parameters
        ##########################################
        #transform index set and elements#########
        ##########################################
        self.iset_dim = self.input_dim + self.output_dim
        self.points, self.values, self.variances, self.prior_mean = self.transform_index_set()
        self.point_number = len(self.points)
        #########################################
        ####compute covariance value prod########
        #########################################
        self.prior_mean,self.prior_covariance,self.covariance_value_prod = self.compute_prior_fvGP_pdf()
        #NOTE TO MYSELF: The prior mean should actually not be computed first,than translated, but should be computed
        #in "compute_prior_fvGP_pdf()".
######################################################################
######################################################################
######################################################################
    def update_gp_data(
        self,
        points,
        values,
        value_positions = None,
        variances = None,
        gp_system_solve_method="inv",
        ):

        """
        This function updates the data in the gp_class.


        Attributes:
            points (N x dim1 numpy array): An array of points.
            values (N x n):                An array of values.

        optional attributes:
            values_positions (N x dim1 x dim2 numpy array): the positions of the outputs in the output space
            variances (N x n):                              variances of the values
            gp_solve_method_solve_method (str):             'inv'/'cg'/'minres'/'rank-n update'; default = 'inv'
        """
        self.points = points
        self.point_number = len(self.points)
        self.values = values
        ##########################################
        #######prepare value positions############
        ##########################################
        if self.output_dim == 1 and isinstance(value_positions, np.ndarray) == False:
            self.value_positions = self.compute_standard_value_positions()
        elif self.output_dim > 1 and isinstance(value_positions, np.ndarray) == False:
            exit(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the FVGP class. EXIT"
            )
        else:
            self.value_positions = value_positions
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.zeros((values.shape))
        else:
            self.variances = variances
        self.mean_vec = self.compute_mean()
        ######################################
        if self.output_num > 1 and gp_system_solve_method == "rank-n update":
            print("CAUTION: It is highly discouraged to use 'rank-n update' in combination with multi-task GPs.")
            print("I will revert to gp_system_solve_method = 'inv'")
            gp_system_solve_method = 'inv'
        self.gp_system_solve_method = gp_system_solve_method
        ######################################
        #####transform to index set###########
        ######################################
        self.points, self.values, self.variances, self.prior_mean = self.transform_index_set()
        self.point_number = len(self.points)
        self.prior_mean,self.prior_covariance,self.covariance_value_prod = self.compute_prior_fvGP_pdf()
    ###################################################################################
    ###################################################################################
    ###################################################################################
    ###################################################################################
    ###################################################################################
    #################TRAINING##########################################################
    ###################################################################################
    def train(self,
        input_hyper_parameter_bounds,
        output_hyper_parameter_bounds,
        init_input_hyper_parameters = None,
        init_output_hyper_parameters = None,
        optimization_method = "global",
        likelihood_pop_size = 20,
        likelihood_optimization_tolerance = 0.1,
        likelihood_optimization_max_iter = 120
        ):
        """
        This function finds the maximum of the log_likelihood and therefore trains the fvGP.
        inputs:
            bounds_input (2d list)
            bounds_output(2d list)
        optional inputs:
            init_input_hyper_parameters (list):  default = None
            init_output_hyper_parameters (list): default = None
            optimization_method : default = "global",
            likelihood_pop_size: default = 20,
            likelihood_optimization_tolerance: default = 0.1,
            likelihood_optimization_max_iter: default = 120

        output:
            None, just updated the class with then new hyper_parameters
        """
        #####safety first:
        if self.gp_system_solve_method == "rank-n update": 
            print("CAUTION: rank-n update is requested. The training will change the hyper parameters, the covariance matrix therefore has to be fully updated. I will do that for you")
            self.gp_system_solve_method = "inv"
        ############################################
        self.hyper_parameter_optimization_bounds = \
        input_hyper_parameter_bounds + output_hyper_parameter_bounds
        if init_input_hyper_parameters is None:
            init_input_hyper_parameters = self.input_hyper_parameters
        if init_output_hyper_parameters is None:
            init_output_hyper_parameters = self.output_hyper_parameters
        init_hyper_parameters = \
            init_input_hyper_parameters + init_output_hyper_parameters
        ######################
        #####TRAINING#########
        ######################
        self.hyper_parameters = list(self.find_hyper_parameters(
        init_hyper_parameters,
        self.hyper_parameter_optimization_bounds,
        optimization_method,
        likelihood_optimization_max_iter,
        likelihood_pop_size,
        likelihood_optimization_tolerance
        ))
        self.input_hyper_parameters = self.hyper_parameters[0:self.input_dim+1]
        self.output_hyper_parameters = self.hyper_parameters[self.input_dim+1:]
        self.prior_mean,self.prior_covariance,self.covariance_value_prod = self.compute_prior_fvGP_pdf()
        ######################
        ######################
        ######################
    ##################################################################################
    def find_hyper_parameters(self,
            hyper_parameters_0,
            hyper_parameter_optimization_bounds,
            hyper_parameter_optimization_mode,
            likelihood_optimization_max_iter,
            likelihood_pop_size,
            likelihood_optimization_tolerance
            ):
        hyper_parameters = self.optimize_log_likelihood(
            self.values,
            self.variances,
            self.prior_mean,
            hyper_parameters_0,
            hyper_parameter_optimization_bounds,
            hyper_parameter_optimization_mode,
            likelihood_optimization_max_iter,
            likelihood_pop_size,
            likelihood_optimization_tolerance
        )
        return hyper_parameters
    ##################################################################################
    def optimize_log_likelihood(
        self,
        values,
        variances,
        mean,
        starting_hps,
        hp_bounds,
        hyper_parameter_optimization_mode,
        likelihood_optimization_max_iter,
        likelihood_pop_size,
        likelihood_optimization_tolerance
        ):

        epsilon = np.inf
        step_size = 1.0
        print(
            "Hyper-parameter tuning in progress. Old hyper-parameters: ",
            starting_hps, " with old log likelihood: ",
            abs(
                self.log_likelihood(
                    starting_hps,
                    values,
                    variances,
                    mean
                )
            )
        )

        ############################
        ####start of optimization:##
        ############################
        if hyper_parameter_optimization_mode == "global":
            print("I am performing a global differential evolution algorithm to find the optimal hyper-parameters.")
            res = differential_evolution(
                self.log_likelihood,
                hp_bounds,
                args=(values, variances, mean),
                disp=True,
                maxiter=likelihood_optimization_max_iter,
                popsize = likelihood_pop_size,
                tol = likelihood_optimization_tolerance,
            )
            print(res) 
            hyper_parameters1 = np.array(res["x"])
            Eval1 = self.log_likelihood(
                hyper_parameters1,
                values,
                variances,
                mean
            )
            hyper_parameters = np.array(hyper_parameters1)
            print("I found hyper-parameters ",hyper_parameters," with likelihood ",
                self.log_likelihood(
                    hyper_parameters,
                    values,
                    variances,
                    mean))
        elif hyper_parameter_optimization_mode == "local":
            hyper_parameters = np.array(starting_hps)
            print("Performing a local update of the hyper parameters.")
            print("Attempting a BFGS optimization...")
            OptimumEvaluation = minimize(
                self.log_likelihood,
                hyper_parameters,
                args=(values, variances, mean),
                method="L-BFGS-B",
                jac=self.log_likelihood_gradient_wrt_hyper_parameters,
                bounds = hp_bounds,
                tol=10 ** -4,
                callback=None,
                options = {"maxiter": likelihood_optimization_max_iter,
                           "gtol": likelihood_optimization_tolerance}
                )

            if OptimumEvaluation["success"] == True:
                print(
                    "Local optimization successfully concluded with result: ",
                    OptimumEvaluation["fun"],
                )
                hyper_parameters = OptimumEvaluation["x"]
            else:
                print("Optimization not successful.")
        elif hyper_parameter_optimization_mode == "hgdl":
            try:
                from hgdl.hgdl import HGDL
            except:# ModuleNotFoundError:
                print('could not locate hgdl for import')
                exit()
            print('bounds are',hp_bounds) 
            a = time.time()
            self.log_likelihood_gradient_wrt_hyper_parameters(self.hyper_parameters,
                    values = values,
                    variances = variances, mean = mean)
            print(time.time() - a)
            a = time.time()
            self.log_likelihood_hessian_wrt_hyper_parameters(self.hyper_parameters,
                    values = values,
                    variances = variances, mean = mean)
            print(time.time() - a)
            import numba as nb
            from functools import partial
            func = partial(self.log_likelihood,values = values,
                    variances = variances, mean = mean)
            grad = partial(self.log_likelihood_gradient_wrt_hyper_parameters,
                    values = values,
                    variances = variances, mean = mean)
            hess = partial(self.log_likelihood_hessian_wrt_hyper_parameters,
                    values = values,
                    variances = variances, mean = mean)

            res = HGDL(func, grad, hess, np.asarray(hp_bounds), numIndividuals=20)
            print(res)
            hyper_parameters = res["best"] #input()
        else:
            print("no optimization mode specified")
        print("New hyper-parameters: ",
            hyper_parameters,
            "with log likelihood: ",
            self.log_likelihood(hyper_parameters,values,
                variances,mean))

        return hyper_parameters
    ##################################################################################
    ### note to me - this is main function 
    def log_likelihood(
        self,
        hyper_parameters,
        values,
        variances,
        mean
    ):
        """computes the log-likelihood
        input:
            hyper parameters
            values
            variances
            means
        output:
            log likelihood(scalar)
        """

        x,K = self._compute_covariance_value_product(hyper_parameters,values, variances, mean)
        y=values
        sign, logdet = self.slogdet(K,compute_device = self.compute_device)
        if sign == 0.0:
            return 0.5 * ((y - mean).T @ x)
        return ((0.5 * ((y - mean).T @ x)) + (0.5 * sign * logdet))[0]
    ##################################################################################
#    @profile
    def log_likelihood_gradient_wrt_hyper_parameters(self, hyper_parameters, values, variances, mean):
        solve_method = "solve"
        x,K = self._compute_covariance_value_product(hyper_parameters,values, variances, mean)
        y = values
        dL_dH = np.empty((len(hyper_parameters)))
        dK_dH = self.gradient_gp_kernel(self.points,self.points, hyper_parameters)
        if solve_method == 'solve':
            #####################################
            ####alternative to avoid inversion###
            #####################################
            for i in range(len(hyper_parameters)):
                x1 = self.solve(K ,dK_dH[i,:, :])
                dL_dH[i] = 0.5 * (
                    ((y - mean).T @ x1 @ x)-(np.trace(x1)))
            #####################################
            #####################################
            #####################################
            #print("first   ",dL_dH)
            return -dL_dH
        if solve_method == 'inv':
            K_inv = self.safe_invert(K)
            for i in range(len(hyper_parameters)):
                dL_dH[i] = 0.5 * (
                    ((y - mean).T @ K_inv @ dK_dH[i,:, :] @ K_inv @ (y - mean))
                    - (np.trace(K_inv @ dK_dH[i,:, :])))
            #print("second: ",dL_dH)
            #exit()
            return -dL_dH
#    @profile
    def log_likelihood_hessian_wrt_hyper_parameters(self, hyper_parameters, values, variances, mean):
        solve_method = "solve"
        x,K = self._compute_covariance_value_product(hyper_parameters,values, variances, mean)
        y = values - mean
        d2L_dH2 = np.empty((len(hyper_parameters),len(hyper_parameters)))
        dK_dH = self.gradient_gp_kernel(self.points,self.points, hyper_parameters)
        d2K_dH2 = self.hessian_gp_kernel(self.points,self.points, hyper_parameters)
        if solve_method == 'solve':
            #####################################
            ####alternative to avoid inversion###
            #####################################
            for i in range(len(hyper_parameters)):
                x1 = self.solve(K,dK_dH[i,:,:])
                for j in range(len(hyper_parameters)):
                    x2 = self.solve(K,dK_dH[j,:,:])
                    x3 = self.solve(K,d2K_dH2[i,j,:,:])
                    d2L_dH2[i,j] = 0.5 * ((y.T @ (-x2 @ x1 @ x - x1 @ x2 @ x + x3 @ x)) - np.trace(-x2 @ x1 + x3))
            #####################################
            #####################################
            #####################################
            #print("first:",d2L_dH2)
            return -d2L_dH2
        if solve_method == 'inv':
            K_inv = self.safe_invert(K)
            for i in range(len(hyper_parameters)):
                for j in range(len(hyper_parameters)):
                    A = -(K_inv @ dK_dH[j,:,:] @ K_inv @ dK_dH[i,:,:] @ K_inv)\
                        -(K_inv @ dK_dH[i,:,:] @ K_inv @ dK_dH[j,:,:] @ K_inv)\
                        +(K_inv @ d2K_dH2[i,j,:,:] @ K_inv)
                    B = np.trace((-K_inv @ dK_dH[j,:,:] @ K_inv @ dK_dH[i,:,:]) + (K_inv @ d2K_dH2[i,j,:,:]))
                    d2L_dH2[i,j] = 0.5 * (((y).T @ A @ (y)) - B)
            #print("second:",d2L_dH2)
            return -d2L_dH2


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
        cov_y,K = self._compute_covariance_value_product(
                self.hyper_parameters,
                self.values,
                self.variances,
                self.prior_mean)
        self.prior_covariance = K
        self.covariance_value_prod = cov_y
#        self.K_inv = K_inv
        return self.prior_mean,K,cov_y
    ##################################################################################
    def K_inv(self):
        return self.safe_invert(self.prior_covariance)

    ##################################################################################
    def _compute_covariance_value_product(self, hyper_parameters,values, variances, mean):
        K = self.compute_covariance(hyper_parameters, variances)
        y = values - mean
        y = y.reshape(-1,1)
        if self.gp_system_solve_method == "solve":
            x = self.solve(K, y, compute_device = self.compute_device)
        #elif self.gp_system_solve_method == "minres":
        #    x, info = minres(K, y - mean)
        #    if info != 0:
        #        print("MINRES solve failed, going back to inversion")
        #        x = self.solve(K, y)
        #elif self.gp_system_solve_method == "cg":
        #    x, info = cg(K, y)
        #    if info != 0:
        #        print("CG solve failed, going back to inversion")
#       #         K_inv = self.safe_invert(K)
        #        x = self.solve(K, y)
        #elif self.gp_system_solve_method == "rank-n update":
        #    K_inv = self.covariance_update(K)
        #    x = K_inv @ (y)
        else:
            print("No solve method specified in _compute_covariance_value_product(). That might indicate a wrong input"); exit()
        return x,K
    ##################################################################################
    def compute_covariance(self, hyper_parameters, variances):
        """computes the covariance matrix from the kernel"""
        #print(hyper_parameters)
        CoVariance = self.kernel(
            self.points, self.points, hyper_parameters, self)
        #plt.imshow(CoVariance)
        #plt.show()
        self.add_to_diag(CoVariance, variances)
        return CoVariance
    ##################################################################################
    def covariance_update(self, CoVariance):
        """performs a rank-n-update of the covariance matrix"""
        OldInverse = np.array(self.K_inv())
        r = len(OldInverse)
        c = len(OldInverse[0])
        F = CoVariance[0:r, c:]
        G = CoVariance[r:, 0:c]
        H = CoVariance[r:, c:]
        EI = OldInverse

        S = H - (G.dot(EI).dot(F))
        SI = self.safe_invert(S)
        CoVarianceInverse = np.block(
            [
                [EI + EI @ F @ SI @G @ EI, -EI @ F @ SI],
                [-SI @ G @ EI, SI],
            ]
        )
        return CoVarianceInverse

    def slogdet(self, A, compute_device = "cpu"):
        """
        check this out
        """

        if compute_device == "cpu":
            A = torch.Tensor(A)
            sign, logdet = torch.slogdet(A)
            return sign.numpy(), logdet.numpy()
        if compute_device == "gpu":
            A = torch.Tensor(A).cuda()
            sign, logdet = torch.slogdet(A)
            return sign.cpu().numpy(), logdet.cpu().numpy()

    def solve(self, A, b, compute_device = "cpu"):
        """
        check out here
        """
        if compute_device == "cpu":
            A = torch.Tensor(A)
            b = torch.Tensor(b)
            try:
                x, lu = torch.solve(b,A)
            except np.linalg.LinAlgError: ###that is not going to work
                x, qr = torch.lstsq(b,A)
            return x.numpy()
        if compute_device == "gpu":
            A = torch.Tensor(A).cuda()
            b = torch.Tensor(b).cuda()
            try:
                x, lu = torch.solve(b,A)
            except np.linalg.LinAlgError:
                x, qr = torch.lstsq(b,A)
            return x.cpu().numpy()



    def safe_invert(self, Matrix):
        """computes in inverse or pseudo-inverse of a matrix"""
        try:
            M = np.linalg.inv(Matrix)
        except:
            print("Matrix not invertible, using pseudo-inverse")
            M = np.linalg.pinv(Matrix)
        return M
    ##################################################################################
    def add_to_diag(self,Matrix, Vector):
        d = np.einsum("ii->i", Matrix)
        d += Vector
        return Matrix

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###############################gp prediction###############################
    ###########################################################################
    ###########################################################################
    def compute_posterior_fvGP_pdf(self, x_input, x_output=np.array([[0.0]]), mode = 'cartesian product',
            compute_entropies = False, compute_prior_covariances = False,
            compute_posterior_covariances = False, compute_means = True):
        """
        Function to compute the variances, covariances and means of the (joint) posterior.
        gp.compute_posterior_fvGP_pdf(x_input, x_output=None, mode = 'cartesian product',compute_entropy = True, compute_covariance = True, compute_mean = True)
        Parameters:
        -----------
            x_input (2d numpy array): a set of elements of the input set, these are coordinates in the input (parameter) space
        Optional:
        ---------
            x_output(2d numpy array):             a set of elements of the output set, these are coordinates in the output space, default: None
            mode (str):                           'cartesian product' or 'stack', tell the algorithm how input and output coordinates will be combined, if 'stack' row number of x_input and x_output has to be the same
            compute_entropies (bool):             True/False, can be turned off if not required for better computational performance, default: True
            compute_prior_covariances (bool):     True/False, can be turned off if not required for better computational performance, default: True
            compute_posterior_covariances (bool): True/False, can be turned off if not required for better computational performance, default: True
            compute_mean(bool):                   True/False, can be turned off if not required for better computational performance, default: True
        Returns:
        --------
        res = {"input points": p,
               "posterior means": mean,
               "posterior covariances": covariance,
               "prior covariances": full_gp_covariances,
               "prior means": None,
               "entropies": entropies
               }
        """
        ####computation we need for all returns:
        if len(x_input.shape) == 1: exit("x_input is not a 2d numpy array, please see docstring for correct usage of gp.compute_posterior_fvGP_pdf() function, thank you")
        n_orig = len(x_input)
        if mode == 'cartesian product':
            new_points = np.zeros((len(x_input) * len(x_output), len(x_input[0]) + len(x_output[0])))
            counter = 0
            for element in itertools.product(x_input, x_output):
                new_points[counter] = np.concatenate([element[0], element[1]], axis=0)
                counter += 1   ###can't we append?
        elif mode == 'stack':
            new_points = np.column_stack([x_input,x_output])
        p = np.array(new_points)
        k = self.kernel(self.points,p,self.hyper_parameters,self)
        kk = self.kernel(p, p,self.hyper_parameters,self)
        #########################################
        if compute_prior_covariances == True: full_gp_covariances = np.asarray([np.block([[self.prior_covariance, k[:,i:i+n_orig]],[k[:,i:i+n_orig].T, kk[i:i+n_orig,i:i+n_orig]]]) for i in range(n_orig)])
        else: full_gp_covariances = None

        if compute_entropies == True:
            entropies = []
            for i in range(n_orig):
                sgn, logdet = self.slogdet(np.block([[self.prior_covariance, k[:,i:i+n_orig]],[k[:,i:i+n_orig].T, kk[i:i+n_orig,i:i+n_orig]]]), compute_device = self.compute_device)
                entropies.append(sgn*logdet)
            entropies = np.asarray(entropies)
        else: entropies = None

        if compute_means == True:
            A = k.T @ self.covariance_value_prod
            if self.gp_mean_function is None: mean = np.reshape(self.prior_mean[0] + A, (n_orig, len(x_output)))
            else: mean = np.reshape(self.gp_mean_function(x_input,self) + A, (n_orig, len(x_output)))
        else:
            mean = None
        if compute_posterior_covariances == True:
            #covariance_k_prod = np.zeros(k.shape)
            #info = np.zeros((len(k[0])))
            covariance_k_prod = self.solve(self.prior_covariance,k,compute_device = self.compute_device)
            #if self.gp_system_solve_method == "solve":
            #elif self.gp_system_solve_method == "inv" or self.gp_system_solve_method == "rank-n update":
            #    covariance_k_prod = self.K_inv() @ k
            #elif self.gp_system_solve_method == "minres":
            #    for i in range(len(k[0])):
            #        covariance_k_prod[:, i], info[i] = minres(self.prior_covariance, k[:, i])
            #elif self.gp_system_solve_method == "cg":
            #    for i in range(len(k[0])):
            #        covariance_k_prod[:, i], info[i] = cg(self.prior_covariance, k[:, i])
            #    if any(info) != 0:
            #        for i in range(len(k[0])):
            #            covariance_k_prod[:, i], info[i] = minres(self.prior_covariance, k[:, i])
            #else:
            #    print(
            #        "no valid solve method defined in fvGP; defined as: ", self.gp_system_solve_method
            #    )
            a = kk - (k.T @ covariance_k_prod)
            diag = np.diag(a)
            diag = np.where(diag<0.0,0.0,diag)
            if any([x < -0.0001 for x in np.diag(a)]):
                print("CAUTION, negative variances encountered. That normally means that the model is unstable.")
                print("Reduce the differentiability of the model, add more noise to the data, or double check the hyper-parameter optimization bounds")
                print("diagonal of the posterior covariance: ",np.diag(a))
            np.fill_diagonal(a,diag)
            M = len(x_output)
            covariance = np.asarray([
                a[i * M : (i + 1) * M, i * M : (i + 1) * M]
                for i in range(int(a.shape[0] / M))
            ])
        else:
            covariance = None
        res = {"input points": p,
               "posterior means": mean,
               "posterior covariances": covariance,
               "prior covariances": full_gp_covariances,
               "prior means": None,
               "entropies": entropies
               }
        return res
    ###########################################################################
    def compute_posterior_fvGP_pdf_gradient(self, x_input, x_output=np.array([[0.0]]), direction = 0, mode = 'cartesian product',
            compute_entropies = False, compute_prior_covariances = False,
            compute_posterior_covariances = False, compute_means = True):
        """
        Function to compute the variances, covariances and means of the (joint) posterior.
        gp.compute_posterior_fvGP_pdf_gradient(x_input, x_output=None, direction = 0, mode = 'cartesian product',compute_entropy = True, compute_covariance = True, compute_mean = True)
        Parameters:
        -----------
            x_input (2d numpy array): a set of elements of the input set, these are coordinates in the input (parameter) space
        Optional:
        ---------
            x_output(2d numpy array): a set of elements of the output set, these are coordinates in the output space, default: None
            direction (int): index of the derivative direction
            mode (str): 'cartesian product' or 'stack', tell the algorithm how input and output coordinates will be combined, if 'stack' row number of x_input and x_output has to be the same
            compute_entropies (bool):             True/False, can be turned off if not required for better computational performance, default: True
            compute_prior_covariances (bool):     True/False, can be turned off if not required for better computational performance, default: True
            compute_posterior_covariances (bool): True/False, can be turned off if not required for better computational performance, default: True
            compute_mean(bool):                   True/False, can be turned off if not required for better computational performance, default: True
        Returns:
        --------
        res = {"input points": p,
               "posterior mean gradients": mean,
               "posterior covariance gradients": covariance,
               "prior covariance gradients": full_gp_covariances,
               "prior mean gradients": None,
               "entropy gradients": entropies
               }
        """
        ####computation we need for all returns:
        if len(x_input.shape) == 1: exit("x_input is not a 2d numpy array, please see docstring for correct usage of gp.fvGP function, thank you")
        n_orig = len(x_input)
        if mode == 'cartesian product':
            new_points = np.zeros((len(x_input) * len(x_output), len(x_input[0]) + len(x_output[0])))
            counter = 0
            for element in itertools.product(x_input, x_output):
                new_points[counter] = np.concatenate([element[0], element[1]], axis=0)
                counter += 1   ###can't we append?
        elif mode == 'stack':
            new_points = np.column_stack([x_input,x_output])
        p = np.array(new_points)
        k = self.kernel(self.points,p,self.hyper_parameters,self)
        k_g = self.d_kernel_dx(p,self.points, direction,self.hyper_parameters).T
        kk =  self.kernel(p, p,self.hyper_parameters,self)
        kk_g =  self.d_kernel_dx(p, p,direction,self.hyper_parameters)
        if compute_prior_covariances == True: full_gp_covariances = np.asarray([np.block([[self.prior_covariance, k_g[:,i:i+n_orig]],[k_g[:,i:i+n_orig].T, kk_g[i:i+n_orig,i:i+n_orig]]]) for i in range(n_orig)])
        else: full_gp_covariances = None

        if compute_entropies == True:
            entropies = []
            for i in range(n_orig):
                Sigma =   np.block([[self.prior_covariance, k[:,i:i+n_orig]],  [k[:,i:i+n_orig].T,   kk[i:i+n_orig,i:i+n_orig]]])
                Sigma_d = np.block([[self.prior_covariance, k_g[:,i:i+n_orig]],[k_g[:,i:i+n_orig].T, kk_g[i:i+n_orig,i:i+n_orig]]])
                entropies.append(np.trace(np.linalg.inv(Sigma)@Sigma_d))
            entropies = np.asarray(entropies)
        else: entropies = None

        if compute_means == True:
            A = k_g.T @ self.covariance_value_prod
            mean = np.reshape(A, (n_orig, len(x_output)))
        else:
            mean = None
        if compute_posterior_covariance == True:
            covariance_k_prod = self.solve(self.prior_covariance,k,compute_device = self.compute_device)
            covariance_k_g_prod = self.solve(self.prior_covariance,k_g,compute_device = self.compute_device)
            #if self.gp_system_solve_method == "inv" or self.gp_system_solve_method == "rank-n update":
            #    covariance_k_prod = self.K_inv() @ k
            #    covariance_k_g_prod = self.K_inv() @ k_g
            #elif self.gp_system_solve_method == "minres":
            #    for i in range(len(k[0])):
            #        covariance_k_prod[:, i], info[i] = minres(self.prior_covariance, k[:, i])
            #        covariance_k_g_prod[:, i], info[i] = minres(self.prior_covariance, k_g[:, i])
            #elif self.gp_system_solve_method == "cg":
            #    for i in range(len(k[0])):
            #        covariance_k_prod[:, i], info[i] = cg(self.prior_covariance, k[:, i])
            #        covariance_k_g_prod[:, i], info[i] = cg(self.prior_covariance, k_g[:, i])
            #    if any(info) != 0:
            #        for i in range(len(k[0])):
            #            covariance_k_prod[:, i], info[i] = minres(self.prior_covariance, k[:, i])
            #            covariance_k_g_prod[:, i], info[i] = minres(self.prior_covariance, k_g[:, i])
            #else:
            #    print(
            #        "no valid solve method defined in fvGP; defined as: ", self.gp_system_solve_method
            #    )
            a = kk_g - ((k_g.T @ covariance_k_prod) + (k.T @ covariance_k_g_prod))
            M = len(x_output)
            covariance = [
                a[i * M : (i + 1) * M, i * M : (i + 1) * M]
                for i in range(int(a.shape[0] / M))
                ]
        else:
            covariance = None

        r1 = np.reshape(A, (n_orig, self.output_num))
        res = {"input points": p,
               "posterior mean gradients": mean,
               "posterior covariance gradients": covariance,
               "prior covariance gradients": full_gp_covariances,
               "prior mean gradients": None,
               "entropy gradients": entropies
               }
        return res

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
    def default_kernel(self,x1,x2,hyper_parameters,obj):
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
        hps = hyper_parameters
        distance_matrix = np.zeros((len(x1),len(x2)))
        for i in range(len(x1[0])):
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
#    @profile
    def d_gp_kernel_dx(self, points1, points2, direction, hyper_parameters):
        new_points = np.array(points1)
        epsilon = 1e-6
        new_points[:,direction] += epsilon
        a = self.kernel(new_points, points2, hyper_parameters,self)
        b = self.kernel(points1,    points2, hyper_parameters,self)
        derivative = ( a - b )/epsilon
        return derivative
#    @profile
    def d_gp_kernel_dh(self, points1, points2, direction, hyper_parameters):
        new_hyper_parameters1 = np.array(hyper_parameters)
        new_hyper_parameters2 = np.array(hyper_parameters)
        epsilon = 1e-6
        new_hyper_parameters1[direction] += epsilon
        new_hyper_parameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyper_parameters1,self)
        b = self.kernel(points1, points2, new_hyper_parameters2,self)
        derivative = ( a - b )/(2.0*epsilon)
        return derivative
#    @profile
    def gradient_gp_kernel(self, points1, points2, hyper_parameters):
        gradient = np.empty((len(hyper_parameters), len(points1),len(points2)))
        for direction in range(len(hyper_parameters)):
            gradient[direction] = self.d_gp_kernel_dh(points1, points2, direction, hyper_parameters)
        return gradient
#    @profile
    def d2_gp_kernel_dh2(self, points1, points2, direction1, direction2, hyper_parameters):
        epsilon = 1e-6
        new_hyper_parameters1 = np.array(hyper_parameters)
        new_hyper_parameters2 = np.array(hyper_parameters)
        new_hyper_parameters3 = np.array(hyper_parameters)
        new_hyper_parameters4 = np.array(hyper_parameters)

        new_hyper_parameters1[direction1] = new_hyper_parameters1[direction1] + epsilon
        new_hyper_parameters1[direction2] = new_hyper_parameters1[direction2] + epsilon

        new_hyper_parameters2[direction1] = new_hyper_parameters2[direction1] + epsilon
        new_hyper_parameters2[direction2] = new_hyper_parameters2[direction2] - epsilon

        new_hyper_parameters3[direction1] = new_hyper_parameters3[direction1] - epsilon
        new_hyper_parameters3[direction2] = new_hyper_parameters3[direction2] + epsilon

        new_hyper_parameters4[direction1] = new_hyper_parameters4[direction1] - epsilon
        new_hyper_parameters4[direction2] = new_hyper_parameters4[direction2] - epsilon

        return (self.kernel(points1,points2,new_hyper_parameters1,self) \
              - self.kernel(points1,points2,new_hyper_parameters2,self) \
              - self.kernel(points1,points2,new_hyper_parameters3,self) \
              + self.kernel(points1,points2,new_hyper_parameters4,self))\
              / (4.0*(epsilon**2))
#    @profile
    def hessian_gp_kernel(self, points1, points2, hyper_parameters):
        hessian = np.zeros((len(hyper_parameters),len(hyper_parameters), len(points1),len(points2)))
        for i in range(len(hyper_parameters)):
            for j in range(i+1):
                hessian[i,j] = hessian[j,i] = self.d2_gp_kernel_dh2(points1, points2, i,j, hyper_parameters)
        return hessian

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################OTHER#######################################################
    ##################################################################################
    ##################################################################################

    def compute_standard_value_positions(self):
        value_pos = np.zeros((self.point_number, self.output_num, self.output_dim))
        for j in range(self.output_num):
            value_pos[:, j, :] = j
        return value_pos

    def transform_index_set(self):
        new_points = np.zeros((self.point_number * self.output_num, self.iset_dim))
        new_values = np.zeros((self.point_number * self.output_num))
        new_variances = np.zeros((self.point_number * self.output_num))
        new_mean = np.zeros((self.point_number * self.output_num))
        for i in range(self.output_num):
            new_points[i * self.point_number : (i + 1) * self.point_number] = \
            np.column_stack([self.points, self.value_positions[:, i, :]])
            new_values[i * self.point_number : (i + 1) * self.point_number] = \
            self.values[:, i]
            new_variances[i * self.point_number : (i + 1) * self.point_number] = \
            self.variances[:, i]
            new_mean[i * self.point_number : (i + 1) * self.point_number] =\
            self.mean_vec[:, i]
        return new_points, new_values, new_variances, new_mean

    def compute_mean(self):
        """evaluates the gp mean function at the data points """
        if self.gp_mean_function is None:
            mean = np.zeros((self.variances.shape))
            for i in range(len(mean[0])):
                mean[:, i] = np.mean(self.values[:, i], axis=0)
            return mean
        else:
            mean = np.zeros((self.variances.shape))
            for i in range(len(mean)):
                mean[i, :] = self.gp_mean_function(np.array([self.points[i]]), self)
            return mean

    ############################################################
    ######################finite difference derivative##########
    ############################################################

###########################################################################
###########################################################################
###########################################################################
#######################################Testing#############################
###########################################################################
###########################################################################
    def test_derivatives(self):
        print("====================================")
        print("====================================")
        print("====================================")
        res = self.compute_posterior_fvGP_pdf(np.array([[0.20,0.15]]),self.value_positions[-1])
        a1 = res["means"]
        b1 = res["covariances"]

        res = self.compute_posterior_fvGP_pdf(np.array([[0.200001,0.15]]),self.value_positions[-1])
        a2 = res["means"]
        b2 = res["covariances"]

        print("====================================")
        res = self.compute_posterior_fvGP_pdf_gradient(np.array([[0.2,0.15]]),self.value_positions[-1],direction = 0)
        print("function values: ",a1,a2)
        print("gradient:")
        print((a2-a1)/0.000001,res["means grad"])
        print("variances:")
        print((b2-b1)/0.000001,res["covariances grad"])
        print("compare values and proceed with enter")
        input()

###########################################################################
#######################################MISC################################
###########################################################################
    def gradient(self, function, point, epsilon = 10e-3,*args):
        """
        This function calculates the gradient of a function by using finite differences

        Extended description of function.

        Parameters:
        function (function object): the function the gradient should be computed of
        point (numpy array 1d): point at which the gradient should be computed

        optional:
        epsilon (float): the distance used for the evaluation of the function

        Returns:
        numpy array of gradient

        """
        gradient = np.zeros((len(point)))
        for i in range(len(point)):
            new_point = np.array(point)
            new_point[i] = new_point[i] + epsilon
            gradient[i] = (function(new_point,args) - function(point,args))/ epsilon
        return gradient
    def hessian(self, function, point, epsilon = 10e-3, *args):
        """
        This function calculates the hessian of a function by using finite differences

        Extended description of function.

        Parameters:
        function (function object): the function, the hessian should be computed of
        point (numpy array 1d): point at which the gradient should be computed

        optional:
        epsilon (float): the distance used for the evaluation of the function

        Returns:
        numpy array of hessian

        """
        hessian = np.zeros((len(point),len(point)))
        for i in range(len(point)):
            for j in range(len(point)):
                new_point1 = np.array(point)
                new_point2 = np.array(point)
                new_point3 = np.array(point)
                new_point4 = np.array(point)

                new_point1[i] = new_point1[i] + epsilon
                new_point1[j] = new_point1[j] + epsilon

                new_point2[i] = new_point2[i] + epsilon
                new_point2[j] = new_point2[j] - epsilon

                new_point3[i] = new_point3[i] - epsilon
                new_point3[j] = new_point3[j] + epsilon

                new_point4[i] = new_point4[i] - epsilon
                new_point4[j] = new_point4[j] - epsilon

                hessian[i,j] = \
                (function(new_point1,args) - function(new_point2,args) - function(new_point3,args) +  function(new_point4,args))\
                / (4.0*(epsilon**2))
        return hessian

###########################################################################
###################################Obsolete################################
###########################################################################

    ##################################################
    #########Kernels##################################
    ##################################################

    ##################################################
    #########Derivatives with respect to x ###########
    ##################################################
    """
    def d_squared_exponential_kernel_dx(self, distance, length):
        dkernel_dx = -(distance / length ** 2) * np.exp(
            -(distance ** 2) / (2.0 * (length ** 2))
        )
        return dkernel_dx

    def d_exponential_kernel_dx(self, distance, length):
        dkernel_dx = -(1.0 / length) * np.exp(-(distance) / (length))
        return dkernel_dx

    def d_matern_kernel_diff1_dx(self, distance, length):
        ee = np.exp(-(np.sqrt(3.0) * distance) / length)
        a = np.sqrt(3.0) / length
        b = a * distance
        dkernel_dx = a * (ee - (ee * (1.0 + b)))
        return dkernel_dx

    def d_matern_kernel_diff2_dx(self, distance, length):
        ee = np.exp(-(np.sqrt(5.0) * distance) / length)
        a = np.sqrt(5.0) / length
        b = a * distance
        c = (10.0 * distance) / (3.0 * length ** 2)
        d = (5.0 * distance ** 2) / (3.0 * length ** 2)
        dkernel_dx = ((a + c) * ee) - ((1.0 + b + d) * a * ee)
        return dkernel_dx

    def d_sparse_kernel_dx(self,distance, radius):
        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d>radius] = radius - 10e-6
        dkernel_dx = 2.0**(3.0/2.0)*d*((3.0*radius**2*np.sqrt(1-d**2/radius**2)-3.0*d**2+3.0*radius**2)*\
                np.log(d/(radius*(np.sqrt(1-d**2/radius**2)+1.0)))+2.0*radius**2*(1.0-d**2/radius**2)**(3.0/2.0)+\
                (radius**2-d**2)*np.sqrt(1.0-d**2/radius**2)-3.0*d**2+3.0*radius**2)/\
                (3.0*np.sqrt(np.pi)*radius**4*np.sqrt(1.0-d**2/radius**2)*(np.sqrt(1.0-d**2/radius**2)+1))
        dkernel_dx[d > radius] = 0.0
        return dkernel_dx
    def dd_sparse_kernel_dxx(self,distance, radius):
        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d>radius] = radius - 10e-6
        r = radius
        ddkernel_dxx = -(2**(3/2)*((np.sqrt(1-d**2/r**2)*(3*r**2*d**4-3*r**4*d**2)\
                -6*r**2*d**4+(1-d**2/r**2)**(3/2)*(6*r**4*d**2-6*r**6)+12*r**4*d**2-6*r**6)\
                *np.log(d/(r*(np.sqrt(1-d**2/r**2)+1)))-2*r**6*(1-d**2/r**2)**(5/2)+6*d**6+\
                np.sqrt(1-d**2/r**2)*(6*r**2*d**4-6*r**4*d**2)-24*r**2*d**4+(1-d**2/r**2)**\
                (3/2)*(16*r**4*d**2-10*r**6)+30*r**4*d**2-12*r**6))/(3*np.sqrt(np.pi)*r**8*\
                (1-d**2/r**2)**(3/2)*(np.sqrt(1-d**2/r**2)+1)**2)
        ddkernel_dxx[d > radius] = 0.0
        return ddkernel_dxx

    ##################################################
    #########Derivatives with respect to length scale#
    ##################################################
    def d_squared_exponential_kernel_dl(self, distance, length):
        # returns the derivative of the 1d squared exponential kernel
        # with respect to length scale
        return self.squared_exponential_kernel(distance, length) * (
            (distance ** 2) / (length ** 3)
        )

    def d_exponential_kernel_dl(self, distance, length):
        # returns the derivative of the 1d exponential kernel
        # with respect to length scale
        return np.exp(-(distance) / (length)) * (distance / (length ** 2))

    def d_matern_kernel_diff1_dl(self, distance, length):
        # returns the derivative of the 1d matern kernel (P1)
        # with respect to length scale
        return np.exp(-((np.sqrt(3.0) * distance) / (length))) * (
            (3.0 * distance ** 2) / (length ** 3)
        )

    def d_matern_kernel_diff2_dl(self, distance, length):
        # returns the derivative of the 1d Matern kernel (P1)
        # with respect to length scale
        a = np.exp(-(np.sqrt(5.0) * distance) / (length))
        return a * (
            (distance ** 2 * ((5.0 * length) + (5.0 ** (3.0 / 2.0) * distance)))
            / (3.0 * length ** 4)
        )

    def d_sparse_kernel_dl(self, distance, radius):
        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d>radius] = radius - 10e-6
        r = radius

        dsk_dl = (np.sqrt(2)*((3*d*(np.sqrt(1-d**2/r**2)+1)*(-d/((np.sqrt(1-d**2/r**2)+1)*r**2)-d**3/\
                ((np.sqrt(1-d**2/r**2)+1)**2*np.sqrt(1-d**2/r**2)*r**4)))/r+(d**2*((2*d**2)/r**2+1))/\
                (np.sqrt(1-d**2/r**2)*r**3)-(4*d**2*np.sqrt(1-d**2/r**2))/r**3-(6*d**2*\
                np.log(d/((np.sqrt(1-d**2/r**2)+1)*r)))/r**3))/(3*np.sqrt(np.pi))

        dsk_dl[d > radius] = 0.0
        return dsk_dl

    def d_sparse_kernel_dl(self, distance, radius):
        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d>radius] = radius - 10e-6
        r = radius

        ddsk_dll = (np.sqrt(2)*d**2*(((18*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))+39)*\
            (1-d**2/r**2)**(5/2)+(18*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))+9)*(1-d**2/r**2)**(3/2)+\
            36*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))+48)*r**8+((-18*d**2*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))\
            -39*d**2)*(1-d**2/r**2)**(5/2)+(-18*d**2*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))-16*d**2)*\
            (1-d**2/r**2)**(3/2)-2*d**2*np.sqrt(1-d**2/r**2)-108*d**2*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))-\
            168*d**2)*r**6+(13*d**4*(1-d**2/r**2)**(3/2)-2*d**4*np.sqrt(1-d**2/r**2)+108*d**4*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))+\
            216*d**4)*r**4+(4*d**6*np.sqrt(1-d**2/r**2)-36*d**6*np.log(d/((np.sqrt(1-d**2/r**2)+1)*r))-120*d**6)*\
            r**2+24*d**8))/(3*np.sqrt(np.pi)*(np.sqrt(1-d**2/r**2)+1)**2*(1-d**2/r**2)**(3/2)*r**10*(r**2-d**2))

        ddsk_dll[d > radius] = 0.0
        return ddsk_dll



    def dd_squared_exponential_kernel_dll(self, distance, length):
        # returns the derivative of the 1d squared exponential kernel
        # with respect to length scale
        return (
            self.squared_exponential_kernel(distance, length) * ((distance ** 4) / (length ** 6))
        ) - (
            3.0
            * selfKernel.SquaredExpDer(distance, length)
            * ((distance ** 2) / (length ** 4))
        )

    def dd_exponential_kernel_dll(self, distance, length):
        # returns the derivative of the 1d exponential kernel
        # with respect to length scale
        ee = self.exponential_kernel(distance, length)
        return ee * (
            (-2.0 * distance / (length ** 3)) + ((distance ** 2) / (length ** 4))
        )

    def dd_matern_kernel_diff1_dll(self, distance, length):
        # returns the derivative of the 1d Matern kernel (P1)
        # with respect to length scale
        ee = np.exp((-np.sqrt(3.0) * distance) / (length))
        return ee * (
            ((3.0 ** (3.0 / 2.0) * distance ** 3) / (length ** 5))
            - ((9.0 * distance ** 2) / (length ** 4))
        )

    def dd_matern_kernel_diff2_dll(self, distance, length):
        # returns the derivative of the 1d Matern kernel (P1)
        # with respect to length scale
        a = np.exp(-(np.sqrt(5.0) * distance) / (length))
        return a * (
            (
                (15.0 * length ** 2)
                + (3.0 * 5.0 ** (3.0 / 2.0) * distance * length)
                - (25.0 * distance ** 2)
            )
            / (3.0 * length ** 6)
        )

    def dd_bump_function_kernel_dll(self, distance, beta, r):
        A = distance ** 2 - r ** 2
        B = BumpFunction(distance, beta, r)
        if distance < r:
            return (
                (2 * beta * r ** 2)
                * (
                    3.0 * distance ** 4
                    + ((2.0 * beta - 2) * r ** 2 * distance ** 2)
                    - r ** 4
                )
                * B
            ) / A ** 4
        else:
            return 0.0
    """
""" derivative test:


        print("gradient test")
        e = 0.01
        d = 0
        hps = np.random.rand(4) * 1.0   #np.array(self.hyper_parameters)
        hps[0] = 200.0
        print("hps:", hps)
        hps1 = np.array(hps)
        hps2 = np.array(hps)
        hps2[d] = hps2[d] + e
        hps1[d] = hps1[d] - e
        print("hps1: ", hps1)
        print("hps2: ", hps2)
        a = self.log_likelihood(hps1,
        self.values,
        self.variances,
        self.prior_mean)
        b = self.log_likelihood(hps2,
        self.values,
        self.variances,
        self.prior_mean)
        print("likelihoods: ",a,b)
        print("test grad: ",( b - a )/(2.0*e))
        print("comp grad: ",self.log_likelihood_gradient_wrt_hyper_parameters(hps,
        self.values,
        self.variances,
        self.prior_mean))
        print("hessian test")
        epsilon = 1e-5
        d1 = 0
        d2 = 1
        hps = np.random.rand(4) * 1.0   #np.array(self.hyper_parameters)
        hps[0] = 200.0
        hps1 = np.array(hps)
        hps2 = np.array(hps)
        hps3 = np.array(hps)
        hps4 = np.array(hps)
        hps1[d1] = hps1[d1] + epsilon
        hps1[d2] = hps1[d2] + epsilon

        hps2[d1] = hps2[d1] + epsilon
        hps2[d2] = hps2[d2] - epsilon

        hps3[d1] = hps3[d1] - epsilon
        hps3[d2] = hps3[d2] + epsilon

        hps4[d1] = hps4[d1] - epsilon
        hps4[d2] = hps4[d2] - epsilon

        f1 = self.log_likelihood(hps1,
        self.values,
        self.variances,
        self.prior_mean)
        f2 = self.log_likelihood(hps2,
        self.values,
        self.variances,
        self.prior_mean)
        f3 = self.log_likelihood(hps3,
        self.values,
        self.variances,
        self.prior_mean)
        f4 = self.log_likelihood(hps4,
        self.values,
        self.variances,
        self.prior_mean)
        print("fin diff hess at ",d1," ",d2,(f1 - f2 - f3 +  f4) / (4.0*(epsilon**2)))
        print(self.log_likelihood_hessian_wrt_hyper_parameters(hps,
        self.values,
        self.variances,
        self.prior_mean))

        plt.show()
        exit()
"""
