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
        init_hyperparameters:          list of (1d numpy array (>0)), one entry per GP

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
        init_hyperparameters,
        variances = None,
        compute_device = "cpu",
        gp_kernel_function = [None],
        gp_mean_function = [None],
        sparse = False,
        normalize_y = False
        ):
        """
        The constructor for the ensemblegp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        self.number_of_GPs = number_of_GPs
        self.ensembleGPs = [GP(input_space_dim,points,values,init_hyperparameters[i],
                               variances = variances,compute_device = compute_device,
                               gp_kernel_function = gp_kernel_function[i]
                               ,gp_mean_function = gp_mean_function[i],
                               sparse = sparse, normalize_y = normalize_y)
                               for i in range(number_of_GPs)]
    def update_EnsembleGP_data():
        return 0

    def stop_training():
        return 0

    def kill_training():
        return 0

    def train():
        return 0

    def train_async():
        return 0
    def update_hyperparameters():
        return 0
    
    def optimize_log_likelihood_async():
        return 0

    def optimize_log_likelihood():
        return 0

    def ensemble_log_likelihood(self,hyperparameters):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        L = 0
        for i in range(self.number_of_GPs)
            L += weight[i] * self.ensembleGPs[i].log_likelihood(hyperparameters[i])
        return L

    def ensemble_log_likelihood_gradient(self,hyperparameters):
        return 0

    def ensemble_log_likelihood_hessian(self,hyperparameters):
        return 0

    ##########################################################
    def posterior_mean(self, x_iset):
        return 0

    def posterior_mean_grad(self, x_iset):
        return 0

    def posterior_covariance(self, x_iset):
        return 0

    def posterior_covariance_grad(self, x_iset):
        return 0
