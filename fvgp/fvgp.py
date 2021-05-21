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
along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
from fvgp.gp import GP


class fvGP(GP):
    """
    fvGP class: Provides all tool for a multi-task GP.
    This class inherits a lot of its method form the GP class.

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space
        dim2: number of dimension of the output space

    Attributes:
        input_space_dim (int):         dim1
        output_space_dim (int):        dim2
        output_number (int):           n
        points (N x dim1 numpy array): 2d numpy array of points
        values (N x n numpy array):    2d numpy array of values
        init_hyperparameters:          1d numpy array

    Optional Attributes:
        value_positions (N x dim1 x dim2 numpy array):  the positions of the outputs in the output space
                                                        default = np.array([[[0],[1],[2],...]])
        variances (N x n numpy array):                  variances of the values, default = array of shape of points
                                                        with 1 % of the values
        compute_device:                                 cpu/gpu, default = cpu
        gp_kernel_function(func):                       None/function defining the 
                                                        kernel def name(x1,x2,hyperparameters,self), default = None
        gp_mean_function(func):                         None/a function def name(x, self), default = None
        sparse (bool):                                  default = False
        normalize_y:                                    default = False, normalizes the values \in [0,1]

    Example:
        obj = fvGP(3,1,2,np.array([[1,2,3],[4,5,6]]),
                         np.array([[2,3],[13,27.2]]),
                         np.array([2,3,4,5]),
                         value_positions = np.array([[[0],[1]],[[0],[1]]]),
                         variances = np.array([[0.001,0.01],[0.1,2]]),
                         gp_kernel_function = kernel_function,
                         gp_mean_function = some_mean_function
        )
    """
    def __init__(
        self,
        input_space_dim,
        output_space_dim,
        output_number,
        points,
        values,
        init_hyperparameters,
        value_positions = None,
        variances = None,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_mean_function = None,
        sparse = False,
        normalize_y = False
        ):
        """
        The constructor for the fvgp class.
        type help(fvGP) for more information about attributes, methods and their parameters
        """
        self.data_x = np.array(points)
        self.data_y = np.array(values)
        self.input_space_dim = input_space_dim
        self.point_number, self.output_num, self.output_dim = len(points), output_number, output_space_dim
        ###check the output dims
        if np.ndim(values) == 1:
            raise ValueError("the output number is 1, you can use GP for single-task GPs")
        if output_number != len(values[0]):
            raise ValueError("the output number is not in agreement with the data values given")
        if output_space_dim == 1 and isinstance(value_positions, np.ndarray) == False:
            self.value_positions = self.compute_standard_value_positions()
        elif self.output_dim > 1 and isinstance(value_positions, np.ndarray) == False:
            raise ValueError(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the fvGP class")
        else:
            self.value_positions = np.array(value_positions)
        if variances is None:
            self.variances = np.ones((self.data_y.shape)) * abs(self.data_y / 100.0)
            print("CAUTION: fvGP reports that you have not provided data variances, they will set to be 1 percent of the data values!")
        else:
            self.variances = np.array(variances)

        self.iset_dim = self.input_space_dim + self.output_dim
        ####transform the space
        self.data_x, self.data_y, self.variances = self.transform_index_set()
        self.point_number = len(self.data_x)

        ####init GP
        GP.__init__(self,self.iset_dim, self.data_x,self.data_y,init_hyperparameters,
                variances = self.variances,compute_device = compute_device,
                gp_kernel_function = gp_kernel_function, gp_mean_function = gp_mean_function,
                sparse = sparse, normalize_y = normalize_y)

        self.hyperparameters = np.array(init_hyperparameters)
        ##########################################
    def update_fvgp_data(
        self,
        points,
        values,
        value_positions = None,
        variances = None,
        ):

        """
        This function updates the data in the gp_class.

        Attributes:
            points (N x dim1 numpy array): An array of points.
            values (N x n):                An array of values.

        optional attributes:
            values_positions (N x dim1 x dim2 numpy array): the positions of the outputs in the output space
            variances (N x n):                              variances of the values
            """
        self.data_x = np.array(points)
        self.data_y = np.array(values)
        ##########################################
        #######prepare value positions############
        ##########################################
        if self.output_dim == 1 and isinstance(value_positions, np.ndarray) == False:
            self.value_positions = self.compute_standard_value_positions()
        elif self.output_dim > 1 and isinstance(value_positions, np.ndarray) == False:
            raise ValueError(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the fvGP class. EXIT"
            )
        else:
            self.value_positions = np.array(value_positions)
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((self.data_y.shape)) * abs(self.data_y / 100.0)
        else:
            self.variances = np.array(variances)
        ######################################
        #####transform to index set###########
        ######################################
        self.point_number = len(self.data_x)
        self.data_x, self.data_y, self.variances = self.transform_index_set()
        self.point_number = len(self.data_x)
        GP.update_gp_data(self,self.data_x, self.data_y, self.variances)

    def compute_standard_value_positions(self):
        value_pos = np.zeros((self.point_number, self.output_num, self.output_dim))
        for j in range(self.output_num):
            value_pos[:, j, :] = j
        return value_pos

    def transform_index_set(self):
        new_points = np.zeros((self.point_number * self.output_num, self.iset_dim))
        new_values = np.zeros((self.point_number * self.output_num))
        new_variances = np.zeros((self.point_number * self.output_num))
        for i in range(self.output_num):
            new_points[i * self.point_number : (i + 1) * self.point_number] = \
            np.column_stack([self.data_x, self.value_positions[:, i, :]])
            new_values[i * self.point_number : (i + 1) * self.point_number] = \
            self.data_y[:, i]
            new_variances[i * self.point_number : (i + 1) * self.point_number] = \
            self.variances[:, i]
        return new_points, new_values, new_variances

    def multi_task_kernel1(self):
        return 0
