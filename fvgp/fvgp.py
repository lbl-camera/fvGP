#!/usr/bin/env python

import dask.distributed as distributed
from loguru import logger

from . import __version__

import matplotlib.pyplot as plt
import numpy as np
import math

import itertools
import time
import torch
from functools import partial
from fvgp.gp import GP


class fvGP(GP):
    """
    This class provides all the tools for a multi-task Gaussian Process (GP).
    This class allows for full HPC support for training. After initialization, this
    class provides all the methods described for the GP class.

    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space.
    output_space_dim : int
        Integer specifying the number of dimensions of the output space. Most often 1.
    output_number : int
        Number of output values.
    points : np.ndarray
        The point positions. Shape (V x D), where D is the `input_space_dim`.
    values : np.ndarray
        The values of the data points. Shape (V,output_number).
    init_hyperparameters : np.ndarray
        Vector of hyperparameters used by the GP initially. The class provides methods to train hyperparameters.
    value_positions : np.ndarray, optional
        A 3-D numpy array of shape (U x output_number x output_dim), so that for each measurement position, the outputs
        are clearly defined by their positions in the output space. The default is np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
        point in the input space. The default is only permissible if output_dim is 1.
    variances : np.ndarray, optional
        An numpy array defining the uncertainties in the data `values`. Shape (V x 1) or (V). Note: if no
        variances are provided they will be set to `abs(np.mean(values) / 100.0`.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
    gp_kernel_function : Callable, optional
        A function that calculates the covariance between datapoints. It accepts as input x1 (a V x D array of positions),
        x2 (a U x D array of positions), hyperparameters (a 1-D array of length D+1 for the default kernel), and a
        `gpcam.gp_optimizer.GPOptimizer` instance. The default is a stationary anisotropic kernel
        (`fvgp.gp.GP.default_kernel`).
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative  of the covariance between datapoints with respect to the hyperparameters.
        If provided, it will be used for local training and can speed up the calculations.
        It accepts as input x1 (a V x D array of positions),
        x2 (a U x D array of positions) and hyperparameters (a 1-D array of length D+1 for the default kernel).
        The default is a finite difference calculation.
        If 'ram_economy' is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and the output
        is a numpy array of shape (V x U).
        If 'ram economy' is False,the function's input is x1, x2, hyperparameters, and the output is
        a numpy array of shape (len(hyperparameters) x U x V)
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at an input position. It accepts as input a
        `gpcam.gp_optimizer.GPOptimizer` instance, an array of positions (of size V x D), and hyperparameters (a 1-D
        array of length D+1 for the default kernel). The return value is a 1-D array of length V. If None is provided,
        `fvgp.gp.GP.default_mean_function` is used.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the prior mean at an input position with respect to the hyperparameters.
        It accepts as input hyperparameters (a 1-D
        array of length D+1 for the default kernel). The return value is a 2-D array of shape (D x len(hyperparameters)). If None is provided,
        a finite difference scheme is used.
    normalize_y : bool, optional
        If True, the data point values will be normalized to max(initial values) = 1. The dfault is False.
    use_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance matrix after each training or update of the dataset,
        which makes computing the posterior covariance faster.
        For larger problems (>2000 data points), the use of inversion should be avoided due to computational instability. The default is
        False. Note, the training will always use a linear solve instead of the inverse for stability reasons.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the marginal log_likelihood is/are used for the training.

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
        gp_kernel_function_grad = None,
        gp_mean_function = None,
        gp_mean_function_grad = None,
        normalize_y = False,
        use_inv = False,
        ram_economy = True
        ):

        self.x_data = np.array(points)
        self.y_data = np.array(values)
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
            self.variances = np.ones((self.y_data.shape)) * abs(self.y_data / 100.0)
            logger.warning("fvGP reports that you have not provided data variances, they will set to be 1 percent of the data values!")
        else:
            self.variances = np.array(variances)

        self.iset_dim = self.input_space_dim + self.output_dim
        ####transform the space
        self.x_data, self.y_data, self.variances = self.transform_index_set()
        self.point_number = len(self.x_data)

        ####init GP
        GP.__init__(self,self.iset_dim, self.x_data,self.y_data,init_hyperparameters,
                variances = self.variances,compute_device = compute_device,
                gp_kernel_function = gp_kernel_function, gp_mean_function = gp_mean_function,
                gp_kernel_function_grad = gp_kernel_function_grad, gp_mean_function_grad = gp_mean_function_grad,
                use_inv = use_inv, normalize_y = normalize_y,ram_economy = ram_economy)

        self.hyperparameters = np.array(init_hyperparameters)

    def update_fvgp_data(
        self,
        points,
        values,
        value_positions = None,
        variances = None,
        ):

        """
        This function updates the data in the fvgp object instance.
        The data will NOT be appended but overwritten!
        Please provide the full updated data set.

        Parameters
        ----------
        points : np.ndarray
            The point positions. Shape (V x D), where D is the `input_space_dim`.
        values : np.ndarray
            The values of the data points. Shape (V,output_number).
        value_positions : np.ndarray, optional
            A 3-D numpy array of shape (U x output_number x output_dim), so that for each measurement position, the outputs
            are clearly defined by their positions in the output space. The default is np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
            point in the input space. The default is only permissible if output_dim is 1.
        variances : np.ndarray, optional
            An numpy array defining the uncertainties in the data `values`. Shape (V x 1) or (V). Note: if no
            variances are provided they will be set to `abs(np.mean(values) / 100.0`.
        """
        self.x_data = np.array(points)
        self.y_data = np.array(values)
        self.point_number = len(self.x_data)
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
            self.variances = np.ones((self.y_data.shape)) * abs(self.y_data / 100.0)
        else:
            self.variances = np.array(variances)
        ######################################
        #####transform to index set###########
        ######################################
        self.x_data, self.y_data, self.variances = self.transform_index_set()
        self.point_number = len(self.x_data)
        GP.update_gp_data(self,self.x_data, self.y_data, self.variances)

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
            np.column_stack([self.x_data, self.value_positions[:, i, :]])
            new_values[i * self.point_number : (i + 1) * self.point_number] = \
            self.y_data[:, i]
            new_variances[i * self.point_number : (i + 1) * self.point_number] = \
            self.variances[:, i]
        return new_points, new_values, new_variances

    def _multi_task_kernel1(self):
        return 0
