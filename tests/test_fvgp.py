#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp.fvgp import fvGP
from fvgp.fvgp import GP
import matplotlib.pyplot as plt
import time
import urllib.request

N = 1000
input_dim = 5


x_data = np.random.rand(N, input_dim)
y_data = np.sin(np.linalg.norm(x_data, axis=1))
x_pred = np.random.rand(10, input_dim)


class Test_fvGP(unittest.TestCase):
    """Tests for `fvgp` package."""

    def test_single_task_init_basic(self):
        my_gp1 = GP(input_dim, x_data, y_data, np.array([1, 1, 1, 1, 1, 1]))
        my_gp1.update_gp_data(x_data, y_data)
        res = my_gp1.posterior_mean(x_pred)
        res = my_gp1.posterior_mean_grad(x_pred,0)
        res = my_gp1.posterior_mean_grad(x_pred)
        res = my_gp1.posterior_covariance(x_pred)
        res = my_gp1.posterior_covariance_grad(x_pred,0)
        res = my_gp1.gp_entropy(x_pred)
        res = my_gp1.shannon_information_gain(x_pred)
        res = my_gp1.squared_exponential_kernel(1,1)
        res = my_gp1.squared_exponential_kernel_robust(1,1)
        res = my_gp1.exponential_kernel(1,1)
        res = my_gp1.exponential_kernel_robust(1,1)
        res = my_gp1.matern_kernel_diff1(1,1)
        res = my_gp1.matern_kernel_diff1_robust(1,1)
        res = my_gp1.matern_kernel_diff2(1,1)
        res = my_gp1.matern_kernel_diff2_robust(1,1)
        res = my_gp1.sparse_kernel(1,1)
        res = my_gp1.periodic_kernel(1,1,1)
        res = my_gp1.default_kernel(x_data,x_data,np.array([1,1,1,1,1,1]),my_gp1)

    def test_single_task_init_advanced(self):
        my_gp2 = GP(input_dim, x_data,y_data,np.array([1, 1, 1, 1, 1, 1]),variances=np.zeros(y_data.shape) + 0.01,
            compute_device="cpu", normalize_y = True, use_inv = True, ram_economy = True)

    def test_train_basic(self):
        my_gp1 = GP(input_dim, x_data, y_data, np.array([1, 1, 1, 1, 1, 1]))
        my_gp1.train(np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
                method = "global", pop_size = 10, tolerance = 0.001,max_iter = 5,deflation_radius = 1.)


    def test_train_hgdl_sync(self):
        my_gp2 = GP(input_dim, x_data,y_data,np.array([1, 1, 1, 1, 1, 1]),variances=np.zeros(y_data.shape) + 0.01,
            compute_device="cpu", normalize_y = True, use_inv = True, ram_economy = True)


        my_gp2.train(np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
                method = "hgdl", tolerance = 0.001, max_iter = 3, deflation_radius = 0.001)


    def test_train_hgdl_async(self):
        my_gp2 = GP(input_dim, x_data,y_data,np.array([1, 1, 1, 1, 1, 1]),variances=np.zeros(y_data.shape) + 0.01,
            compute_device="cpu", normalize_y = True, use_inv = True, ram_economy = True)

        opt_obj = my_gp2.train_async(np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
                max_iter = 5000, deflation_radius = 0.001)

        time.sleep(5)
        my_gp2.update_hyperparameters(opt_obj)
        my_gp2.kill_training(opt_obj)

    def test_multi_task(self):
        y_data = np.zeros((N,2))
        y_data[:,0] = np.sin(np.linalg.norm(x_data, axis=1))
        y_data[:,1] = np.cos(np.linalg.norm(x_data, axis=1))

        my_fvgp = fvGP(input_dim,1,2, x_data, y_data, np.array([1, 1, 1, 1, 1, 1]))
        my_fvgp.train(np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
                method = "global", pop_size = 10, tolerance = 0.001,max_iter = 5,deflation_radius = 1.)
