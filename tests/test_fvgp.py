#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp.fvgp import fvGP
from fvgp.fvgp import GP
import matplotlib.pyplot as plt
import time
import urllib.request

class TestfvGP(unittest.TestCase):
    """Tests for `fvgp` package."""

    def test_initialization(self):
        print("=========================================")
        print("init test started ...")
        print("=========================================")
        N = 100
        points = np.empty((N,2))
        points = np.random.uniform(low = -1, high = 1, size = points.shape)
        values = func(points)
        my_gp = GP(2,points,values,np.array([2,2]),
            gp_kernel_function = stationary_kernel,
            compute_device = "cpu")
        print("Base GP successfully initiated")
        values = np.empty((N,2))
        values[:,0] = func(points)
        values[:,1] = func(points) + 10.0
        my_fvgp = fvGP(2,1,2,points,values,np.array([2,2]),
            gp_kernel_function = stationary_kernel,
            compute_device = "cpu")
        print("fvGP successfully initiated")
        res = my_fvgp.posterior_mean(np.array([[3,3,0],[3,3,0]]))
        print(res)
        print("=========================================")
        print("init test successful")
        print("=========================================")
    ############################################################
    def test_1d_single_task(self,N = 100, training_method = "global"):
        print("=========================================")
        print("1d single task test started ...")
        print("=========================================")
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = func(points)
        print("shape of values:   ",values.shape)
        my_gp = GP(1,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        my_gp.train(np.array([[100.0,200.0],[5.0,100.0]]),
                init_hyperparameters = np.array([110.0,8.0]),
                method = training_method,
                pop_size = 20,
                tolerance = 0.0001,
                max_iter = 2)
        print("=========================================")
        print("1d single task test successful")
        print("=========================================")
    ############################################################
    def test_1d_multi_task(self,N = 100, training_method = "global"):
        print("=========================================")
        print("1d multi-task test started ....")
        print("=========================================")
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = np.empty((len(points),2))
        values[:,0] = func(points)
        values[:,1] = func(points) + 2.0
        print("shape of values:   ",values.shape)
        my_gp = fvGP(1,1,2,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        my_gp.train(np.array([[100.0,200.0],[5.0,100.0],[5.0,100.0]]),
                init_hyperparameters = np.array([110.0,8.0,8.0]),
                method = training_method,
                pop_size = 20,
                tolerance = 0.0001,
                max_iter = 2)
        print("TRAINING STOPPED")
        print("=========================================")
        print("1d multi task test successful")
        print("=========================================")

    def test_1d_single_task_async(self,N = 100):
        print("=========================================")
        print("1d async test started ....")
        print("=========================================")
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = func(points)
        print("shape of values:   ",values.shape)
        my_gp = GP(1,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        opt_obj = my_gp.train_async(np.array([[100.0,200.0],[5.0,100.0]]),
                init_hyperparameters = np.array([110.0,8.0]),
                max_iter = 2)
        print("lets see how the hyper-parameters are changing")
        for i in range(100):
            # time.sleep(10)
            my_gp.update_hyperparameters(opt_obj)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print("|latest hyper parameters:| ",my_gp.hyperparameters)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
        my_gp.stop_training(opt_obj)
        my_gp.kill_training(opt_obj)
        print("=========================================")
        print("1d single task async training test successful")
        print("=========================================")
    ############################################################
    def test_us_topo(self,method = "global"):
        print("=========================================")
        print("multi-task test started ...")
        print("=========================================")
        urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1BMNsdv168PoxNCHsNWR_znpDswjdFxXI', 'us_topo.npy')

        a = np.load("us_topo.npy")
        points = a[::16,0:2]
        values = a[::16,2:3]
        print("length of data set: ", len(points))
        my_gp = GP(2,points,values,np.array([1,1,1]), sparse = False)
        bounds = np.array([[10,10000000],[1,10000],[1,10000]])
        my_gp.train(bounds, method = method,
                max_iter = 1)
        print("=========================================")
        print("US topo test successfully concluded")
        print("=========================================")
    ############################################################


def func(points):
    if len(points[0]) == 1: return 3.0*points[:,0] + 0.3 * np.sin(10.0*points[:,0])
    elif len(points[0]) == 2: return 3.0*points[:,0] + 0.3 * np.sin(10.0*points[:,1])

def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)
