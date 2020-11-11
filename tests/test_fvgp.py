#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp.fvgp import FVGP
import matplotlib.pyplot as plt
import time

class TestfvGP(unittest.TestCase):
    """Tests for `gpcam` package."""

    def test_initialization(self,points,values):
        """Test something."""
        my_gp = FVGP(len(points[0]),1,len(values[0]),points,values,
            gp_kernel_function = stationary_kernel,
            compute_device = "cpu")
        print("success")
    ############################################################
    def test_1d_single_task(self,N = 100, training_method = "global"):
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = func(points)
        my_gp = FVGP(1,1,1,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        my_gp.train([[100.0,200.0],[5.0,10.0]],
                init_hyperparameters = [110.0,8.0],
                optimization_method = training_method,
                optimization_pop_size = 20,
                optimization_tolerance = 0.0001,
                optimization_max_iter = 200,
                dask_client = True)
        if training_method == "hgdl":
            print("lets see how the hyper-parameters are changing")
            for i in range(10):
                time.sleep(1)
                my_gp.update_hyperparameters()
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("|latest hyper parameters:| ",my_gp.hyperparameters)
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
            my_gp.stop_training()

        print("working on the prediction...")
        x_input = np.empty((1000,1))
        x_input[:,0] = np.linspace(0,2.0,1000)
        y = func(x_input)
        pred1_mean = my_gp.posterior_mean(x_input)
        pred1_cov = my_gp.posterior_covariance(x_input)
        sig = np.empty((len(x_input)))
        for i in range(len(x_input)):
            ##shannon ig always gives back the information gain for all input points
            sig[i] = my_gp.shannon_information_gain(np.array([x_input[i]]))["sig"]
        plt.figure(figsize = (10,4))
        plt.plot(x_input,pred1_mean["f(x)"], label = "posterior mean",linewidth = 3.0)
        plt.plot(x_input,y, label = "ground truth",linewidth = 3.0)
        plt.plot(x_input,sig + 10.0, label = "shannon ig", linewidth = 3.0)
        m = pred1_mean["f(x)"]
        s = np.diag(pred1_cov["S"])
        plt.plot(x_input, 1000.0*s, label = "std", linewidth = 3.0)
        plt.fill_between(x_input[:,0], m-3.0*np.sqrt(s), m+3.0*np.sqrt(s), alpha = 0.5, label = "95% confidence interval")
        plt.scatter(points,values, label = "data",linewidth = 3.0)
        plt.legend()
        print("computing probability of the given values")
        comp_mean_vec = np.array([2.0,1.0])
        comp_var = np.zeros((2, 2))
        np.fill_diagonal(comp_var,np.random.rand(len(comp_var)))
        x_input_prob = np.array([[0.55],[1.4]])
        print("mean: ",comp_mean_vec)
        print("var: ",comp_var)
        print("points: ", x_input_prob)
        s = my_gp.posterior_probability(x_input_prob, comp_mean_vec, comp_var)
        print("s: ",s)
        plt.savefig('plot.png')
        plt.show()
    ############################################################
    def test_us_topo(self,method = "global",dask_client = False):
        a = np.load("us_topo.npy")
        points = a[::16,0:2]
        values = a[::16,2:3]
        print("length of data set: ", len(points))
        my_gp = FVGP(2,1,1,points,values,np.array([1,1,1]), sparse = False)
        bounds = np.array([[10,10000000],[1,10000],[1,10000]])
        my_gp.train(bounds, optimization_method = method, 
                optimization_max_iter = 20,
                optimization_pop_size = 4,
                dask_client = dask_client)
        if method == "hgdl":
            print("lets see how the hyper-parameters are changing")
            for i in range(30):
                time.sleep(1)
                my_gp.update_hyperparameters()
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("|latest hyper parameters:| ",my_gp.hyperparameters)
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
            my_gp.stop_training()


    ############################################################







def func(points):
    return 3.0*points + 0.3*np.sin(10.0*points)

def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)
