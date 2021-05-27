#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp.fvgp import fvGP
from fvgp.fvgp import GP
import matplotlib.pyplot as plt
import time

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
        my_gp.train([[100.0,200.0],[5.0,100.0]],
                init_hyperparameters = [110.0,8.0],
                method = training_method,
                pop_size = 20,
                tolerance = 0.0001,
                max_iter = 2)
        self.visualize(my_gp)
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
        my_gp.train([[100.0,200.0],[5.0,100.0],[5.0,100.0]],
                init_hyperparameters = [110.0,8.0,8.0],
                method = training_method,
                pop_size = 20,
                tolerance = 0.0001,
                max_iter = 2)
        print("TRAINING STOPPED")
        self.visualize_multi_task(my_gp)
        print("=========================================")
        print("1d multi task test successful")
        print("=========================================")

    def test_1d_single_task_async(self,N = 100):
        print("=========================================")
        print("1d async test started ....")
        print("=========================================")
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = func(points)[:,0]
        print("shape of values:   ",values.shape)
        my_gp = GP(1,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        opt_obj = my_gp.train_async([[100.0,200.0],[5.0,100.0]],
                init_hyperparameters = [110.0,8.0],
                pop_size = 20,
                tolerance = 0.0001,
                max_iter = 2)
        print("lets see how the hyper-parameters are changing")
        for i in range(100):
            time.sleep(10)
            my_gp.update_hyperparameters(opt_obj)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print("|latest hyper parameters:| ",my_gp.hyperparameters)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
        my_gp.stop_training(opt_obj)
        my_gp.kill_training(opt_obj)
        print("TRAINING STOPPED")
        self.visualize(my_gp)
        print("=========================================")
        print("1d single task async training test successful")
        print("=========================================")
    ############################################################
    def test_us_topo(self,method = "global"):
        print("=========================================")
        print("multi-task test started ...")
        print("=========================================")
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
    def visualize(self, my_gp):
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
        plt.plot(x_input,sig, label = "shannon ig", linewidth = 3.0)
        m = pred1_mean["f(x)"]
        s = np.diag(pred1_cov["S(x)"])
        plt.plot(x_input, s, label = "std", linewidth = 3.0)
        plt.fill_between(x_input[:,0], m-3.0*np.sqrt(s), m+3.0*np.sqrt(s), alpha = 0.5, label = "95% confidence interval")
        plt.scatter(my_gp.data_x[:,0],my_gp.data_y[0:len(x_input)], label = "data",linewidth = 3.0)
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
        #plt.savefig('plot.png')
        plt.show()
        #############################################################
    def visualize_multi_task(self, my_gp):
        print("working on the prediction...")
        x_input = np.empty((2000,2))
        x_input[0:1000,0] = np.linspace(0,2.0,1000)
        x_input[0:1000,1] = 0
        x_input[1000:,0] = np.linspace(0,2.0,1000)
        x_input[1000:,1] = 1
        y1 = func(x_input[0:1000,0:1])
        y2 = func(x_input[1000:,0:1]) + 2.0
        pred1_mean = my_gp.posterior_mean(x_input)
        pred1_cov = my_gp.posterior_covariance(x_input)
        sig = np.empty((len(x_input)))
        for i in range(len(x_input)):
            ##shannon ig always gives back the information gain for all input points
            sig[i] = my_gp.shannon_information_gain(np.array([x_input[i]]))["sig"]
        plt.figure(figsize = (10,4))
        plt.plot(x_input[0:1000,0],pred1_mean["f(x)"][0:1000], label = "posterior mean task 1",linewidth = 3.0)
        plt.plot(x_input[1000:,0],pred1_mean["f(x)"][1000:], label = "posterior mean task 2",linewidth = 3.0)
        plt.plot(x_input[0:1000,0],y1, label = "ground truth task 1",linewidth = 3.0)
        plt.plot(x_input[1000:,0], y2, label = "ground truth task 2",linewidth = 3.0)
        m1 = pred1_mean["f(x)"][0:1000]
        m2 = pred1_mean["f(x)"][1000:]
        s1 = np.diag(pred1_cov["S(x)"])[0:1000]
        s2 = np.diag(pred1_cov["S(x)"])[1000:]
        plt.fill_between(x_input[0:1000,0], m1-3.0*np.sqrt(s1), m1+3.0*np.sqrt(s1), alpha = 0.5, label = "95% confidence interval task 1")
        plt.fill_between(x_input[1000:,0], m2-3.0*np.sqrt(s2), m2+3.0*np.sqrt(s2), alpha = 0.5, label = "95% confidence interval task 2")
        plt.legend()
        plt.show()




def func(points):
    if len(points[0]) == 1: return 3.0*points[:,0] + 0.3 * np.sin(10.0*points[:,0])
    elif len(points[0]) == 2: return 3.0*points[:,0] + 0.3 * np.sin(10.0*points[:,1])

def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)
