#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp.fvgp import fvGP
from fvgp.fvgp import GP
import matplotlib.pyplot as plt
import time

class TestfvGP(unittest.TestCase):
    """Tests for `gpcam` package."""

    def test_initialization(self):
        """Test something."""
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
        print("init test successful")
    ############################################################
    def test_1d_single_task(self,N = 100, training_method = "global"):
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = func(points)
        print("values:   ",values)
        my_gp = GP(1,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        my_gp.train([[100.0,200.0],[5.0,100.0]],
                init_hyperparameters = [110.0,8.0],
                optimization_method = training_method,
                optimization_pop_size = 20,
                optimization_tolerance = 0.0001,
                optimization_max_iter = 2)
        if training_method == "hgdl":
            print("lets see how the hyper-parameters are changing")
            for i in range(5):
                time.sleep(1)
                my_gp.update_hyperparameters()
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("|latest hyper parameters:| ",my_gp.hyperparameters)
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
            my_gp.stop_training()
            print("TRAINING STOPPED")
        self.visualize(my_gp)
        print("1d test test successful")
    ############################################################
    def test_1d_multi_task(self,N = 100, training_method = "global"):
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = np.empty((len(points),2))
        values[:,0] = func(points)
        values[:,1] = func(points) + 2.0
        print("values:   ",values)
        my_gp = fvGP(1,1,2,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        my_gp.train([[100.0,200.0],[5.0,100.0],[5.0,100.0]],
                init_hyperparameters = [110.0,8.0,8.0],
                optimization_method = training_method,
                optimization_pop_size = 20,
                optimization_tolerance = 0.0001,
                optimization_max_iter = 2)
        if training_method == "hgdl":
            print("lets see how the hyper-parameters are changing")
            for i in range(5):
                time.sleep(1)
                my_gp.update_hyperparameters()
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("|latest hyper parameters:| ",my_gp.hyperparameters)
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
            my_gp.stop_training()
            print("TRAINING STOPPED")
        self.visualize_multi_task(my_gp)
        print("1d test test successful")

    def test_1d_single_task_async(self,N = 100):
        points = np.empty((N,1))
        points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
        values = func(points)
        print("values:   ",values)
        my_gp = GP(1,points,values,np.ones((2)),
                gp_kernel_function = None,
                compute_device = "cpu")
        my_gp.train_async([[100.0,200.0],[5.0,100.0]],
                init_hyperparameters = [110.0,8.0],
                optimization_pop_size = 20,
                optimization_tolerance = 0.0001,
                optimization_max_iter = 2)
        print("lets see how the hyper-parameters are changing")
        for i in range(100):
            time.sleep(2)
            my_gp.update_hyperparameters()
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print("|latest hyper parameters:| ",my_gp.hyperparameters)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
        my_gp.stop_training()
        print("TRAINING STOPPED")
        self.visualize(my_gp)
        print("1d test test successful")
    ############################################################
    def test_us_topo(self,method = "global",dask_client = None):
        a = np.load("us_topo.npy")
        points = a[::16,0:2]
        values = a[::16,2:3]
        print("length of data set: ", len(points))
        my_gp = GP(2,points,values,np.array([1,1,1]), sparse = False)
        bounds = np.array([[10,10000000],[1,10000],[1,10000]])
        my_gp.train(bounds, optimization_method = method,
                optimization_max_iter = 20)
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
    def test_derivatives(self,direction):
        a = np.load("us_topo.npy")
        points = a[::64,0:2]
        values = a[::64,2:3]
        print("length of data set: ", len(points))
        my_gp = GP(2,points,values,np.array([1,1,1]), sparse = False)
        bounds = np.array([[10,10000000],[1,10000],[1,10000]])
        my_gp.train(bounds, optimization_method = method,
                optimization_max_iter = 20,
                optimization_pop_size = 4)
        print("ranges x:", np.min(points[:,0]),np.max(points[:,0]))
        print("ranges y:", np.min(points[:,1]),np.max(points[:,1]))
        eps = 1e-6
        x = np.array([[50.0,100.0]])
        x1= np.array(x)
        x2= np.array(x)
        x1[:,direction] = x1[:,direction] + eps
        x2[:,direction] = x2[:,direction] - eps
        #######posterior mean#######
        print("=============================")
        print("posterior mean gradient test:")
        print("=============================")
        fin_dif = (my_gp.posterior_mean(x1)["f(x)"] - my_gp.posterior_mean(x2)["f(x)"])/(2.0*eps)
        ana_dif = my_gp.posterior_mean_grad(x,direction)["df/dx"]
        print("finite difference mean gradient:   ", fin_dif)
        print("analytic difference mean gradient: ", ana_dif)
        input("check results and continue with enter...")
        print("=============================")
        print("posterior variance gradient test:")
        print("=============================")
        fin_dif = (my_gp.posterior_covariance(x1)["v(x)"] - my_gp.posterior_covariance(x2)["v(x)"])/(2.0*eps)
        ana_dif = my_gp.posterior_covariance_grad(x,direction)["dv/dx"]
        print("finite difference variance gradient:   ", fin_dif)
        print("analytic difference variance gradient: ", ana_dif)
        input("check results and continue with enter...")
        print("=============================")
        print("prior gradient test:")
        print("=============================")
        fin_dif = (my_gp.gp_prior(x1)["S(x)"] - my_gp.gp_prior(x2)["S(x)"])/(2.0*eps)
        ana_dif = my_gp.gp_prior_grad(x,direction)["dS/dx"]
        print("finite difference prior gradient:   ", fin_dif)
        print("analytic difference prior gradient: ", ana_dif)
        input("check results and continue with enter...")
        print("=============================")
        print("entropy gradient test:")
        print("=============================")
        fin_dif = (my_gp.gp_entropy(x1) - my_gp.gp_entropy(x2))/(2.0*eps)
        ana_dif = my_gp.gp_entropy_grad(x,direction)
        print("finite difference entropy gradient:   ", fin_dif)
        print("analytic difference entropy gradient: ", ana_dif)
        input("check results and continue with enter...")
        print("=============================")
        print("kl-divergence gradient test:")
        print("=============================")
        comp_mean = 5.0
        comp_cov = np.array([[2.0]])
        fin_dif = (my_gp.gp_kl_div(x1,comp_mean,comp_cov)["kl-div"] - \
                my_gp.gp_kl_div(x2,comp_mean,comp_cov)["kl-div"])/(2.0*eps)
        ana_dif = my_gp.gp_kl_div_grad(x,comp_mean,comp_cov,direction)["kl-div grad"]
        print("finite difference kl-div gradient:   ", fin_dif)
        print("analytic difference kl-div gradient: ", ana_dif)
        input("check results and continue with enter...")
        print("=============================")
        print("shannon info g gradient test:")
        print("=============================")
        fin_dif = (my_gp.shannon_information_gain(x1)["sig"] - my_gp.shannon_information_gain(x2)["sig"])/(2.0*eps)
        ana_dif =  my_gp.shannon_information_gain_grad(x,direction)["sig grad"]
        print("finite difference sig gradient:   ", fin_dif)
        print("analytic difference sig gradient: ", ana_dif)
        input("check results and continue with enter...")
        ################################################
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
        plt.savefig('plot.png')
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
        #plt.plot(x_input[0:1000,0],sig[0:1000], label = "shannon ig task 1", linewidth = 3.0)
        #plt.plot(x_input[1000:,0],sig[1000:], label = "shannon ig task 2", linewidth = 3.0)
        m1 = pred1_mean["f(x)"][0:1000]
        m2 = pred1_mean["f(x)"][1000:]
        s1 = np.diag(pred1_cov["S(x)"])[0:1000]
        s2 = np.diag(pred1_cov["S(x)"])[1000:]
        #plt.plot(x_input[0:1000,0], s1, label = "std task 1", linewidth = 3.0)
        #plt.plot(x_input[1000:,0], s2, label = "std task 2", linewidth = 3.0)
        plt.fill_between(x_input[0:1000,0], m1-3.0*np.sqrt(s1), m1+3.0*np.sqrt(s1), alpha = 0.5, label = "95% confidence interval task 1")
        plt.fill_between(x_input[1000:,0], m2-3.0*np.sqrt(s2), m2+3.0*np.sqrt(s2), alpha = 0.5, label = "95% confidence interval task 2")
        #plt.scatter(my_gp.data_x[:,0],my_gp.data_y[0:len(x_input)], label = "data",linewidth = 3.0)
        plt.legend()
        #print("computing probability of the given values")
        #comp_mean_vec = np.array([2.0,1.0])
        #comp_var = np.zeros((2, 2))
        #np.fill_diagonal(comp_var,np.random.rand(len(comp_var)))
        #x_input_prob = np.array([[0.55],[1.4]])
        #print("mean: ",comp_mean_vec)
        #print("var: ",comp_var)
        #print("points: ", x_input_prob)
        #s = my_gp.posterior_probability(x_input_prob, comp_mean_vec, comp_var)
        #print("s: ",s)
        #plt.savefig('plot.png')
        plt.show()




def func(points):
    if len(points[0]) == 1: return 3.0*points[:,0] + 0.3 * np.sin(10.0*points[:,0])
    elif len(points[0]) == 2: return 3.0*points[:,0] + 0.3 * np.sin(10.0*points[:,1])

def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)
