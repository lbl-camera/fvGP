#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp import fvGP
from fvgp import GP2 as GP
import matplotlib.pyplot as plt
import time
import urllib.request

from dask.distributed import Client
import socket
import time
import argparse
import datetime
import sys
from dask.distributed import performance_report
from distributed.utils_test import gen_cluster, client, loop, cluster_fixture, loop_in_thread, cleanup





N = 100
input_dim = 5


x_data = np.random.rand(N, input_dim)
y_data = np.sin(np.linalg.norm(x_data, axis=1))

x_new = np.random.rand(3, input_dim)
y_new = np.sin(np.linalg.norm(x_new, axis=1))


x_pred = np.random.rand(10, input_dim)


"""Tests for `fvgp` package."""
def test_single_task_init_basic():
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), compute_device = 'cpu', info = True)
    my_gp1 = GP(x_data, y_data, info = True)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), calc_inv = False, info = True)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), args = {'a':2.})
    my_gp1.update_gp_data(x_data, y_data, append = True)
    my_gp1.update_gp_data(x_data, y_data, append = False)
    
    my_gp1 = GP(x_data, y_data, noise_variances = np.zeros(y_data.shape) + 0.01,init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), args = {'a':2.})
    my_gp1.update_gp_data(x_data, y_data, noise_variances_new = np.zeros(y_data.shape) + 0.01, append = True)
    my_gp1.update_gp_data(x_data, y_data, noise_variances_new = np.zeros(y_data.shape) + 0.01, append = False)
    
    res = my_gp1.posterior_mean(x_pred)
    res = my_gp1.posterior_mean_grad(x_pred,direction=0)
    res = my_gp1.posterior_mean_grad(x_pred)
    res = my_gp1.posterior_covariance(x_pred)
    res = my_gp1.posterior_covariance_grad(x_pred,direction=0)
    res = my_gp1.gp_entropy(x_pred)
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
    res = my_gp1.default_kernel(x_data,x_data,np.array([1.,1.,1.,1.,1.,1.]),my_gp1)

def test_single_task_init_advanced():
    my_gp2 = GP(input_dim, x_data,y_data,np.array([1, 1, 1, 1, 1, 1]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)

def test_train_basic():
    my_gp1 = GP(input_dim, x_data, y_data, np.array([1., 1., 1., 1., 1., 1.]))
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "local", pop_size = 10, tolerance = 0.001,max_iter = 2)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "global", pop_size = 10, tolerance = 0.001,max_iter = 2)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "hgdl", pop_size = 10, tolerance = 0.001,max_iter = 2)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "mcmc", pop_size = 10, tolerance = 0.001,max_iter = 2)

    res = my_gp1.posterior_mean(np.random.rand(len(x_data),len(x_data[0])))
    res = my_gp1.posterior_mean_grad(np.random.rand(10,len(x_data[0])))
    res = my_gp1.posterior_covariance(np.random.rand(10,len(x_data[0])))
    res = my_gp1.posterior_covariance_grad(np.random.rand(10,len(x_data[0])))
    res = my_gp1.joint_gp_prior(np.random.rand(10,len(x_data[0])))
    res = my_gp1.joint_gp_prior_grad(np.random.rand(10,len(x_data[0])),0)
    res = my_gp1.gp_entropy(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_entropy_grad(np.random.rand(10,len(x_data[0])),0)
    res = my_gp1.gp_relative_information_entropy(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_relative_information_entropy_set(np.random.rand(10,len(x_data[0])))

    A = np.random.rand(10,10)
    B = A.T @ A
    res = my_gp1.entropy(B)
    res = my_gp1.gp_kl_div(np.random.rand(10,len(x_data[0])), np.random.rand(10), B)
    res = my_gp1.gp_kl_div_grad(np.random.rand(10,len(x_data[0])), np.random.rand(10), B,0)
    res = my_gp1.posterior_probability(np.random.rand(10,len(x_data[0])), np.random.rand(10), B)
    res = my_gp1.posterior_probability_grad(np.random.rand(10,len(x_data[0])), np.random.rand(10), B, direction = 0)

    res = my_gp1.squared_exponential_kernel(1.,1.)
    res = my_gp1.squared_exponential_kernel_robust(1.,1.)
    res = my_gp1.exponential_kernel(1.,1.)
    res = my_gp1.exponential_kernel_robust(1.,1.)
    distance = 1.
    length = 1.5
    phi = 2.
    l = 2.
    w = 5.
    p = 1.
    radius = 3.

    res = my_gp1.matern_kernel_diff1(distance, length)
    res = my_gp1.matern_kernel_diff1_robust(distance, phi)
    res = my_gp1.matern_kernel_diff2(distance, length)

    res = my_gp1.matern_kernel_diff2_robust(distance, phi)
    res = my_gp1.sparse_kernel(distance, radius)
    res = my_gp1.periodic_kernel(distance, length, p)

    res = my_gp1.linear_kernel(2.,2.2, 1.,1.,1.)
    res = my_gp1.dot_product_kernel(np.random.rand(2),np.random.rand(2),1.,np.array([[1.,0.],[0.,2.]]))
    res = my_gp1.polynomial_kernel(np.random.rand(2),np.random.rand(2), 2)
    res = my_gp1.default_kernel(x_data,x_data,np.ones((6)),my_gp1)
    res = my_gp1.non_stat_kernel(x_data,x_data,np.random.rand(10,5),np.random.rand(10),0.5)
    res = my_gp1.non_stat_kernel_gradient(x_data,x_data,np.random.rand(10,5),np.random.rand(10),0.5)
    res = my_gp1.wendland_anisotropic(x_data,x_data,np.ones((6)), my_gp1)

def test_train_hgdl():
    my_gp2 = GP(input_dim, x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]), noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)


    my_gp2.train(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "hgdl", tolerance = 0.001, max_iter = 2)


def test_train_hgdl_async():
    my_gp2 = GP(input_dim, x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)

    opt_obj = my_gp2.train_async(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            max_iter = 5000)

    time.sleep(3)
    my_gp2.update_hyperparameters(opt_obj)
    my_gp2.stop_training(opt_obj)
    my_gp2.kill_training(opt_obj)
    my_gp2.set_hyperparameters(np.array([1., 1., 1., 1., 1., 1.]))
    my_gp2.get_hyperparameters()
    my_gp2.get_prior_pdf()
    my_gp2.test_log_likelihood_gradient(np.array([1., 1., 1., 1., 1., 1.]))


def test_multi_task():
    def mkernel(x1,x2,hps,obj):
        d = obj.get_distance_matrix(x1,x2)
        return hps[0] * obj.matern_kernel_diff1(d,hps[1])
    y_data = np.zeros((N,2))
    y_data[:,0] = np.sin(np.linalg.norm(x_data, axis=1))
    y_data[:,1] = np.cos(np.linalg.norm(x_data, axis=1))

    my_fvgp = fvGP(input_dim,1,2, x_data, y_data, init_hyperparameters = np.array([1, 1]), hyperparameter_bounds = np.array([[0.,1.],[0.,1.]]), gp_kernel_function=mkernel)
    my_fvgp.update_gp_data(x_data, y_data, append = True)
    my_fvgp.update_gp_data(x_data, y_data, append = False)
    my_fvgp.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10]]),
            method = "global", pop_size = 10, tolerance = 0.001, max_iter = 2)
    my_fvgp.posterior_mean(np.random.rand(2,5), x_out = np.zeros((1,1)))["f(x)"]
    
    #my_fvgp = fvGP(input_dim,1,2, x_data, y_data, init_hyperparameters = np.array([1, 1]))


def test_gp2Scale(client):
    input_dim = 1
    N = 2000
    x_data = np.random.rand(N,input_dim)
    y_data = np.sin(np.linalg.norm(x_data,axis = 1) * 5.0)

    x_new = np.random.rand(3, input_dim)
    y_new = np.sin(np.linalg.norm(x_new, axis=1))


    hps_n = 2

    hps_bounds = np.array([[0.1,10.],    ##signal var of stat kernel
                           [0.001,0.02]     ##length scale for stat kernel
                            ])

    init_hps = np.random.uniform(size = len(hps_bounds), low = hps_bounds[:,0], high = hps_bounds[:,1])
    my_gp2S = GP(1, x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client)
    
    my_gp2S.update_gp_data(x_data,y_data, append = True)
    my_gp2S.update_gp_data(x_new,y_new, append = True)

    my_gp2S.train(hyperparameter_bounds=hps_bounds, max_iter = 2, init_hyperparameters = init_hps)

    def obj_func(hps,args):
        return my_gp2S.log_likelihood(hyperparameters=hps[0:2])


    from fvgp.gpMCMC import ProposalDistribution

    init_s = (np.diag(hps_bounds[:,1]-hps_bounds[:,0])/100.)**2

    from fvgp import gpMCMC
    def proposal_distribution(x0, obj):
        cov = obj.prop_args["prop_Sigma"]
        proposal_hps = np.zeros((len(x0)))
        proposal_hps = np.random.multivariate_normal(
            mean = x0, cov = cov, size = 1).reshape(len(x0))
        return proposal_hps

    def in_bounds(v,bounds):
        if any(v<bounds[:,0]) or any(v>bounds[:,1]): return False
        return True
    def prior_function(theta,args):
        bounds = args["bounds"]
        if in_bounds(theta, bounds): 
            return 0. + np.sum(np.log(theta)/2.)
        else: 
            return -np.inf
    pd = ProposalDistribution(proposal_distribution, [0,1], 
                            init_prop_Sigma = init_s, adapt_callable="default")




    my_mcmc = gpMCMC(obj_func, len(hps_bounds), prior_function, [pd], 
                    args={"bounds":hps_bounds})

    hps = np.random.uniform(
                            low = hps_bounds[:,0], 
                            high = hps_bounds[:,1], 
                            size = len(hps_bounds))
    mcmc_result = my_mcmc.run_mcmc(x0=hps, info=True, n_updates=10, break_condition="default")

    x_pred = np.linspace(0,1,1000)
    mean1 = my_gp2S.posterior_mean(x_pred.reshape(-1,1))["f(x)"]
    var1 =  my_gp2S.posterior_covariance(x_pred.reshape(-1,1))["v(x)"]
