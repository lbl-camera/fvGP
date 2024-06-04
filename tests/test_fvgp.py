#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp import fvGP
from fvgp import GP
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
from fvgp.gp_kernels import *
from fvgp.gp_lin_alg import *
from scipy import sparse
from fvgp.gp_kernels import *



N = 100
input_dim = 5


x_data = np.random.rand(N, input_dim)
y_data = np.sin(np.linalg.norm(x_data, axis=1))

x_new = np.random.rand(3, input_dim)
y_new = np.sin(np.linalg.norm(x_new, axis=1))


x_pred = np.random.rand(10, input_dim)


"""Tests for `fvgp` package."""


def test_lin_alg():
    B = np.random.rand(100,100)
    A = B @ B.T + np.identity(100)
    B = A[0:90,0:90]
    c = calculate_Chol_factor(B)
    k = A[0:90,90:]
    kk = A[90:,90:]
    C = cholesky_update_rank_n(c,k,kk)
    LU = compute_LU_factor(sparse.coo_matrix(A))
    s = calculate_LU_solve(LU, np.random.rand(len(A)))
    l = calculate_LU_logdet(LU)
    dd = update_Chol_factor(c, A)
    ss = calculate_Chol_solve(dd, np.random.rand(len(A)))
    ll = calculate_Chol_logdet(dd)
    ll = spai(A,20)
    calculate_sparse_conj_grad(sparse.coo_matrix(A),np.random.rand(len(A)))
    logd = calculate_logdet(B)
    update_logdet(logd, np.linalg.inv(B), A)
    i = calculate_inv(B)
    update_inv(i, A)
    solve(A, np.random.rand(len(A)))
    is_sparse(A)
    how_sparse_is(A)


def test_single_task_init_basic():
    def kernel(x1,x2,hps,obj):
        d = get_distance_matrix(x1,x2)
        return hps[0] * matern_kernel_diff1(d,3.)
    def noise(x,hps,obj):
        return np.identity(len(x))
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), compute_device = 'cpu', info = True)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), gp_kernel_function = kernel,
            gp_noise_function=noise, compute_device = 'cpu', info = True, ram_economy=True)
    my_gp1.marginal_density.neg_log_likelihood_hessian(hyperparameters=my_gp1.get_hyperparameters())
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), gp_kernel_function = kernel,
            gp_noise_function=noise, compute_device = 'cpu', info = True, ram_economy=False)
    my_gp1.marginal_density.neg_log_likelihood_hessian(hyperparameters=my_gp1.get_hyperparameters())
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
    res = squared_exponential_kernel(1,1)
    res = squared_exponential_kernel_robust(1,1)
    res = exponential_kernel(1,1)
    res = exponential_kernel_robust(1,1)
    res = matern_kernel_diff1(1,1)
    res = matern_kernel_diff1_robust(1,1)
    res = matern_kernel_diff2(1,1)
    res = matern_kernel_diff2_robust(1,1)
    res = sparse_kernel(1,1)
    res = periodic_kernel(1,1,1)
    res = my_gp1.prior._default_kernel(x_data,x_data,np.array([1.,1.,1.,1.,1.,1.]),my_gp1)
    my_gp1.crps(x_data[0:2] + 1., np.array([1.,2.]))
    my_gp1.rmse(x_data[0:2] + 1., np.array([1.,2.]))
    my_gp1.make_2d_x_pred(np.array([1.,2.]),np.array([3.,4]))
    my_gp1.make_1d_x_pred(np.array([1.,2.]))
    my_gp1._get_default_hyperparameter_bounds()


def test_single_task_init_advanced():
    my_gp2 = GP(x_data,y_data,np.array([1, 1, 1, 1, 1, 1]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)

def test_train_basic():
    my_gp1 = GP(x_data, y_data, np.array([1., 1., 1., 1., 1., 1.]))
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

    res = squared_exponential_kernel(1.,1.)
    res = squared_exponential_kernel_robust(1.,1.)
    res = exponential_kernel(1.,1.)
    res = exponential_kernel_robust(1.,1.)
    distance = 1.
    length = 1.5
    phi = 2.
    l = 2.
    w = 5.
    p = 1.
    radius = 3.

    res = matern_kernel_diff1(distance, length)
    res = matern_kernel_diff1_robust(distance, phi)
    res = matern_kernel_diff2(distance, length)

    res = matern_kernel_diff2_robust(distance, phi)
    res = sparse_kernel(distance, radius)
    res = periodic_kernel(distance, length, p)

    res = linear_kernel(2.,2.2, 1.,1.,1.)
    res = dot_product_kernel(np.random.rand(2),np.random.rand(2),1.,np.array([[1.,0.],[0.,2.]]))
    res = polynomial_kernel(np.random.rand(2),np.random.rand(2), 2)
    res = my_gp1.prior._default_kernel(x_data,x_data,np.ones((6)),my_gp1)
    res = non_stat_kernel(x_data,x_data,np.random.rand(10,5),np.random.rand(10),0.5)
    res = non_stat_kernel_gradient(x_data,x_data,np.random.rand(10,5),np.random.rand(10),0.5)
    res = wendland_anisotropic(x_data,x_data,np.ones((6)), my_gp1)

def test_train_hgdl():
    my_gp2 = GP(x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]), noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)


    my_gp2.train(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "hgdl", tolerance = 0.001, max_iter = 2)


def test_train_hgdl_async():
    my_gp2 = GP(x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)

    opt_obj = my_gp2.train_async(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            max_iter = 50)

    time.sleep(3)
    my_gp2.update_hyperparameters(opt_obj)
    my_gp2.stop_training(opt_obj)
    my_gp2.kill_training(opt_obj)
    my_gp2.set_hyperparameters(np.array([1., 1., 1., 1., 1., 1.]))
    my_gp2.get_hyperparameters()
    my_gp2.get_prior_pdf()
    my_gp2.marginal_density.test_log_likelihood_gradient(np.array([1., 1., 1., 1., 1., 1.]))


def test_multi_task():
    def mkernel(x1,x2,hps,obj):
        d = get_distance_matrix(x1,x2)
        return hps[0] * matern_kernel_diff1(d,hps[1])
    y_data = np.zeros((N,2))
    y_data[:,0] = np.sin(np.linalg.norm(x_data, axis=1))
    y_data[:,1] = np.cos(np.linalg.norm(x_data, axis=1))

    my_fvgp = fvGP(x_data, y_data, init_hyperparameters = np.array([1, 1]), gp_kernel_function=mkernel)
    my_fvgp.update_gp_data(x_data, y_data, append = True)
    my_fvgp.update_gp_data(x_data, y_data, append = False)
    my_fvgp.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10]]),
            method = "global", pop_size = 10, tolerance = 0.001, max_iter = 2)
    my_fvgp.posterior_mean(np.random.rand(10,5), x_out = np.array([0,1]))["f(x)"]
    my_fvgp.posterior_mean_grad(np.random.rand(10,5), x_out = np.array([0,1]))["df/dx"]
    my_fvgp.posterior_covariance(np.random.rand(10,5), x_out = np.array([0,1]))["v(x)"]
    


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
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client)
    
    my_gp2S.update_gp_data(x_data,y_data, append = False)
    my_gp2S.update_gp_data(x_new,y_new, append = True)

    my_gp2S.train(hyperparameter_bounds=hps_bounds, max_iter = 2, init_hyperparameters = init_hps)

    def obj_func(hps,args):
        return my_gp2S.log_likelihood(hyperparameters=hps[0:2])

    from fvgp.gpMCMC import ProposalDistribution
    init_s = (np.diag(hps_bounds[:,1]-hps_bounds[:,0])/100.)**2

    from fvgp import gpMCMC
    def proposal_distribution(x0, hps, obj):
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
                            init_prop_Sigma = init_s, adapt_callable="normal")




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
