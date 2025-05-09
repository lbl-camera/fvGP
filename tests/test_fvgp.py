#!/usr/bin/env python

"""Tests for `fvgp` package."""


import unittest
import numpy as np
from fvgp import fvGP
from fvgp import GP
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
from fvgp.kernels import *
from fvgp.gp_lin_alg import *
from scipy import sparse
from fvgp import deep_kernel_network


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
    c = calculate_Chol_factor(B, args = {"xz": 3.})
    k = A[0:90,90:]
    kk = A[90:,90:]
    C = cholesky_update_rank_n(c,k,kk, args = {"xz": 3.})
    LU = calculate_sparse_LU_factor(sparse.coo_matrix(A), args = {"xz": 3.})
    s = calculate_LU_solve(LU, np.random.rand(len(A)), args = {"xz": 3.})
    l = calculate_LU_logdet(LU, args = {"xz": 3.})
    dd = update_Chol_factor(c, A, args = {"xz": 3.})
    ss = calculate_Chol_solve(dd, np.random.rand(len(A)), args = {"xz": 3.})
    ll = calculate_Chol_logdet(dd, args = {"xz": 3.})
    ll = spai(sparse.coo_matrix(A),20, args = {"xz": 3.})
    calculate_sparse_minres(sparse.coo_matrix(A),np.random.rand(len(A)), args = {"xz": 3.})
    calculate_sparse_conj_grad(sparse.coo_matrix(A),np.random.rand(len(A)), args = {"xz": 3.})
    logd = calculate_logdet(B, args = {"xz": 3.})
    update_logdet(logd, np.linalg.inv(B), A, args = {"xz": 3.})
    i = calculate_inv(B, args = {"xz": 3.})
    update_inv(i, A, args = {"xz": 3.})
    solve(A, np.random.rand(len(A)), args = {"xz": 3.})
    calculate_sparse_solve(sparse.coo_matrix(A), np.random.rand(len(A)), args = {"ds":3.})
    calculate_logdet(A, compute_device='gpu')
    calculate_inv(A, compute_device='gpu')
    b = np.random.rand(len(A))
    solve(A, b, compute_device='gpu')
    solve(A, b, compute_device='multi-gpu')
    is_sparse(A)
    how_sparse_is(A)


def test_single_task_init_basic():
    def kernel(x1,x2,hps):
        d = get_distance_matrix(x1,x2)
        return hps[0] * matern_kernel_diff1(d,3.)
    def noise(x,hps):
        return np.ones((len(x)))
    def prior_mean(x,hps):
        return np.zeros(len(x))

    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), compute_device = 'cpu')
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), kernel_function = kernel,
            noise_function=noise, compute_device = 'cpu', ram_economy=True)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), kernel_function = kernel,
            noise_function=noise, compute_device = 'gpu', ram_economy=True)

    my_gp1.marginal_density.neg_log_likelihood_hessian(hyperparameters=my_gp1.get_hyperparameters())
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), kernel_function = kernel,
            noise_function=noise, prior_mean_function = prior_mean, compute_device = 'cpu', ram_economy=False)
    my_gp1.marginal_density.neg_log_likelihood_hessian(hyperparameters=my_gp1.get_hyperparameters())
    my_gp1 = GP(x_data, y_data)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), calc_inv = False)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1.train()
    my_gp1.update_gp_data(x_data, y_data, append = True)
    my_gp1.update_gp_data(x_data, y_data, append = False)
    
    my_gp1 = GP(x_data, y_data, noise_variances = np.zeros(y_data.shape) + 0.01,init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), args = {"xyz":3.})
    my_gp1.update_gp_data(x_data, y_data, noise_variances_new = np.zeros(y_data.shape) + 0.01, append = True)
    my_gp1.update_gp_data(x_data, y_data, noise_variances_new = np.zeros(y_data.shape) + 0.01, append = False)
    my_gp1.set_args({"dcf":4.})
    my_gp1.get_args()
    
    res = my_gp1.posterior_mean(x_pred)
    res = my_gp1.posterior_mean(x_pred, hyperparameters = np.ones((6)))
    res = my_gp1.posterior_mean_grad(x_pred,direction=0)
    res = my_gp1.posterior_mean_grad(x_pred)
    res = my_gp1.posterior_covariance(x_pred)
    res = my_gp1.posterior_covariance(x_pred, add_noise = True)
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
    wendland_kernel(get_anisotropic_distance_matrix(np.ones((2,2)), np.ones((2,2))+1., np.array([1,1])))
    
    a = np.random.rand(10)
    wasserstein_1d(a,a.copy())

    b = np.random.rand(10,100)
    wasserstein_1d_outer_vec(b,b.copy())

    res = my_gp1.prior._default_kernel(x_data,x_data,np.array([1.,1.,1.,1.,1.,1.]))
    my_gp1.crps(x_data[0:2] + 1., np.array([1.,2.]))
    my_gp1.rmse(x_data[0:2] + 1., np.array([1.,2.]))
    my_gp1.nlpd(x_data[0:2] + 1., np.array([1.,2.]))
    my_gp1.make_2d_x_pred(np.array([1.,2.]),np.array([3.,4]))
    my_gp1.make_1d_x_pred(np.array([1.,2.]))
    my_gp1._get_default_hyperparameter_bounds()


def test_single_task_init_advanced():
    my_gp2 = GP(x_data,y_data,np.array([1, 1, 1, 1, 1, 1]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)

def test_train_basic(client):
    def noiseC(x,hps):
        return np.identity((len(x)))

    my_gp1 = GP(x_data, y_data, np.array([1., 1., 1., 1., 1., 1.]), noise_function = noiseC)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "local", pop_size = 10, tolerance = 0.001,max_iter = 2, dask_client=client)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "global", pop_size = 10, tolerance = 0.001,max_iter = 2, dask_client=client)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "hgdl", pop_size = 10, tolerance = 0.001,max_iter = 2, dask_client=client)
    my_gp1.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "mcmc", pop_size = 10, tolerance = 0.001,max_iter = 20, dask_client=client)
    my_gp1.test_log_likelihood_gradient(np.array([1., 1., 1., 1., 1., 1.]))

    res = my_gp1.posterior_mean(np.random.rand(len(x_data),len(x_data[0])))
    res = my_gp1.posterior_mean_grad(np.random.rand(10,len(x_data[0])))
    res = my_gp1.posterior_mean_grad(np.random.rand(10,len(x_data[0])), hyperparameters = np.array([1., 1., 1., 1., 1., 1.]))
    res = my_gp1.posterior_covariance(np.random.rand(10,len(x_data[0])))
    res = my_gp1.posterior_covariance_grad(np.random.rand(10,len(x_data[0])))
    res = my_gp1.joint_gp_prior(np.random.rand(10,len(x_data[0])))
    res = my_gp1.joint_gp_prior_grad(np.random.rand(10,len(x_data[0])),0)
    res = my_gp1.gp_entropy(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_entropy_grad(np.random.rand(10,len(x_data[0])),0)
    res = my_gp1.gp_relative_information_entropy(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_relative_information_entropy_set(np.random.rand(10,len(x_data[0])))
    
    res = my_gp1.gp_mutual_information(np.random.rand(10,len(x_data[0])), add_noise = False)
    res = my_gp1.gp_mutual_information(np.random.rand(10,len(x_data[0])), add_noise = True)
    res = my_gp1.gp_total_correlation(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_total_correlation(np.random.rand(10,len(x_data[0])), add_noise = True)
    res = my_gp1.gp_relative_information_entropy(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_relative_information_entropy(np.random.rand(10,len(x_data[0])), add_noise = True)

    res = my_gp1.gp_relative_information_entropy_set(np.random.rand(10,len(x_data[0])))
    res = my_gp1.gp_relative_information_entropy_set(np.random.rand(10,len(x_data[0])), add_noise = True)



    A = np.random.rand(10,10)
    B = A.T @ A
    res = my_gp1.gp_kl_div(np.random.rand(10,len(x_data[0])), np.random.rand(10), B)
    res = my_gp1.posterior_probability(np.random.rand(10,len(x_data[0])), np.random.rand(10), B)

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
    res = my_gp1.prior._default_kernel(x_data,x_data,np.ones((6)))
    res = non_stat_kernel(x_data,x_data,np.random.rand(10,5),np.random.rand(10),0.5)
    res = non_stat_kernel_gradient(x_data,x_data,np.random.rand(10,5),np.random.rand(10),0.5)
    res = wendland_anisotropic(x_data,x_data,np.ones((6)))

def test_train_hgdl(client):
    my_gp2 = GP(x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]), noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)


    my_gp2.train(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "hgdl", tolerance = 0.001, max_iter = 2, dask_client=client)


def test_train_hgdl_async(client):
    my_gp2 = GP(x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", calc_inv = True, ram_economy = True)

    opt_obj = my_gp2.train_async(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            max_iter = 50, dask_client=client)
    opt_obj = my_gp2.train_async(max_iter = 5, dask_client=client)


    time.sleep(5)
    my_gp2.update_hyperparameters(opt_obj)
    my_gp2.stop_training(opt_obj)
    my_gp2.kill_training(opt_obj)
    my_gp2.set_hyperparameters(np.array([1., 1., 1., 1., 1., 1.]))
    my_gp2.get_hyperparameters()
    my_gp2.get_prior_pdf()
    my_gp2.marginal_density.test_log_likelihood_gradient(np.array([1., 1., 1., 1., 1., 1.]))


def test_multi_task(client):
    def mkernel(x1,x2,hps):
        d = get_distance_matrix(x1,x2)
        return hps[0] * matern_kernel_diff1(d,hps[1])
    y_data = np.zeros((N,2))
    y_data[:,0] = np.sin(np.linalg.norm(x_data, axis=1))
    y_data[:,1] = np.cos(np.linalg.norm(x_data, axis=1))

    my_fvgp = fvGP(x_data, y_data, init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp.update_gp_data(x_data, y_data, append = True)
    my_fvgp.update_gp_data(x_data, y_data, append = False)
    my_fvgp.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,10]]),
            method = "global", pop_size = 10, tolerance = 0.001, max_iter = 2, dask_client=client, info = True)
    my_fvgp.posterior_mean(np.random.rand(10,5), x_out = np.array([0,1]))["m(x)"]
    my_fvgp.posterior_mean(np.random.rand(10,5))["m(x)"]
    my_fvgp.posterior_mean_grad(np.random.rand(10,5), x_out = np.array([0,1]))["dm/dx"]
    my_fvgp.posterior_mean_grad(np.random.rand(10,5))["dm/dx"]
    my_fvgp.posterior_covariance(np.random.rand(10,5), x_out = np.array([0,1]))["v(x)"]
    my_fvgp.posterior_covariance(np.random.rand(10,5))["v(x)"]
    my_fvgp.posterior_covariance_grad(np.random.rand(10,5))
    my_fvgp.posterior_covariance_grad(np.random.rand(10,5), x_out = np.array([0,1]))

    my_fvgp.joint_gp_prior(np.random.rand(10,5))
    my_fvgp.joint_gp_prior(np.random.rand(10,5), x_out = np.array([0,1]))

    my_fvgp.joint_gp_prior_grad(np.random.rand(10,5), 0)
    my_fvgp.joint_gp_prior_grad(np.random.rand(10,5), 0, x_out = np.array([0,1]))

    my_fvgp.gp_entropy(np.random.rand(10,5))
    my_fvgp.gp_entropy_grad(np.random.rand(10,5), 0)
    my_fvgp.gp_entropy(np.random.rand(10,5), x_out = np.array([0,1]))
    my_fvgp.gp_entropy_grad(np.random.rand(10,5),0, x_out = np.array([0,1]))

    A = np.random.rand(20,20)
    B = A.T @ A


    my_fvgp.gp_kl_div(np.random.rand(10,5), np.random.rand(20), B)
    my_fvgp.gp_kl_div(np.random.rand(10,5), np.random.rand(20), B ,x_out = np.array([0,1]))

    my_fvgp.gp_mutual_information(np.random.rand(10,5))
    my_fvgp.gp_mutual_information(np.random.rand(10,5), x_out = np.array([0,1]))


    my_fvgp.gp_total_correlation(np.random.rand(10,5))
    my_fvgp.gp_total_correlation(np.random.rand(10,5), x_out = np.array([0,1]))


    my_fvgp.gp_relative_information_entropy(np.random.rand(10,5))
    my_fvgp.gp_relative_information_entropy(np.random.rand(10,5), x_out = np.array([0,1]))

    my_fvgp.gp_relative_information_entropy_set(np.random.rand(10,5))
    my_fvgp.gp_relative_information_entropy_set(np.random.rand(10,5), x_out = np.array([0,1]))

    my_fvgp.posterior_probability(np.random.rand(10,5), np.random.rand(20), B)
    my_fvgp.posterior_probability(np.random.rand(10,5), np.random.rand(20), B, x_out = np.array([0,1]))


    my_fvgp = fvGP(np.random.rand(3,5), np.random.rand(3,2), noise_variances = None, init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp = fvGP(np.random.rand(3,5), np.random.rand(3,2), noise_variances = None, init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp = fvGP(np.random.rand(3,5), np.array([[3.,4.],[1.,6.],[87.,3.]]), noise_variances = None, init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp = fvGP(np.random.rand(3,5), np.array([[3.,4.],[1.,6.],[87.,3.]]), noise_variances = None, init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp = fvGP(np.random.rand(3,5), np.array([[3.,4.],[1.,6.],[87.,3.]]), noise_variances = np.random.rand(3,2), init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp = fvGP(np.random.rand(3,5), np.array([[3.,4.],[1.,6.],[87., np.nan]]), noise_variances = np.array([[.1,.2],[.1,.2],[.1, np.nan]]), init_hyperparameters = np.array([1, 1]), kernel_function=mkernel)
    my_fvgp.update_gp_data(np.random.rand(3,5), np.array([[3.,4.],[1.,6.],[87., np.nan]]), noise_variances_new = np.array([[.1,.2],[.1,.2],[.1, np.nan]]), append = True)
    my_fvgp.update_gp_data(np.random.rand(3,5), np.array([[3.,4.],[1.,6.],[87., np.nan]]), noise_variances_new = np.array([[.1,.2],[.1,.2],[.1, np.nan]]), append = False)
    


def test_gp2Scale(client):
    from imate import logdet as imate_logdet
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
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseLU")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseCG")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseMINRES")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseCGpre")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseCGpre")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = False, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="Inv")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseMINRESpre")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseSolve")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)

    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 1000, gp2Scale_dask_client=client)

    my_gp2S.update_gp_data(x_data,y_data, append = False)
    my_gp2S.update_gp_data(x_new,y_new, append = True)

    my_gp2S.train(hyperparameter_bounds=hps_bounds, max_iter = 2, init_hyperparameters = init_hps, info = True)

    def obj_func(hps,args):
        return my_gp2S.log_likelihood(hyperparameters=hps[0:2])

    from fvgp import ProposalDistribution
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
    pd = ProposalDistribution([0,1] ,proposal_dist=proposal_distribution,
                            init_prop_Sigma = init_s, adapt_callable="normal")




    my_mcmc = gpMCMC(obj_func, prior_function, [pd],
                    args={"bounds":hps_bounds})

    hps = np.random.uniform(
                            low = hps_bounds[:,0], 
                            high = hps_bounds[:,1], 
                            size = len(hps_bounds))
    mcmc_result = my_mcmc.run_mcmc(x0=hps, n_updates=10, break_condition="default")
    my_gp2S.set_hyperparameters(mcmc_result["x"][-1])
    my_gp2S.get_gp2Scale_exec_time(1.,10)
    x_pred = np.linspace(0,1,1000)
    mean1 = my_gp2S.posterior_mean(x_pred.reshape(-1,1))["m(x)"]
    var1 =  my_gp2S.posterior_covariance(x_pred.reshape(-1,1))["v(x)"]
    
    pd = ProposalDistribution([0,1], init_prop_Sigma = init_s, adapt_callable = "normal")
    my_mcmc = gpMCMC(obj_func, prior_function, [pd],
                    args={"bounds":hps_bounds})

    mcmc_result = my_mcmc.run_mcmc(x0=hps, n_updates=20, break_condition="default")
    
    pd = ProposalDistribution([0,1], init_prop_Sigma = init_s, adapt_callable = "normal")
    my_mcmc = gpMCMC(obj_func, prior_function, [pd],
                    args={"bounds":hps_bounds})

    mcmc_result = my_mcmc.run_mcmc(x0=hps, info=True, n_updates=10, break_condition="default")

