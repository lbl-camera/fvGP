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


def test_gpu_lin_algebra():
    """
    Tests all GPU-accelerated linear algebra functions in gp_lin_alg.py.

    The function first detects which GPU engines are available (torch with a CUDA device,
    and/or cupy). If neither is available the function returns immediately and is a no-op.
    For every available engine the GPU code paths are executed and their numerical results
    are compared against the CPU reference to verify correctness.

    Covered functions
    -----------------
    Both engines  : calculate_Chol_factor, calculate_Chol_logdet, calculate_Chol_solve,
                    cholesky_update_rank_1, cholesky_update_rank_n, solve, matmul, matmul3
    torch only    : calculate_logdet, calculate_inv  (these hardcode torch internally)
    """
    import importlib

    # Detect available GPU engines
    engines = []
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            if torch.cuda.is_available():
                engines.append("torch")
        except Exception as e:
            print(f"Error occurred while checking torch GPU availability: {e}")
    else:
        print("torch not installed; skipping torch tests")
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy as cp
            cp.zeros(1)  # trigger device initialization; raises if no GPU
            engines.append("cupy")
        except Exception as e:
            print(f"Error occurred while checking cupy GPU availability: {e}")
    else:
        print("cupy not installed; skipping cupy tests")

    if not engines:
        return  # no GPU present – skip silently
    print(engines, "GPU engines detected; running GPU linear algebra tests")

    # ------------------------------------------------------------------ #
    # Build deterministic, well-conditioned PD test matrices              #
    # ------------------------------------------------------------------ #
    np.random.seed(0)
    B = np.random.rand(20, 20)
    A = (B @ B.T + np.eye(20) * 5.).astype(np.float64)
    b = np.random.rand(20)

    # Smaller matrix for rank-update tests: we have a 9x9 factor and extend to 10x10
    B_s = np.random.rand(10, 10)
    A_s = (B_s @ B_s.T + np.eye(10) * 3.).astype(np.float64)
    A9 = A_s[:9, :9]
    k = A_s[:9, 9:]   # (9, 1) cross-covariance vector
    kk = A_s[9:, 9:]  # (1, 1) new-point variance

    C = np.random.rand(20, 15)

    # ------------------------------------------------------------------ #
    # CPU reference values                                                #
    # ------------------------------------------------------------------ #
    c_cpu = calculate_Chol_factor(A, compute_device="cpu")
    logdet_chol_cpu = calculate_Chol_logdet(c_cpu, compute_device="cpu")
    solve_chol_cpu = calculate_Chol_solve(c_cpu, b.copy(), compute_device="cpu")

    c9_cpu = calculate_Chol_factor(A9, compute_device="cpu")
    rank1_cpu = cholesky_update_rank_1(c9_cpu, k[:, 0], float(kk[0, 0]), compute_device="cpu")
    rankn_cpu = cholesky_update_rank_n(c9_cpu, k, kk, compute_device="cpu")

    solve_cpu = solve(A, b, compute_device="cpu")
    mm_cpu = matmul(A, C, compute_device="cpu")
    mm3_cpu = matmul3(A, A, b.reshape(-1, 1), compute_device="cpu")

    # ------------------------------------------------------------------ #
    # Per-engine GPU tests                                                #
    # ------------------------------------------------------------------ #
    for engine in engines:
        args = {"GPU_engine": engine}
        print("testing GPU engine:", engine)

        # calculate_Chol_factor
        c_gpu = calculate_Chol_factor(A, compute_device="gpu", args=args)
        assert isinstance(c_gpu, np.ndarray), f"{engine}: Chol factor wrong type"
        assert c_gpu.shape == A.shape, f"{engine}: Chol factor wrong shape"

        # calculate_Chol_logdet
        logdet_gpu = calculate_Chol_logdet(c_gpu, compute_device="gpu", args=args)
        assert np.isscalar(logdet_gpu), f"{engine}: Chol logdet is not scalar"
        assert np.isclose(logdet_gpu, logdet_chol_cpu, rtol=1e-5), \
            f"{engine}: Chol logdet mismatch  gpu={logdet_gpu:.6f}  cpu={logdet_chol_cpu:.6f}"

        # calculate_Chol_solve
        solve_chol_gpu = calculate_Chol_solve(c_gpu, b.copy(), compute_device="gpu", args=args)
        assert solve_chol_gpu.shape == solve_chol_cpu.shape, \
            f"{engine}: Chol solve shape mismatch"
        assert np.allclose(solve_chol_gpu, solve_chol_cpu, rtol=1e-5), \
            f"{engine}: Chol solve mismatch"

        # cholesky_update_rank_1
        c9_gpu = calculate_Chol_factor(A9, compute_device="gpu", args=args)
        rank1_gpu = cholesky_update_rank_1(
            c9_gpu, k[:, 0], float(kk[0, 0]), compute_device="gpu", args=args)
        assert rank1_gpu.shape == rank1_cpu.shape, \
            f"{engine}: rank-1 update shape mismatch"
        assert np.allclose(np.abs(np.diag(rank1_gpu)), np.abs(np.diag(rank1_cpu)), rtol=1e-5), \
            f"{engine}: rank-1 update diagonal mismatch"

        # cholesky_update_rank_n
        rankn_gpu = cholesky_update_rank_n(c9_gpu, k, kk, compute_device="gpu", args=args)
        assert rankn_gpu.shape == rankn_cpu.shape, \
            f"{engine}: rank-n update shape mismatch"
        assert np.allclose(np.abs(np.diag(rankn_gpu)), np.abs(np.diag(rankn_cpu)), rtol=1e-5), \
            f"{engine}: rank-n update diagonal mismatch"

        # solve
        solve_gpu = solve(A, b, compute_device="gpu", args=args)
        assert np.allclose(solve_gpu, solve_cpu, rtol=1e-5), f"{engine}: solve mismatch"

        # matmul
        mm_gpu = matmul(A, C, compute_device="gpu", args=args)
        assert np.allclose(mm_gpu, mm_cpu, rtol=1e-5), f"{engine}: matmul mismatch"

        # matmul3
        mm3_gpu = matmul3(A, A, b.reshape(-1, 1), compute_device="gpu", args=args)
        assert np.allclose(mm3_gpu, mm3_cpu, rtol=1e-5), f"{engine}: matmul3 mismatch"

    # ------------------------------------------------------------------ #
    # torch-only functions (hardcode torch; no get_gpu_engine dispatch)  #
    # ------------------------------------------------------------------ #
    if "torch" in engines:
        logdet_torch = calculate_logdet(A, compute_device="gpu")
        logdet_ref = calculate_logdet(A, compute_device="cpu")
        assert np.isclose(logdet_torch, logdet_ref, rtol=1e-5), \
            f"torch: calculate_logdet mismatch  gpu={logdet_torch:.6f}  cpu={logdet_ref:.6f}"

        inv_torch = calculate_inv(A, compute_device="gpu")
        inv_cpu = calculate_inv(A, compute_device="cpu")
        assert np.allclose(inv_torch, inv_cpu, rtol=1e-5), "torch: calculate_inv mismatch"


def test_lin_alg():
    B = np.random.rand(100,100)
    A = B @ B.T + np.identity(100)
    B = A[0:90,0:90]
    c = calculate_Chol_factor(B, args = {"xz": 3.})
    k = A[0:90,90:]
    kk = A[90:,90:]
    C = cholesky_update_rank_n(c,k,kk, args = {"xz": 3.})
    LU = calculate_sparse_LU_factor(sparse.csr_matrix(A), args = {"xz": 3.})
    s = calculate_LU_solve(LU, np.random.rand(len(A)), args = {"xz": 3.})
    l = calculate_LU_logdet(LU, args = {"xz": 3.})
    dd = update_Chol_factor(c, A, args = {"xz": 3.})
    ss = calculate_Chol_solve(dd, np.random.rand(len(A)), args = {"xz": 3.})
    ll = calculate_Chol_logdet(dd, args = {"xz": 3.})
    ll = spai(sparse.csr_matrix(A),20, args = {"xz": 3.})
    calculate_sparse_minres(sparse.csr_matrix(A),np.random.rand(len(A)), args = {"xz": 3.})
    calculate_sparse_conj_grad(sparse.csr_matrix(A),np.random.rand(len(A)), args = {"xz": 3.})
    logd = calculate_logdet(B, args = {"xz": 3.})
    update_logdet(logd, np.linalg.inv(B), A, args = {"xz": 3.})
    i = calculate_inv(B, args = {"xz": 3.})
    update_inv(i, A, args = {"xz": 3.})
    solve(A, np.random.rand(len(A)), args = {"xz": 3.})
    calculate_sparse_solve(sparse.csr_matrix(A), np.random.rand(len(A)), args = {"ds":3.})
    calculate_logdet(A, compute_device='cpu')
    calculate_inv(A, compute_device='cpu')
    b = np.random.rand(len(A))
    solve(A, b, compute_device='cpu')
    solve(A, b, compute_device='cpu')
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
    my_gp1 = GP(x_data, np.column_stack([y_data, y_data+1.]), init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), kernel_function = kernel,
            noise_function=noise, compute_device = 'cpu', ram_economy=True)

    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), kernel_function = kernel,
            noise_function=noise, compute_device = 'cpu', ram_economy=True)

    my_gp1.marginal_likelihood.neg_log_likelihood_hessian(hyperparameters=my_gp1.hyperparameters)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), kernel_function = kernel,
            noise_function=noise, prior_mean_function = prior_mean, compute_device = 'cpu', ram_economy=False)
    my_gp1.marginal_likelihood.neg_log_likelihood_hessian(hyperparameters=my_gp1.hyperparameters)
    my_gp1 = GP(x_data, y_data)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), calc_inv = False)
    res = my_gp1.posterior_covariance(x_pred, variance_only = True)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), calc_inv = True)
    res = my_gp1.posterior_covariance(x_pred, variance_only = True)

    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1.train(max_iter = 100)
    my_gp1.train(method = "adam", max_iter = 3)
    my_gp1.update_gp_data(x_data, y_data, append = True)
    my_gp1.update_gp_data(x_data, y_data, append = False)
    my_gp1.make_2d_x_pred([0,1], [0,1], resx=100, resy=100)
    
    my_gp1 = GP(x_data, y_data, noise_variances = np.zeros(y_data.shape) + 0.01,init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), args = {"xyz":3.})
    my_gp1.update_gp_data(x_data, y_data, noise_variances_new = np.zeros(y_data.shape) + 0.01, append = True)
    my_gp1.update_gp_data(x_data, y_data, noise_variances_new = np.zeros(y_data.shape) + 0.01, append = False)
    my_gp1.set_args({"dcf":4.})
    my_gp1.args
    assert my_gp1.args == my_gp1.args == {"dcf":4.}
    assert my_gp1.args == my_gp1.prior.args
    assert my_gp1.args == my_gp1.likelihood.args
    assert my_gp1.args == my_gp1.marginal_likelihood.args
    assert my_gp1.args == my_gp1.trainer.args
    assert my_gp1.args == my_gp1.posterior.args
    assert my_gp1.args == my_gp1.marginal_likelihood.KVlinalg.args

    my_gp1 = GP(x_data, y_data, noise_variances = np.zeros(y_data.shape) + 0.01,init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1.set_args({"dcf":4.})
    assert my_gp1.args == my_gp1.args == {"dcf":4.}
    assert my_gp1.args == my_gp1.prior.args
    assert my_gp1.args == my_gp1.likelihood.args
    assert my_gp1.args == my_gp1.marginal_likelihood.args
    assert my_gp1.args == my_gp1.trainer.args
    assert my_gp1.args == my_gp1.posterior.args
    assert my_gp1.args == my_gp1.marginal_likelihood.KVlinalg.args


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
    sle_kernel(np.ones((2,2)), np.ones((2,2))+1.21, np.array([1,1,1,1]), args = {"x_data": np.random.rand(10,2)})
    
    a = np.random.rand(10)
    wasserstein_1d(a,a.copy())

    b = np.random.rand(10,100)
    wasserstein_1d_outer_vec(b,b.copy())

    res = my_gp1.prior._default_kernel(x_data,x_data,np.array([1.,1.,1.,1.,1.,1.]))
    x_m = x_data[0:2] + 1.
    y_m = np.array([1., 2.])

    my_gp1.crps(x_m, y_m)
    my_gp1.rmse(x_m, y_m)
    my_gp1.nrmse(x_m, y_m)
    my_gp1.nlpd(x_m, y_m)
    my_gp1.r2(x_m, y_m)
    my_gp1.picp(x_m, y_m, interval=0.95)

    # mae
    mae_val = my_gp1.mae(x_m, y_m)
    assert np.isscalar(mae_val) and mae_val >= 0.

    # mape
    mape_val = my_gp1.mape(x_m, y_m)
    assert np.isscalar(mape_val) and mape_val >= 0.

    # msll
    msll_val = my_gp1.msll(x_m, y_m)
    assert np.isscalar(msll_val)

    # mpiw
    mpiw_val = my_gp1.mpiw(x_m, interval=0.95)
    assert np.isscalar(mpiw_val) and mpiw_val > 0.

    # interval_score
    is_val = my_gp1.interval_score(x_m, y_m, interval=0.95)
    assert np.isscalar(is_val) and is_val > 0.

    # coverage_curve: default intervals
    cc = my_gp1.coverage_curve(x_m, y_m)
    assert "target_coverage" in cc and "measured_coverage" in cc
    assert len(cc["target_coverage"]) == len(cc["measured_coverage"]) == 19
    assert all(0. <= v <= 1. for v in cc["measured_coverage"])

    # coverage_curve: custom intervals
    custom = np.array([0.5, 0.9])
    cc2 = my_gp1.coverage_curve(x_m, y_m, intervals=custom)
    assert len(cc2["target_coverage"]) == len(cc2["measured_coverage"]) == 2
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

    opt_obj = my_gp2.train(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            max_iter = 50, dask_client=client, method = "hgdl", asynchronous=True)
    opt_obj = my_gp2.train(max_iter = 5, dask_client=client, asynchronous=True, method="hgdl")


    time.sleep(5)
    my_gp2.update_hyperparameters(opt_obj)
    my_gp2.stop_training(opt_obj)
    my_gp2.kill_client(opt_obj)
    my_gp2.set_hyperparameters(np.array([1., 1., 1., 1., 1., 1.]))
    my_gp2.hyperparameters
    my_gp2.get_prior_pdf()
    my_gp2.marginal_likelihood.test_log_likelihood_gradient(np.array([1., 1., 1., 1., 1., 1.]))


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
    my_fvgp.train(hyperparameter_bounds=np.array([[0.01,1],[0.01,1]]),
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
    input_dim = 1
    N = 200
    x_data = np.random.rand(N,input_dim)
    y_data = np.sin(np.linalg.norm(x_data,axis = 1) * 5.0)

    x_new = np.random.rand(3, input_dim)
    y_new = np.sin(np.linalg.norm(x_new, axis=1))


    hps_n = 2

    hps_bounds = np.array([[0.1,10.],    ##signal var of stat kernel
                           [0.001,0.02]     ##length scale for stat kernel
                            ])

    init_hps = np.random.uniform(size = len(hps_bounds), low = hps_bounds[:,0], high = hps_bounds[:,1])
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseLU")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseCG")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseMINRES")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseCGpre")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseCGpre")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = False, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="Inv")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
    my_gp2S.neg_log_likelihood_gradient()
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseMINRESpre")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)
    
    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client, gp2Scale_linalg_mode="sparseSolve")
    my_gp2S.log_likelihood(hyperparameters = init_hps)
    my_gp2S.update_gp_data(x_new,y_new, append = True)

    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, gp2Scale_dask_client=client)

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
    def prior_function(theta, bounds, args):
        if in_bounds(theta, bounds):
            return 0. + np.sum(np.log(theta)/2.)
        else:
            return -np.inf
    pd = ProposalDistribution([0,1], proposal_dist=proposal_distribution,
                              init_prop_Sigma=init_s, adapt_callable="normal")

    my_mcmc = gpMCMC(obj_func, bounds=hps_bounds, prior_function=prior_function,
                     proposal_distributions=[pd])

    hps = np.random.uniform(
                            low = hps_bounds[:,0],
                            high = hps_bounds[:,1],
                            size = len(hps_bounds))
    mcmc_result = my_mcmc.run_mcmc(x0=hps, n_updates=10, break_condition="default")
    my_gp2S.set_hyperparameters(mcmc_result["x"][-1])
    my_gp2S.get_gp2Scale_exec_time(1.,10)
    x_pred = np.linspace(0,1,100)
    mean1 = my_gp2S.posterior_mean(x_pred.reshape(-1,1))["m(x)"]
    var1 =  my_gp2S.posterior_covariance(x_pred.reshape(-1,1))["v(x)"]

    pd = ProposalDistribution([0,1], init_prop_Sigma=init_s, adapt_callable="normal")
    my_mcmc = gpMCMC(obj_func, bounds=hps_bounds, prior_function=prior_function,
                     proposal_distributions=[pd])
    mcmc_result = my_mcmc.run_mcmc(x0=hps, n_updates=20, break_condition="default")

    pd = ProposalDistribution([0,1], init_prop_Sigma=init_s, adapt_callable="normal")
    my_mcmc = gpMCMC(obj_func, bounds=hps_bounds, prior_function=prior_function,
                     proposal_distributions=[pd])
    mcmc_result = my_mcmc.run_mcmc(x0=hps, info=True, n_updates=10, break_condition="default")


def test_pickle():
    import numpy as np
    from fvgp import GP
    import pickle

    #initialize some data
    x_data = np.random.uniform(size = (10,3))
    y_data = np.sin(np.linalg.norm(x_data, axis = 1))

    #TEST0
    #tests empty gp pickling
    my_gpo = GP(x_data, y_data)
    pickle.loads(pickle.dumps(my_gpo))

    #TEST1
    #initialize the GPOptimizer
    my_gpo = GP(x_data, y_data, args = {'a':2.,'b':3.})

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)


    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)

    #TEST2
    #initialize the GPOptimizer
    my_gpo = GP(x_data,y_data,
        init_hyperparameters = np.ones((4))/10.,  # We need enough of those for kernel, noise, and prior mean functions
        args = {"df":3})
    

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)


    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)
    assert my_gpo.args == my_gpo2.args

    #TEST3
    #initialize the GPOptimizer

    my_gpo = GP(x_data,y_data,
        init_hyperparameters = np.ones((4))/10.,  # We need enough of those for kernel, noise, and prior mean functions
        )

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)


    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)



    #TEST4
    #initialize the GPOptimizer

    my_gpo = fvGP(x_data,np.random.rand(len(x_data),2),
        init_hyperparameters = np.ones((5))/10.,  # We need enough of those for kernel, noise, and prior mean functions
        )

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)


    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)
    assert my_gpo.input_set_dim == my_gpo2.input_set_dim
    assert my_gpo.index_set_dim == my_gpo2.index_set_dim

    my_gpo = fvGP(x_data,np.random.rand(len(x_data),2),
        init_hyperparameters = np.ones((5))/10.,  # We need enough of those for kernel, noise, and prior mean functions
        args = {"sfdf": 4.})
    def is_pickle_equal(obj):
        # Get class and instance attributes before pickling
        cls = type(obj)
        before_class = {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}.keys()
        before_instance = dict(obj.__dict__).keys()

        # Pickle and unpickle
        obj2 = pickle.loads(pickle.dumps(obj))

        # Get attributes after pickling
        cls2 = type(obj2)
        after_class = {k: v for k, v in cls2.__dict__.items() if not k.startswith('__')}.keys()
        after_instance = dict(obj2.__dict__).keys()

        # Compare everything
        if before_class != after_class: print(before_class, after_class)
        if before_instance != after_instance: print(before_instance, after_instance)

        return before_class == after_class and before_instance == after_instance

    assert is_pickle_equal(my_gpo)
    assert is_pickle_equal(my_gpo.prior)
    assert is_pickle_equal(my_gpo.likelihood)
    assert is_pickle_equal(my_gpo.marginal_likelihood)
    assert is_pickle_equal(my_gpo.trainer)
    assert is_pickle_equal(my_gpo.posterior)
    assert is_pickle_equal(my_gpo.data)
    assert is_pickle_equal(my_gpo.marginal_likelihood.KVlinalg)


def test_gpMCMC():
    """Test the gpMCMC class directly with the new API (bounds explicit, not via args)."""
    from fvgp import gpMCMC, ProposalDistribution

    bounds = np.array([[0.01, 5.], [0.01, 5.]])
    hps = np.array([1., 1.])

    def log_likelihood(hps, args):
        return -0.5 * np.sum(hps ** 2)

    # Default uniform prior (only bounds, no prior_function)
    my_mcmc = gpMCMC(log_likelihood, bounds=bounds)
    res = my_mcmc.run_mcmc(x0=hps, n_updates=20, break_condition=None)
    assert "median(x)" in res
    assert res["median(x)"].shape == hps.shape

    # Default break condition
    my_mcmc = gpMCMC(log_likelihood, bounds=bounds)
    res = my_mcmc.run_mcmc(x0=hps, n_updates=20, break_condition="default")
    assert "median(x)" in res

    # Custom prior_function with new signature (theta, bounds, args)
    def custom_prior(theta, bounds, args):
        if np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])):
            return 0.
        return -np.inf

    my_mcmc = gpMCMC(log_likelihood, bounds=bounds, prior_function=custom_prior)
    res = my_mcmc.run_mcmc(x0=hps, n_updates=20, break_condition=None)
    assert "median(x)" in res

    # Custom proposal distribution
    init_s = np.diag((bounds[:, 1] - bounds[:, 0]) * 0.1) ** 2
    pd = ProposalDistribution([0, 1], init_prop_Sigma=init_s, adapt_callable="normal")
    my_mcmc = gpMCMC(log_likelihood, bounds=bounds, prior_function=custom_prior,
                     proposal_distributions=[pd])
    res = my_mcmc.run_mcmc(x0=hps, n_updates=30, break_condition=None)
    assert "x" in res and len(res["x"]) > 1

    # Callable break condition
    def stop_early(obj):
        return len(obj.trace["f(x)"]) >= 5

    my_mcmc = gpMCMC(log_likelihood, bounds=bounds)
    res = my_mcmc.run_mcmc(x0=hps, n_updates=100, break_condition=stop_early)
    assert len(res["f(x)"]) <= 6   # stopped early


def test_train_async_mcmc(client):
    """Async MCMC training: submit, poll, stop."""
    my_gp = GP(x_data, y_data, init_hyperparameters=np.array([1., 1., 1., 1., 1., 1.]),
               noise_variances=np.zeros(y_data.shape) + 0.01)
    bounds = np.array([[0.01, 10.]] * 6)
    opt_obj = my_gp.train(hyperparameter_bounds=bounds, max_iter=200,
                          dask_client=client, method="mcmc", asynchronous=True)
    time.sleep(4)
    my_gp.update_hyperparameters(opt_obj)
    my_gp.stop_training(opt_obj)
    assert my_gp.hyperparameters.shape == (6,)


def test_train_async_adam(client):
    """Async Adam training: submit, poll, stop."""
    my_gp = GP(x_data, y_data, init_hyperparameters=np.array([1., 1., 1., 1., 1., 1.]),
               noise_variances=np.zeros(y_data.shape) + 0.01)
    bounds = np.array([[0.01, 10.]] * 6)
    opt_obj = my_gp.train(hyperparameter_bounds=bounds, max_iter=50,
                          dask_client=client, method="adam", asynchronous=True)
    time.sleep(3)
    my_gp.update_hyperparameters(opt_obj)
    my_gp.stop_training(opt_obj)
    assert my_gp.hyperparameters.shape == (6,)

