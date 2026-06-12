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
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    res = my_gp1.posterior_covariance(x_pred, variance_only = True)
    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]), linalg_mode = "CholInv")
    res = my_gp1.posterior_covariance(x_pred, variance_only = True)

    my_gp1 = GP(x_data, y_data, init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1.train(max_iter = 100)
    my_gp1.train(method = "adam", max_iter = 3)
    my_gp1.update_gp_data(x_data, y_data, append = True, rank_n_update = True)
    my_gp1.update_gp_data(x_data, y_data, append = True, rank_n_update = False)
    my_gp1.update_gp_data(x_data, y_data, append = False, rank_n_update = True)
    my_gp1.update_gp_data(x_data, y_data, append = False, rank_n_update = False)
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
    assert my_gp1.args == my_gp1.kv.args

    my_gp1 = GP(x_data, y_data, noise_variances = np.zeros(y_data.shape) + 0.01,init_hyperparameters = np.array([1, 1, 1, 1, 1, 1]))
    my_gp1.set_args({"dcf":4.})
    assert my_gp1.args == my_gp1.args == {"dcf":4.}
    assert my_gp1.args == my_gp1.prior.args
    assert my_gp1.args == my_gp1.likelihood.args
    assert my_gp1.args == my_gp1.marginal_likelihood.args
    assert my_gp1.args == my_gp1.trainer.args
    assert my_gp1.args == my_gp1.posterior.args
    assert my_gp1.args == my_gp1.kv.args


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
        compute_device="cpu", linalg_mode = "CholInv", ram_economy = True)


def test_linalg_modes():
    from scipy.linalg import cho_factor, cho_solve
    import importlib as _il

    hps = np.ones(6)

    # Modes where log_likelihood works without optional dependencies
    modes_full = ["Chol", "CholInv", "Inv", "sparseLU"]
    # Modes whose logdet requires the optional imate package; test everything except log_likelihood
    modes_no_logdet = ["sparseCG", "sparseMINRES", "sparseCGpre", "sparseMINRESpre", "sparseSolve"]

    for mode in modes_full:
        gp = GP(x_data, y_data, init_hyperparameters=hps, linalg_mode=mode)
        gp.log_likelihood()
        gp.posterior_mean(x_pred)
        gp.posterior_covariance(x_pred, variance_only=True)
        gp.update_gp_data(x_data, y_data, append=True)
        gp.update_gp_data(x_data, y_data, append=False)

    for mode in modes_no_logdet:
        gp = GP(x_data, y_data, init_hyperparameters=hps, linalg_mode=mode)
        gp.posterior_mean(x_pred)
        gp.posterior_covariance(x_pred, variance_only=True)
        gp.update_gp_data(x_data, y_data, append=True)
        gp.update_gp_data(x_data, y_data, append=False)

    # Preconditioner-type aliases on the *pre solvers.  Each alias must resolve
    # to the canonical mode + matching args["sparse_preconditioner_type"], and
    # the GP must function end-to-end (posterior + data updates) with that
    # preconditioner backing the iterative solve.
    canonical_to_aliases = {
        "sparseCGpre":     ["ilu", "native_ic", "block_jacobi", "schwarz"],
        "sparseMINRESpre": ["ilu", "native_ic", "block_jacobi", "schwarz"],
    }
    if _il.util.find_spec("ilupp") is not None:
        canonical_to_aliases["sparseCGpre"].extend(["ichol", "ichol0"])
        canonical_to_aliases["sparseMINRESpre"].extend(["ichol", "ichol0"])
    if _il.util.find_spec("pyamg") is not None:
        canonical_to_aliases["sparseCGpre"].append("amg")
        canonical_to_aliases["sparseMINRESpre"].append("amg")
    canonical_to_type = {
        "ilu": "ilu",
        "ichol": "ichol",
        "ichol0": "ichol0",
        "native_ic": "native_incomplete_cholesky",
        "block_jacobi": "block_jacobi",
        "schwarz": "additive_schwarz",
        "amg": "amg",
    }
    for canonical, alias_types in canonical_to_aliases.items():
        for alias_type in alias_types:
            mode = f"{canonical}_{alias_type}"
            gp = GP(x_data, y_data, init_hyperparameters=hps, linalg_mode=mode)
            assert gp.kv.mode == canonical
            assert gp.data.args.get("sparse_preconditioner_type") == canonical_to_type[alias_type]
            gp.posterior_mean(x_pred)
            gp.posterior_covariance(x_pred, variance_only=True)
            gp.update_gp_data(x_data, y_data, append=True)
            gp.update_gp_data(x_data, y_data, append=False)

    # Custom 3-callable interface
    f_factor = lambda K: cho_factor(K)
    f_solve = lambda obj, b: cho_solve(obj, b)
    f_logdet = lambda obj: 2.0 * float(np.sum(np.log(np.diag(obj[0]))))
    gp = GP(x_data, y_data, init_hyperparameters=hps, linalg_mode=[f_factor, f_solve, f_logdet])
    gp.log_likelihood()
    gp.posterior_mean(x_pred)
    gp.posterior_covariance(x_pred, variance_only=True)
    gp.update_gp_data(x_data, y_data, append=True)
    gp.update_gp_data(x_data, y_data, append=False)


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
        compute_device="cpu", linalg_mode = "CholInv", ram_economy = True)


    my_gp2.train(hyperparameter_bounds=np.array([[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10],[0.01,10]]),
            method = "hgdl", tolerance = 0.001, max_iter = 2, dask_client=client)


def test_train_hgdl_async(client):
    my_gp2 = GP(x_data,y_data,init_hyperparameters = np.array([1., 1., 1., 1., 1., 1.]),noise_variances=np.zeros(y_data.shape) + 0.01,
        compute_device="cpu", linalg_mode = "CholInv", ram_economy = True)

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

    # Each iteration exercises a different linalg_mode (which is fixed at __init__).
    # Between iterations: drop the previous GP, force-collect, and round-trip the
    # scheduler so any pending `_dec_ref` for its scatter future has fired before the
    # next GP's init scatter starts.  Otherwise those dec_refs race against the new
    # scatter's replicate inside the scheduler.
    import gc
    modes = [
        ("sparseLU",        True),
        ("sparseCG",        True),
        ("sparseMINRES",    True),
        ("sparseCGpre",     True),
        ("Chol",            True),
        ("CholInv",         True),
        ("Inv",             False),
        ("sparseMINRESpre", True),
        ("sparseSolve",     True),
    ]
    for mode, gp2s in modes:
        my_gp2S = GP(x_data, y_data, init_hps, gp2Scale=gp2s, gp2Scale_batch_size=100,
                     dask_client=client, linalg_mode=mode)
        my_gp2S.log_likelihood(hyperparameters=init_hps)
        if mode == "Inv":
            my_gp2S.neg_log_likelihood_gradient(hyperparameters=init_hps)
            my_gp2S.neg_log_likelihood_gradient()
        my_gp2S.update_gp_data(x_new, y_new, append=True)
        del my_gp2S
        gc.collect()
        client.run(lambda: None)

    my_gp2S = GP(x_data,y_data,init_hps, gp2Scale = True, gp2Scale_batch_size= 100, dask_client=client)

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


def test_ggmp():
    from fvgp import ggmp
    GGMP = ggmp.GGMP
    hyperparameters = ggmp.hyperparameters
    NormalLikelihood = ggmp.NormalLikelihood
    constant_mean = ggmp.constant_mean
    _get_key = ggmp._get_key
    gaussian_pdf = ggmp.gaussian_pdf
    _normalize_pdf = ggmp._normalize_pdf
    empirical_pdf_from_samples = ggmp.empirical_pdf_from_samples
    fit_gmm_fixed_weights = ggmp.fit_gmm_fixed_weights
    _as_2d = ggmp._as_2d
    _covariances_to_full = ggmp._covariances_to_full
    _sym_psd = ggmp._sym_psd
    _sqrtm_psd = ggmp._sqrtm_psd
    gaussian_w2_squared = ggmp.gaussian_w2_squared
    align_gmm_components_hungarian = ggmp.align_gmm_components_hungarian
    align_local_gmms_sequence = ggmp.align_local_gmms_sequence
    _log_mvn_density = ggmp._log_mvn_density
    optimize_weights_em_multivariate_samples = ggmp.optimize_weights_em_multivariate_samples
    loglik_multivariate_mixture_samples = ggmp.loglik_multivariate_mixture_samples
    sample_gmm_multivariate = ggmp.sample_gmm_multivariate
    energy_distance_multivariate = ggmp.energy_distance_multivariate
    sliced_wasserstein_distance = ggmp.sliced_wasserstein_distance
    mmd_rbf = ggmp.mmd_rbf
    fit_gmm_free_weights_multivariate = ggmp.fit_gmm_free_weights_multivariate
    fit_local_gmms_multivariate = ggmp.fit_local_gmms_multivariate

    rng = np.random.default_rng(0)
    N = 6   # stations
    K = 2   # GMM components

    # ------------------------------------------------------------------
    # NormalLikelihood
    # ------------------------------------------------------------------
    nl_mean = rng.standard_normal(N)
    nl_var  = np.abs(rng.standard_normal(N)) + 0.1
    nl = NormalLikelihood(nl_mean, nl_var, 0.5)
    assert nl.dim == N
    nl.set_moments(nl_mean + 1, nl_var * 2)
    nl.set_weight(0.3)
    assert nl.weight == 0.3
    vec = nl.unravel()
    assert len(vec) == 2 * N
    m2, v2 = nl.ravel(vec)
    assert len(m2) == N and len(v2) == N

    # ------------------------------------------------------------------
    # hyperparameters: K=2 components, 1-D x_data → 3 hps each
    # (signal_var, length_scale, prior_mean via constant_mean)
    # ------------------------------------------------------------------
    n_hps = 3
    weights = np.ones(K) / K
    weights_bounds = np.array([[0.01, 1.0]] * K)
    hps_list = [np.array([1.0, 0.5, 0.0])] * K
    hps_bounds = [np.array([[0.1, 10.], [0.01, 2.], [-5., 5.]])] * K
    hps_obj = hyperparameters(weights, weights_bounds, hps_list, hps_bounds)

    v = hps_obj.vectorized_hps
    assert len(v) == K + K * n_hps
    w2, h2 = hps_obj.devectorize_hps(v)
    assert len(w2) == K and len(h2) == K

    b = hps_obj.vectorized_bounds
    wb2, hb2 = hps_obj.devectorize_bounds(b)
    assert len(wb2) == K and len(hb2) == K

    hps_obj.set(weights, hps_list)
    assert np.allclose(hps_obj.vectorized_hps, v)

    # ------------------------------------------------------------------
    # Build small dataset: 1-D station locations, simple Gaussian PDFs
    # ------------------------------------------------------------------
    xs = np.sort(rng.random((N, 1)), axis=0)
    domain = np.linspace(-3, 3, 50)
    y_data = [
        (domain, np.exp(-0.5 * (domain - rng.uniform(-1, 1)) ** 2))
        for _ in range(N)
    ]

    # ------------------------------------------------------------------
    # GGMP construction
    # ------------------------------------------------------------------
    g = GGMP(xs, y_data, hps_obj=hps_obj, likelihood_terms=K)

    # __getattr__ fallback should not raise
    g.nonexistent_method()

    # ------------------------------------------------------------------
    # initLikelihoods — default initialization
    # ------------------------------------------------------------------
    lks = g.initLikelihoods()
    assert len(lks) == K
    assert all(isinstance(lk, NormalLikelihood) for lk in lks)

    # initLikelihoods — explicit mean / std / weights
    g2 = GGMP(xs, y_data, hps_obj=hps_obj, likelihood_terms=K)
    g2.initLikelihoods(
        init_mean=[np.zeros(N)] * K,
        init_std=[np.ones(N) * 0.5] * K,
        weights=weights,
    )
    assert len(g2.likelihoods) == K

    # ------------------------------------------------------------------
    # initGPs
    # ------------------------------------------------------------------
    g.initGPs()
    assert len(g.gps) == K
    assert g.gps is g._component_GPs

    # ------------------------------------------------------------------
    # build_pairwise_data_generating_normals
    # ------------------------------------------------------------------
    joints = g.build_pairwise_data_generating_normals(0, 1)
    assert len(joints) == K
    assert "mean" in joints[0] and "cov" in joints[0] and "weight" in joints[0]
    assert joints[0]["mean"].shape == (2,)
    assert joints[0]["cov"].shape == (2, 2)

    # ------------------------------------------------------------------
    # _as_float
    # ------------------------------------------------------------------
    assert g._as_float(1.5) == 1.5
    assert g._as_float(np.float64(2.0)) == 2.0
    assert g._as_float(np.array(3.0)) == 3.0
    assert g._as_float(np.array([4.0])) == 4.0
    assert g._as_float(np.array([1.0, 2.0]), reduce="sum") == 3.0
    assert g._as_float(np.array([1.0, 3.0]), reduce="mean") == 2.0

    # ------------------------------------------------------------------
    # _gp_log_likelihood
    # ------------------------------------------------------------------
    ll = g._gp_log_likelihood(g.gps[0])
    assert np.isscalar(ll) and np.isfinite(ll)

    # ------------------------------------------------------------------
    # _safe_set_hyperparameters (update + no-op)
    # ------------------------------------------------------------------
    new_hps = np.array([1.2, 0.4, 0.1])
    g._safe_set_hyperparameters(g.gps[0], new_hps)
    g._safe_set_hyperparameters(g.gps[0], new_hps)  # same → no-op

    # ------------------------------------------------------------------
    # constant_mean
    # ------------------------------------------------------------------
    cm = constant_mean(xs, np.array([1.0, 0.5, 2.0]))
    assert cm.shape == (N,) and np.allclose(cm, 2.0)

    # ------------------------------------------------------------------
    # _get_key
    # ------------------------------------------------------------------
    assert _get_key({"m(x)": 1, "other": 9}, ["m(x)", "mean"]) == 1
    assert _get_key({"mean": 2}, ["m(x)", "mean"]) == 2
    assert _get_key(5, ["m(x)"]) == 5   # non-dict passthrough

    # ------------------------------------------------------------------
    # gaussian_pdf
    # ------------------------------------------------------------------
    pdf_vals = gaussian_pdf(np.linspace(-2, 2, 20), 0.0, 1.0)
    assert pdf_vals.shape == (20,) and np.all(pdf_vals > 0)

    # ------------------------------------------------------------------
    # _normalize_pdf
    # ------------------------------------------------------------------
    d_n, p_n, dx_n = _normalize_pdf(domain, np.exp(-0.5 * domain ** 2))
    assert np.isclose(np.sum(p_n * dx_n), 1.0, atol=1e-3)
    # zero-mass edge case: should not raise
    d_z, p_z, dx_z = _normalize_pdf(domain, np.zeros_like(domain))
    assert np.isfinite(p_z).all()

    # ------------------------------------------------------------------
    # empirical_pdf_from_samples
    # ------------------------------------------------------------------
    samples_1d = rng.standard_normal(200)
    centers, dens = empirical_pdf_from_samples(samples_1d, bins=30)
    assert len(centers) == 30 and np.all(dens > 0)

    # ------------------------------------------------------------------
    # fit_gmm_fixed_weights
    # ------------------------------------------------------------------
    y_samp = rng.standard_normal(60)
    w_fixed = np.array([0.5, 0.5])
    means_fit, vars_fit = fit_gmm_fixed_weights(y_samp, 2, w_fixed, max_iter=10)
    assert len(means_fit) == 2 and len(vars_fit) == 2
    assert means_fit[0] <= means_fit[1]  # sorted by mean

    # ------------------------------------------------------------------
    # _as_2d
    # ------------------------------------------------------------------
    assert _as_2d(np.ones(5)).shape == (5, 1)
    assert _as_2d(np.ones((3, 2))).shape == (3, 2)

    # ------------------------------------------------------------------
    # _covariances_to_full — all four covariance types
    # ------------------------------------------------------------------
    assert _covariances_to_full(np.ones((2, 3)), covariance_type="diag",       K=2, d=3).shape == (2, 3, 3)
    assert _covariances_to_full(np.array([1., 2.]), covariance_type="spherical", K=2, d=3).shape == (2, 3, 3)
    assert _covariances_to_full(np.eye(3)[None].repeat(2, 0), covariance_type="full", K=2, d=3).shape == (2, 3, 3)
    assert _covariances_to_full(np.eye(3), covariance_type="tied",             K=2, d=3).shape == (2, 3, 3)

    # ------------------------------------------------------------------
    # _sym_psd / _sqrtm_psd
    # ------------------------------------------------------------------
    A = rng.standard_normal((3, 3))
    S = _sym_psd(A)
    assert np.allclose(S, S.T)

    B = A @ A.T + np.eye(3)
    sqrtB = _sqrtm_psd(B)
    assert np.allclose(sqrtB @ sqrtB, B, atol=1e-8)

    # ------------------------------------------------------------------
    # gaussian_w2_squared
    # ------------------------------------------------------------------
    assert np.isclose(gaussian_w2_squared(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2)), 0.0, atol=1e-8)
    assert gaussian_w2_squared(np.array([2., 0.]), np.eye(2), np.zeros(2), np.eye(2)) > 0.0

    # ------------------------------------------------------------------
    # align_gmm_components_hungarian
    # ------------------------------------------------------------------
    m_ref = np.array([[0.], [3.]])
    c_ref = np.eye(1)[None].repeat(2, 0)
    m_cur = np.array([[3.1], [0.1]])  # reversed order
    perm = align_gmm_components_hungarian(m_ref, c_ref, m_cur, c_ref)
    assert perm.shape == (2,)
    assert perm[0] == 1 and perm[1] == 0  # should swap back

    perm2, cost2 = align_gmm_components_hungarian(m_ref, c_ref, m_cur, c_ref, return_cost=True)
    assert cost2.shape == (2, 2)

    # ------------------------------------------------------------------
    # align_local_gmms_sequence
    # ------------------------------------------------------------------
    wl = [np.array([0.5, 0.5])] * 4
    ml = [np.array([[0.], [3.]])] * 4
    cl = [np.eye(1)[None].repeat(2, 0)] * 4
    res = align_local_gmms_sequence(wl, ml, cl, reference="previous")
    assert len(res["means"]) == 4 and len(res["perms"]) == 4

    res2 = align_local_gmms_sequence(wl, ml, cl, reference="first")
    assert len(res2["means"]) == 4

    # ------------------------------------------------------------------
    # _log_mvn_density
    # ------------------------------------------------------------------
    y_mv = rng.standard_normal((10, 2))
    ldens = _log_mvn_density(y_mv, np.zeros(2), np.eye(2))
    assert ldens.shape == (10,) and np.all(np.isfinite(ldens))

    # ------------------------------------------------------------------
    # optimize_weights_em_multivariate_samples
    # ------------------------------------------------------------------
    y_list_mv  = [rng.standard_normal((20, 2)) for _ in range(4)]
    means_em   = [np.array([[-1., 0.], [1., 0.]])] * 4   # (K, d) per station
    covs_em    = [np.eye(2)[None].repeat(2, 0)] * 4        # (K, d, d) per station
    w_opt, w_hist, obj_hist = optimize_weights_em_multivariate_samples(
        y_list_mv, means_em, covs_em, K=2, max_iter=5, log_every=0)
    assert w_opt.shape == (2,) and np.isclose(w_opt.sum(), 1.0, atol=1e-8)

    # ------------------------------------------------------------------
    # loglik_multivariate_mixture_samples
    # ------------------------------------------------------------------
    ll_ps = loglik_multivariate_mixture_samples(
        y_mv,
        np.array([0.5, 0.5]),
        np.array([[0., 0.], [1., 1.]]),
        np.eye(2)[None].repeat(2, 0),
    )
    assert ll_ps.shape == (10,) and np.all(np.isfinite(ll_ps))

    # ------------------------------------------------------------------
    # sample_gmm_multivariate
    # ------------------------------------------------------------------
    samp = sample_gmm_multivariate(
        np.array([0.5, 0.5]),
        np.array([[0., 0.], [2., 2.]]),
        np.eye(2)[None].repeat(2, 0),
        n_samples=30,
        random_state=1,
    )
    assert samp.shape == (30, 2)

    # ------------------------------------------------------------------
    # energy_distance_multivariate
    # ------------------------------------------------------------------
    ed = energy_distance_multivariate(y_mv, rng.standard_normal((8, 2)))
    assert np.isscalar(ed) and ed >= 0.0

    # ------------------------------------------------------------------
    # sliced_wasserstein_distance
    # ------------------------------------------------------------------
    swd = sliced_wasserstein_distance(y_mv, rng.standard_normal((8, 2)), n_projections=8)
    assert np.isscalar(swd) and swd >= 0.0

    # ------------------------------------------------------------------
    # mmd_rbf
    # ------------------------------------------------------------------
    mmd = mmd_rbf(y_mv, rng.standard_normal((8, 2)))
    assert np.isscalar(mmd) and np.isfinite(mmd)

    # ------------------------------------------------------------------
    # fit_gmm_free_weights_multivariate / fit_local_gmms_multivariate
    # (require scikit-learn; skip gracefully if not installed)
    # ------------------------------------------------------------------
    try:
        w_free, m_free, c_free, info = fit_gmm_free_weights_multivariate(
            rng.standard_normal((60, 2)), K=2, n_init=2, max_iter=20)
        assert w_free.shape == (2,) and np.isclose(w_free.sum(), 1.0, atol=1e-8)
        assert "converged" in info and "bic" in info

        local = fit_local_gmms_multivariate(
            [rng.standard_normal((40, 1)) for _ in range(3)], K=2, n_init=2, max_iter=20)
        assert "weights" in local and len(local["weights"]) == 3
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # prepare_station_terms_density + optimize_weights_em_density
    # ------------------------------------------------------------------
    prepare_station_terms_density = ggmp.prepare_station_terms_density
    optimize_weights_em_density = ggmp.optimize_weights_em_density

    terms_d, ll_comp = prepare_station_terms_density(g, hps_list)
    assert len(terms_d) == N
    p0, dx0, lpdf0 = terms_d[0]
    assert p0.shape == dx0.shape
    assert lpdf0.shape == (len(domain), K)
    assert ll_comp.shape == (K,) and np.all(np.isfinite(ll_comp))

    w_d, w_hist_d, obj_hist_d = optimize_weights_em_density(
        terms_d, K=K, weight_floor=1e-9, max_iter=10, tol_l1=1e-10, log_every=0)
    assert w_d.shape == (K,) and np.isclose(w_d.sum(), 1.0, atol=1e-8)
    assert len(w_hist_d) > 0 and len(obj_hist_d) > 0

    # ------------------------------------------------------------------
    # train — phase 1 only (train_weights=False)
    # ------------------------------------------------------------------
    synced = g.train(method="local", max_iter=5, train_weights=False)
    assert len(synced) == K
    assert all(len(h) == n_hps for h in synced)

    # ------------------------------------------------------------------
    # train — phase 2 density (default)
    # ------------------------------------------------------------------
    synced2 = g.train(method="local", max_iter=5, train_weights=True, weight_method="density",
                      weight_max_iter=5)
    assert len(synced2) == K
    weights_after = np.array([g.likelihoods[k].weight for k in range(K)])
    assert np.isclose(weights_after.sum(), 1.0, atol=1e-8)

    # ------------------------------------------------------------------
    # train — phase 2 samples
    # ------------------------------------------------------------------
    y_samples_train = [rng.standard_normal(30) for _ in range(N)]
    synced3 = g.train(method="local", max_iter=5, train_weights=True,
                      weight_method="samples", weight_max_iter=5,
                      y_samples=y_samples_train)
    assert len(synced3) == K

    # train with unknown weight_method raises
    try:
        g.train(method="local", max_iter=2, train_weights=True, weight_method="bad")
        assert False, "should have raised"
    except ValueError:
        pass

    # train with weight_method='samples' but no y_samples raises
    try:
        g.train(method="local", max_iter=2, train_weights=True, weight_method="samples")
        assert False, "should have raised"
    except ValueError:
        pass

    # ------------------------------------------------------------------
    # posterior_mean / posterior_variance
    # ------------------------------------------------------------------
    x_pred = np.linspace(0, 1, 4).reshape(-1, 1)
    pm = g.posterior_mean(x_pred)
    assert pm.shape == (4,) and np.all(np.isfinite(pm))

    pv = g.posterior_variance(x_pred)
    assert pv.shape == (4,) and np.all(pv >= 0)

    # ------------------------------------------------------------------
    # bhattacharyya_distance / kl_divergence / wasserstein_1d
    # ------------------------------------------------------------------
    bhattacharyya_distance = ggmp.bhattacharyya_distance
    kl_divergence = ggmp.kl_divergence
    wasserstein_1d = ggmp.wasserstein_1d

    p_ref = np.exp(-0.5 * domain ** 2)
    q_ref = np.exp(-0.5 * (domain - 1) ** 2)

    bd = bhattacharyya_distance(domain, p_ref, q_ref)
    assert np.isscalar(bd) and bd >= 0.0
    assert np.isclose(bhattacharyya_distance(domain, p_ref, p_ref), 0.0, atol=1e-6)

    kl = kl_divergence(domain, p_ref, q_ref)
    assert np.isscalar(kl) and kl >= 0.0

    w1 = wasserstein_1d(domain, p_ref, q_ref)
    assert np.isscalar(w1) and w1 >= 0.0
    assert np.isclose(wasserstein_1d(domain, p_ref, p_ref), 0.0, atol=1e-6)


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
    my_gpo2 = pickle.loads(pickle.dumps(my_gpo))
    assert my_gpo2.marginal_likelihood is my_gpo2.marginal_likelihood

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
    assert is_pickle_equal(my_gpo.kv)


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


# =========================================================================
# Tests for the new linear-algebra capabilities (preconditioner framework,
# block CG, multi-column x0 normalization, GPU detection helpers).
# =========================================================================

import importlib as _importlib_for_tests
import pytest
from fvgp import gp_lin_alg as _gp_lin_alg


def _gpu_engines_available():
    engines = []
    if _importlib_for_tests.util.find_spec("torch") is not None:
        try:
            import torch
            if torch.cuda.is_available():
                engines.append("torch")
            else:
                mps_backend = getattr(torch.backends, "mps", None)
                if mps_backend is not None and torch.backends.mps.is_available():
                    engines.append("torch")
        except Exception:
            pass
    if _importlib_for_tests.util.find_spec("cupy") is not None:
        try:
            import cupy as cp
            cp.zeros(1)
            engines.append("cupy")
        except Exception:
            pass
    return engines


def _make_test_spd_sparse(n=40, seed=0):
    rng = np.random.RandomState(seed)
    A = sparse.random(n, n, density=0.15, random_state=rng, format="csr")
    A = (A + A.T) * 0.5
    A = A + (abs(A).sum(axis=1).A1.max() + 1.0) * sparse.eye(n, format="csr")
    return A.tocsr()


def test_normalize_sparse_preconditioner_type():
    assert normalize_sparse_preconditioner_type("ILU") == "ilu"
    assert normalize_sparse_preconditioner_type("ic") == "ichol"
    assert normalize_sparse_preconditioner_type("ichol") == "ichol"
    assert normalize_sparse_preconditioner_type("ichol0") == "ichol0"
    assert normalize_sparse_preconditioner_type("native_ic") == "native_incomplete_cholesky"
    assert normalize_sparse_preconditioner_type("native_ichol") == "native_incomplete_cholesky"
    assert normalize_sparse_preconditioner_type("BlockJacobi") == "block_jacobi"
    assert normalize_sparse_preconditioner_type("schwarz") == "additive_schwarz"
    assert normalize_sparse_preconditioner_type("AMG") == "amg"
    with pytest.raises(ValueError):
        normalize_sparse_preconditioner_type("nope")


def test_resolve_gp2scale_linalg_mode():
    # Pass-through for unknown / non-prefixed strings
    mode, args = resolve_gp2scale_linalg_mode("Chol")
    assert mode == "Chol" and "sparse_preconditioner_type" not in args

    # Alias resolution for CG / MINRES preconditioner suffixes
    mode, args = resolve_gp2scale_linalg_mode("sparseCGpre_amg")
    assert mode == "sparseCGpre" and args["sparse_preconditioner_type"] == "amg"

    mode, args = resolve_gp2scale_linalg_mode("sparseMINRESpre_ichol")
    assert mode == "sparseMINRESpre" and args["sparse_preconditioner_type"] == "ichol"

    mode, args = resolve_gp2scale_linalg_mode("sparseMINRESpre_ic")
    assert mode == "sparseMINRESpre" and args["sparse_preconditioner_type"] == "ichol"

    mode, args = resolve_gp2scale_linalg_mode("sparseMINRESpre_native_ic")
    assert mode == "sparseMINRESpre" and args["sparse_preconditioner_type"] == "native_incomplete_cholesky"

    mode, args = resolve_gp2scale_linalg_mode("sparseCGpre_native_ichol")
    assert mode == "sparseCGpre" and args["sparse_preconditioner_type"] == "native_incomplete_cholesky"

    # Consistent explicit type is allowed
    mode, args = resolve_gp2scale_linalg_mode(
        "sparseCGpre_ilu", args={"sparse_preconditioner_type": "ilu"}
    )
    assert mode == "sparseCGpre" and args["sparse_preconditioner_type"] == "ilu"

    # Conflicting explicit type raises
    with pytest.raises(ValueError):
        resolve_gp2scale_linalg_mode(
            "sparseCGpre_ilu", args={"sparse_preconditioner_type": "amg"}
        )


def test_calculate_sparse_preconditioner_ilu():
    A = _make_test_spd_sparse(n=30)
    factor, op = calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "ilu"})
    # The ILU operator should approximately invert A; test by checking the
    # residual when using it as a preconditioner on a CG solve
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_calculate_sparse_preconditioner_native_ic0():
    A = _make_test_spd_sparse(n=30)
    factor, op = calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "native_ic"})
    assert factor["type"] == "native_incomplete_cholesky"
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_missing_ilupp_message_for_ic_aliases(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ilupp":
            raise ImportError("simulated missing ilupp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    A = _make_test_spd_sparse(n=10)

    with pytest.raises(ImportError, match="pip install ilupp"):
        calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "ic"})

    with pytest.raises(ImportError, match="pip install ilupp"):
        calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "ichol0"})


def test_calculate_sparse_preconditioner_ichol0():
    pytest.importorskip("ilupp")
    A = _make_test_spd_sparse(n=30)
    factor, op = calculate_sparse_preconditioner(
        A, args={"sparse_preconditioner_type": "ichol0"}
    )
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_calculate_sparse_preconditioner_ichol():
    pytest.importorskip("ilupp")
    A = _make_test_spd_sparse(n=30)
    factor, op = calculate_sparse_preconditioner(
        A,
        args={
            "sparse_preconditioner_type": "ichol",
            "sparse_preconditioner_ichol_fill_in": 8,
            "sparse_preconditioner_ichol_threshold": 1e-3,
        },
    )
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_ichol0_shift_retry_on_factor_failure(monkeypatch):
    """First attempt fails -> shift-retry helper bumps the diagonal and succeeds."""
    pytest.importorskip("ilupp")
    import ilupp

    real_factor_cls = ilupp.IChol0Preconditioner
    call_count = {"n": 0}

    def fake_factor(A):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated non-PD pivot")
        return real_factor_cls(A)

    monkeypatch.setattr(ilupp, "IChol0Preconditioner", fake_factor)

    A = _make_test_spd_sparse(n=20)
    factor, op = calculate_sparse_preconditioner(
        A,
        args={
            "sparse_preconditioner_type": "ichol0",
            "sparse_preconditioner_shift": 1e-6,
            "sparse_preconditioner_shift_attempts": 4,
        },
    )
    # First attempt failed, retry succeeded -> at least 2 calls.
    assert call_count["n"] >= 2
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_calculate_sparse_preconditioner_block_jacobi():
    A = _make_test_spd_sparse(n=30)
    factor, op = calculate_sparse_preconditioner(
        A, args={"sparse_preconditioner_type": "block_jacobi", "sparse_preconditioner_block_size": 5}
    )
    assert factor["type"] == "block_jacobi"
    # Block partition covers all rows exactly once
    covered = np.concatenate(factor["blocks"])
    assert sorted(covered.tolist()) == list(range(A.shape[0]))
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8, "sparse_cg_maxiter": 500})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_calculate_sparse_preconditioner_additive_schwarz():
    A = _make_test_spd_sparse(n=30)
    factor, op = calculate_sparse_preconditioner(
        A,
        args={
            "sparse_preconditioner_type": "additive_schwarz",
            "sparse_preconditioner_block_size": 5,
            "sparse_preconditioner_schwarz_overlap": 1,
        },
    )
    assert factor["type"] == "additive_schwarz"
    assert factor["overlap"] == 1
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8, "sparse_cg_maxiter": 500})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_calculate_sparse_preconditioner_amg():
    if _importlib_for_tests.util.find_spec("pyamg") is None:
        pytest.skip("pyamg not installed")
    A = _make_test_spd_sparse(n=40)
    factor, op = calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "amg"})
    b = np.random.rand(A.shape[0])
    x = calculate_sparse_conj_grad(A, b, M=op, args={"sparse_cg_tol": 1e-8})
    res = np.linalg.norm(A @ x[:, 0] - b) / np.linalg.norm(b)
    assert res < 1e-6


def test_calculate_sparse_preconditioner_unknown_type():
    A = _make_test_spd_sparse(n=10)
    with pytest.raises(ValueError):
        calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "nope"})


def test_block_conjugate_gradient_multi_rhs():
    A = _make_test_spd_sparse(n=30)
    rng = np.random.RandomState(1)
    B = rng.randn(A.shape[0], 4)
    # Block-CG path
    X_block = calculate_sparse_conj_grad(
        A, B, args={"sparse_block_krylov": True, "sparse_cg_tol": 1e-8}
    )
    assert X_block.shape == B.shape
    res = np.linalg.norm(A @ X_block - B) / np.linalg.norm(B)
    assert res < 1e-6
    # Single-column path should give consistent result
    X_single = calculate_sparse_conj_grad(A, B, args={"sparse_cg_tol": 1e-8})
    assert np.allclose(X_block, X_single, atol=1e-4)


def test_sparse_solvers_multi_column_x0():
    """The merged solvers must accept a 2-d x0 with mismatched leading dim."""
    A = _make_test_spd_sparse(n=20)
    rng = np.random.RandomState(2)
    B = rng.randn(A.shape[0], 3)
    # Short x0 (15 rows) should get zero-padded to 20 internally
    x0_short = rng.randn(15, 3)
    X = calculate_sparse_conj_grad(A, B, x0=x0_short, args={"sparse_cg_tol": 1e-8})
    assert X.shape == B.shape
    assert np.linalg.norm(A @ X - B) / np.linalg.norm(B) < 1e-6
    # Single-column x0 should broadcast to all RHS columns
    x0_one_col = rng.randn(A.shape[0], 1)
    X2 = calculate_sparse_minres(A, B, x0=x0_one_col, args={"sparse_minres_tol": 1e-8})
    assert X2.shape == B.shape


def test_sparse_solvers_maxiter():
    """maxiter caps iterations even when tolerance is unmet — should emit a warning."""
    A = _make_test_spd_sparse(n=30)
    b = np.random.rand(A.shape[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_sparse_conj_grad(A, b, args={"sparse_cg_tol": 1e-15, "sparse_cg_maxiter": 1})
        assert any("CG not successful" in str(w.message) for w in caught)


def test_sparse_conj_grad_legacy_tolerance_keys():
    """Backward-compat: cg_minres_tol and sparse_minres_tol still work for CG."""
    A = _make_test_spd_sparse(n=15)
    b = np.random.rand(A.shape[0])
    # Each should produce a usable solution
    x1 = calculate_sparse_conj_grad(A, b, args={"cg_minres_tol": 1e-8})
    x2 = calculate_sparse_conj_grad(A, b, args={"sparse_minres_tol": 1e-8})
    assert np.linalg.norm(A @ x1[:, 0] - b) / np.linalg.norm(b) < 1e-6
    assert np.linalg.norm(A @ x2[:, 0] - b) / np.linalg.norm(b) < 1e-6


def test_gpu_engine_detection_no_args():
    """get_gpu_engine returns None when no usable GPU backend is detected."""
    engines = _gpu_engines_available()
    detected = _gp_lin_alg.get_gpu_engine(None)
    if engines:
        assert detected in engines
    else:
        assert detected is None


def test_gpu_engine_unknown_request():
    """Explicit unsupported engine returns None rather than raising."""
    assert _gp_lin_alg.get_gpu_engine({"GPU_engine": "tensorflow"}) is None


def test_gpu_cpu_fallback_warning():
    """When compute_device='gpu' is requested but no GPU backend is usable,
    the dense GPU paths must fall back to CPU with a UserWarning, not crash."""
    if _gpu_engines_available():
        pytest.skip("GPU backend available; this test exercises CPU fallback")
    A = np.eye(5) * 2.0 + np.ones((5, 5)) * 0.01
    b = np.random.rand(5)
    # Disable real backends explicitly by requesting an unknown engine
    args = {"GPU_engine": "tensorflow"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        L = calculate_Chol_factor(A, compute_device="gpu", args=args)
        calculate_Chol_solve(L, b, compute_device="gpu", args=args)
        calculate_Chol_logdet(L, compute_device="gpu", args=args)
        matmul(A, A, compute_device="gpu", args=args)
        matmul3(A, A, A, compute_device="gpu", args=args)
    fallback_msgs = [w.message.args[0] for w in caught if isinstance(w.category, type) and issubclass(w.category, UserWarning)]
    # At least four fallback warnings should have fired (one per function above)
    assert sum("Falling back to CPU" in m for m in fallback_msgs) >= 4


# -------- GPU-only paths (run only when a real GPU backend is present) --------

def test_calculate_logdet_cupy():
    """cupy logdet path; previously this function was torch-only."""
    if "cupy" not in _gpu_engines_available():
        pytest.skip("cupy GPU not available")
    np.random.seed(0)
    B = np.random.rand(15, 15)
    A = (B @ B.T + np.eye(15) * 5.0).astype(np.float64)
    cpu_ld = calculate_logdet(A, compute_device="cpu")
    gpu_ld = calculate_logdet(A, compute_device="gpu", args={"GPU_engine": "cupy"})
    assert np.isclose(cpu_ld, gpu_ld, rtol=1e-5)


def test_calculate_inv_cupy():
    if "cupy" not in _gpu_engines_available():
        pytest.skip("cupy GPU not available")
    np.random.seed(0)
    B = np.random.rand(15, 15)
    A = (B @ B.T + np.eye(15) * 5.0).astype(np.float64)
    cpu_inv = calculate_inv(A, compute_device="cpu")
    gpu_inv = calculate_inv(A, compute_device="gpu", args={"GPU_engine": "cupy"})
    assert np.allclose(cpu_inv, gpu_inv, rtol=1e-5)


def test_solve_cupy():
    if "cupy" not in _gpu_engines_available():
        pytest.skip("cupy GPU not available")
    np.random.seed(0)
    B = np.random.rand(15, 15)
    A = (B @ B.T + np.eye(15) * 5.0).astype(np.float64)
    b = np.random.rand(15)
    cpu_x = solve(A, b, compute_device="cpu")
    gpu_x = solve(A, b, compute_device="gpu", args={"GPU_engine": "cupy"})
    assert np.allclose(cpu_x, gpu_x, rtol=1e-5)


def test_torch_device_selection_mps_or_cuda():
    """_torch_gpu_device honors GPU_device requests when the device exists."""
    if _importlib_for_tests.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import torch
    if not torch.cuda.is_available() and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        pytest.skip("no torch GPU/MPS available")
    device = _gp_lin_alg._torch_gpu_device(None)
    assert device is not None
    assert device.type in ("cuda", "mps")


# =========================================================================
# Tests for the new kernel capabilities (support-aware Wendland sparse
# kernels, GPU detection helpers).
# =========================================================================

from fvgp import kernels as _kernels


def test_wendland_support_aware_cpu_matches_dense():
    """Output-sensitive sparse kernel must equal the dense reference exactly."""
    rng = np.random.RandomState(0)
    x1 = rng.rand(40, 3)
    x2 = rng.rand(30, 3)
    hps = np.array([1.7, 0.3, 0.4, 0.5])
    K_dense = wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)
    K_sparse = wendland_anisotropic_gp2Scale_cpu_sparse(x1, x2, hps)
    assert sparse.issparse(K_sparse)
    assert K_sparse.shape == K_dense.shape
    assert np.allclose(K_dense, K_sparse.toarray(), atol=1e-12)


def test_wendland_support_aware_cpu_self_block():
    """K(x, x) sparse vs dense agreement on a self-block (diagonal full of amplitude)."""
    rng = np.random.RandomState(1)
    x = rng.rand(25, 2)
    hps = np.array([2.5, 0.6, 0.4])
    K_dense = wendland_anisotropic_gp2Scale_cpu(x, x, hps)
    K_sparse = wendland_anisotropic_gp2Scale_cpu_sparse(x, x, hps)
    diff = K_dense - K_sparse.toarray()
    assert np.max(np.abs(diff)) < 1e-12
    # Diagonal equals amplitude for self-distance 0
    assert np.allclose(np.diag(K_sparse.toarray()), hps[0])


def test_wendland_support_aware_cpu_disjoint_blocks():
    """Blocks separated beyond the support radius yield an all-zero sparse block."""
    # Two clusters far apart in whitened coordinates: with length scale 0.1 along
    # each axis, points at separation 10.0 are >> support radius 1.
    x1 = np.array([[0.0, 0.0], [0.0, 0.05]])
    x2 = np.array([[10.0, 10.0], [10.0, 10.05]])
    hps = np.array([1.0, 0.1, 0.1])
    K_sparse = wendland_anisotropic_gp2Scale_cpu_sparse(x1, x2, hps)
    assert K_sparse.nnz == 0
    assert K_sparse.shape == (2, 2)


def test_wendland_support_aware_cpu_empty_input():
    """Empty input arrays return an empty sparse block of correct shape."""
    hps = np.array([1.0, 0.5, 0.5])
    K = wendland_anisotropic_gp2Scale_cpu_sparse(np.zeros((0, 2)), np.zeros((0, 2)), hps)
    assert K.shape == (0, 0)


def test_kernels_gpu_engine_detection():
    """The kernels module's GPU engine helper should agree with availability."""
    engine = _kernels._get_default_gpu_engine()
    if _gpu_engines_available():
        assert engine in ("torch", "cupy")
    else:
        assert engine is None


def test_wendland_anisotropic_gp2Scale_gpu_fallback():
    """When no GPU backend is available, the GPU Wendland falls back to CPU
    with a UserWarning and returns the same array."""
    if _gpu_engines_available():
        pytest.skip("GPU backend available; this test exercises CPU fallback")
    rng = np.random.RandomState(3)
    x1 = rng.rand(20, 2)
    x2 = rng.rand(15, 2)
    hps = np.array([1.0, 0.3, 0.3])
    K_cpu = wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        K_gpu = wendland_anisotropic_gp2Scale_gpu(x1, x2, hps)
        assert any("falling back to the CPU" in str(w.message) for w in caught)
    assert np.allclose(K_cpu, K_gpu)


def test_wendland_anisotropic_gp2Scale_gpu_matches_cpu():
    """When a torch or cupy GPU is available, the GPU Wendland matches the CPU."""
    if not _gpu_engines_available():
        pytest.skip("no GPU backend available")
    rng = np.random.RandomState(4)
    x1 = rng.rand(20, 2)
    x2 = rng.rand(15, 2)
    hps = np.array([1.0, 0.3, 0.3])
    K_cpu = wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)
    K_gpu = wendland_anisotropic_gp2Scale_gpu(x1, x2, hps)
    # GPU path internally uses float32; allow modest tolerance
    assert np.allclose(K_cpu, K_gpu, atol=1e-4)


def test_wendland_support_aware_gpu_sparse_matches_cpu_sparse():
    """GPU support-aware sparse Wendland matches the CPU sparse variant
    (or falls back to it with a warning when no GPU is available)."""
    rng = np.random.RandomState(5)
    x1 = rng.rand(30, 3)
    x2 = rng.rand(25, 3)
    hps = np.array([1.4, 0.3, 0.4, 0.5])
    K_cpu = wendland_anisotropic_gp2Scale_cpu_sparse(x1, x2, hps)
    K_gpu = wendland_anisotropic_gp2Scale_gpu_sparse(x1, x2, hps)
    # GPU path uses float32, so allow a slightly looser tolerance
    assert np.allclose(K_cpu.toarray(), K_gpu.toarray(), atol=1e-4)


# =========================================================================
# Tests for the new preconditioner cache + warm-start integration in
# GPkv / GPMarginalLikelihood (the training-path acceleration wiring).
# =========================================================================

def _make_test_gp(linalg_mode, args=None, n=60, noise=0.05, seed=11):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, 2)
    y = np.sin(np.linalg.norm(x, axis=1) * 4.0) + noise * rng.randn(n)
    hps = np.array([1.0, 0.4, 0.4])
    extra = {} if args is None else dict(args)
    return GP(x, y, init_hyperparameters=hps, linalg_mode=linalg_mode,
              args=extra, compute_device="cpu"), hps


def test_kv_preconditioner_cache_reuse_counter():
    """Refresh interval > 1 lets repeated update_KV calls reuse the cached
    preconditioner rather than rebuilding from scratch.

    Counting note: init runs set_KV (force-builds, counter=0) AND a follow-up
    solve in _refresh (which reuses → counter=1).  So the counter starts at 1
    after construction, not 0.  With refresh_interval=4, three more reuses are
    available (counter 1→2→3) before the fourth call rebuilds.
    """
    gp, hps = _make_test_gp("sparseCGpre", args={"sparse_preconditioner_refresh_interval": 4})
    kv = gp.kv
    assert kv.Preconditioner_operator is not None
    op0 = kv.Preconditioner_operator
    assert kv.Preconditioner_reuse_counter == 1

    KV = kv.addKV(kv.K, kv.V)
    kv.update_KV(KV)
    assert kv.Preconditioner_operator is op0
    assert kv.Preconditioner_reuse_counter == 2

    kv.update_KV(KV)
    assert kv.Preconditioner_operator is op0
    assert kv.Preconditioner_reuse_counter == 3

    # Now reuse_counter >= refresh_interval-1 (= 3): next call rebuilds
    kv.update_KV(KV)
    assert kv.Preconditioner_operator is not None
    assert kv.Preconditioner_operator is not op0
    assert kv.Preconditioner_reuse_counter == 0


def test_kv_preconditioner_signature_invalidates_cache():
    """Changing a sparse_preconditioner_* arg invalidates the cached operator."""
    gp, hps = _make_test_gp("sparseCGpre", args={"sparse_preconditioner_refresh_interval": 5})
    kv = gp.kv
    op0 = kv.Preconditioner_operator
    assert op0 is not None

    # Mutate args to flip the preconditioner type
    gp.data.args["sparse_preconditioner_type"] = "native_ic"
    KV = kv.addKV(kv.K, kv.V)
    kv.update_KV(KV)
    assert kv.Preconditioner_operator is not None
    assert kv.Preconditioner_operator is not op0  # rebuilt
    assert kv.Preconditioner_reuse_counter == 0


def test_kv_set_KV_force_refreshes_preconditioner():
    """set_KV models a real state change and must always rebuild the preconditioner."""
    gp, hps = _make_test_gp("sparseCGpre", args={"sparse_preconditioner_refresh_interval": 99})
    kv = gp.kv
    op0 = kv.Preconditioner_operator
    KV = kv.addKV(kv.K, kv.V)
    kv.set_KV(KV)
    assert kv.Preconditioner_operator is not None
    assert kv.Preconditioner_operator is not op0
    assert kv.Preconditioner_reuse_counter == 0


def test_kv_mode_alias_resolution_at_init():
    """`sparseCGpre_amg` at GP construction → mode `sparseCGpre` + args injected."""
    if _importlib_for_tests.util.find_spec("pyamg") is None:
        pytest.skip("pyamg not installed")
    gp, _ = _make_test_gp("sparseCGpre_amg")
    assert gp.kv.mode == "sparseCGpre"
    assert gp.data.args.get("sparse_preconditioner_type") == "amg"


def test_compute_new_KVlogdet_matches_baseline():
    """Cached + warm-started compute_new_KVlogdet_KVinvY must equal the
    uncached, cold-start baseline numerically."""
    # Baseline run — refresh every call, no warm-start
    gp_base, hps = _make_test_gp("sparseCGpre")
    # Configured run — interval=4, warm-start on
    gp_opt, _ = _make_test_gp("sparseCGpre",
                              args={"sparse_preconditioner_refresh_interval": 4,
                                    "sparse_krylov_warm_start": True})

    # Step through a sequence of nearby hyperparameter values
    test_hps_list = [hps * (1.0 + 0.02 * i) for i in range(6)]
    base_logdets = []
    opt_logdets = []
    for hps_i in test_hps_list:
        K = gp_base.prior.compute_prior_covariance_matrix(gp_base.x_data, hps_i)
        V = gp_base.likelihood.calculate_V(gp_base.x_data, hps_i)
        m = gp_base.prior.compute_mean(gp_base.x_data, hps_i)
        _, ld_base = gp_base.marginal_likelihood.compute_new_KVlogdet_KVinvY(K, V, m)
        _, ld_opt = gp_opt.marginal_likelihood.compute_new_KVlogdet_KVinvY(K, V, m)
        base_logdets.append(ld_base)
        opt_logdets.append(ld_opt)

    # Stochastic-Lanczos logdet is noisy; the iterative KVinvY solve is
    # tolerance-controlled.  The values should agree to a few percent.
    base_arr = np.array(base_logdets)
    opt_arr = np.array(opt_logdets)
    assert np.allclose(base_arr, opt_arr, rtol=0.1)


def test_warm_start_updates_cached_KVinvY():
    """When sparse_krylov_warm_start=True, the marginal likelihood caches the
    most recent KVinvY for use as x0 on the next call."""
    gp, hps = _make_test_gp("sparseCG",
                            args={"sparse_krylov_warm_start": True})
    ml = gp.marginal_likelihood
    assert ml._warm_start_KVinvY is None  # not seeded by init
    # Call compute_new_KVlogdet_KVinvY with the committed hps
    K, V, m = gp.K, gp.V, gp.prior.m
    ml.compute_new_KVlogdet_KVinvY(K, V, m)
    assert ml._warm_start_KVinvY is not None
    # A second call should keep updating the cache (overwrites)
    cached1 = ml._warm_start_KVinvY.copy()
    ml.compute_new_KVlogdet_KVinvY(K, V, m)
    assert ml._warm_start_KVinvY is not None
    # Shape matches the y_data shape
    assert ml._warm_start_KVinvY.shape == gp.y_data.shape


def test_warm_start_off_by_default():
    """Without the flag, no warm-start state is built up."""
    gp, hps = _make_test_gp("sparseCG")
    ml = gp.marginal_likelihood
    K, V, m = gp.K, gp.V, gp.prior.m
    ml.compute_new_KVlogdet_KVinvY(K, V, m)
    assert ml._warm_start_KVinvY is None


def test_preconditioner_build_failure_falls_back():
    """A broken preconditioner builder must trigger a UserWarning and the
    iterative solve still runs unpreconditioned (returns a usable KVinvY)."""
    # 'amg' will fail if pyamg is missing — exercise the fallback path
    if _importlib_for_tests.util.find_spec("pyamg") is not None:
        pytest.skip("pyamg installed; failure path not exercised")
    gp, hps = _make_test_gp("sparseCGpre", args={"sparse_preconditioner_type": "amg"})
    K, V, m = gp.K, gp.V, gp.prior.m
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KVinvY, _ = gp.marginal_likelihood.compute_new_KVlogdet_KVinvY(K, V, m)
    # Solve still produces an array of the right shape
    assert KVinvY.shape == gp.y_data.shape
    # And the build-failure warning fired
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("Failed to build sparse preconditioner" in m for m in msgs)
