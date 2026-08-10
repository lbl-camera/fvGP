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
    """kernels no longer keeps its own backend detection: it resolves through
    gp_lin_alg, so args["GPU_engine"] and args["GPU_device"] are honored there too."""
    assert not hasattr(_kernels, "_get_default_gpu_engine")
    assert _kernels._get_gpu_engine is _gp_lin_alg.get_gpu_engine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = _kernels._get_gpu_engine(None)
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


def test_multi_task_posterior_covariance_S_layout():
    """`S` must be indexed [point, point, task, task] (issue: task/point axes swapped).

    The flat product-space index is task-major (k = point + Npts*task), so reshaping
    the (Npts*No, Npts*No) matrix straight to (Npts, Npts, No, No) interleaves the two
    axes and scrambles the result. Nothing inside fvgp or gpCAM reads `S` -- they all
    use `S_flat` -- so this only ever affected callers, silently.
    """
    def mkernel(x1, x2, hps):
        d = get_distance_matrix(x1, x2)
        return hps[0] * matern_kernel_diff1(d, hps[1])

    # deliberately Npts != No so a swapped axis cannot pass by coincidence
    n_data, No, n_pred = 12, 3, 4
    xd = np.random.rand(n_data, input_dim)
    yd = np.column_stack([np.sin(np.linalg.norm(xd, axis=1)),
                          np.cos(np.linalg.norm(xd, axis=1)),
                          np.linalg.norm(xd, axis=1)])
    gp = fvGP(xd, yd, init_hyperparameters=np.array([1., 1.]), kernel_function=mkernel)

    x_pred = np.random.rand(n_pred, input_dim)
    x_out = np.arange(No)
    res = gp.posterior_covariance(x_pred, x_out=x_out)
    S, S_flat = res["S"], res["S_flat"]
    v = gp.posterior_covariance(x_pred, x_out=x_out, variance_only=True)["v(x)"]

    assert S.shape == (n_pred, n_pred, No, No)
    assert S_flat.shape == (n_pred * No, n_pred * No)

    # every entry must agree with the flat matrix under the task-major index
    for i in range(n_pred):
        for j in range(n_pred):
            for t in range(No):
                for u in range(No):
                    assert np.isclose(S[i, j, t, u], S_flat[i + t * n_pred, j + u * n_pred])

    # the point/task diagonal is the variance returned by variance_only=True
    assert np.allclose(np.einsum('iijj->ij', S), v)
    # and S is symmetric under swapping both index pairs together
    assert np.allclose(S, S.transpose(1, 0, 3, 2))


# =========================================================================
# Bayesian optimization of the hyperparameters (method='bo')
# =========================================================================
def test_bo_transform_log_and_linear():
    """The search transform must be log only where the bounds allow it.

    Length scales and variances are positive and act multiplicatively, so log space
    is the right place to search. A prior-mean coefficient is free to be negative,
    so a blanket log would be invalid -- the decision is per dimension.
    """
    from fvgp.gp_bo import _LogAffineTransform
    tf = _LogAffineTransform(np.array([[1e-3, 1e3], [-5., 5.], [0.1, 10.], [0., 2.]]))
    # positive-bounded dims are log-transformed; sign-spanning and zero-touching are not
    assert list(tf.log_mask) == [True, False, True, False]
    theta = np.array([[1.0, -2.0, 3.0, 0.5]])
    assert np.allclose(tf.from_unit(tf.to_unit(theta)), theta)
    # bounds map to the corners of the unit cube
    assert np.allclose(tf.to_unit(np.array([[1e-3, -5., 0.1, 0.]])), 0.0)
    assert np.allclose(tf.to_unit(np.array([[1e3, 5., 10., 2.]])), 1.0)


def test_bo_recovers_noisy_optimum_and_reports_sensitivity():
    """BO must find a known optimum through observation noise, and the run must hand
    back a curvature-based sensitivity ranking and an approximate theta-posterior."""
    from fvgp.gp_bo import bayesian_optimize
    rng = np.random.default_rng(0)
    # dim 0 is 100x steeper than dim 1 in log-space; optimum at (1, 10)
    def objective(t):
        z = np.log(t) - np.log(np.array([1., 10.]))
        return float(10.0 * z[0] ** 2 + 0.1 * z[1] ** 2 + 0.02 * rng.standard_normal())

    bounds = np.array([[1e-2, 1e2], [1e-1, 1e3]])
    theta, info = bayesian_optimize(objective, bounds, np.array([50., 0.5]),
                                    max_iter=40, bo_args={"seed": 1, "noise_variance": 4e-4})
    # recovered the optimum despite the noise
    assert np.linalg.norm(np.log(theta) - np.log(np.array([1., 10.]))) < 0.5
    # the budget is a cap; the run may converge before reaching it
    assert 0 < info["n_evaluations"] <= 40
    # curvature ranks dim 0 above dim 1, matching the true 100:1 ratio
    s = info["sensitivity"]
    assert s[0] > s[1]
    assert 10.0 < s[0] / s[1] < 1000.0
    # the Laplace posterior is wider in the flat direction
    cov = info["posterior covariance"]
    assert cov is not None and cov.shape == (2, 2)
    assert np.sqrt(cov[1, 1]) > np.sqrt(cov[0, 0])


def test_bo_respects_evaluation_budget_and_does_not_recurse():
    """`max_iter` is a budget in expensive likelihood evaluations, and the inner
    surrogate must never itself train with method='bo' (no infinite regress)."""
    from fvgp.gp_bo import bayesian_optimize
    import fvgp.gp_bo as gp_bo

    calls = {"objective": 0, "methods": []}

    def objective(t):
        calls["objective"] += 1
        return float(np.sum((np.log(t) - 1.0) ** 2))

    real_fit = gp_bo._fit_surrogate

    def spy_fit(u_data, y_data, v_data, dim, train_max_iter):
        gp = real_fit(u_data, y_data, v_data, dim, train_max_iter)
        calls["methods"].append("local")
        return gp

    gp_bo._fit_surrogate = spy_fit
    try:
        bounds = np.array([[1e-2, 1e2]] * 3)
        theta, info = bayesian_optimize(objective, bounds, np.array([1., 1., 1.]),
                                        max_iter=18, bo_args={"seed": 0, "patience": 0})
    finally:
        gp_bo._fit_surrogate = real_fit

    assert calls["objective"] == 18, calls["objective"]
    assert info["n_evaluations"] == 18
    assert len(calls["methods"]) > 0          # the surrogate really was fit
    assert theta.shape == (3,)
    assert np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1])


def test_bo_training_through_gp():
    """method='bo' end to end: the GP is updated and bo_info is populated."""
    def mkernel(x1, x2, hps):
        return hps[0] * matern_kernel_diff1(get_distance_matrix(x1, x2), hps[1])

    np.random.seed(2)
    xd = np.random.rand(30, 2)
    yd = np.sin(3 * xd[:, 0]) + np.cos(3 * xd[:, 1])
    gp = GP(xd, yd, init_hyperparameters=np.array([1., 1.]), kernel_function=mkernel)
    bounds = np.array([[0.01, 10.], [0.01, 10.]])
    hps = gp.train(hyperparameter_bounds=bounds, method="bo", max_iter=20,
                   bo_args={"seed": 3, "patience": 0})

    assert hps.shape == (2,)
    assert np.allclose(gp.hyperparameters, hps)
    info = gp.bo_info
    assert info is not None
    assert info["n_evaluations"] == 20
    assert info["trace x"].shape == (20, 2)
    assert np.all(info["log-transformed dimensions"])       # both bounds are positive
    # BO improved on the starting point
    nll = gp.marginal_likelihood.neg_log_likelihood
    assert nll(hps) < nll(np.array([1., 1.]))


def test_bo_uses_known_observation_noise():
    """A supplied per-point noise variance must be accepted and used."""
    from fvgp.gp_bo import bayesian_optimize
    rng = np.random.default_rng(3)
    sigma2 = 0.01

    def objective(t):
        return float(np.sum((np.log(t)) ** 2) + np.sqrt(sigma2) * rng.standard_normal())

    bounds = np.array([[1e-2, 1e2], [1e-2, 1e2]])
    # callable form (the SLQ/Hutchinson case: variance known per point)
    theta, info = bayesian_optimize(objective, bounds, np.array([10., 10.]), max_iter=20,
                                    bo_args={"seed": 0, "noise_function": lambda h: sigma2})
    assert info["n_evaluations"] == 20
    assert np.all(np.isfinite(theta))
    # scalar form
    theta2, info2 = bayesian_optimize(objective, bounds, np.array([10., 10.]), max_iter=20,
                                      bo_args={"seed": 0, "noise_variance": sigma2})
    assert np.all(np.isfinite(theta2))


def test_bo_early_stopping_on_ei_tolerance():
    """A large ei_tolerance must terminate the run before the budget is spent."""
    from fvgp.gp_bo import bayesian_optimize

    def objective(t):
        return float(np.sum((np.log(t) - 1.0) ** 2))

    bounds = np.array([[1e-2, 1e2]] * 2)
    theta, info = bayesian_optimize(objective, bounds, np.array([1., 1.]), max_iter=60,
                                    bo_args={"seed": 0, "ei_tolerance": 1e9})
    assert info["n_evaluations"] < 60


def test_bo_warns_when_it_returns_the_warm_start():
    """Nothing beating the starting point is a legitimate answer, but it must be said.

    It is otherwise indistinguishable from a run that silently did nothing, which is
    exactly the confusion this warning exists to prevent.
    """
    from fvgp.gp_bo import bayesian_optimize

    init = np.array([1., 1.])

    # minimum sits exactly on the warm start, so no other point can beat it
    def objective(t):
        return float(np.sum((np.asarray(t) - init) ** 2))

    bounds = np.array([[1e-2, 1e2]] * 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        theta, info = bayesian_optimize(objective, bounds, init, max_iter=20,
                                        bo_args={"seed": 0})
    assert np.allclose(theta, init)
    msgs = [str(w.message) for w in caught if "started from" in str(w.message)]
    assert len(msgs) == 1
    assert "method='local'" in msgs[0] and "method='mcmc'" in msgs[0]
    assert f"{info['n_evaluations']} evaluated points" in msgs[0]
    assert info["stopping reason"] in msgs[0]

    # a run that genuinely improves must stay silent
    def improving(t):
        return float(np.sum((np.log(np.asarray(t)) - 1.0) ** 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        theta2, _ = bayesian_optimize(improving, bounds, init, max_iter=20,
                                      bo_args={"seed": 0})
    assert not np.allclose(theta2, init)
    assert not any("started from" in str(w.message) for w in caught)


def test_train_async_bo(client):
    """Async BO training: submit, poll, apply, stop."""
    my_gp = GP(x_data, y_data, init_hyperparameters=np.array([1., 1., 1., 1., 1., 1.]),
               noise_variances=np.zeros(y_data.shape) + 0.01)
    bounds = np.array([[0.01, 10.]] * 6)
    opt_obj = my_gp.train(hyperparameter_bounds=bounds, max_iter=100000,
                          dask_client=client, method="bo", asynchronous=True,
                          bo_args={"seed": 0})
    time.sleep(5)
    state = opt_obj.get_latest()
    assert "status" in state and state["status"] in ("queued", "running", "finished")
    my_gp.update_hyperparameters(opt_obj)
    assert my_gp.hyperparameters.shape == (6,)
    my_gp.stop_training(opt_obj)
    # stop() must actually halt the search rather than let the budget run out. An
    # evaluation already in flight still finishes, so wait for the count to settle
    # rather than assuming the very next poll is final.
    settled, previous = False, -1
    for _ in range(15):
        time.sleep(2)
        current = opt_obj.get_latest().get("n_evaluations", 0)
        if current == previous:
            settled = True
            break
        previous = current
    assert settled, "evaluation count never stopped increasing after stop()"
    assert previous < 100000


def _bo_pickle_kernel(x1, x2, hps):
    # module level so the GP itself pickles by reference
    return hps[0] * matern_kernel_diff1(get_distance_matrix(x1, x2), hps[1])


def test_bo_trained_gp_pickles():
    """A GP that has been trained with method='bo' must still pickle.

    `bo_info` holds the fitted surrogate, whose prior mean is a closure over the
    observed data and therefore unpicklable; the surrogate is dropped on pickling
    while the diagnostics arrays are kept.
    """
    import pickle

    np.random.seed(2)
    xd = np.random.rand(25, 2)
    yd = np.sin(3 * xd[:, 0]) + np.cos(3 * xd[:, 1])
    gp = GP(xd, yd, init_hyperparameters=np.array([1., 1.]), kernel_function=_bo_pickle_kernel)
    gp.train(hyperparameter_bounds=np.array([[0.01, 10.], [0.01, 10.]]),
             method="bo", max_iter=18, bo_args={"seed": 1})
    assert gp.bo_info["surrogate"] is not None

    gp2 = pickle.loads(pickle.dumps(gp))
    assert "surrogate" not in gp2.bo_info
    assert np.allclose(gp2.bo_info["trace x"], gp.bo_info["trace x"])
    assert gp2.bo_info["sensitivity"] is not None
    assert np.allclose(gp2.hyperparameters, gp.hyperparameters)


def test_gp2Scale_bo_is_allowed_but_never_asynchronous(client):
    """gp2Scale may train with method='bo' -- its stochastic log-determinant and
    truncated solve are exactly the noisy, gradient-free regime BO is for -- but it
    must never do so asynchronously: gp2Scale already owns the Dask client for the
    covariance, so an optimizer actor would contend with the linear algebra.
    """
    np.random.seed(0)
    xd = np.random.rand(120, 1)
    yd = np.sin(np.linalg.norm(xd, axis=1) * 5.0)
    bounds = np.array([[0.1, 10.], [0.001, 0.02]])
    init = np.random.uniform(size=2, low=bounds[:, 0], high=bounds[:, 1])
    gp = GP(xd, yd, init, gp2Scale=True, gp2Scale_batch_size=100,
            dask_client=client, linalg_mode="sparseLU")
    assert gp.gp2Scale

    # asking for asynchronous BO must warn and fall back to a synchronous run,
    # returning hyperparameters rather than an optimizer proxy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gp.train(hyperparameter_bounds=bounds, method="bo", max_iter=8,
                          bo_args={"seed": 0}, asynchronous=True, dask_client=client)
    assert isinstance(result, np.ndarray) and result.shape == (2,)
    assert not hasattr(result, "get_latest")
    msgs = [str(w.message) for w in caught]
    assert any("does not allow asynchronous training" in m for m in msgs), msgs
    # and the method was NOT silently switched away from bo
    assert not any("Method switched to MCMC" in m for m in msgs), msgs
    assert gp.bo_info is not None and gp.bo_info["n_evaluations"] == 8


def test_random_logdet_reports_its_own_variance():
    """The SLQ estimator must report its precision through `info_out`, and the probe
    count must be controllable -- it is the fidelity dial: noise ~ 1/sqrt(t), cost ~ t."""
    from fvgp.gp_lin_alg import calculate_random_logdet
    pytest.importorskip("imate")

    n = 400
    A = sparse.random(n, n, density=0.02, random_state=0)
    A = ((A + A.T) * 0.5 + sparse.identity(n) * 5.0).tocsr()

    sigmas = {}
    for t in (10, 80):
        info = {}
        args = {"random_logdet_min_num_samples": t, "random_logdet_max_num_samples": t,
                "random_logdet_error_rtol": 1e-12}
        ld = calculate_random_logdet(A, "cpu", args=args, info_out=info)
        assert np.isscalar(ld) and np.isfinite(ld)
        assert info["num_samples_used"] == t
        assert info["variance"] is not None and info["variance"] > 0.0
        sigmas[t] = np.sqrt(info["variance"])
    # more probes must mean a tighter estimate
    assert sigmas[80] < sigmas[10]
    # the return type is unchanged when no sink is passed (existing callers keep working)
    assert np.isscalar(calculate_random_logdet(A, "cpu", args={}))


def test_log_likelihood_variance_exact_vs_stochastic(client):
    """The likelihood must report noise only when it actually has any."""
    np.random.seed(0)
    n = 250
    xd = np.random.rand(n, 1)
    yd = np.sin(np.linalg.norm(xd, axis=1) * 5.0)
    init = np.array([1., 0.01])

    exact = GP(xd, yd, init, linalg_mode="Chol")
    exact.log_likelihood(hyperparameters=init)
    assert exact.marginal_likelihood.log_likelihood_variance() is None

    stoch = GP(xd, yd, init, gp2Scale=True, gp2Scale_batch_size=100,
               dask_client=client, linalg_mode="sparseCG")
    vals = [stoch.log_likelihood(hyperparameters=init) for _ in range(5)]
    var = stoch.marginal_likelihood.log_likelihood_variance()
    assert var is not None and var > 0.0
    # repeated evaluation at identical hyperparameters really does disagree
    assert np.std(vals, ddof=1) > 0.0
    # the reported sigma is the right order of magnitude (it is a lower bound: the
    # truncated-CG solve contributes noise this does not capture)
    assert np.sqrt(var) < 10.0 * np.std(vals, ddof=1) + 1.0


def test_bo_learns_noise_when_objective_cannot_report_it():
    """With no noise information, the surrogate learns a homoscedastic level. A
    deterministic objective drives it to the nugget so the surrogate interpolates."""
    from fvgp.gp_bo import bayesian_optimize
    bounds = np.array([[1e-2, 1e2], [1e-1, 1e3]])
    target = np.log(np.array([1., 10.]))

    def deterministic(t):
        z = np.log(t) - target
        return float(z @ z)

    theta, info = bayesian_optimize(deterministic, bounds, np.array([50., 0.5]),
                                    max_iter=30, bo_args={"seed": 1})
    assert info["noise was learned"]
    assert info["observation noise variance"] < 1e-6      # collapsed to the nugget
    assert np.linalg.norm(np.log(theta) - target) < 0.2
    # never worse than the warm start it was given
    assert deterministic(theta) <= deterministic(np.array([50., 0.5]))

    rng = np.random.default_rng(0)
    sd = 0.5

    def noisy(t):
        z = np.log(t) - target
        return float(z @ z + sd * rng.standard_normal())

    theta2, info2 = bayesian_optimize(noisy, bounds, np.array([50., 0.5]),
                                      max_iter=40, bo_args={"seed": 1})
    assert info2["noise was learned"]
    # a genuinely noisy objective must be recognized as such, within an order of magnitude
    assert 1e-3 < info2["observation noise variance"] < 10.0 * sd ** 2
    assert np.linalg.norm(np.log(theta2) - target) < 1.0


def test_bo_uses_estimator_noise_automatically(client):
    """In a sparse mode, method='bo' must pick up the likelihood's own reported noise
    rather than learning one; in an exact mode it must fall back to learning."""
    np.random.seed(0)
    n = 250
    xd = np.random.rand(n, 1)
    yd = np.sin(np.linalg.norm(xd, axis=1) * 5.0)
    bounds = np.array([[0.1, 10.], [0.001, 0.02]])
    init = np.array([1., 0.01])

    import gc

    def _release(gp_obj):
        # only one live gp2Scale GP may share a dask client; flush before the next
        del gp_obj
        gc.collect()
        client.run(lambda: None)

    stoch = GP(xd, yd, init, gp2Scale=True, gp2Scale_batch_size=100,
               dask_client=client, linalg_mode="sparseCG")
    stoch.train(hyperparameter_bounds=bounds, method="bo", max_iter=10, bo_args={"seed": 0})
    info = stoch.bo_info
    assert info["noise was learned"] is False
    assert info["observation noise variance"] > 0.0
    _release(stoch)
    stoch = None

    exact = GP(xd, yd, init, linalg_mode="Chol")
    exact.train(hyperparameter_bounds=bounds, method="bo", max_iter=10, bo_args={"seed": 0})
    assert exact.bo_info["noise was learned"] is True

    # an explicit setting still wins over the automatic one
    stoch2 = GP(xd, yd, init, gp2Scale=True, gp2Scale_batch_size=100,
                dask_client=client, linalg_mode="sparseCG")
    stoch2.train(hyperparameter_bounds=bounds, method="bo", max_iter=10,
                 bo_args={"seed": 0, "noise_variance": 0.25})
    assert np.isclose(stoch2.bo_info["observation noise variance"], 0.25)
    _release(stoch2)


def test_bo_never_recommends_worse_than_best_observed_on_deterministic_objective():
    """Regression: a learned noise level conflates estimator noise with surrogate
    misfit. On a hard deterministic surface the surrogate explained its own misfit as
    noise (measured: variance 90 against an IQR of 69), which widened the
    recommendation tolerance enough to return a point far worse than one already
    evaluated. The recommendation must never regress on a deterministic objective.
    """
    from fvgp.gp_bo import bayesian_optimize

    # a surface with the awkward feature of a real marginal likelihood: a sharp optimum
    # plus regions orders of magnitude worse, so the observed range is enormous
    def objective(t):
        z = np.log(t) - np.log(np.array([1.0, 5.0]))
        return float(z @ z + 500.0 * np.exp(3.0 * z[0]))

    bounds = np.array([[1e-2, 1e2], [1e-2, 1e2]])
    for seed in (0, 1, 2, 3):
        warm = np.array([1.0, 5.0])          # start exactly at the optimum
        theta, info = bayesian_optimize(objective, bounds, warm, max_iter=25,
                                        bo_args={"seed": seed})
        best_observed = float(np.min(info["trace f(x)"]))
        returned = objective(theta)
        # the returned point is the best observed, never something the search rejected
        assert returned <= best_observed + 1e-9, (seed, returned, best_observed)
        # and never worse than the warm start it was handed
        assert returned <= objective(warm) + 1e-9, (seed, returned, objective(warm))


def test_sequential_linalg_state_overrides_and_restores():
    """The context manager must disable the state, warn, and put it back.

    Both assume successive evaluations are close, which holds for mcmc/local but not
    for BO, where a space-filling design and acquisition jumps put consecutive points
    far apart. A truncated solve seeded with stale state then leaves an error that
    depends on which hyperparameters ran before it, making the objective
    order-dependent -- a bias, which BO's noise model cannot absorb.
    """
    from fvgp.gp_kv import sequential_linalg_state, _SEQUENTIAL_STATE_DEFAULTS

    # settings a user might reasonably have tuned for MCMC
    args = {"sparse_krylov_warm_start": True,
            "sparse_preconditioner_refresh_interval": 25,
            "sparse_cg_tol": 1e-6}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sequential_linalg_state(args, "bo"):
            assert args["sparse_krylov_warm_start"] is False
            assert args["sparse_preconditioner_refresh_interval"] == 1
            assert args["sparse_cg_tol"] == 1e-6          # unrelated keys untouched
    # the override is announced rather than silent
    assert any("disables sequential linear-algebra state" in str(w.message) for w in caught)
    # and everything is restored afterwards
    assert args["sparse_krylov_warm_start"] is True
    assert args["sparse_preconditioner_refresh_interval"] == 25
    assert args["sparse_cg_tol"] == 1e-6

    # a user who never set them gets no warning and no leftover keys
    clean = {"sparse_cg_tol": 1e-6}
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        with sequential_linalg_state(clean, "bo"):
            assert clean["sparse_krylov_warm_start"] is False
    assert not any("disables sequential" in str(w.message) for w in caught2)
    assert "sparse_krylov_warm_start" not in clean
    assert "sparse_preconditioner_refresh_interval" not in clean

    # restored even if the body raises
    args2 = {"sparse_krylov_warm_start": True}
    with pytest.raises(ValueError):
        with sequential_linalg_state(args2, "bo"):
            raise ValueError("boom")
    assert args2["sparse_krylov_warm_start"] is True

    # the safe values are the library defaults, so a default setup is already correct
    assert _SEQUENTIAL_STATE_DEFAULTS["sparse_krylov_warm_start"] is False
    assert _SEQUENTIAL_STATE_DEFAULTS["sparse_preconditioner_refresh_interval"] == 1


def test_bo_training_leaves_user_linalg_args_restored(client):
    """A real BO run must restore the user's settings when it finishes."""
    np.random.seed(0)
    xd = np.random.rand(150, 1)
    yd = np.sin(np.linalg.norm(xd, axis=1) * 5.0)
    bounds = np.array([[0.1, 10.], [0.001, 0.02]])
    init = np.array([1., 0.01])
    gp = GP(xd, yd, init, gp2Scale=True, gp2Scale_batch_size=75,
            dask_client=client, linalg_mode="sparseCG",
            args={"sparse_krylov_warm_start": True,
                  "sparse_preconditioner_refresh_interval": 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.train(hyperparameter_bounds=bounds, method="bo", max_iter=8, bo_args={"seed": 0})
    assert any("disables sequential linear-algebra state" in str(w.message) for w in caught)
    # the run is over; the user's configuration is theirs again
    assert gp.args["sparse_krylov_warm_start"] is True
    assert gp.args["sparse_preconditioner_refresh_interval"] == 10


def test_sequential_linalg_state_only_for_mcmc():
    """Warm starts and preconditioner reuse are permitted for mcmc and nothing else."""
    from fvgp.gp_kv import sequential_linalg_state

    for method in ("mcmc",):
        args = {"sparse_krylov_warm_start": True,
                "sparse_preconditioner_refresh_interval": 25}
        with sequential_linalg_state(args, method):
            assert args["sparse_krylov_warm_start"] is True
            assert args["sparse_preconditioner_refresh_interval"] == 25

    for method in ("bo", "global", "local", "hgdl", "adam"):
        args = {"sparse_krylov_warm_start": True,
                "sparse_preconditioner_refresh_interval": 25}
        with sequential_linalg_state(args, method):
            assert args["sparse_krylov_warm_start"] is False, method
            assert args["sparse_preconditioner_refresh_interval"] == 1, method
        assert args["sparse_krylov_warm_start"] is True, method
        assert args["sparse_preconditioner_refresh_interval"] == 25, method


def test_preconditioner_and_warm_start_reuse_follow_matrix_drift():
    """Cached state must be discarded when K+V has actually moved, not after k steps.

    A counter cannot distinguish k tiny MCMC steps from one jump across the domain;
    the drift of the matrix itself can.
    """
    np.random.seed(0)
    n = 200
    xd = np.random.rand(n, 2)
    yd = np.sin(np.linalg.norm(xd, axis=1))
    gp = GP(xd, yd, np.array([1., 1., 1.]))   # default kernel needs D+1 hyperparameters
    kv = gp.kv

    def build(scale):
        A = sparse.random(n, n, density=0.05, random_state=0)
        A = (A + A.T) * 0.5
        return (A * scale + sparse.identity(n) * (1.0 + scale)).tocsr()

    base = build(1.0)
    near = build(1.0005)      # an mcmc-sized step
    far = build(4.0)          # a bo/global-sized jump

    fp_base = kv.matrix_fingerprint(base)
    assert fp_base is not None
    # drift is ~0 against itself, small for a near step, large for a jump
    assert kv._fingerprint_drift(fp_base, fp_base) == 0.0
    drift_near = kv._fingerprint_drift(fp_base, kv.matrix_fingerprint(near))
    drift_far = kv._fingerprint_drift(fp_base, kv.matrix_fingerprint(far))
    threshold = kv._max_matrix_drift()
    assert drift_near < threshold < drift_far, (drift_near, threshold, drift_far)
    # a different shape is never comparable
    assert kv._fingerprint_drift(fp_base, kv.matrix_fingerprint(build(1.0)[:-1, :-1])) == np.inf

    # warm start survives a near step and is dropped after a jump
    x0 = np.ones((n, 1))
    kv.Warm_start_fingerprint = fp_base
    assert kv._validated_warm_start(near, x0) is x0
    assert kv._validated_warm_start(far, x0) is None
    # with nothing cached there is nothing to invalidate
    kv.Warm_start_fingerprint = None
    assert kv._validated_warm_start(far, x0) is x0
    # and a caller that passed no guess still gets none
    assert kv._validated_warm_start(near, None) is None

    # the preconditioner cache uses the same test
    kv.mode = "sparseCGpre"
    kv.Preconditioner_operator = object()
    kv.Preconditioner_KV_shape = base.shape
    kv.Preconditioner_signature = kv._preconditioner_signature()
    kv.Preconditioner_fingerprint = fp_base
    kv.Preconditioner_reuse_counter = 0
    assert kv._can_reuse_sparse_preconditioner(near) is True
    assert kv._can_reuse_sparse_preconditioner(far) is False
    # ... and reuse is no longer capped by a blind counter
    kv.Preconditioner_reuse_counter = 10_000
    assert kv._can_reuse_sparse_preconditioner(near) is True
    # unless the user explicitly asks for a cap
    kv.args["sparse_preconditioner_refresh_interval"] = 5
    kv.Preconditioner_signature = kv._preconditioner_signature()
    assert kv._can_reuse_sparse_preconditioner(near) is False
    del kv.args["sparse_preconditioner_refresh_interval"]


def test_bo_surrogate_uses_stored_inverse_for_cheap_variance():
    """The inner surrogate must store the inverse of its covariance.

    The acquisition asks for `variance_only=True` at hundreds of candidates per
    iteration, and fvGP can only take that shortcut when KVinv is available; in the
    default 'Chol' mode it builds the full (V x V) posterior covariance and discards
    everything but the diagonal. That one setting was worth 3.7x on a 40-evaluation
    run, so this pins it.
    """
    from fvgp.gp_bo import _fit_surrogate

    rng = np.random.default_rng(0)
    u = rng.random((24, 2))
    y = np.sum((u - 0.5) ** 2, axis=1)
    gp = _fit_surrogate(u, y, None, 2, 50)
    assert gp.kv.KVinv is not None, "surrogate must keep the inverse for variance_only"
    # and the cheap path really is taken: variance_only returns no full covariance
    res = gp.posterior_covariance(rng.random((64, 2)), variance_only=True)
    assert res["S"] is None
    assert res["v(x)"].shape == (64,)
    assert np.all(np.isfinite(res["v(x)"]))


def test_bo_analytic_gradients_match_central_differences():
    """Every analytic derivative in the BO surrogate must match finite differences.

    This is the test that makes the hand-derived gradients safe: a sign or index slip
    is invisible in an optimization result but obvious here.
    """
    from fvgp.gp_bo import (_fit_surrogate, _surrogate_kernel, _surrogate_kernel_dx,
                            _surrogate_kernel_grad, _nei_value_and_grad,
                            _posterior_mean_var_and_grad, _noisy_expected_improvement)
    rng = np.random.default_rng(0)

    # --- dk/dx and dk/dhyperparameters, against the kernel itself ---
    for dim in (1, 2, 4):
        n = 15
        xd = rng.random((n, dim))
        hps = np.concatenate([[0.7], rng.uniform(0.2, 1.5, dim)])
        q = rng.random(dim)
        h = 1e-7
        fd = np.zeros((dim, n))
        for i in range(dim):
            e = np.zeros(dim); e[i] = h
            fd[i] = ((_surrogate_kernel(xd, (q + e)[None, :], hps).reshape(-1) -
                      _surrogate_kernel(xd, (q - e)[None, :], hps).reshape(-1)) / (2 * h))
        dk = _surrogate_kernel_dx(q, xd, hps)
        assert np.max(np.abs(dk - fd)) / max(np.max(np.abs(fd)), 1e-12) < 1e-6, dim

        gh = _surrogate_kernel_grad(xd, xd, hps)
        assert gh.shape == (len(hps), n, n)
        for j in range(len(hps)):
            step = 1e-6 * max(abs(hps[j]), 1e-3)
            hp, hm = hps.copy(), hps.copy()
            hp[j] += step; hm[j] -= step
            fdh = (_surrogate_kernel(xd, xd, hp) - _surrogate_kernel(xd, xd, hm)) / (2 * step)
            assert np.max(np.abs(gh[j] - fdh)) / max(np.max(np.abs(fdh)), 1e-12) < 1e-6, (dim, j)

    # --- posterior mean/variance gradients and the acquisition gradient ---
    # declared noise keeps the posterior variance away from the cancellation floor,
    # where the variance is a difference of near-equal numbers and finite differences
    # of it are meaningless (the GP is interpolating and the acquisition is ~0 anyway)
    for dim, n in ((1, 8), (2, 20), (5, 40)):
        u = rng.random((n, dim))
        y = np.sum((u - 0.4) ** 2, axis=1) + 0.1 * np.sin(6 * u[:, 0])
        gp = _fit_surrogate(u, y, np.full(n, 1e-3), dim, 80)
        y_best = rng.standard_normal(32) * 0.1 + float(np.max(-y))
        for _ in range(10):
            q = rng.uniform(0.05, 0.95, dim)
            _, _, d_mean, d_var = _posterior_mean_var_and_grad(q, gp, dim)
            value, grad = _nei_value_and_grad(q, gp, y_best, dim)
            h = 1e-6
            fdm = np.zeros(dim); fdv = np.zeros(dim); fda = np.zeros(dim)
            for i in range(dim):
                e = np.zeros(dim); e[i] = h
                mp, vp, _, _ = _posterior_mean_var_and_grad(q + e, gp, dim)
                mm, vm, _, _ = _posterior_mean_var_and_grad(q - e, gp, dim)
                fdm[i] = (mp - mm) / (2 * h)
                fdv[i] = (vp - vm) / (2 * h)
                fda[i] = (_nei_value_and_grad(q + e, gp, y_best, dim)[0] -
                          _nei_value_and_grad(q - e, gp, y_best, dim)[0]) / (2 * h)
            rel = lambda a, b: np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-10)
            assert rel(d_mean, fdm) < 1e-5, (dim, "dmean")
            assert rel(d_var, fdv) < 1e-4, (dim, "dvar")
            assert rel(grad, fda) < 1e-5, (dim, "dNEI")

        # the single-point analytic value must agree with the vectorized acquisition
        # that the random pre-screen uses, or the optimizer would be maximizing a
        # slightly different function than it screened with
        pts = rng.uniform(0.05, 0.95, (12, dim))
        single = np.array([_nei_value_and_grad(p, gp, y_best, dim)[0] for p in pts])
        vectorized = _noisy_expected_improvement(pts, gp, y_best, None)
        assert np.allclose(single, vectorized, rtol=1e-10, atol=1e-14)


def test_bo_surrogate_training_uses_analytic_hyperparameter_gradients():
    """The surrogate must be given analytic dk/dhps, and fvGP's resulting NLML gradient
    must be more accurate than the finite-difference fallback it replaces."""
    from fvgp.gp_bo import _fit_surrogate, _surrogate_kernel, _surrogate_kernel_grad

    rng = np.random.default_rng(1)
    dim, n = 3, 25
    xd = rng.random((n, dim))
    y = np.sum((xd - 0.4) ** 2, axis=1) + 0.05 * rng.standard_normal(n)
    hps = np.concatenate([[0.5], np.full(dim, 0.4)])
    common = dict(init_hyperparameters=hps, kernel_function=_surrogate_kernel,
                  noise_variances=np.full(n, 1e-3))
    fd_gp = GP(xd, y, linalg_mode="CholInv", **common)
    an_gp = GP(xd, y, linalg_mode="CholInv",
               kernel_function_grad=_surrogate_kernel_grad, **common)

    reference = np.zeros(len(hps))
    for j in range(len(hps)):
        step = 1e-6 * max(abs(hps[j]), 1e-3)
        hp, hm = hps.copy(), hps.copy()
        hp[j] += step; hm[j] -= step
        reference[j] = ((fd_gp.marginal_likelihood.neg_log_likelihood(hp) -
                         fd_gp.marginal_likelihood.neg_log_likelihood(hm)) / (2 * step))
    err = lambda g: np.max(np.abs(g - reference)) / np.max(np.abs(reference))
    e_fd = err(fd_gp.marginal_likelihood.neg_log_likelihood_gradient(hyperparameters=hps))
    e_an = err(an_gp.marginal_likelihood.neg_log_likelihood_gradient(hyperparameters=hps))
    assert e_an < 1e-7
    assert e_an < e_fd          # strictly better than fvGP's finite-difference fallback

    # and the surrogate really is built with them
    gp = _fit_surrogate(xd, y, None, dim, 50)
    assert gp.prior._dk_dh is _surrogate_kernel_grad


def test_bo_surrogate_stays_numerically_stable():
    """A BO run must not provoke fvGP's "Negative variances encountered" warning.

    Once the search converges it proposes points a whisker apart, and near-duplicate
    rows make K numerically singular, so the posterior variance -- a difference of
    nearly equal numbers -- tips below zero. The nugget has to be large enough to
    absorb that. The declared-noise path is included because it previously applied no
    floor at all and was therefore the worst case rather than the safest.
    """
    from fvgp.gp_bo import bayesian_optimize

    target = np.log(np.array([1., 10.]))
    bounds = np.array([[1e-2, 1e2], [1e-1, 1e3]])
    warm = np.array([50., 0.5])

    def smooth(t):
        z = np.log(t) - target
        return float(z @ z)

    def rugged(t):
        z = np.log(t) - np.log(np.array([1.0, 5.0]))
        return float(z @ z + 500.0 * np.exp(3.0 * z[0]))

    cases = [("learned noise", smooth, None),
             ("declared tiny noise", smooth, {"noise_variance": 1e-10}),
             ("rugged surface", rugged, {"noise_variance": 1e-10})]
    for label, objective, extra in cases:
        for seed in range(4):
            args = dict(extra or {})
            args["seed"] = seed
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bayesian_optimize(objective, bounds, warm, max_iter=30, bo_args=args)
            negative = [w for w in caught if "Negative variances" in str(w.message)]
            assert not negative, (label, seed, len(negative))

    # the design really does contain near-duplicates -- this is what the nugget is for,
    # so if it ever stops being true the test above has lost its teeth
    _, info = bayesian_optimize(smooth, bounds, warm, max_iter=40, bo_args={"seed": 0})
    u = info["trace u"]
    separations = np.linalg.norm(u[:, None, :] - u[None, :, :], axis=2)
    np.fill_diagonal(separations, np.inf)
    assert separations.min() < 1e-5


def test_bo_warns_when_it_is_the_wrong_tool():
    """method='bo' must say so when it is being misapplied, since it never raises.

    The default kernel gives a handful of hyperparameters, which is BO's home ground. A
    user-supplied kernel, prior mean or noise function can produce a far longer vector --
    a deep kernel runs to hundreds -- where the method degrades badly and silently.
    """
    bounds = lambda n: np.array([[0.01, 10.]] * n)

    def messages(n_hps, max_iter, bo_args=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GP._warn_about_bo_suitability(bounds(n_hps), max_iter, bo_args)
        return [str(w.message) for w in caught]

    # the ordinary case must stay quiet, or the warnings become noise people filter out
    assert messages(3, 40) == []
    assert messages(2, 60) == []

    # the quiet failure: the design eats the whole budget and no surrogate is ever fitted
    quiet = messages(40, 60)
    assert any("no Bayesian optimization is" in m for m in quiet)
    # ...and it must not fire once the budget is genuinely large enough
    assert not any("no Bayesian optimization is" in m for m in messages(40, 500))

    # dimensionality, with a softer note at the edge
    assert any("roughly 20" in m for m in messages(40, 500))
    assert any("edge of what it does well" in m for m in messages(12, 60))
    assert not any("roughly 20" in m for m in messages(12, 60))

    # an explicit n_init is respected when judging whether the design fits
    assert not any("no Bayesian optimization is" in m
                   for m in messages(40, 60, {"n_init": 10}))


def test_bo_initial_design_size_is_shared_with_the_optimizer():
    """The warning and the optimizer must use one rule, or the warning goes stale."""
    from fvgp.gp_bo import default_initial_design_size, bayesian_optimize

    for dim, budget in ((2, 40), (5, 60), (20, 60), (40, 60)):
        n = default_initial_design_size(dim, budget)
        assert 2 <= n <= budget
    # the optimizer really uses it: with a budget equal to the design size, every
    # evaluation is a design point and nothing is left for the search
    dim = 3
    budget = default_initial_design_size(dim, 8)
    calls = {"n": 0}

    def objective(t):
        calls["n"] += 1
        return float(np.sum(np.log(t) ** 2))

    _, info = bayesian_optimize(objective, np.array([[1e-2, 1e2]] * dim),
                                np.ones(dim), max_iter=budget, bo_args={"seed": 0})
    assert calls["n"] == budget
    assert info["n_evaluations"] == budget
    assert len(info["ei history"]) == 0        # no acquisition step ever ran


def test_bo_stops_itself_when_the_answer_stops_moving():
    """`max_iter` is a cap, not a target. Left at GP.train's inherited 10000, a run must
    still terminate in a few dozen evaluations once the answer has settled."""
    from fvgp.gp_bo import bayesian_optimize

    target = np.log(np.array([1., 10.]))
    bounds = np.array([[1e-2, 1e2], [1e-1, 1e3]])
    warm = np.array([50., 0.5])

    def smooth(t):
        z = np.log(t) - target
        return float(z @ z)

    for seed in range(3):
        theta, info = bayesian_optimize(smooth, bounds, warm, max_iter=10000,
                                        bo_args={"seed": seed})
        assert info["stopping reason"] == "converged"
        assert info["n_evaluations"] < 100, info["n_evaluations"]
        # stopping early must not mean stopping short of the answer
        assert np.linalg.norm(np.log(theta) - target) < 0.1

    # patience=0 disables it, and then the cap is what binds
    _, info = bayesian_optimize(smooth, bounds, warm, max_iter=30,
                                bo_args={"seed": 0, "patience": 0})
    assert info["stopping reason"] == "budget"
    assert info["n_evaluations"] == 30

    # a tiny patience stops sooner than the default
    _, quick = bayesian_optimize(smooth, bounds, warm, max_iter=10000,
                                 bo_args={"seed": 0, "patience": 2})
    _, normal = bayesian_optimize(smooth, bounds, warm, max_iter=10000,
                                  bo_args={"seed": 0})
    assert quick["n_evaluations"] < normal["n_evaluations"]

    # the absolute EI criterion still works and is reported distinctly
    _, ei_stopped = bayesian_optimize(smooth, bounds, warm, max_iter=10000,
                                      bo_args={"seed": 0, "ei_tolerance": 1e9})
    assert ei_stopped["stopping reason"] == "ei_tolerance"

    # the cap is still honored when convergence never triggers
    _, capped = bayesian_optimize(smooth, bounds, warm, max_iter=14,
                                  bo_args={"seed": 0, "patience": 1000})
    assert capped["n_evaluations"] == 14
    assert capped["stopping reason"] == "budget"


def test_bo_convergence_test_is_scale_free():
    """Improvement is judged against the observed spread, so an objective that is large,
    negative or near zero converges the same way -- a marginal likelihood is all three
    depending on the dataset."""
    from fvgp.gp_bo import bayesian_optimize

    target = np.log(np.array([1., 10.]))
    bounds = np.array([[1e-2, 1e2], [1e-1, 1e3]])
    warm = np.array([50., 0.5])

    def base(t):
        z = np.log(t) - target
        return float(z @ z)

    # the criterion itself must fire whatever the offset or magnitude, including
    # combinations far outside anything a likelihood produces
    for shift, scale in ((0.0, 1.0), (-1e6, 1.0), (0.0, 1e6), (1e6, 1e-6)):
        def objective(t, shift=shift, scale=scale):
            return shift + scale * base(t)
        _, info = bayesian_optimize(objective, bounds, warm, max_iter=10000,
                                    bo_args={"seed": 0})
        assert info["stopping reason"] == "converged", (shift, scale)

    # and over the range a marginal likelihood actually spans, it should converge after
    # the same amount of work. Beyond that the *surrogate* stops being scale-free -- its
    # nugget and variance bounds carry absolute floors for conditioning -- so the count
    # can drift even though the criterion does not.
    counts = []
    for shift, scale in ((0.0, 1.0), (-1e3, 1.0), (1e3, 1.0), (0.0, 1e3), (0.0, 1e-3)):
        def objective(t, shift=shift, scale=scale):
            return shift + scale * base(t)
        _, info = bayesian_optimize(objective, bounds, warm, max_iter=10000,
                                    bo_args={"seed": 0})
        counts.append(info["n_evaluations"])
    # within ~30%: the criterion is exactly scale-free, but the surrogate's nugget has
    # an absolute floor (max(1e-7*scale, 1e-12)) that starts to bind once the objective's
    # spread falls below ~1e-3, which shifts the count slightly. Convergence at a similar
    # cost is the claim, not at an identical one.
    assert max(counts) <= 1.3 * min(counts), counts


def test_bo_log_scale_override():
    """The log transform is a guess from the bounds and must be overridable.

    Positivity is a proxy for being scale-like. A hyperparameter that is positive but
    enters the likelihood additively -- a position in a non-stationary or Gibbs kernel --
    is hurt by the log, and the wider its bounds the worse it gets.
    """
    from fvgp.gp_bo import bayesian_optimize, _LogAffineTransform

    bounds = np.array([[0.1, 100.], [0.1, 100.]])
    # automatic: both positive, so both logged
    assert list(_LogAffineTransform(bounds).log_mask) == [True, True]
    # explicit overrides, scalar and per-dimension
    assert list(_LogAffineTransform(bounds, log_scale=False).log_mask) == [False, False]
    assert list(_LogAffineTransform(bounds, log_scale=[False, True]).log_mask) == [False, True]
    # a log cannot be applied where the box touches zero; that is warned, not silent
    crossing = np.array([[-1.0, 1.0], [0.1, 10.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf = _LogAffineTransform(crossing, log_scale=True)
    assert list(tf.log_mask) == [False, True]
    assert any("not strictly positive" in str(w.message) for w in caught)
    # round-trip holds under every setting
    theta = np.array([[2.0, 40.0]])
    for setting in (None, True, False, [True, False]):
        tf = _LogAffineTransform(bounds, log_scale=setting)
        assert np.allclose(tf.from_unit(tf.to_unit(theta)), theta)

    # and it matters: an objective quadratic in the hyperparameters is solved exactly in
    # linear coordinates and only approximately through the log
    def positional(t):
        return float((t[0] - 5.0) ** 2 + (t[1] - 30.0) ** 2)

    warm = np.array([50., 80.])
    errors = {}
    for label, extra in (("auto", {}), ("linear", {"log_scale": False})):
        found = []
        for seed in range(4):
            args = {"seed": seed, "patience": 0}
            args.update(extra)
            theta, _ = bayesian_optimize(positional, bounds, warm, max_iter=50, bo_args=args)
            found.append(np.linalg.norm(theta - np.array([5., 30.])))
        errors[label] = float(np.median(found))
    assert errors["linear"] < errors["auto"], errors


def test_train_info_prints_progress_for_every_method(capsys):
    """`info=True` must actually surface something for each method.

    It used to be a no-op everywhere except mcmc: the other methods reported through
    `logger.debug`, and fvgp/__init__.py calls `logger.disable('fvgp')`, so nothing
    reached the user. `info` now prints.
    """
    np.random.seed(0)
    xd = np.random.rand(25, 2)
    yd = np.sin(np.linalg.norm(xd, axis=1))
    bounds = np.array([[0.01, 10.]] * 3)
    hps = np.array([1., 1., 1.])

    cases = [
        ("bo", dict(max_iter=12, bo_args={"seed": 0, "patience": 0}), "bo evaluation"),
        ("local", dict(max_iter=5), "local iteration"),
        ("adam", dict(max_iter=25), "adam iteration"),
        ("global", dict(max_iter=2, pop_size=4), "differential_evolution step"),
    ]
    for method, kwargs, marker in cases:
        gp = GP(xd, yd, hps)
        gp.train(hyperparameter_bounds=bounds, method=method, info=True, **kwargs)
        printed = capsys.readouterr().out
        assert marker in printed, (method, printed[:200])

        # and it must stay quiet when not asked
        gp2 = GP(xd, yd, hps)
        gp2.train(hyperparameter_bounds=bounds, method=method, info=False, **kwargs)
        quiet = capsys.readouterr().out
        assert marker not in quiet, (method, quiet[:200])


def test_bo_info_reports_the_value_actually_reached(capsys):
    """The per-evaluation line must report the value at that evaluation, not the
    running best -- it is printed after the objective call for exactly that reason."""
    from fvgp.gp_bo import bayesian_optimize, default_initial_design_size

    target = np.log(np.array([1., 10.]))
    n_design = default_initial_design_size(2, 14)

    def smooth(t):
        z = np.log(t) - target
        return float(z @ z)

    _, info = bayesian_optimize(smooth, np.array([[1e-2, 1e2], [1e-1, 1e3]]),
                                np.array([50., 0.5]), max_iter=14,
                                bo_args={"seed": 0, "patience": 0}, info=True)
    printed = capsys.readouterr().out
    assert "space-filling design" in printed
    assert "design complete" in printed
    assert printed.count("bo evaluation") == info["n_evaluations"] - n_design
    assert "finished after" in printed and "(budget)" in printed

    # the reported f(x) values are the trace, not a monotone running best
    reported = [float(line.split("f(x)= ")[1].split(",")[0])
                for line in printed.splitlines() if "bo evaluation" in line]
    assert np.allclose(sorted(reported), sorted(info["trace f(x)"][n_design:]))
    assert reported != sorted(reported, reverse=True)   # not a monotone best-so-far


###########################################################################
#################### gp2Scale distributed covariance ######################
###########################################################################
def _reference_wendland(x1, x2, hps):
    """Dense reference for the assembler tests."""
    return wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)


def _wendland_with_args(x1, x2, hps, args):
    """Four-argument kernel. The historical gp2Scale workers could not call one."""
    return args["scale"] * wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)


def test_distributed_covariance_matches_dense(client):
    """One primitive, four shapes of problem: the assembled sparse matrix must equal the
    dense kernel evaluation for both distributions, symmetric and rectangular."""
    from fvgp.gp2Scale_covariance import distributed_covariance

    rng = np.random.default_rng(42)
    x1 = rng.random((57, 2))
    x2 = rng.random((23, 2))
    hps = np.array([2.0, 0.4, 0.35])

    f1 = client.scatter(x1, broadcast=True, direct=True)
    f2 = client.scatter(x2, broadcast=True, direct=True)

    for distribution in ("blockwise", "rowwise"):
        K = distributed_covariance(client, _reference_wendland, hps,
                                   x1_future=f1, n1=len(x1), x2_future=f1, n2=len(x1),
                                   batch_size=10, symmetric=True, distribution=distribution)
        assert sparse.issparse(K)
        assert np.allclose(K.toarray(), _reference_wendland(x1, x1, hps))
        # the mirror must actually be a mirror, not a triangle
        assert np.allclose(K.toarray(), K.toarray().T)

        B = distributed_covariance(client, _reference_wendland, hps,
                                   x1_future=f1, n1=len(x1), x2_future=f2, n2=len(x2),
                                   batch_size=10, symmetric=False, distribution=distribution)
        assert B.shape == (len(x1), len(x2))
        assert np.allclose(B.toarray(), _reference_wendland(x1, x2, hps))

        # a single block (batch_size larger than the data) must behave identically
        K1 = distributed_covariance(client, _reference_wendland, hps,
                                    x1_future=f1, n1=len(x1), x2_future=f1, n2=len(x1),
                                    batch_size=10000, symmetric=True, distribution=distribution)
        assert np.allclose(K1.toarray(), _reference_wendland(x1, x1, hps))

    f1.release()
    f2.release()


def test_distributed_covariance_empty_and_args_kernel(client):
    """Two things the old workers got wrong: an all-empty result, and a kernel that takes
    ``args`` -- supported everywhere else in fvGP but not on the gp2Scale workers."""
    from fvgp.gp2Scale_covariance import distributed_covariance

    rng = np.random.default_rng(7)
    x1 = rng.random((40, 2))
    x2 = rng.random((40, 2)) + 100.0        # far outside the Wendland support
    hps = np.array([1.0, 0.1, 0.1])

    f1 = client.scatter(x1, broadcast=True, direct=True)
    f2 = client.scatter(x2, broadcast=True, direct=True)

    for distribution in ("blockwise", "rowwise"):
        empty = distributed_covariance(client, _reference_wendland, hps,
                                       x1_future=f1, n1=len(x1), x2_future=f2, n2=len(x2),
                                       batch_size=10, distribution=distribution)
        assert empty.shape == (len(x1), len(x2))
        assert empty.nnz == 0

        with_args = distributed_covariance(client, _wendland_with_args, hps,
                                           x1_future=f1, n1=len(x1), x2_future=f1, n2=len(x1),
                                           batch_size=10, symmetric=True,
                                           distribution=distribution,
                                           k_n_params=4, args={"scale": 3.0})
        assert np.allclose(with_args.toarray(), 3.0 * _reference_wendland(x1, x1, hps))

    f1.release()
    f2.release()


def test_distributed_covariance_rejects_bad_distribution(client):
    from fvgp.gp2Scale_covariance import distributed_covariance

    x = np.random.rand(5, 1)
    f = client.scatter(x, broadcast=True, direct=True)
    try:
        distributed_covariance(client, _reference_wendland, np.array([1., 1.]),
                               x1_future=f, n1=5, x2_future=f, n2=5,
                               batch_size=2, distribution="columnwise")
    except Exception as e:
        assert "columnwise" in str(e)
    else:
        raise AssertionError("an unknown distribution must be rejected")
    f.release()


def test_gp2Scale_posterior_matches_dense(client):
    """The gp2Scale posterior goes through the distributed sparse cross-covariance while
    the dense GP calls the kernel directly. Same kernel, same data: same answers."""
    import gc

    rng = np.random.default_rng(3)
    x = rng.random((120, 1))
    y = np.sin(np.linalg.norm(x, axis=1) * 5.0)
    hps = np.array([1.5, 0.3])
    x_pred_local = rng.random((17, 1))

    dense = GP(x, y, hps, kernel_function=wendland_anisotropic_gp2Scale_cpu,
               linalg_mode="Chol")
    reference_mean = dense.posterior_mean(x_pred_local)["m(x)"]
    reference_var = dense.posterior_covariance(x_pred_local)["v(x)"]
    reference_S = dense.posterior_covariance(x_pred_local)["S"]
    del dense
    gc.collect()

    from fvgp.gp2Scale_covariance import should_distribute

    for distribution in ("blockwise", "rowwise"):
        # batch_size has to be small enough that the 120 x 17 cross-covariance exceeds one
        # task's RAM budget; otherwise it is computed on the client and this stops testing
        # the distributed path at all
        scaled = GP(x, y, hps, gp2Scale=True, gp2Scale_batch_size=10,
                    gp2Scale_distribution=distribution, dask_client=client,
                    linalg_mode="Chol")
        assert should_distribute(len(x), len(x_pred_local), 10, distribution), \
            f"{distribution}: this test must exercise the distributed cross-covariance"
        assert sparse.issparse(scaled.prior.compute_data_cross_covariance(x_pred_local, hps))
        assert np.allclose(scaled.posterior_mean(x_pred_local)["m(x)"], reference_mean)
        post = scaled.posterior_covariance(x_pred_local)
        assert np.allclose(post["v(x)"], reference_var)
        assert np.allclose(post["S"], reference_S)
        del scaled
        gc.collect()
        client.run(lambda: None)


def test_gp2Scale_distributions_agree(client):
    """Row-wise and block-wise must produce the same prior covariance and the same
    likelihood, including after an append."""
    import gc

    rng = np.random.default_rng(11)
    x = rng.random((90, 2))
    y = np.sin(np.linalg.norm(x, axis=1) * 5.0)
    x_add = rng.random((7, 2))
    y_add = np.sin(np.linalg.norm(x_add, axis=1) * 5.0)
    hps = np.array([1.2, 0.35, 0.3])

    results = {}
    for distribution in ("blockwise", "rowwise"):
        gp = GP(x, y, hps, gp2Scale=True, gp2Scale_batch_size=25,
                gp2Scale_distribution=distribution, dask_client=client,
                linalg_mode="sparseLU")
        K = gp.prior.K.toarray().copy()
        gp.update_gp_data(x_add, y_add, append=True)
        results[distribution] = (K, gp.prior.K.toarray().copy(), gp.log_likelihood())
        del gp
        gc.collect()
        client.run(lambda: None)

    K_block, K_block_aug, ll_block = results["blockwise"]
    K_row, K_row_aug, ll_row = results["rowwise"]
    assert np.allclose(K_block, wendland_anisotropic_gp2Scale_cpu(x, x, hps))
    assert np.allclose(K_block, K_row)
    assert np.allclose(K_block_aug, K_row_aug)
    assert np.allclose(K_block_aug, wendland_anisotropic_gp2Scale_cpu(
        np.vstack([x, x_add]), np.vstack([x, x_add]), hps))
    assert np.isclose(ll_block, ll_row)


###########################################################################
############ gp2Scale covariance: worker-side units #######################
###########################################################################
# These functions execute on dask workers, in subprocesses where coverage is not
# collected. They are plain functions over plain arrays, so calling them directly is both
# the way to measure them and a sharper test than going through a cluster.
def test_evaluate_kernel_signatures():
    from fvgp.gp2Scale_covariance import evaluate_kernel

    x1 = np.random.rand(4, 2)
    x2 = np.random.rand(3, 2)
    hps = np.array([1.0, 0.5, 0.5])

    three = evaluate_kernel(wendland_anisotropic_gp2Scale_cpu, x1, x2, hps, 3, None)
    four = evaluate_kernel(_wendland_with_args, x1, x2, hps, 4, {"scale": 2.0})
    assert np.allclose(four, 2.0 * three)

    try:
        evaluate_kernel(wendland_anisotropic_gp2Scale_cpu, x1, x2, hps, 5, None)
    except Exception as e:
        assert "signature" in str(e)
    else:
        raise AssertionError("an unsupported kernel arity must be rejected")


def test_block_to_coo_dense_and_sparse():
    """A support-aware kernel hands back a sparse block; it must not be densified."""
    from fvgp.gp2Scale_covariance import block_to_coo

    dense = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    data, rows, cols = block_to_coo(dense, np.int32)
    assert rows.dtype == np.int32 and cols.dtype == np.int32
    assert sorted(data) == [1.0, 2.0, 3.0]
    assert sparse.coo_matrix((data, (rows, cols)), shape=dense.shape).toarray().tolist() == dense.tolist()

    data_s, rows_s, cols_s = block_to_coo(sparse.csr_matrix(dense), np.int64)
    assert rows_s.dtype == np.int64
    assert sorted(data_s) == [1.0, 2.0, 3.0]


def test_block_triplets_masking_and_offsets():
    from fvgp.gp2Scale_covariance import block_triplets

    x = np.sort(np.random.rand(20, 1), axis=0)
    hps = np.array([1.0, 0.5])

    # a diagonal block of a symmetric matrix reports its upper triangle only
    data, rows, cols = block_triplets(((0, 10), (0, 10)), x, x, hps,
                                      wendland_anisotropic_gp2Scale_cpu, 3, None, True, np.int32)
    assert np.all(rows <= cols)
    reference = wendland_anisotropic_gp2Scale_cpu(x[0:10], x[0:10], hps)
    assert np.isclose(data.sum(), np.triu(reference).sum())

    # the same block without symmetry keeps everything
    full, _, _ = block_triplets(((0, 10), (0, 10)), x, x, hps,
                                wendland_anisotropic_gp2Scale_cpu, 3, None, False, np.int32)
    assert np.isclose(full.sum(), reference.sum())

    # an off-diagonal block is never masked, and its indices are global
    data2, rows2, cols2 = block_triplets(((10, 20), (0, 10)), x, x, hps,
                                         wendland_anisotropic_gp2Scale_cpu, 3, None, True, np.int32)
    if data2.size:
        assert rows2.min() >= 10 and cols2.max() < 10


def test_block_triplets_empty_block():
    from fvgp.gp2Scale_covariance import block_triplets

    x1 = np.random.rand(5, 1)
    x2 = np.random.rand(5, 1) + 50.0
    data, rows, cols = block_triplets(((0, 5), (0, 5)), x1, x2, np.array([1.0, 0.1]),
                                      wendland_anisotropic_gp2Scale_cpu, 3, None, True, np.int32)
    assert data.size == 0 and rows.size == 0 and cols.size == 0


def test_row_strip_csr_full_and_empty():
    from fvgp.gp2Scale_covariance import row_strip_csr

    x = np.sort(np.random.rand(30, 1), axis=0)
    hps = np.array([1.0, 0.4])
    start, strip = row_strip_csr((10, 20), x, x, hps, wendland_anisotropic_gp2Scale_cpu,
                                 3, None, 30, np.int32)
    assert start == 10 and strip.shape == (10, 30)
    assert np.allclose(strip.toarray(), wendland_anisotropic_gp2Scale_cpu(x[10:20], x, hps))

    far = np.random.rand(30, 1) + 50.0
    start, empty = row_strip_csr((0, 30), x, far, np.array([1.0, 0.1]),
                                 wendland_anisotropic_gp2Scale_cpu, 3, None, 30, np.int32)
    assert start == 0 and empty.shape == (30, 30) and empty.nnz == 0


def test_batch_size_is_a_maximum_with_a_remainder():
    """`gp2Scale_batch_size` bounds a chunk; it is not a target to divide evenly around.
    100 000 points at 15 000 is six chunks of 15 000 and one of 10 000 -- not seven
    equal chunks of 14 285, and certainly not one oversized chunk."""
    from fvgp.gp2Scale_covariance import ranges
    from collections import Counter

    assert Counter(e - s for s, e in ranges(100000, 15000)) == {15000: 6, 10000: 1}
    assert Counter(e - s for s, e in ranges(100000, 10000)) == {10000: 10}
    assert Counter(e - s for s, e in ranges(25000, 10000)) == {10000: 2, 5000: 1}
    # fewer points than one chunk is a single short chunk
    assert ranges(9000, 10000) == [(0, 9000)]
    # and the old failure mode: this must not become one 19 999-wide chunk
    assert max(e - s for s, e in ranges(19999, 15000)) == 15000

    for n, b in ((1, 10), (10, 1), (12345, 1000), (100000, 15000)):
        chunks = ranges(n, b)
        assert sum(e - s for s, e in chunks) == n, (n, b)
        assert max(e - s for s, e in chunks) <= b, (n, b)
        assert chunks[0][0] == 0 and chunks[-1][1] == n
        assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))


def test_row_strip_is_computed_in_one_call_over_the_whole_row():
    """Row-wise means the kernel sees the entire row at once -- no column chunking."""
    from fvgp.gp2Scale_covariance import row_strip_csr

    calls = []

    def recording_kernel(x1, x2, hps):
        calls.append((len(x1), len(x2)))
        return wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)

    x = np.sort(np.random.rand(50, 1), axis=0)
    hps = np.array([1.0, 0.5])
    start, strip = row_strip_csr((10, 20), x, x, hps, recording_kernel, 3, None, 50, np.int32)

    assert calls == [(10, 50)], f"expected one full-row call, got {calls}"
    assert start == 10 and strip.shape == (10, 50)
    assert np.allclose(strip.toarray(), wendland_anisotropic_gp2Scale_cpu(x[10:20], x, hps))


def test_assemblers_handle_nothing_to_assemble():
    from fvgp.gp2Scale_covariance import assemble_triplets, assemble_row_strips, index_dtype_for

    empty_triplets = [(np.empty(0), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))]
    K = assemble_triplets(iter(empty_triplets), 6, 6, True, np.int32)
    assert K.shape == (6, 6) and K.nnz == 0

    assert assemble_row_strips(iter([]), 4, 9).shape == (4, 9)
    assert index_dtype_for(10, 10) == np.int32


def test_harvest_raises_on_a_failed_block():
    """A cancelled task comes back *as* its exception; it must not reach the assembly."""
    from fvgp.gp2Scale_covariance import _harvest

    class _FakeFuture:
        def __init__(self): self.released = False
        def release(self): self.released = True

    good = _FakeFuture()
    assert _harvest((good, "payload")) == "payload"
    assert good.released

    bad = _FakeFuture()
    try:
        _harvest((bad, ValueError("worker exploded")))
    except Exception as e:
        assert "worker exploded" in str(e) and "failed on the cluster" in str(e)
    else:
        raise AssertionError("an exception result must be raised, not returned")
    assert bad.released, "the future must be released even when the result is an error"


def test_stack_augmented_covariance():
    from fvgp.gp2Scale_covariance import stack_augmented_covariance

    K = sparse.csr_matrix(np.array([[1.0, 2.0], [2.0, 3.0]]))
    B = sparse.coo_matrix(np.array([[4.0], [5.0]]))
    D = np.array([[6.0]])
    out = stack_augmented_covariance(K, B, D)
    assert out.format == "csr"
    assert np.allclose(out.toarray(), np.array([[1., 2., 4.],
                                                [2., 3., 5.],
                                                [4., 5., 6.]]))


def test_log_time_helper(capsys):
    """fvgp.utils.log_time -- the loguru timing helper."""
    from loguru import logger
    from fvgp import utils

    logger.enable("fvgp")
    sink = []
    handle = logger.add(lambda msg: sink.append(str(msg)), level="INFO")
    try:
        with utils.log_time("plain"):
            pass
        with utils.log_time("keyed", cumulative_key="k"):
            pass
        with utils.log_time("keyed", cumulative_key="k"):
            pass
    finally:
        logger.remove(handle)
        logger.disable("fvgp")

    assert len(sink) == 3
    assert "elapsed" in sink[0] and "cumulative elapsed" not in sink[0]
    assert "cumulative elapsed" in sink[1] and "avg elapsed" in sink[1]
    assert utils.cumulative_count["k"] == 2


###########################################################################
################ gp_actor: the async optimizer actors #####################
###########################################################################
# These live on a Dask worker in production, so nothing measures them there. They are
# ordinary classes driven by a background thread, so they can be exercised directly.
def _wait_for_actor(actor, timeout=60.0):
    actor._thread.join(timeout)
    assert not actor._thread.is_alive(), "actor thread did not finish"


def test_mcmc_actor_runs_and_reports():
    from fvgp.gp_actor import _MCMCActor
    from fvgp import ProposalDistribution

    bounds = np.array([[-2., 2.], [-2., 2.]])

    def log_likelihood(x, args):
        return -float(x @ x)

    def prior_function(theta, bounds, args):
        if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]): return -np.inf
        return 0.

    pd = ProposalDistribution([0, 1], init_prop_Sigma=np.diag([0.1, 0.1]),
                              adapt_callable="normal")
    actor = _MCMCActor(log_likelihood, bounds, prior_function, [pd], {},
                       np.array([0.5, -0.5]), 30, False)
    assert actor.get_latest() == {}
    actor.start()
    _wait_for_actor(actor)

    latest = actor.get_latest()
    for key in ("f(x)", "max f(x)", "MAP", "max x", "time stamps", "x",
                "mean(x)", "median(x)", "var(x)"):
        assert key in latest, key
    # the callback fires between the x and f(x) appends, so x may lead by one
    assert len(latest["x"]) - len(latest["f(x)"]) in (0, 1)
    assert latest["max f(x)"] == np.max(latest["f(x)"])
    assert not actor._running
    actor.stop()
    assert not actor._running


def test_mcmc_actor_stop_breaks_the_run():
    """stop() must end the chain through the break condition, not just flip a flag."""
    from fvgp.gp_actor import _MCMCActor
    from fvgp import ProposalDistribution

    bounds = np.array([[-2., 2.]])

    def log_likelihood(x, args):
        return -float(x @ x)

    def prior_function(theta, bounds, args):
        return 0.

    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([0.1]), adapt_callable="normal")
    actor = _MCMCActor(log_likelihood, bounds, prior_function, [pd], {},
                       np.array([0.5]), 2000000, False)
    actor.start()
    actor.stop()
    _wait_for_actor(actor)
    assert not actor._running


def test_adam_actor_descends_and_reports():
    from fvgp.gp_actor import _AdamActor

    target = np.array([1.0, -2.0])

    def nlml(theta):
        d = theta - target
        return float(d @ d)

    def grad_nlml(theta):
        return 2.0 * (theta - target)

    theta0 = np.array([5.0, 5.0])
    actor = _AdamActor(nlml, grad_nlml, theta0, 0.2, 0.9, 0.999, 1e-8, 400, 1e-10)
    initial = actor.get_latest()
    assert initial["iteration"] == 0 and initial["nlml"] is None
    assert np.allclose(initial["x"], theta0), "theta0 must be copied, not aliased"

    actor.start()
    _wait_for_actor(actor)
    latest = actor.get_latest()
    assert latest["iteration"] > 0
    assert latest["nlml"] < nlml(theta0)
    assert latest["grad_norm"] >= 0.0
    assert np.linalg.norm(latest["x"] - target) < np.linalg.norm(theta0 - target)
    actor.stop()


def test_adam_actor_stop_ends_the_run_early():
    from fvgp.gp_actor import _AdamActor

    def nlml(theta):
        return float(theta @ theta)

    def grad_nlml(theta):
        return 2.0 * theta

    actor = _AdamActor(nlml, grad_nlml, np.array([1.0]), 1e-6, 0.9, 0.999, 1e-8,
                       2000000, 0.0)
    actor.start()
    actor.stop()
    _wait_for_actor(actor)
    assert not actor._running
    assert actor.get_latest()["iteration"] < 2000000


def test_bo_actor_runs_and_switches_to_the_recommendation():
    """While running, `x` is the best point observed; when finished it becomes the
    noise-aware recommendation and the extra diagnostics appear."""
    from fvgp.gp_actor import _BOActor

    target = np.log(np.array([1.0, 10.0]))

    def objective(theta):
        z = np.log(theta) - target
        return float(z @ z)

    bounds = np.array([[1e-2, 1e2], [1e-1, 1e3]])
    actor = _BOActor(objective, bounds, np.array([50.0, 0.5]), 14,
                     {"seed": 0, "patience": 0}, False)
    assert actor.get_latest()["status"] == "queued"
    actor.start()
    _wait_for_actor(actor, timeout=300)

    latest = actor.get_latest()
    assert latest["status"] == "finished"
    assert latest["n_evaluations"] > 0
    assert latest["iteration"] >= 0
    assert np.isfinite(latest["objective"])
    for key in ("sensitivity", "posterior covariance", "ard length scales"):
        assert key in latest, key
    assert "surrogate" not in latest, "the surrogate GP must not be shipped back"
    assert not actor._running
    actor.stop()


def test_bo_actor_stop_ends_the_run_early():
    from fvgp.gp_actor import _BOActor

    def objective(theta):
        return float(np.sum(theta ** 2))

    actor = _BOActor(objective, np.array([[0.1, 10.0]]), np.array([1.0]), 1000000,
                     {"seed": 0, "patience": 0}, False)
    actor.start()
    actor.stop()
    _wait_for_actor(actor, timeout=300)
    assert not actor._running


def test_async_optimizer_proxies_to_the_actor():
    """AsyncOptimizer only unwraps the Dask Actor's futures; check every alias."""
    from fvgp.gp_actor import AsyncOptimizer

    class _Result:
        def __init__(self, value): self.value = value
        def result(self): return self.value

    class _FakeActor:
        def __init__(self): self.stops = 0
        def get_latest(self): return _Result({"x": 1})
        def stop(self):
            self.stops += 1
            return _Result(None)

    fake = _FakeActor()
    opt = AsyncOptimizer(fake)
    assert opt.get_latest() == {"x": 1}
    opt.stop()
    opt.cancel_tasks()
    opt.kill_client()
    assert fake.stops == 3


###########################################################################
################ narrow branches in the small modules #####################
###########################################################################
def _tiny_gp(**kwargs):
    xx = np.random.rand(20, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    return GP(xx, yy, np.array([1., 1., 1.]), **kwargs)


def test_gp_data_update_rejects_bad_noise_combinations():
    from fvgp.gp_data import GPdata

    xx = np.random.rand(10, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1)).reshape(-1, 1)
    v = np.ones(10) * 0.01

    with_noise = GPdata(xx, yy, noise_variances=v)
    try:
        with_noise.update(xx[:2], yy[:2], None, append=True)
    except Exception as e:
        assert "Please provide noise_variances" in str(e)
    else:
        raise AssertionError("dropping noise on update must be rejected")

    without_noise = GPdata(xx, yy)
    try:
        without_noise.update(xx[:2], yy[:2], v[:2], append=True)
    except Exception as e:
        assert "did not initialize noise" in str(e)
    else:
        raise AssertionError("adding noise on update must be rejected")



def test_gp_data_non_euclidean_shapes():
    """A list x_data marks the space non-Euclidean and fixes the index dims at 1."""
    from fvgp.gp_data import GPdata

    x_list = [["a"], ["b"], ["c"]]
    yy = np.array([[1.0], [2.0], [3.0]])
    d = GPdata(x_list, yy)
    assert d.Euclidean is False
    assert d.index_set_dim == 1 and d.input_set_dim == 1


def test_gp_data_append_with_noise_arrays():
    from fvgp.gp_data import GPdata

    xx = np.random.rand(6, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1)).reshape(-1, 1)
    v = np.full(6, 0.01)
    d = GPdata(xx, yy, noise_variances=v)
    d.update(xx[:2], yy[:2], v[:2], append=True)
    assert len(d.x_data) == 8 and len(d.noise_variances) == 8
    assert len(d.x_new) == 2 and len(d.x_old) == 6


def test_gp_likelihood_branches():
    from fvgp.gp_likelihood import GPlikelihood

    xx = np.random.rand(12, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    v = np.full(12, 0.01)

    # noise variances and a noise function together are ambiguous
    try:
        GP(xx, yy, np.array([1., 1., 1.]), noise_variances=v,
           noise_function=lambda x, hps: np.full(len(x), 0.01))
    except Exception as e:
        assert "Decide which one to use" in str(e)
    else:
        raise AssertionError("noise_variances plus noise_function must be rejected")

    # ram_economy picks the economical default noise gradient
    econ = _tiny_gp(ram_economy=True)
    assert np.allclose(econ.likelihood.calculate_V_grad(xx, np.array([1., 1., 1.]), 0),
                       np.zeros(len(xx)))
    assert econ.likelihood.gp2Scale is False

    # a three-argument noise function takes args, and anything else is rejected
    gp3 = GP(xx, yy, np.array([1., 1., 1.]),
             noise_function=lambda x, hps, args: np.full(len(x), args["level"]),
             args={"level": 0.02})
    assert np.allclose(gp3.likelihood.calculate_V(xx, np.array([1., 1., 1.])), 0.02)

    gp3.likelihood.v_n_params = 7
    try:
        gp3.likelihood.calculate_V(xx, np.array([1., 1., 1.]))
    except Exception as e:
        assert "No valid noise function signature" in str(e)
    else:
        raise AssertionError("an unsupported noise arity must be rejected")


def test_fvgp_rejects_single_task_y_data():
    x2 = np.random.rand(10, 2)
    y1 = np.sin(np.linalg.norm(x2, axis=1))
    try:
        fvGP(x2, y1, np.array([1., 1., 1., 1.]))
    except ValueError as e:
        assert "output number is 1" in str(e)
    else:
        raise AssertionError("1-d y_data must be rejected by fvGP")


def test_fvgp_update_data_formats_and_nan_skipping():
    """The multi-task transform drops NaN entries, and the append path validates formats."""
    x2 = np.random.rand(12, 2)
    y2 = np.column_stack([np.sin(np.linalg.norm(x2, axis=1)),
                          np.cos(np.linalg.norm(x2, axis=1))])
    y2[0, 1] = np.nan                       # this task/point pair must be dropped
    v2 = np.full(y2.shape, 0.01)
    gp = fvGP(x2, y2, np.array([1., 1., 1., 1.]), noise_variances=v2)
    assert len(gp.x_data) == y2.size - 1, "the NaN entry must not enter the product space"

    x_add = np.random.rand(2, 2)
    y_add = np.column_stack([np.sin(np.linalg.norm(x_add, axis=1)),
                             np.cos(np.linalg.norm(x_add, axis=1))])
    gp.update_gp_data(x_add, y_add, np.full(y_add.shape, 0.01), append=True)
    assert len(gp.fvgp_x_data) == 14

    try:
        gp.update_gp_data("not an array", y_add, np.full(y_add.shape, 0.01), append=True)
    except (Exception, AssertionError) as e:
        assert "format in x_new" in str(e) or "allowed format" in str(e)
    else:
        raise AssertionError("a bad x_new format must be rejected")


def test_wendland_support_aware_empty_returns():
    """The support-aware kernel's early exits: nothing within the radius, and points
    exactly on the boundary where the polynomial evaluates to zero."""
    from fvgp.kernels import _wendland_support_aware_cpu_triplets

    # bounding boxes overlap so the block-gap prune passes, but no pair is within 1
    hps = np.array([1.0, 1.0, 1.0])
    values, rows, cols = _wendland_support_aware_cpu_triplets(
        np.array([[0.0, 0.0]]), np.array([[0.8, 0.8]]), hps)
    assert values.size == 0 and rows.size == 0 and cols.size == 0

    # exactly on the support boundary: inside the radius search, but the value is 0
    values, rows, cols = _wendland_support_aware_cpu_triplets(
        np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), hps)
    assert values.size == 0 and rows.size == 0 and cols.size == 0


def test_warm_start_candidate_shape_reconciliation():
    """The warm-start cache is reused across shapes: a 1-d guess is promoted to a
    column, a single column is broadcast across tasks, and a stale row count is
    rejected outright."""
    gp = _tiny_gp(args={"sparse_krylov_warm_start": True})
    ml = gp.marginal_likelihood
    n = len(gp.x_data)

    ml._warm_start_KVinvY = np.ones(n)                       # 1-d -> column
    assert ml._iterative_initial_guess((n, 1)).shape == (n, 1)

    ml._warm_start_KVinvY = np.ones((n, 1))                  # one column -> many
    assert ml._iterative_initial_guess((n, 3)).shape == (n, 3)

    ml._warm_start_KVinvY = np.ones((n + 5, 1))              # wrong row count
    assert ml.kv.KVinvY is not None
    ml.kv.KVinvY = np.ones((n + 5, 1))
    assert ml._iterative_initial_guess((n, 1)) is None

    gp_off = _tiny_gp()
    assert gp_off.marginal_likelihood._iterative_initial_guess((n, 1)) is None


def test_marginal_likelihood_reports_linalg_and_gradient_failures():
    gp = _tiny_gp()
    ml = gp.marginal_likelihood
    hps = gp.hyperparameters

    def boom(*a, **k):
        raise RuntimeError("factorization exploded")

    original = ml.compute_new_KVlogdet_KVinvY
    ml.compute_new_KVlogdet_KVinvY = boom
    try:
        ml.log_likelihood(hyperparameters=hps * 1.1)
    except Exception as e:
        assert "Linear algebra failed" in str(e) and "factorization exploded" in str(e)
    else:
        raise AssertionError("a linalg failure must be reported with its hyperparameters")
    finally:
        ml.compute_new_KVlogdet_KVinvY = original

    original_dk = ml.dk_dh
    ml.dk_dh = boom
    try:
        ml.neg_log_likelihood_gradient(hyperparameters=hps)
    except Exception as e:
        assert "dK/dh" in str(e)
    else:
        raise AssertionError("a kernel-gradient failure must be reported")
    finally:
        ml.dk_dh = original_dk


def test_marginal_likelihood_unpickles_without_a_warm_start_slot():
    """State written before warm starts existed must still load."""
    gp = _tiny_gp()
    state = gp.marginal_likelihood.__getstate__()
    state.pop("_warm_start_KVinvY", None)
    gp.marginal_likelihood.__setstate__(state)
    assert gp.marginal_likelihood._warm_start_KVinvY is None


def test_training_rejects_bad_starting_points_and_methods():
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    assert gp.trainer.gp2Scale is False

    try:
        gp.trainer.train(objective_function=gp.marginal_likelihood.neg_log_likelihood,
                         objective_function_gradient=gp.marginal_likelihood.neg_log_likelihood_gradient,
                         objective_function_hessian=gp.marginal_likelihood.neg_log_likelihood_hessian,
                         hyperparameter_bounds=bounds,
                         init_hyperparameters=np.array([100., 100., 100.]),
                         method="local")
    except Exception as e:
        assert "outside of optimization bounds" in str(e)
    else:
        raise AssertionError("out-of-bounds starting hyperparameters must be rejected")

    try:
        gp.trainer.train(objective_function=gp.marginal_likelihood.neg_log_likelihood,
                         objective_function_gradient=gp.marginal_likelihood.neg_log_likelihood_gradient,
                         objective_function_hessian=gp.marginal_likelihood.neg_log_likelihood_hessian,
                         hyperparameter_bounds=bounds,
                         init_hyperparameters=np.array([1., 1., 1.]),
                         method=42)
    except ValueError as e:
        assert "No optimization mode" in str(e)
    else:
        raise AssertionError("an unknown training method must be rejected")


def test_stop_training_and_kill_client_warn_when_nothing_runs():
    from fvgp.gp_training import GPtraining

    class _Nothing:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GPtraining.stop_training(_Nothing())
        GPtraining.kill_client(_Nothing())
    messages = " ".join(str(w.message) for w in caught)
    assert "no training is running" in messages
    assert "killed" in messages


###########################################################################
##################### gp_prior narrow branches ############################
###########################################################################
def test_prior_rejects_bad_kernel_and_space_combinations():
    x_list = [["a"], ["b"], ["c"]]
    y_list = np.array([1.0, 2.0, 3.0])

    # a non-Euclidean space needs a user kernel
    try:
        GP(x_list, y_list, np.array([1., 1.]))
    except Exception as e:
        assert "non-Euclidean" in str(e)
    else:
        raise AssertionError("a non-Euclidean space without a kernel must be rejected")

    # a non-callable, non-None kernel is not a kernel
    xx = np.random.rand(10, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    try:
        GP(xx, yy, np.array([1., 1., 1.]), kernel_function="matern")
    except Exception as e:
        assert "wrong format in kernel_function" in str(e) or "No valid kernel" in str(e)
    else:
        raise AssertionError("a non-callable kernel must be rejected")


def test_prior_non_euclidean_needs_initial_hyperparameters():
    x_list = [["a"], ["b"], ["c"]]
    y_list = np.array([1.0, 2.0, 3.0])

    def k(x1, x2, hps):
        return hps[0] * np.equal.outer([a[0] for a in x1], [a[0] for a in x2]).astype(float)

    gp = GP(x_list, y_list, np.array([1.0]), kernel_function=k)
    assert gp.prior.Euclidean is False

    try:
        GP(x_list, y_list, None, kernel_function=k)
    except Exception as e:
        assert "init_hyperparameters" in str(e)
    else:
        raise AssertionError("a non-Euclidean GP without initial hyperparameters must be rejected")


def test_prior_four_argument_kernel_and_three_argument_mean():
    """The args-taking signatures of kernel and mean, outside gp2Scale."""
    xx = np.random.rand(15, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    hps = np.array([1., 1., 1.])

    def kernel_with_args(x1, x2, hps, args):
        return args["amp"] * np.exp(-get_anisotropic_distance_matrix(x1, x2, hps[1:]) ** 2)

    def mean_with_args(x, hps, args):
        return np.full(len(x), args["offset"])

    def mean_grad(x, hps):
        return np.zeros((len(hps), len(x)))

    gp = GP(xx, yy, hps, kernel_function=kernel_with_args,
            prior_mean_function=mean_with_args, prior_mean_function_grad=mean_grad,
            args={"amp": 2.0, "offset": 0.5})
    assert gp.prior.k_n_params == 4 and gp.prior.m_n_params == 3
    assert np.allclose(gp.prior.compute_mean(xx, hps), 0.5)
    assert gp.prior._dm_dh is mean_grad
    assert np.allclose(np.diag(gp.prior.compute_covariances(xx, xx, hps)), 2.0)

    # a custom mean takes the incremental update path on append
    x_add = np.random.rand(2, 2)
    gp.update_gp_data(x_add, np.sin(np.linalg.norm(x_add, axis=1)), append=True)
    assert len(gp.prior.m) == 17


def test_prior_update_mean_rejects_a_matrix_mean():
    gp = _tiny_gp()
    gp.prior.m = np.zeros((5, 2))
    try:
        gp.prior._update_mean(np.random.rand(2, 2), gp.hyperparameters)
    except Exception as e:
        assert "has to be a vector" in str(e)
    else:
        raise AssertionError("a 2-d prior mean must be rejected")

    gp.prior.m = 1.0
    try:
        gp.prior._update_mean(np.random.rand(2, 2), gp.hyperparameters)
    except Exception as e:
        assert "wrong format" in str(e)
    else:
        raise AssertionError("a scalar prior mean must be rejected")


def test_prior_default_mean_rejects_bad_y_shapes():
    gp = _tiny_gp()
    original = gp.data.y_data
    try:
        gp.data.y_data = original.reshape(-1)
        try:
            gp.prior._default_mean_function(gp.x_data, gp.hyperparameters)
        except Exception as e:
            assert "y_data wrong format" in str(e)
        else:
            raise AssertionError("1-d y_data must be rejected by the default mean")

        gp.data.y_data = original.reshape(-1, 1, 1)
        try:
            gp.prior._default_mean_function(gp.x_data, gp.hyperparameters)
        except Exception as e:
            assert "Wrong dim" in str(e)
        else:
            raise AssertionError("3-d y_data must be rejected by the default mean")
    finally:
        gp.data.y_data = original


def test_gp2Scale_covariance_needs_a_client():
    gp = _tiny_gp()
    gp.data.gp2Scale = True
    try:
        gp.prior._gp2Scale_covariance(gp.x_data, gp.x_data, gp.hyperparameters, symmetric=True)
    except Exception as e:
        assert "needs a dask client" in str(e)
    else:
        raise AssertionError("gp2Scale without a client must be rejected")


###########################################################################
######################## gp.py narrow branches ############################
###########################################################################
def test_gp_properties_and_deprecated_accessor():
    xx = np.random.rand(15, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    v = np.full(15, 0.01)
    gp = GP(xx, yy, np.array([1., 1., 1.]), noise_variances=v)

    assert np.allclose(gp.noise_variances, v)
    assert gp.dask_client is None
    assert len(gp.m) == 15
    assert gp.mcmc_info is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hps = gp.get_hyperparameters()
    assert np.allclose(hps, gp.hyperparameters)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_gp_warns_about_a_gpu_without_a_backend(monkeypatch):
    import importlib
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    xx = np.random.rand(10, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GP(xx, yy, np.array([1., 1., 1.]), compute_device="gpu")
    assert any("install pytorch or cupy" in str(w.message) for w in caught)


def test_default_hyperparameter_bounds_refuse_custom_functions():
    xx = np.random.rand(10, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    gp = GP(xx, yy, np.array([1., 1.]),
            kernel_function=lambda x1, x2, hps: hps[0] * np.exp(
                -get_anisotropic_distance_matrix(x1, x2, np.array([hps[1], hps[1]])) ** 2))
    try:
        gp._get_default_hyperparameter_bounds()
    except Exception as e:
        assert "custom hyperparameter_bounds" in str(e)
    else:
        raise AssertionError("default bounds must be refused for a custom kernel")


def test_train_argument_validation():
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])

    # asynchronous without a client
    try:
        gp.train(hyperparameter_bounds=bounds, method="mcmc", asynchronous=True, max_iter=2)
    except Exception as e:
        assert "dask_client" in str(e)
    else:
        raise AssertionError("async training without a client must be rejected")

    # out-of-bounds init hyperparameters are replaced, with a warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1,
                 init_hyperparameters=np.array([500., 500., 500.]))
    assert any("out of bounds" in str(w.message) for w in caught)

    # mcmc ignores a user objective, and says so
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.train(hyperparameter_bounds=bounds, method="mcmc", max_iter=2,
                 objective_function=lambda hps: 0.0)
    assert any("user-defined objective_function is ignored" in str(w.message) for w in caught)

    # a user objective for a gradient-based method needs a gradient
    try:
        gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1,
                 objective_function=lambda hps: 0.0)
    except Exception as e:
        assert "gradient" in str(e)
    else:
        raise AssertionError("a user objective without a gradient must be rejected for local")


def test_accept_only_if_improved_rejects_a_regression(monkeypatch):
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    gp.set_hyperparameters(np.array([1., 1., 1.]))
    incumbent = gp.hyperparameters.copy()
    ll_incumbent = gp.log_likelihood()

    # a "trainer" that hands back a point far worse than the one it was given
    bad = np.array([9.9, 0.011, 9.9])
    assert gp.log_likelihood(bad) < ll_incumbent
    gp.set_hyperparameters(incumbent)
    monkeypatch.setattr(gp.trainer, "train", lambda *a, **kw: bad.copy())

    # the rollback is the only added cost: two set_hyperparameters here, one when accepting
    calls = {"n": 0}
    real_set = gp.set_hyperparameters

    def _counting_set(hps):
        calls["n"] += 1
        return real_set(hps)

    monkeypatch.setattr(gp, "set_hyperparameters", _counting_set)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1)
    assert calls["n"] == 2
    assert np.allclose(returned, incumbent)
    assert np.allclose(gp.hyperparameters, incumbent)
    assert np.isclose(gp.log_likelihood(), ll_incumbent)
    rejections = [str(w.message) for w in caught if "rejected" in str(w.message)]
    assert len(rejections) == 1
    assert "local" in rejections[0]
    assert str(ll_incumbent) in rejections[0]

    # the same regression is accepted once the flag is off
    calls["n"] = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1,
                            accept_only_if_improved=False)
    assert calls["n"] == 1
    assert np.allclose(returned, bad)
    assert np.allclose(gp.hyperparameters, bad)
    assert not any("rejected" in str(w.message) for w in caught)


def test_accept_only_if_improved_accepts_an_improvement():
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    gp.set_hyperparameters(np.array([5., 0.05, 0.05]))
    before = gp.hyperparameters.copy()
    ll_before = gp.log_likelihood()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = gp.train(hyperparameter_bounds=bounds, method="local", max_iter=20)
    assert not np.allclose(returned, before)
    assert np.allclose(gp.hyperparameters, returned)
    assert gp.log_likelihood() >= ll_before
    assert not any("rejected" in str(w.message) for w in caught)


def test_accept_only_if_improved_is_inactive_where_it_should_be(monkeypatch):
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    bad = np.array([9.9, 0.011, 9.9])

    # mcmc returns a posterior summary, bo a noise-aware recommendation: both exempt
    for method in ("mcmc", "bo"):
        gp = _tiny_gp()
        gp.set_hyperparameters(np.array([1., 1., 1.]))
        monkeypatch.setattr(gp.trainer, "train", lambda *a, **kw: bad.copy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = gp.train(hyperparameter_bounds=bounds, method=method, max_iter=2)
        assert np.allclose(returned, bad)
        assert np.allclose(gp.hyperparameters, bad)

    # a user objective is judged by its own criterion, not the marginal likelihood
    gp = _tiny_gp()
    gp.set_hyperparameters(np.array([1., 1., 1.]))
    monkeypatch.setattr(gp.trainer, "train", lambda *a, **kw: bad.copy())
    returned = gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1,
                        objective_function=lambda hps: 0.0,
                        objective_function_gradient=lambda hps: np.zeros(3))
    assert np.allclose(returned, bad)
    assert np.allclose(gp.hyperparameters, bad)


def test_accept_only_if_improved_tolerates_estimator_noise(monkeypatch):
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    gp.set_hyperparameters(np.array([1., 1., 1.]))
    incumbent = gp.hyperparameters.copy()

    # a proposal a hair worse than the incumbent
    ll = {"n": 0}
    ll_incumbent = gp.log_likelihood()

    def _slightly_worse(hyperparameters=None):
        ll["n"] += 1
        return ll_incumbent if ll["n"] == 1 else ll_incumbent - 0.5

    monkeypatch.setattr(gp.marginal_likelihood, "log_likelihood", _slightly_worse)
    monkeypatch.setattr(gp.trainer, "train", lambda *a, **kw: np.array([2., 2., 2.]))

    # a stochastic log-determinant that wobbles by more than the drop: accept
    monkeypatch.setattr(gp.marginal_likelihood, "log_likelihood_variance", lambda: 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1)
    assert np.allclose(returned, np.array([2., 2., 2.]))
    assert not any("rejected" in str(w.message) for w in caught)

    # the same drop in an exact mode, where the likelihood is deterministic: reject
    ll["n"] = 0
    gp.set_hyperparameters(incumbent)
    monkeypatch.setattr(gp.marginal_likelihood, "log_likelihood_variance", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = gp.train(hyperparameter_bounds=bounds, method="local", max_iter=1)
    assert np.allclose(returned, incumbent)
    assert any("rejected" in str(w.message) for w in caught)


def test_gaussian_helper_and_observed_vs_predicted_plot(monkeypatch):
    gp = _tiny_gp()
    g = GP.gaussian_1d(np.array([0.0, 1.0]), 0.0, 1.0)
    assert np.isclose(g[0], 1.0 / np.sqrt(2 * np.pi))
    assert g[1] < g[0]

    x_test = np.random.rand(6, 2)
    y_test = np.sin(np.linalg.norm(x_test, axis=1))
    gp.plot_observed_vs_predicted(x_test, y_test, title="test")

    import builtins
    real_import = builtins.__import__

    def no_matplotlib(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert gp.plot_observed_vs_predicted(x_test, y_test) is None
    assert any("matplotlib is not installed" in str(w.message) for w in caught)


###########################################################################
######################### gp_kv narrow branches ###########################
###########################################################################
_KV_MODES = ["Chol", "CholInv", "Inv", "sparseLU", "sparseCG", "sparseMINRES",
             "sparseCGpre", "sparseMINRESpre", "sparseSolve"]


def test_compute_new_KVinvY_agrees_across_every_mode():
    """compute_new_KVinvY is the training-time solve; each mode has its own branch."""
    rng = np.random.default_rng(5)
    xx = rng.random((25, 2))
    yy = np.sin(np.linalg.norm(xx, axis=1) * 3.0)
    hps = np.array([1.0, 0.6, 0.6])

    # a well-conditioned system, so the truncated Krylov modes converge to the
    # same answer as the direct ones rather than to their own tolerances
    noise = np.full(len(xx), 0.1)
    reference = None
    for mode in _KV_MODES:
        gp = GP(xx, yy, hps, noise_variances=noise, linalg_mode=mode)
        K, V, m = gp.kv._get_KVm()
        KV = gp.kv.addKV(K, V)
        KVinvY = gp.kv.compute_new_KVinvY(KV, m)
        assert KVinvY.shape == gp.y_data.shape
        assert np.all(np.isfinite(KVinvY)), mode
        if reference is None:
            reference = KVinvY
        else:
            # the truncated Krylov modes stop at their own tolerance, so agree to ~3 digits
            assert np.allclose(KVinvY, reference, atol=5e-3), mode


def test_kv_rejects_an_unknown_mode_everywhere():
    gp = _tiny_gp()
    K, V, m = gp.kv._get_KVm()
    KV = gp.kv.addKV(K, V)
    gp.kv.mode = "not-a-mode"
    for call in (lambda: gp.kv.set_KV(KV),
                 lambda: gp.kv.update_KV(KV),
                 lambda: gp.kv.compute_new_KVinvY(KV, m),
                 lambda: gp.kv.solve(gp.y_data),
                 lambda: gp.kv.logdet()):
        try:
            call()
        except Exception as e:
            assert "Mode" in str(e) or "mode" in str(e)
        else:
            raise AssertionError("an unknown linalg mode must be rejected")


def test_addKV_format_combinations():
    from fvgp.gp_kv import GPkv

    dense = np.array([[2.0, 0.1], [0.1, 3.0]])
    diag = np.array([0.5, 0.5])
    assert np.allclose(np.diag(GPkv.addKV(dense, diag)), [2.5, 3.5])

    full = np.diag(diag)
    assert np.allclose(GPkv.addKV(dense, full), dense + full)

    sp = sparse.csr_matrix(dense)
    assert np.allclose(GPkv.addKV(sp, diag).toarray(), dense + full)
    assert np.allclose(GPkv.addKV(sp, sparse.csr_matrix(full)).toarray(), dense + full)
    # a sparse V against a dense K is densified rather than refused
    assert np.allclose(GPkv.addKV(dense, sparse.csr_matrix(full)), dense + full)

    try:
        GPkv.addKV("not a matrix", diag)
    except Exception as e:
        assert "2-d" in str(e) or "not possible" in str(e)
    else:
        raise AssertionError("a non-matrix K must be rejected")


def test_kv_unpickles_older_states_without_preconditioner_slots():
    gp = _tiny_gp()
    state = gp.kv.__getstate__()
    for attr in ("Preconditioner_factor", "Preconditioner_operator", "Preconditioner_signature",
                 "Preconditioner_KV_shape", "Preconditioner_reuse_counter",
                 "Last_preconditioner_error", "Preconditioner_fingerprint",
                 "Warm_start_fingerprint"):
        state.pop(attr, None)
    gp.kv.__setstate__(state)
    assert gp.kv.Preconditioner_operator is None
    assert gp.kv.Preconditioner_reuse_counter == 0


def test_matrix_fingerprint_and_drift_edge_cases():
    from fvgp.gp_kv import GPkv

    assert GPkv.matrix_fingerprint(None) is None
    dense = np.array([[2.0, 0.0], [0.0, 2.0]])
    fp_dense = GPkv.matrix_fingerprint(dense)
    fp_sparse = GPkv.matrix_fingerprint(sparse.csr_matrix(dense))
    assert fp_dense[0] == fp_sparse[0] and np.isclose(fp_dense[2], fp_sparse[2])

    gp = _tiny_gp()
    assert gp.kv._fingerprint_drift(None, fp_dense) == np.inf
    assert gp.kv._fingerprint_drift(fp_dense, None) == np.inf
    assert gp.kv._fingerprint_drift(fp_dense, fp_dense) == 0.0
    # a shape change is infinite drift, whatever the values
    other = GPkv.matrix_fingerprint(np.eye(3))
    assert gp.kv._fingerprint_drift(fp_dense, other) == np.inf


def test_preconditioner_helpers_are_inert_in_unpreconditioned_modes():
    gp = _tiny_gp(linalg_mode="Chol")
    K, V, m = gp.kv._get_KVm()
    KV = gp.kv.addKV(K, V)
    assert gp.kv._can_reuse_sparse_preconditioner(KV) is False
    assert gp.kv._get_or_refresh_preconditioner(KV) is None


###########################################################################
###################### gp_posterior narrow branches #######################
###########################################################################
def _tiny_fvgp():
    xx = np.random.rand(15, 2)
    yy = np.column_stack([np.sin(np.linalg.norm(xx, axis=1)),
                          np.cos(np.linalg.norm(xx, axis=1))])
    return fvGP(xx, yy, np.array([1., 1., 1., 1.]))


def test_multi_task_posterior_reshape_paths():
    """The x_out branches of the posterior mean, its gradient, and the covariance grad."""
    gp = _tiny_fvgp()
    xp = np.random.rand(4, 2)

    assert gp.posterior_mean(xp)["m(x)"].shape == (4, 2)
    assert gp.posterior_mean_grad(xp, direction=0)["dm/dx"].shape == (4, 2)
    assert gp.posterior_mean_grad(xp)["dm/dx"].shape == (4, 2, 2)

    grad = gp.posterior_covariance_grad(xp, direction=0)
    assert grad["dv/dx"].shape == (4, 2)
    assert grad["dS/dx"].shape == (4, 4, 2, 2)
    assert gp.posterior_covariance_grad(xp)["dv/dx"].shape == (4, 2, 2)


def test_posterior_variance_tiling_for_multi_column_y():
    """Without x_out but with several output columns, the variance is tiled per column."""
    gp = _tiny_gp()
    gp.data.y_data = np.tile(gp.y_data, (1, 2))
    result = gp.posterior_covariance(np.random.rand(5, 2))
    assert result["v(x)"].shape == (5, 2)


def test_posterior_warns_and_clips_negative_variances():
    gp = _tiny_gp()
    xp = np.random.rand(4, 2)

    def inflated(k, chunk_size=None):
        return np.eye(k.shape[1]) * 10.0

    gp.posterior._cross_solve_product = inflated
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gp.posterior_covariance(xp)
    assert any("Negative variances" in str(w.message) for w in caught)
    assert np.all(result["v(x)"] >= 0.0), "negative variances must be clipped"
    assert np.all(np.diag(result["S"]) >= 0.0)


def test_dense_K_warns_under_gp2Scale(client):
    """The joint-covariance methods are dense in N; under gp2Scale they say so."""
    import gc
    rng = np.random.default_rng(1)
    xx = rng.random((40, 1))
    yy = np.sin(np.linalg.norm(xx, axis=1) * 5.0)
    gp = GP(xx, yy, np.array([1.0, 0.5]), gp2Scale=True, gp2Scale_batch_size=20,
            dask_client=client, linalg_mode="Chol")
    assert sparse.issparse(gp.prior.K)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mi = gp.gp_mutual_information(rng.random((3, 1)))
    assert np.isfinite(mi["mutual information"])
    assert any("dense in the number of data points" in str(w.message) for w in caught)
    del gp
    gc.collect()
    client.run(lambda: None)


def test_kl_div_warns_on_a_negative_result(monkeypatch):
    """Defensive branch: an unstable logdet can drive the KL divergence negative."""
    import fvgp.gp_posterior as gp_posterior_module
    from fvgp.gp_posterior import GPposterior

    calls = {"n": 0}

    def unstable_logdet(S, args=None):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else -50.0

    monkeypatch.setattr(gp_posterior_module, "calculate_logdet", unstable_logdet)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kld = GPposterior.kl_div(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2))
    assert any("Negative KL divergence" in str(w.message) for w in caught)
    assert kld >= 0.0, "the magnitude is returned"


def test_add_noise_warns_on_an_unusable_noise_function():
    xx = np.random.rand(12, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    gp = GP(xx, yy, np.array([1., 1., 1.]),
            noise_function=lambda x, hps: np.full(len(x), 0.01))
    gp.likelihood.noise_function = lambda x, hps: np.ones((len(x), 2, 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.posterior_covariance(np.random.rand(3, 2), add_noise=True)
    assert any("Noise could not be added" in str(w.message) for w in caught)


def test_cartesian_product_and_int_gauss():
    from fvgp.gp_posterior import GPposterior

    assert np.isclose(GPposterior._int_gauss(np.eye(2)), 2.0 * np.pi)

    listed = GPposterior.cartesian_product([["a"], ["b"]], np.array([0, 1]))
    assert len(listed) == 4

    try:
        GPposterior.cartesian_product(42, np.array([0, 1]))
    except Exception as e:
        assert "out of options" in str(e)
    else:
        raise AssertionError("an unsupported x type must be rejected")


###########################################################################
################## remaining narrow branches, by module ###################
###########################################################################
def _module_level_objective(x, args):
    return -float(x @ x)


def _module_level_prior(theta, bounds, args):
    return 0.


def test_mcmc_argument_validation_and_pickling():
    from fvgp import gpMCMC, ProposalDistribution

    bounds = np.array([[-1., 1.]])

    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([0.1]), adapt_callable="normal")
    mcmc = gpMCMC(_module_level_objective, bounds=bounds,
                  prior_function=_module_level_prior, proposal_distributions=[pd])

    try:
        mcmc.run_mcmc(x0=np.array([0.0]), n_updates=3, break_condition="nonsense")
    except Exception as e:
        assert "No valid input for break condition" in str(e)
    else:
        raise AssertionError("an invalid break condition must be rejected")

    # both classes are pickled by value; round-trip them
    import pickle
    assert pickle.loads(pickle.dumps(mcmc)).bounds.shape == bounds.shape
    assert pickle.loads(pickle.dumps(pd)).indices == [0]


def test_proposal_distribution_configuration():
    from fvgp.gp_mcmc import ProposalDistribution

    # no proposal distribution at all
    try:
        ProposalDistribution([0], proposal_dist=None, adapt_callable="normal")
    except Exception as e:
        assert "No proposal distribution specified" in str(e)
    else:
        raise AssertionError("a missing proposal distribution must be rejected")

    # the normal default warns when no covariance is supplied
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pd = ProposalDistribution([0, 1], proposal_dist="normal")
    assert any("normal proposal distribution" in str(w.message) for w in caught)
    assert pd.prop_args["prop_Sigma"].shape == (2, 2)

    # a callable adapter is used as given
    def my_adapt(end, mcmc_obj):
        return None

    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([1.0]), adapt_callable=my_adapt)
    assert pd.adapt is my_adapt

    # an unknown adapter string is rejected; anything else falls back to no adaptation
    try:
        ProposalDistribution([0], init_prop_Sigma=np.diag([1.0]),
                             proposal_dist=lambda x0, hps, obj: x0, adapt_callable="nonsense")
    except Exception as e:
        assert "Invalid string provided for adapt" in str(e)
    else:
        raise AssertionError("an unknown adapt string must be rejected")

    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([1.0]),
                              proposal_dist=lambda x0, hps, obj: x0, adapt_callable=None)
    assert pd._no_adapt(0, None) is None


def test_mcmc_default_break_condition_needs_a_long_chain():
    from fvgp import gpMCMC, ProposalDistribution

    bounds = np.array([[-1., 1.]])
    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([0.05]), adapt_callable="normal")
    mcmc = gpMCMC(lambda x, args: -float(x @ x), bounds=bounds,
                  prior_function=lambda t, b, a: 0., proposal_distributions=[pd])
    mcmc.run_mcmc(x0=np.array([0.0]), n_updates=5, break_condition="default")
    # under 1000 updates the default break condition never fires
    assert len(mcmc.trace["f(x)"]) == 4


def test_wendland_support_aware_filters_points_outside_the_radius():
    """query_ball_tree can return a point whose recomputed distance is just outside 1."""
    from fvgp.kernels import _wendland_support_aware_cpu_triplets

    hps = np.array([1.0, 1.0, 1.0])
    x1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    x2 = np.array([[0.5, 0.0], [1.0, 0.0]])
    values, rows, cols = _wendland_support_aware_cpu_triplets(x1, x2, hps)
    # the pair at distance exactly 1 contributes nothing; the one at 0.5 does
    assert values.size == 2
    assert set(cols.tolist()) == {0}


def test_fvgp_append_with_list_inputs_and_noise():
    x2 = np.random.rand(10, 2)
    y2 = np.column_stack([np.sin(np.linalg.norm(x2, axis=1)),
                          np.cos(np.linalg.norm(x2, axis=1))])
    v2 = np.full(y2.shape, 0.01)
    gp = fvGP(x2, y2, np.array([1., 1., 1., 1.]), noise_variances=v2)

    x_add = np.random.rand(2, 2)
    y_add = np.column_stack([np.sin(np.linalg.norm(x_add, axis=1)),
                             np.cos(np.linalg.norm(x_add, axis=1))])
    gp.update_gp_data(x_add, y_add, np.full(y_add.shape, 0.01), append=True)
    assert len(gp.fvgp_noise_variances) == 12

    # noise variances are always arrays; missing tasks are signalled with np.nan
    try:
        gp.update_gp_data(x_add, y_add, "bad noise", append=True)
    except (Exception, AssertionError) as e:
        assert "must be np.ndarray" in str(e)
    else:
        raise AssertionError("a non-array noise must be rejected")


def test_prior_rejects_unsupported_kernel_and_mean_arities():
    gp = _tiny_gp()
    gp.prior.k_n_params = 9
    try:
        gp.prior.compute_covariances(gp.x_data, gp.x_data, gp.hyperparameters)
    except Exception as e:
        assert "No valid kernel function signature" in str(e)
    else:
        raise AssertionError("an unsupported kernel arity must be rejected")

    gp.prior.m_n_params = 9
    try:
        gp.prior.compute_mean(gp.x_data, gp.hyperparameters)
    except Exception as e:
        assert "No valid mean function signature" in str(e)
    else:
        raise AssertionError("an unsupported mean arity must be rejected")


def _string_distance(s1, s2):
    difference = abs(len(s1) - len(s2))
    common = min(len(s1), len(s2))
    s1, s2 = s1[:common], s2[:common]
    for i in range(len(s1)):
        if s1[i] != s2[i]: difference += 1.
    return difference


def _string_kernel(x1, x2, hps):
    """Single-task: the points are the objects themselves."""
    d = np.zeros((len(x1), len(x2)))
    for i, a in enumerate(x1):
        for j, b in enumerate(x2):
            d[i, j] = _string_distance(a, b)
    return hps[0] * matern_kernel_diff1(d, hps[1])


def _string_kernel_multi_task(x1, x2, hps):
    """Multi-task: each point is a [object, task] pair from the index-set transform."""
    d = np.zeros((len(x1), len(x2)))
    for i, a in enumerate(x1):
        for j, b in enumerate(x2):
            d[i, j] = _string_distance(a[0], b[0])
    return hps[0] * matern_kernel_diff1(d, hps[1])


def test_non_euclidean_single_task_end_to_end():
    """A non-Euclidean input space is a flat list of arbitrary objects."""
    x_data = ['hello', 'world', 'this', 'is', 'fvgp']
    y_data = np.array([2., 1.9, 1.8, 3.0, 5.])
    gp = GP(x_data, y_data, init_hyperparameters=np.ones(2),
            kernel_function=_string_kernel)
    assert gp.prior.Euclidean is False

    assert np.isfinite(gp.posterior_mean(['full'])["m(x)"]).all()
    assert np.all(gp.posterior_covariance(['full'])["v(x)"] >= 0.0)

    # appending must accept the same flat-list form the constructor took
    gp.update_gp_data(['newone'], np.array([3.0]), append=True)
    assert len(gp.x_data) == 6 and gp.x_data[-1] == 'newone'
    assert len(gp.prior.m) == 6 and gp.prior.K.shape == (6, 6)
    assert np.isfinite(gp.posterior_mean(['full'])["m(x)"]).all()

    # and overwriting, too
    gp.update_gp_data(['a', 'bb'], np.array([1.0, 2.0]), append=False)
    assert len(gp.x_data) == 2


def test_non_euclidean_multi_task_end_to_end():
    """Non-Euclidean and multi-task together: the product space is [object, task]."""
    x_data = ['frf', 'ferfe', 'ferf', 'febhn']
    y_data = np.random.rand(len(x_data), 5)
    gp = fvGP(x_data, y_data, init_hyperparameters=np.ones(2),
              kernel_function=_string_kernel_multi_task)

    assert len(gp.x_data) == len(x_data) * 5
    assert gp.x_data[0][0] == 'frf' and gp.x_data[0][1] == 0
    assert gp.prior.Euclidean is False

    gp.train(hyperparameter_bounds=np.array([[0.001, 100.], [0.001, 100.]]), max_iter=2)

    x_pred = ["dwed", "dwe"]
    assert gp.posterior_mean(x_pred, x_out=np.array([0, 1, 2, 3]))["m(x)"].shape == (2, 4)
    assert gp.posterior_mean(x_pred)["m(x)"].shape == (2, 5)
    assert gp.posterior_covariance(x_pred)["v(x)"].shape == (2, 5)

    # appending takes the flat list of new objects, one row of tasks each
    gp.update_gp_data(['newstring'], np.random.rand(1, 5), append=True)
    assert len(gp.fvgp_x_data) == 5
    assert len(gp.x_data) == 25
    assert len(gp.prior.m) == 25 and gp.prior.K.shape == (25, 25)
    assert gp.posterior_mean(x_pred)["m(x)"].shape == (2, 5)


def test_marginal_likelihood_gradient_without_ram_economy():
    """The non-ram-economy gradient path with a user kernel gradient per direction."""
    xx = np.random.rand(12, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    hps = np.array([1.0, 0.5, 0.5])

    def kernel(x1, x2, hps):
        return hps[0] * np.exp(-get_anisotropic_distance_matrix(x1, x2, hps[1:]) ** 2)

    def kernel_grad(x1, x2, hps, direction):
        eps = 1e-6
        up, down = np.array(hps, dtype=float), np.array(hps, dtype=float)
        up[direction] += eps
        down[direction] -= eps
        return (kernel(x1, x2, up) - kernel(x1, x2, down)) / (2 * eps)

    gp = GP(xx, yy, hps, kernel_function=kernel, kernel_function_grad=kernel_grad,
            ram_economy=True)
    grad = gp.neg_log_likelihood_gradient(hyperparameters=hps)
    assert grad.shape == (3,) and np.all(np.isfinite(grad))


###########################################################################
###################### gp_lin_alg: CPU-side branches ######################
###########################################################################
def _spd(n=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    return A @ A.T + n * np.eye(n)


def test_non_positive_definite_messages_are_diagnostic():
    """A failed factorization must say why, not just that it failed."""
    from fvgp.gp_lin_alg import calculate_Chol_factor, NonPositiveDefiniteError

    bad = np.array([[1.0, 2.0], [2.0, 1.0]])           # indefinite
    try:
        calculate_Chol_factor(bad)
    except NonPositiveDefiniteError as e:
        text = str(e)
        assert "2" in text                              # the dimension
        assert "diag" in text.lower() or "symmetr" in text.lower()
    else:
        raise AssertionError("an indefinite matrix must raise NonPositiveDefiniteError")


def test_rank_1_update_reports_a_non_positive_discriminant():
    from fvgp.gp_lin_alg import cholesky_update_rank_1, NonPositiveDefiniteError

    L = np.linalg.cholesky(_spd(4))
    b = np.zeros(4)
    c = -1e6                                            # drives the discriminant negative
    try:
        cholesky_update_rank_1(L, b, c)
    except NonPositiveDefiniteError as e:
        assert "discriminant" in str(e).lower() or "positive" in str(e).lower()
    else:
        raise AssertionError("a negative discriminant must raise NonPositiveDefiniteError")


def test_linalg_entry_points_reject_an_unknown_compute_device():
    from fvgp.gp_lin_alg import (calculate_Chol_factor, calculate_Chol_solve,
                                 calculate_Chol_logdet, calculate_inv_from_chol,
                                 cholesky_update_rank_1, matmul, matmul3)

    A = _spd(5)
    L = np.linalg.cholesky(A)
    b = np.ones((5, 1))
    for call in (lambda: calculate_Chol_factor(A, compute_device="quantum"),
                 lambda: calculate_Chol_solve(L, b, compute_device="quantum"),
                 lambda: calculate_Chol_logdet(L, compute_device="quantum"),
                 lambda: calculate_inv_from_chol(L, compute_device="quantum"),
                 lambda: cholesky_update_rank_1(L, np.ones(5), 1.0, compute_device="quantum"),
                 lambda: matmul(A, A, compute_device="quantum"),
                 lambda: matmul3(A, A, A, compute_device="quantum")):
        try:
            call()
        except Exception as e:
            assert "compute device" in str(e)
        else:
            raise AssertionError("an unknown compute device must be rejected")


def test_gpu_requests_fall_back_to_the_cpu_without_a_backend():
    """Asking for a GPU on a CPU-only machine must degrade, not crash."""
    from fvgp.gp_lin_alg import (calculate_logdet, calculate_inv, matmul, matmul3,
                                 cholesky_update_rank_1, calculate_inv_from_chol,
                                 get_gpu_engine, _cupy_gpu_available, _imate_gpu_enabled)

    A = _spd(6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(calculate_logdet(A, compute_device="gpu"),
                          np.linalg.slogdet(A)[1])
        assert np.allclose(calculate_inv(A, compute_device="gpu"), np.linalg.inv(A), atol=1e-8)
        assert np.allclose(matmul(A, A, compute_device="gpu"), A @ A)
        assert np.allclose(matmul3(A, A, A, compute_device="gpu"), A @ A @ A)
        L = np.linalg.cholesky(A)
        assert np.allclose(calculate_inv_from_chol(L, compute_device="gpu"),
                           np.linalg.inv(A), atol=1e-8)
        assert cholesky_update_rank_1(L, np.ones(6), 1.0, compute_device="gpu") is None

    # backend detection itself must answer rather than raise
    assert _cupy_gpu_available() in (True, False)
    assert _imate_gpu_enabled() in (True, False)
    for request in ("torch", "cupy", None):
        assert get_gpu_engine({"GPU_engine": request} if request else {}) in (None, "torch", "cupy")


def test_solve_handles_dtypes_singular_systems_and_bad_methods():
    from fvgp.gp_lin_alg import solve

    A = _spd(5).astype(np.float64)
    b = np.ones((5, 1), dtype=np.float32)
    assert solve(A, b).shape == (5, 1)                  # b is cast to A's dtype

    singular = np.zeros((3, 3))
    singular[0, 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = solve(singular, np.ones((3, 1)))             # least-squares fallback
    assert x.shape == (3, 1) and np.all(np.isfinite(x))

    try:
        solve(A, np.ones((5, 1)), compute_device="telepathy")
    except Exception as e:
        assert "No valid solve method" in str(e)
    else:
        raise AssertionError("an unknown compute device must be rejected by solve")


def test_sparse_preconditioner_guidance_covers_every_type():
    from fvgp.gp_lin_alg import sparse_preconditioner_failure_guidance

    for kind in ("ilu", "ic", "block_jacobi", "additive_schwarz", "amg", object()):
        text = sparse_preconditioner_failure_guidance({"sparse_preconditioner_type": kind})
        assert isinstance(text, str) and text


def test_krylov_helpers_normalize_their_inputs():
    from fvgp.gp_lin_alg import (_resolve_krylov_maxiter, _normalize_initial_guess,
                                 _column_initial_guess, _apply_linear_operator,
                                 _resolve_krylov_mode, is_sparse)

    assert _resolve_krylov_maxiter({"sparse_krylov_maxiter": None}) is None
    assert _resolve_krylov_maxiter({"sparse_krylov_maxiter": 7}) == 7

    # a 1-d guess becomes a column, and an over-long one is truncated
    assert _normalize_initial_guess(np.ones(4), (4, 1)).shape == (4, 1)
    assert _normalize_initial_guess(np.ones((9, 1)), (4, 1)).shape == (4, 1)
    try:
        _normalize_initial_guess(np.ones((4, 5)), (4, 2))
    except ValueError as e:
        assert "initial guess" in str(e).lower() or "shape" in str(e).lower()
    else:
        raise AssertionError("an incompatible initial guess must be rejected")

    assert _column_initial_guess(None, 0) is None
    assert _column_initial_guess(np.ones((4, 2)), 1).shape in ((4,), (4, 1))

    assert _apply_linear_operator(None, np.empty((0, 2))).shape == (0, 2)

    try:
        _resolve_krylov_mode({"sparse_krylov_mode": "nonsense"})
    except ValueError as e:
        assert "nonsense" in str(e)
    else:
        raise AssertionError("an unknown Krylov mode must be rejected")

    # is_sparse is a density test (<1% nonzero), not a type test
    mostly_zero = np.zeros((40, 40))
    mostly_zero[0, 0] = 1.0
    assert is_sparse(mostly_zero) is True
    assert is_sparse(np.ones((40, 40))) is False


def test_block_conjugate_gradient_single_and_multi_rhs():
    from fvgp.gp_lin_alg import _block_conjugate_gradient

    A = sparse.csr_matrix(_spd(15, seed=3))
    dense = A.toarray()

    single = np.ones((15, 1))
    X, code = _block_conjugate_gradient(A, single, 1e-10)
    assert X.shape == (15, 1)
    assert np.allclose(dense @ X, single, atol=1e-6)

    multi = np.random.default_rng(4).random((15, 3))
    X, code = _block_conjugate_gradient(A, multi, 1e-10)
    assert X.shape == (15, 3)
    assert np.allclose(dense @ X, multi, atol=1e-6)

    # an exactly-solved right-hand side returns immediately
    X0, code0 = _block_conjugate_gradient(A, np.zeros((15, 2)), 1e-10)
    assert np.allclose(X0, 0.0) and code0 == 0


def test_sparse_conj_grad_falls_back_when_block_cg_fails(monkeypatch):
    import fvgp.gp_lin_alg as la

    A = sparse.csr_matrix(_spd(10, seed=6))
    b = np.random.default_rng(7).random((10, 2))

    def exploding_block_cg(*a, **k):
        raise RuntimeError("block CG broke down")

    monkeypatch.setattr(la, "_block_conjugate_gradient", exploding_block_cg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = la.calculate_sparse_conj_grad(A, b, args={"sparse_krylov_mode": "block"})
    assert x.shape == (10, 2)
    assert np.allclose(A.toarray() @ x, b, atol=1e-5)
    assert any("falling back to columnwise CG" in str(w.message) for w in caught)


def test_shifted_dense_cholesky_gives_up_with_a_clear_error():
    from fvgp.gp_lin_alg import _shifted_dense_cholesky

    hopeless = np.full((3, 3), -1e30)
    try:
        _shifted_dense_cholesky(hopeless, args={"sparse_preconditioner_shift_attempts": 2})
    except np.linalg.LinAlgError as e:
        assert "shifted retries" in str(e)
    else:
        raise AssertionError("an unfixable matrix must report that the retries failed")


def test_ic0_factor_error_paths():
    from fvgp.gp_lin_alg import _build_ic0_factor

    # a missing diagonal entry
    missing_diagonal = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 2.0]]))
    try:
        _build_ic0_factor(missing_diagonal, args={"sparse_preconditioner_shift_attempts": 1})
    except np.linalg.LinAlgError as e:
        assert "IC(0)" in str(e)
    else:
        raise AssertionError("IC(0) on a matrix with a zero diagonal must fail loudly")


def test_dtype_adapted_operator_handles_matrices():
    from fvgp.gp_lin_alg import _build_dtype_adapted_operator

    def solve32(v):
        return v * 2.0

    op = _build_dtype_adapted_operator((3, 3), solve32, factor_dtype=np.float64)
    assert np.allclose(op.matmat(np.ones((3, 2), dtype=np.float32)), 2.0)
    assert np.allclose(op.matvec(np.ones(3, dtype=np.float32)), 2.0)


def test_ilu_preconditioner_honors_its_tuning_arguments():
    from fvgp.gp_lin_alg import calculate_sparse_preconditioner

    A = sparse.csr_matrix(_spd(12, seed=8))
    _, operator = calculate_sparse_preconditioner(A, args={
        "sparse_preconditioner_type": "ilu",
        "sparse_preconditioner_drop_rule": "basic",
        "sparse_preconditioner_permc_spec": "NATURAL",
        "sparse_preconditioner_diag_pivot_thresh": 0.1,
    })
    assert operator.shape == (12, 12)
    assert np.all(np.isfinite(operator.matvec(np.ones(12))))


def test_unknown_preconditioner_type_is_rejected():
    from fvgp.gp_lin_alg import calculate_sparse_preconditioner

    A = sparse.csr_matrix(_spd(6, seed=9))
    try:
        calculate_sparse_preconditioner(A, args={"sparse_preconditioner_type": "telekinesis"})
    except ValueError as e:
        assert "telekinesis" in str(e)
    else:
        raise AssertionError("an unknown preconditioner type must be rejected")


###########################################################################
############### gp_lin_alg + gp_kv: the last branches #####################
###########################################################################
def test_torch_device_resolution_without_a_gpu():
    """Every GPU_device request must resolve to a device or to None, never raise."""
    from fvgp.gp_lin_alg import _torch_gpu_device

    assert _torch_gpu_device({"GPU_device": "cuda"}) is None      # no CUDA here
    assert _torch_gpu_device({"GPU_device": "mps"}) is None       # no MPS here
    assert _torch_gpu_device({"GPU_device": "not-a-device"}) is None
    cpu_device = _torch_gpu_device({"GPU_device": "cpu"})
    assert cpu_device is None or str(cpu_device) == "cpu"


def test_solvers_cast_the_right_hand_side_to_the_factor_dtype():
    from fvgp.gp_lin_alg import (calculate_Chol_factor, calculate_Chol_solve,
                                 calculate_sparse_LU_factor, calculate_LU_solve)

    A = _spd(6)
    factor = calculate_Chol_factor(A)
    x = calculate_Chol_solve(factor, np.ones((6, 1), dtype=np.float32))
    assert x.shape == (6, 1) and np.all(np.isfinite(x))

    lu = calculate_sparse_LU_factor(sparse.csr_matrix(A))
    x = calculate_LU_solve(lu, np.ones((6, 1), dtype=np.float32))
    assert x.shape == (6, 1) and np.all(np.isfinite(x))


def test_linalg_entry_points_accept_an_unrecognized_device_as_cpu():
    """calculate_logdet / calculate_inv / matmul treat anything else as the CPU path."""
    from fvgp.gp_lin_alg import calculate_logdet, calculate_inv, matmul, matmul3

    A = _spd(5)
    assert np.isclose(calculate_logdet(A, compute_device="other"), np.linalg.slogdet(A)[1])
    assert np.allclose(calculate_inv(A, compute_device="other"), np.linalg.inv(A), atol=1e-8)
    assert np.allclose(matmul(A, A), A @ A)
    assert np.allclose(matmul3(A, A, A), A @ A @ A)


def test_resolve_gp2scale_linalg_mode_passes_callables_through():
    from fvgp.gp_lin_alg import resolve_gp2scale_linalg_mode

    triple = (lambda KV: KV, lambda f, b: b, lambda f: 0.0)
    mode, args = resolve_gp2scale_linalg_mode(triple, {})
    assert mode is triple


def test_preconditioner_operator_and_initial_guess_helpers():
    from fvgp.gp_lin_alg import (_apply_preconditioner, _apply_linear_operator,
                                 _column_initial_guess)
    from scipy.sparse.linalg import LinearOperator

    residual = np.ones((4, 2))
    assert np.allclose(_apply_preconditioner(None, residual), residual)

    doubling = LinearOperator((4, 4), matvec=lambda v: 2.0 * v)
    assert np.allclose(_apply_preconditioner(doubling, residual), 2.0 * residual)

    # an operator whose matmat fails must fall back to column-by-column application
    class _BrokenMatmat:
        shape = (4, 4)
        def matmat(self, X): raise RuntimeError("matmat unsupported")
        def __matmul__(self, v): return 3.0 * v

    assert np.allclose(_apply_linear_operator(_BrokenMatmat(), residual), 3.0 * residual)

    # a 1-d initial guess is shared by every column
    guess = np.ones(4)
    assert _column_initial_guess(guess, 3) is guess


def test_amg_preconditioner_honors_its_tuning_arguments():
    pyamg = pytest.importorskip("pyamg")
    from fvgp.gp_lin_alg import calculate_sparse_preconditioner

    A = sparse.csr_matrix(_spd(20, seed=11))
    _, operator = calculate_sparse_preconditioner(A, args={
        "sparse_preconditioner_type": "amg",
        "sparse_preconditioner_amg_strength": "symmetric",
        "sparse_preconditioner_amg_symmetry": "hermitian",
        "sparse_preconditioner_amg_presmoother": ("gauss_seidel", {"sweep": "symmetric"}),
        "sparse_preconditioner_amg_postsmoother": ("gauss_seidel", {"sweep": "symmetric"}),
        "sparse_preconditioner_amg_cycle": "W",
    })
    assert np.all(np.isfinite(operator.matvec(np.ones(20))))


def test_ilupp_shift_retry_gives_up_with_a_labelled_error():
    pytest.importorskip("ilupp")
    from fvgp.gp_lin_alg import _shift_retry_ilupp_factor

    def always_fails(A):
        raise RuntimeError("nope")

    A = sparse.csr_matrix(_spd(5, seed=12))
    try:
        _shift_retry_ilupp_factor(A, always_fails, "ICholT",
                                  {"sparse_preconditioner_shift_attempts": 2})
    except np.linalg.LinAlgError as e:
        assert "ICholT" in str(e) and "shifted retries" in str(e)
    else:
        raise AssertionError("an unbuildable preconditioner must report the label and retries")


def test_ic0_rejects_a_non_positive_pivot():
    from fvgp.gp_lin_alg import _build_ic0_factor

    # symmetric, has a full diagonal, but is indefinite: the pivot goes non-positive
    indefinite = sparse.csr_matrix(np.array([[1.0, 4.0], [4.0, 1.0]]))
    try:
        _build_ic0_factor(indefinite, args={"sparse_preconditioner_shift_attempts": 1})
    except np.linalg.LinAlgError as e:
        assert "IC(0)" in str(e)
    else:
        raise AssertionError("a non-positive IC(0) pivot must be reported")


def test_kv_chol_and_inv_updates_with_a_shrinking_matrix():
    """update_KV must refactorize from scratch when the matrix has not grown."""
    for mode in ("Chol", "CholInv", "Inv"):
        gp = _tiny_gp(linalg_mode=mode)
        K, V, m = gp.kv._get_KVm()
        KV = gp.kv.addKV(K, V)
        smaller = KV[:10, :10]
        gp.kv.update_KV(smaller)
        if mode == "Chol":
            assert gp.kv.Chol_factor.shape == (10, 10)
        elif mode == "Inv":
            assert gp.kv.KVinv.shape == (10, 10)


def test_kv_custom_callable_mode_end_to_end():
    """A 3-tuple of callables replaces the whole factorization interface."""
    calls = {"factor": 0, "solve": 0, "logdet": 0}

    def factorize(KV):
        calls["factor"] += 1
        return np.linalg.inv(KV.toarray() if sparse.issparse(KV) else KV)

    def do_solve(factor, b):
        calls["solve"] += 1
        return factor @ b

    def do_logdet(factor):
        calls["logdet"] += 1
        return -float(np.linalg.slogdet(factor)[1])

    gp = _tiny_gp(linalg_mode=(factorize, do_solve, do_logdet))
    K, V, m = gp.kv._get_KVm()
    KV = gp.kv.addKV(K, V)
    assert gp.kv.compute_new_KVinvY(KV, m).shape == gp.y_data.shape
    KVinvY, logdet = gp.kv.compute_new_KVlogdet_KVinvY(K, V, m)
    assert KVinvY.shape == gp.y_data.shape and np.isfinite(logdet)
    assert calls["factor"] > 0 and calls["solve"] > 0 and calls["logdet"] > 0


def test_addKV_rejects_a_two_dimensional_non_matrix():
    """Square and 2-d, but neither an ndarray nor scipy sparse (a torch tensor, say)."""
    from fvgp.gp_kv import GPkv

    class _NotAMatrix:
        ndim = 2
        shape = (2, 2)

    try:
        GPkv.addKV(_NotAMatrix(), np.array([0.1, 0.1]))
    except Exception as e:
        assert "not possible" in str(e)
    else:
        raise AssertionError("an unsupported K type must be rejected")


def test_kv_setstate_on_a_bare_instance_fills_in_defaults():
    from fvgp.gp_kv import GPkv

    gp = _tiny_gp()
    state = gp.kv.__getstate__()
    for attr in ("Preconditioner_factor", "Preconditioner_operator", "Preconditioner_signature",
                 "Preconditioner_KV_shape", "Preconditioner_reuse_counter",
                 "Last_preconditioner_error", "Preconditioner_fingerprint",
                 "Warm_start_fingerprint"):
        state.pop(attr, None)

    bare = GPkv.__new__(GPkv)
    bare.__setstate__(state)
    assert bare.Preconditioner_operator is None
    assert bare.Preconditioner_reuse_counter == 0
    assert bare.Warm_start_fingerprint is None


def test_preconditioner_build_failure_resets_the_cache(monkeypatch):
    import fvgp.gp_kv as kv_module

    gp = _tiny_gp(linalg_mode="sparseCGpre")
    K, V, m = gp.kv._get_KVm()
    KV = sparse.csr_matrix(gp.kv.addKV(K, V))

    def exploding_builder(KV, args=None):
        raise RuntimeError("preconditioner unavailable")

    monkeypatch.setattr(kv_module, "calculate_sparse_preconditioner", exploding_builder)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        operator = gp.kv._get_or_refresh_preconditioner(KV, force_refresh=True)
    assert operator is None
    assert any("Failed to build sparse preconditioner" in str(w.message) for w in caught)
    assert gp.kv.Preconditioner_operator is None
    assert gp.kv.Last_preconditioner_error is None or "unavailable" in str(gp.kv.Last_preconditioner_error)
    assert gp.kv._can_reuse_sparse_preconditioner(KV) is False


def test_gp2Scale_mode_selection_thresholds():
    """The automatic gp2Scale mode depends on size and sparsity."""
    gp = _tiny_gp()
    gp.data.gp2Scale = True
    dense_small = sparse.csr_matrix(np.ones((50, 50)))
    assert gp.kv._set_gp2Scale_mode(dense_small) == "Chol"


###########################################################################
###################### the last remaining branches ########################
###########################################################################
def test_second_gp2Scale_gp_on_one_client_is_refused(client):
    import gc
    rng = np.random.default_rng(2)
    xx = rng.random((30, 1))
    yy = np.sin(np.linalg.norm(xx, axis=1) * 5.0)
    first = GP(xx, yy, np.array([1.0, 0.5]), gp2Scale=True, gp2Scale_batch_size=15,
               dask_client=client, linalg_mode="Chol")
    try:
        GP(xx, yy, np.array([1.0, 0.5]), gp2Scale=True, gp2Scale_batch_size=15,
           dask_client=client, linalg_mode="Chol")
    except Exception as e:
        assert "already active on this dask client" in str(e)
    else:
        raise AssertionError("a second live gp2Scale GP on one client must be refused")
    finally:
        del first
        gc.collect()
        client.run(lambda: None)


def test_gp2Scale_client_bootstrap_requires_imate(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_imate(name, *a, **k):
        if name == "imate":
            raise ImportError("no imate")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_imate)
    gp = _tiny_gp()
    try:
        gp.initialize_gp2Scale_dask_client(True, None)
    except Exception as e:
        assert "install imate" in str(e)
    else:
        raise AssertionError("gp2Scale without imate must be reported")


def test_gp2Scale_client_bootstrap_creates_one_when_absent():
    gp = _tiny_gp()
    created = gp.initialize_gp2Scale_dask_client(True, None)
    try:
        assert created is not None
    finally:
        if created is not None:
            created.close()


def test_prior_selects_the_gpu_wendland_kernel(client):
    """compute_device='gpu' picks the GPU Wendland, which falls back to the CPU here."""
    import gc
    rng = np.random.default_rng(13)
    xx = rng.random((30, 1))
    yy = np.sin(np.linalg.norm(xx, axis=1) * 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GP(xx, yy, np.array([1.0, 0.5]), gp2Scale=True, gp2Scale_batch_size=15,
                compute_device="gpu", dask_client=client, linalg_mode="Chol")
    assert gp.prior.kernel is wendland_anisotropic_gp2Scale_gpu
    del gp
    gc.collect()
    client.run(lambda: None)


def test_mcmc_default_break_condition_on_a_long_stable_chain():
    from fvgp import gpMCMC, ProposalDistribution

    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([0.05]), adapt_callable="normal")
    mcmc = gpMCMC(_module_level_objective, bounds=np.array([[-1., 1.]]),
                  prior_function=_module_level_prior, proposal_distributions=[pd])
    mcmc.trace = {"f(x)": list(np.zeros(3000)), "x": [np.array([0.0])] * 3000,
                  "time stamp": [0.0] * 3000}
    assert mcmc._default_break_condition(mcmc) is True

    mcmc.trace["f(x)"] = list(np.arange(3000, dtype=float))
    assert mcmc._default_break_condition(mcmc) is False


def test_proposal_distribution_accepts_explicit_prop_args():
    from fvgp.gp_mcmc import ProposalDistribution

    # the normal adapter overwrites prop_Sigma with init_prop_Sigma on purpose
    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([1.0]),
                              adapt_callable="normal",
                              prop_args={"prop_Sigma": np.diag([2.0])})
    assert np.allclose(pd.prop_args["prop_Sigma"], np.diag([1.0]))
    assert "sigma_m" in pd.prop_args

    # a callable adapter leaves the supplied prop_args untouched
    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([1.0]),
                              proposal_dist=lambda x0, hps, obj: x0,
                              adapt_callable=lambda end, obj: None,
                              prop_args={"prop_Sigma": np.diag([2.0])})
    assert np.allclose(pd.prop_args["prop_Sigma"], np.diag([2.0]))


def test_hgdl_paths_validate_their_starting_point(client):
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    try:
        gp.trainer.hgdl_async(
            objective_function=gp.marginal_likelihood.neg_log_likelihood,
            objective_function_gradient=gp.marginal_likelihood.neg_log_likelihood_gradient,
            objective_function_hessian=gp.marginal_likelihood.neg_log_likelihood_hessian,
            hyperparameter_bounds=bounds,
            init_hyperparameters=np.array([500., 500., 500.]),
            dask_client=client)
    except Exception as e:
        assert "outside of optimization bounds" in str(e)
    else:
        raise AssertionError("hgdl must reject an out-of-bounds starting point")


def test_hgdl_reports_a_broken_objective(client):
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])

    class _BrokenOptimizer:
        def get_final(self):
            raise RuntimeError("objective blew up")

    try:
        gp.trainer.train(
            objective_function=gp.marginal_likelihood.neg_log_likelihood,
            objective_function_gradient=gp.marginal_likelihood.neg_log_likelihood_gradient,
            objective_function_hessian=gp.marginal_likelihood.neg_log_likelihood_hessian,
            hyperparameter_bounds=bounds,
            init_hyperparameters=np.array([1., 1., 1.]),
            method="hgdl", max_iter=1, dask_client=None)
    except Exception as e:
        assert "gone wrong" in str(e) or "dask" in str(e).lower() or "client" in str(e).lower()


def test_fvgp_index_set_transform_handles_list_inputs():
    """The product-space transform keeps list-valued (non-Euclidean) points as lists."""
    x2 = np.random.rand(4, 2)
    y2 = np.column_stack([np.sin(np.linalg.norm(x2, axis=1)),
                          np.cos(np.linalg.norm(x2, axis=1))])
    gp = fvGP(x2, y2, np.array([1., 1., 1., 1.]))

    x_list = [["a"], ["b"], ["c"]]
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    v = np.full(y.shape, 0.1)
    new_x, new_y, new_v = gp._transform_index_set2(x_list, y, v)
    assert isinstance(new_x, list) and len(new_x) == 6
    assert new_x[0] == [["a"], 0] and new_x[-1] == [["c"], 1]
    assert new_y.shape == (6,) and new_v.shape == (6,)


def test_marginal_likelihood_unpickles_onto_a_bare_instance():
    from fvgp.gp_marginal_likelihood import GPMarginalLikelihood

    gp = _tiny_gp()
    state = gp.marginal_likelihood.__getstate__()
    state.pop("_warm_start_KVinvY", None)
    bare = GPMarginalLikelihood.__new__(GPMarginalLikelihood)
    bare.__setstate__(state)
    assert bare._warm_start_KVinvY is None


def test_posterior_mean_returns_a_matrix_for_multi_column_solutions():
    """Several right-hand sides and no x_out: the mean keeps its columns."""
    gp = _tiny_gp()
    gp.kv.KVinvY = np.tile(gp.kv.KVinvY, (1, 2))
    result = gp.posterior_mean(np.random.rand(4, 2))
    assert result["m(x)"].shape == (4, 2)
    assert result["m(x)_flat"].shape == (4, 2)


def test_variance_only_multi_task_posterior_has_no_covariance_matrix():
    xx = np.random.rand(15, 2)
    yy = np.column_stack([np.sin(np.linalg.norm(xx, axis=1)),
                          np.cos(np.linalg.norm(xx, axis=1))])
    # the variance-only fast path exists only where the explicit inverse is cached
    gp = fvGP(xx, yy, np.array([1., 1., 1., 1.]), linalg_mode="Inv")
    result = gp.posterior_covariance(np.random.rand(4, 2), variance_only=True)
    assert result["v(x)"].shape == (4, 2)
    assert result["S"] is None


###########################################################################
######################## the final few branches ###########################
###########################################################################
def test_fvgp_update_rejects_bad_new_data_formats():
    x2 = np.random.rand(8, 2)
    y2 = np.column_stack([np.sin(np.linalg.norm(x2, axis=1)),
                          np.cos(np.linalg.norm(x2, axis=1))])
    gp = fvGP(x2, y2, np.array([1., 1., 1., 1.]), noise_variances=np.full(y2.shape, 0.01))

    x_add = np.random.rand(2, 2)
    y_add = np.column_stack([np.sin(np.linalg.norm(x_add, axis=1)),
                             np.cos(np.linalg.norm(x_add, axis=1))])
    try:
        gp.update_gp_data(x_add, y_add, {"not": "a noise array"}, append=True)
    except (Exception, AssertionError) as e:
        assert "format" in str(e) or "np.ndarray" in str(e)
    else:
        raise AssertionError("a dict noise must be rejected")


def test_gp2Scale_mode_selection_covers_every_threshold():
    gp = _tiny_gp()
    gp.data.gp2Scale = True
    n = len(gp.x_data)
    very_sparse = sparse.eye(n, format="csr")
    dense_small = sparse.csr_matrix(np.ones((n, n)))
    assert gp.kv._set_gp2Scale_mode(very_sparse) in ("sparseLU", "Chol", "sparseMINRES")
    assert gp.kv._set_gp2Scale_mode(dense_small) == "Chol"


def test_compute_new_KVinvY_uses_the_gp2Scale_mode(client):
    """Under gp2Scale the mode is re-derived per call from the sparsity of K+V."""
    import gc
    rng = np.random.default_rng(21)
    xx = rng.random((30, 1))
    yy = np.sin(np.linalg.norm(xx, axis=1) * 5.0)
    gp = GP(xx, yy, np.array([1.0, 0.5]), gp2Scale=True, gp2Scale_batch_size=15,
            dask_client=client, linalg_mode="Chol")
    K, V, m = gp.kv._get_KVm()
    KV = gp.kv.addKV(K, V)
    assert gp.kv.compute_new_KVinvY(KV, m).shape == gp.y_data.shape
    del gp
    gc.collect()
    client.run(lambda: None)


def test_compute_new_KVlogdet_rejects_an_unknown_mode():
    gp = _tiny_gp()
    K, V, m = gp.kv._get_KVm()
    gp.kv.mode = "not-a-mode"
    try:
        gp.kv.compute_new_KVlogdet_KVinvY(K, V, m)
    except Exception as e:
        assert "No mode" in str(e) or "Mode" in str(e)
    else:
        raise AssertionError("an unknown mode must be rejected by compute_new_KVlogdet_KVinvY")


def test_block_jacobi_and_schwarz_preconditioners_on_a_disconnected_graph():
    """Both partitioners must cope with a matrix whose graph falls apart."""
    from fvgp.gp_lin_alg import calculate_sparse_preconditioner

    # two disconnected clusters, so the BFS partitioner restarts and the overlap
    # expansion runs out of frontier
    block = _spd(6, seed=22)
    A = sparse.csr_matrix(np.block([[block, np.zeros((6, 6))],
                                    [np.zeros((6, 6)), block]]))
    for kind, extra in (("block_jacobi", {}),
                        ("additive_schwarz", {"sparse_preconditioner_overlap": 3})):
        args = {"sparse_preconditioner_type": kind,
                "sparse_preconditioner_block_size": 4}
        args.update(extra)
        _, operator = calculate_sparse_preconditioner(A, args=args)
        assert np.all(np.isfinite(operator.matvec(np.ones(12))))


def test_mcmc_accepts_a_hugely_favourable_proposal():
    """A very large likelihood ratio short-circuits the exponential."""
    from fvgp import gpMCMC, ProposalDistribution

    def steep(x, args):
        return -1e6 * float(x @ x)

    pd = ProposalDistribution([0], init_prop_Sigma=np.diag([1.0]), adapt_callable="normal")
    mcmc = gpMCMC(steep, bounds=np.array([[-5., 5.]]),
                  prior_function=_module_level_prior, proposal_distributions=[pd])
    mcmc.run_mcmc(x0=np.array([4.0]), n_updates=30)
    assert len(mcmc.trace["f(x)"]) == 29


def test_marginal_likelihood_gradient_ram_economy_with_matrix_noise():
    """ram_economy with a 2-d noise gradient takes the non-diagonal branch."""
    xx = np.random.rand(12, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    hps = np.array([1.0, 0.5, 0.5])

    def kernel(x1, x2, hps):
        return hps[0] * np.exp(-get_anisotropic_distance_matrix(x1, x2, hps[1:]) ** 2)

    def kernel_grad(x1, x2, hps, direction):
        eps = 1e-6
        up, down = np.array(hps, dtype=float), np.array(hps, dtype=float)
        up[direction] += eps
        down[direction] -= eps
        return (kernel(x1, x2, up) - kernel(x1, x2, down)) / (2 * eps)

    def noise(x, hps):
        return np.full(len(x), 0.01)

    def noise_grad(x, hps, direction):
        return np.zeros((len(x), len(x)))          # 2-d: a full matrix derivative

    gp = GP(xx, yy, hps, kernel_function=kernel, kernel_function_grad=kernel_grad,
            noise_function=noise, noise_function_grad=noise_grad, ram_economy=True)
    grad = gp.neg_log_likelihood_gradient(hyperparameters=hps)
    assert grad.shape == (3,) and np.all(np.isfinite(grad))


def test_bo_surrogate_helpers_handle_degenerate_inputs():
    from fvgp.gp_bo import _posterior_mean_var_and_grad, _polynomial_mean, bayesian_optimize

    # a constant objective: zero residual variance, zero spread, non-finite guards
    calls = {"n": 0}

    def constant(theta):
        calls["n"] += 1
        return 1.0

    theta, info = bayesian_optimize(constant, np.array([[0.1, 10.0], [0.1, 10.0]]),
                                    np.array([1.0, 1.0]), max_iter=12,
                                    bo_args={"seed": 0, "patience": 0})
    assert theta.shape == (2,)
    assert info["n_evaluations"] == calls["n"]
    assert np.all(np.isfinite(info["sensitivity"]))


def test_bo_stops_when_asked():
    from fvgp.gp_bo import bayesian_optimize

    def objective(theta):
        return float(np.sum(theta ** 2))

    # stop once the space-filling design is done, so there is something to report
    seen = {"n": 0}

    def stop_after_the_design():
        seen["n"] += 1
        return seen["n"] > 1

    theta, info = bayesian_optimize(objective, np.array([[0.1, 10.0]]), np.array([1.0]),
                                    max_iter=200, bo_args={"seed": 0, "patience": 0},
                                    early_stop=stop_after_the_design)
    assert info["stopped early"] is True
    assert theta.shape == (1,)


def test_bo_handles_a_non_finite_objective():
    from fvgp.gp_bo import bayesian_optimize

    def sometimes_infinite(theta):
        return np.inf if theta[0] > 5.0 else float(theta[0])

    theta, info = bayesian_optimize(sometimes_infinite, np.array([[0.1, 10.0]]),
                                    np.array([1.0]), max_iter=12,
                                    bo_args={"seed": 0, "patience": 0})
    assert np.all(np.isfinite(info["trace f(x)"]))


def test_dask_client_bootstrap_survives_a_failing_client(monkeypatch):
    """When gp2Scale cannot start a local client, it logs and hands back None."""
    import fvgp.gp as gp_module

    def failing_client(*a, **k):
        raise RuntimeError("no scheduler available")

    monkeypatch.setattr(gp_module, "Client", failing_client)
    gp = _tiny_gp()
    assert gp.initialize_gp2Scale_dask_client(True, None) is None


def test_hgdl_final_result_failure_is_reported(monkeypatch):
    """If HGDL cannot produce a final result, say so rather than leaking a KeyError."""
    import fvgp.gp_training as training_module

    class _BrokenHGDL:
        def __init__(self, *a, **k): pass
        def optimize(self, *a, **k): return None
        def get_final(self): raise RuntimeError("objective blew up")

    monkeypatch.setattr(training_module, "HGDL", _BrokenHGDL)
    gp = _tiny_gp()
    bounds = np.array([[0.01, 10.], [0.01, 10.], [0.01, 10.]])
    try:
        gp.trainer.train(
            objective_function=gp.marginal_likelihood.neg_log_likelihood,
            objective_function_gradient=gp.marginal_likelihood.neg_log_likelihood_gradient,
            objective_function_hessian=gp.marginal_likelihood.neg_log_likelihood_hessian,
            hyperparameter_bounds=bounds,
            init_hyperparameters=np.array([1., 1., 1.]),
            method="hgdl", max_iter=1)
    except Exception as e:
        assert "gone wrong" in str(e)
    else:
        raise AssertionError("a broken HGDL result must be reported")


def test_wendland_support_aware_empty_neighbor_lists():
    """Overlapping bounding boxes, but no pair actually inside the support."""
    from fvgp.kernels import _wendland_support_aware_cpu_triplets

    hps = np.array([1.0, 1.0, 1.0])
    x1 = np.array([[0.0, 0.0], [1.4, 1.4]])
    x2 = np.array([[0.0, 1.4], [1.4, 0.0]])
    values, rows, cols = _wendland_support_aware_cpu_triplets(x1, x2, hps)
    assert values.size == 0 and rows.size == 0 and cols.size == 0


def test_graph_block_partitioner_on_a_disconnected_graph():
    """A block that fills up mid-BFS leaves queued-but-assigned nodes to skip, and
    an overlap expansion runs out of frontier once the component is exhausted."""
    from fvgp.gp_lin_alg import _build_graph_blocks, _expand_block_overlap

    # a path graph: BFS fills a block and abandons queued neighbours to the next seed
    n = 12
    path = sparse.diags([np.ones(n - 1), np.full(n, 4.0), np.ones(n - 1)],
                        [-1, 0, 1], format="csr")
    blocks = _build_graph_blocks(path, block_size=3)
    assert sum(len(b) for b in blocks) == n
    assert sorted(np.concatenate(blocks).tolist()) == list(range(n))

    # asking for more overlap than the component has exhausts the frontier
    expanded = _expand_block_overlap(path, np.array([0]), overlap=50)
    assert len(expanded) == n


def test_block_cg_reports_a_breakdown_on_dependent_right_hand_sides():
    """Duplicate columns make the block Gram matrix singular; CG must exit, not raise."""
    from fvgp.gp_lin_alg import _block_conjugate_gradient

    A = sparse.csr_matrix(_spd(12, seed=31))
    column = np.random.default_rng(32).random((12, 1))
    dependent = np.hstack([column, column, column])
    X, exit_code = _block_conjugate_gradient(A, dependent, 1e-14, maxiter=50)
    assert X.shape == (12, 3)
    assert exit_code in (0, 2, 3)


def test_gp2Scale_mode_falls_through_to_sparse_minres():
    gp = _tiny_gp()
    gp.data.gp2Scale = True
    gp.data.x_data = np.zeros((3000, 1))          # large, and not sparse enough for LU
    assert gp.kv._set_gp2Scale_mode(sparse.eye(3000, format="csr")) == "sparseMINRES"


def test_bo_variance_floor_and_acquisition_without_a_gradient():
    from fvgp.gp_bo import _posterior_mean_var_and_grad, _maximize_acquisition

    # an acquisition with no analytic gradient falls back to a gradient-free optimizer
    rng = np.random.default_rng(33)

    def acq(u):
        u = np.atleast_2d(u)
        return -np.sum((u - 0.5) ** 2, axis=1)

    best_u, best_v = _maximize_acquisition(acq, 2, rng, n_restarts=2, n_raw=32)
    assert best_u.shape == (2,) and np.isfinite(best_v)


def test_bo_gradient_is_flattened_where_the_variance_hits_its_floor():
    """At an observed point the posterior variance collapses; the reported gradient of
    the variance must go to zero rather than to numerical noise."""
    from fvgp.gp_bo import _fit_surrogate, _posterior_mean_var_and_grad

    rng = np.random.default_rng(34)
    u = rng.random((8, 2))
    y = np.sum(u ** 2, axis=1)
    v = np.zeros(len(u))
    gp = _fit_surrogate(u, y, v, 2, 50)
    mean, var, d_mean, d_var = _posterior_mean_var_and_grad(u[0], gp, 2)
    assert var > 0.0
    assert d_mean.shape == (2,) and d_var.shape == (2,)


def test_marginal_likelihood_reports_a_gradient_failure_in_ram_economy_mode():
    xx = np.random.rand(10, 2)
    yy = np.sin(np.linalg.norm(xx, axis=1))
    hps = np.array([1.0, 0.5, 0.5])

    def kernel(x1, x2, hps):
        return hps[0] * np.exp(-get_anisotropic_distance_matrix(x1, x2, hps[1:]) ** 2)

    def kernel_grad(x1, x2, hps, direction):
        raise RuntimeError("gradient unavailable")

    gp = GP(xx, yy, hps, kernel_function=kernel, kernel_function_grad=kernel_grad,
            ram_economy=True)
    try:
        gp.neg_log_likelihood_gradient(hyperparameters=hps)
    except Exception as e:
        assert "ram_economy" in str(e)
    else:
        raise AssertionError("a ram-economy gradient failure must be reported")


def test_graph_partitioner_skips_already_assigned_neighbours():
    """A queued neighbour can be claimed by an earlier block before it is popped."""
    from fvgp.gp_lin_alg import _build_graph_blocks

    # a star: the hub is queued by several leaves, but only one block may claim it
    n = 8
    A = np.eye(n) * 5.0
    A[0, 1:] = 1.0
    A[1:, 0] = 1.0
    blocks = _build_graph_blocks(sparse.csr_matrix(A), block_size=2)
    assert sum(len(b) for b in blocks) == n
    assert sorted(np.concatenate(blocks).tolist()) == list(range(n))


def test_block_cg_breaks_down_in_the_beta_solve():
    """A rank-deficient block makes the second Gram solve singular too."""
    from fvgp.gp_lin_alg import _block_conjugate_gradient

    A = sparse.csr_matrix(_spd(20, seed=41))
    column = np.random.default_rng(42).random((20, 1))
    dependent = np.hstack([column] * 4)
    X, exit_code = _block_conjugate_gradient(A, dependent, 1e-16, maxiter=200)
    assert X.shape == (20, 4) and exit_code in (0, 2, 3)


def test_bo_objective_that_is_never_finite():
    from fvgp.gp_bo import bayesian_optimize

    theta, info = bayesian_optimize(lambda t: np.inf, np.array([[0.1, 10.0]]),
                                    np.array([1.0]), max_iter=10,
                                    bo_args={"seed": 0, "patience": 0})
    assert np.all(np.isfinite(info["trace f(x)"])), "non-finite values must be clamped"


def test_bo_early_stop_between_iterations():
    from fvgp.gp_bo import bayesian_optimize

    seen = {"n": 0}

    def stop_soon():
        seen["n"] += 1
        return seen["n"] > 2

    theta, info = bayesian_optimize(lambda t: float(np.sum(t ** 2)),
                                    np.array([[0.1, 10.0], [0.1, 10.0]]),
                                    np.array([1.0, 1.0]), max_iter=500,
                                    bo_args={"seed": 0, "patience": 0},
                                    early_stop=stop_soon)
    assert info["stopped early"] is True
    assert info["n_evaluations"] < 500


def test_bo_sensitivity_falls_back_to_the_length_scales(monkeypatch):
    """When the Laplace curvature cannot be formed, sensitivity comes from the ARD scales."""
    import fvgp.gp_bo as bo_module

    def no_laplace(gp, u_best, tf):
        raise RuntimeError("curvature unavailable")

    monkeypatch.setattr(bo_module, "_laplace_posterior", no_laplace)
    theta, info = bo_module.bayesian_optimize(
        lambda t: float(np.sum(np.log(t) ** 2)), np.array([[0.1, 10.0], [0.1, 10.0]]),
        np.array([1.0, 1.0]), max_iter=12, bo_args={"seed": 0, "patience": 0})
    assert np.all(np.isfinite(info["sensitivity"]))
    assert np.allclose(info["sensitivity"], 1.0 / np.maximum(info["ard length scales"], 1e-12))


###########################################################################
########### non-Euclidean points really are arbitrary objects #############
###########################################################################
def _object_label(o):
    """A point may be any object; pull a comparable label out of it."""
    if isinstance(o, dict): return o["name"]
    if isinstance(o, (list, tuple)): return o[0]
    if hasattr(o, "name"): return o.name
    return o


class _PointObject:
    def __init__(self, name): self.name = name


def _label_kernel(x1, x2, hps):
    d = np.zeros((len(x1), len(x2)))
    for i, a in enumerate(x1):
        for j, b in enumerate(x2):
            d[i, j] = 0.0 if _object_label(a) == _object_label(b) else 1.0
    return hps[0] * matern_kernel_diff1(d, hps[1])


def _label_kernel_multi_task(x1, x2, hps):
    d = np.zeros((len(x1), len(x2)))
    for i, a in enumerate(x1):
        for j, b in enumerate(x2):
            d[i, j] = 0.0 if _object_label(a[0]) == _object_label(b[0]) else 1.0
    return hps[0] * matern_kernel_diff1(d, hps[1])


_OBJECT_POINT_SETS = {
    "strings":        ([ 'aa', 'bb', 'cc' ],                         ['dd']),
    "lists":          ([['aa'], ['bb'], ['cc']],                     [['dd']]),
    "longer lists":   ([['aa', 1], ['bb', 2], ['cc', 3]],            [['dd', 4]]),
    "ragged lists":   ([['aa'], ['bb', 'x'], ['cc', 'y', 'z']],      [['dd', 'w']]),
    "tuples":         ([('aa',), ('bb',), ('cc',)],                  [('dd',)]),
    "dicts":          ([{'name': 'aa'}, {'name': 'bb'}, {'name': 'cc'}], [{'name': 'dd'}]),
    "objects":        ([_PointObject('aa'), _PointObject('bb'), _PointObject('cc')],
                       [_PointObject('dd')]),
}


def test_non_euclidean_points_of_any_object_type():
    """The input space is a list of *arbitrary* objects -- including objects that are
    themselves sequences, and sequences of differing length. Nothing may introspect
    them with numpy, which would either mis-read or refuse a ragged list."""
    y = np.array([1.0, 2.0, 3.0])
    for name, (x, x_new) in _OBJECT_POINT_SETS.items():
        gp = GP(x, y, np.ones(2), kernel_function=_label_kernel)
        assert gp.prior.Euclidean is False, name
        assert np.isfinite(gp.posterior_mean(x_new)["m(x)"]).all(), name
        assert np.all(gp.posterior_covariance(x_new)["v(x)"] >= 0.0), name

        gp.update_gp_data(x_new, np.array([4.0]), append=True)
        assert len(gp.x_data) == 4, name
        assert gp.prior.K.shape == (4, 4), name
        assert np.isfinite(gp.posterior_mean(x_new)["m(x)"]).all(), name


def test_non_euclidean_multi_task_points_of_any_object_type():
    for name, (x, x_new) in _OBJECT_POINT_SETS.items():
        y = np.random.rand(3, 2)
        gp = fvGP(x, y, np.ones(2), kernel_function=_label_kernel_multi_task)
        assert len(gp.x_data) == 6, name
        assert np.isfinite(gp.posterior_mean(x_new)["m(x)"]).all(), name

        gp.update_gp_data(x_new, np.random.rand(1, 2), append=True)
        assert len(gp.x_data) == 8, name
        assert gp.prior.K.shape == (8, 8), name


def test_gp2Scale_with_non_euclidean_object_points(client):
    """gp2Scale scatters the point set to the workers. A list is scattered element-wise
    by dask unless it is kept together, so this is where object-typed points break
    first -- and the result must still match the dense computation."""
    import gc

    letters = "abcdefghij"
    x = [[letters[i % 10]] * (1 + i % 3) for i in range(30)]   # ragged, non-Euclidean
    y = np.random.rand(30)

    def compact_kernel(x1, x2, hps):
        d = np.zeros((len(x1), len(x2)))
        for i, a in enumerate(x1):
            for j, b in enumerate(x2):
                d[i, j] = 0.0 if a[0] == b[0] else 2.0
        d[d > 1.] = 1.
        return hps[0] * (1. - d) ** 8 * (32. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)

    hps = np.array([1.0, 0.5])
    gp = GP(x, y, hps, gp2Scale=True, gp2Scale_batch_size=10, dask_client=client,
            kernel_function=compact_kernel, linalg_mode="Chol")
    assert sparse.issparse(gp.prior.K)
    # the distributed assembly must equal the direct evaluation
    assert np.allclose(gp.prior.K.toarray(), compact_kernel(x, x, hps))

    assert np.isfinite(gp.posterior_mean([['a'], ['b', 'b']])["m(x)"]).all()
    gp.update_gp_data([['z', 'z', 'z']], np.array([0.5]), append=True)
    assert len(gp.x_data) == 31 and gp.prior.K.shape == (31, 31)

    del gp
    gc.collect()
    client.run(lambda: None)


###########################################################################
############ an unsatisfiable GPU request must be reported ################
###########################################################################
def test_gpu_engine_unavailable_reason_distinguishes_the_failure_modes():
    """A user needs to tell 'the package is missing' from 'the package is there but
    there is no device' -- the fix is different."""
    from fvgp.gp_lin_alg import gpu_engine_unavailable_reason
    import importlib

    for engine in ("torch", "pytorch"):
        reason = gpu_engine_unavailable_reason(engine)
        if importlib.util.find_spec("torch") is None:
            assert reason is not None and "not installed" in reason
        else:
            # installed here, so either it works or it has no device -- never "not installed"
            assert reason is None or "no usable CUDA or MPS device" in reason

    reason = gpu_engine_unavailable_reason("cupy")
    if importlib.util.find_spec("cupy") is None:
        assert reason == "cupy is not installed"

    unknown = gpu_engine_unavailable_reason("tensorflow")
    assert "not a supported GPU engine" in unknown and "tensorflow" in unknown


def test_requested_gpu_engine_warns_when_it_cannot_be_honored():
    """Silently computing on the CPU after the user asked for a GPU helps nobody."""
    from fvgp.gp_lin_alg import get_gpu_engine

    for requested in ("cupy", "torch", "pytorch", "tensorflow"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine = get_gpu_engine({"GPU_engine": requested})
        if engine is None:
            messages = [str(w.message) for w in caught]
            assert any(f"`{requested}` GPU engine" in m for m in messages), requested
            assert any("Falling back to the CPU" in m for m in messages), requested
        else:
            assert engine in ("torch", "cupy")


def test_gpu_request_without_an_engine_warns_when_nothing_is_usable():
    from fvgp.gp_lin_alg import get_gpu_engine

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = get_gpu_engine({})
    if engine is None:
        assert any("no usable GPU backend was found" in str(w.message) for w in caught)


def test_gpu_engine_aliases_resolve_to_the_same_backend():
    """'pytorch' is what the docs call it; 'torch' is what the arg took."""
    from fvgp.gp_lin_alg import _GPU_ENGINE_ALIASES

    assert _GPU_ENGINE_ALIASES["pytorch"] == "torch"
    assert _GPU_ENGINE_ALIASES["torch"] == "torch"
    assert _GPU_ENGINE_ALIASES["cupy"] == "cupy"


def test_gpu_kernels_take_args_so_the_engine_request_reaches_them():
    """The GPU Wendland kernels resolve their backend through the same helper, which
    means they honor args['GPU_engine'] -- and warn when they cannot."""
    import inspect
    from fvgp.kernels import (wendland_anisotropic_gp2Scale_gpu,
                              wendland_anisotropic_gp2Scale_gpu_sparse,
                              wendland_anisotropic_gp2Scale_cpu)

    for kernel in (wendland_anisotropic_gp2Scale_gpu, wendland_anisotropic_gp2Scale_gpu_sparse):
        params = list(inspect.signature(kernel).parameters)
        assert params == ["x1", "x2", "hps", "args"], kernel.__name__

    x = np.random.rand(6, 2)
    hps = np.array([1.0, 0.5, 0.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = wendland_anisotropic_gp2Scale_gpu(x, x, hps, {"GPU_engine": "cupy"})
    if not _gpu_engines_available():
        assert any("`cupy` GPU engine" in str(w.message) for w in caught)
        assert np.allclose(out, wendland_anisotropic_gp2Scale_cpu(x, x, hps))


def test_gpu_wendland_kernel_arity_is_seen_by_the_prior(client):
    """A four-argument kernel means GPprior passes `args` through, so the engine
    request survives all the way to the distributed workers."""
    import gc
    rng = np.random.default_rng(0)
    x = rng.random((30, 1))
    y = np.sin(np.linalg.norm(x, axis=1) * 5.0)
    hps = np.array([1.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GP(x, y, hps, gp2Scale=True, gp2Scale_batch_size=15, compute_device="gpu",
                dask_client=client, linalg_mode="Chol", args={"GPU_engine": "cupy"})
        assert gp.prior.k_n_params == 4, "args must reach the kernel"
        assert gp.prior.kernel is wendland_anisotropic_gp2Scale_gpu
        # no GPU here, so the kernel fell back to the CPU implementation -- correctly
        assert np.allclose(gp.prior.K.toarray(),
                           wendland_anisotropic_gp2Scale_cpu(x, x, hps))
    del gp
    gc.collect()
    client.run(lambda: None)


def test_imate_gpu_gate_does_not_consult_torch_or_cupy():
    """imate ships its own CUDA backend. Gating it on pytorch/cupy would switch off a
    perfectly usable GPU on a machine where neither package can see one."""
    import inspect
    from fvgp.gp_lin_alg import _imate_gpu_enabled

    assert list(inspect.signature(_imate_gpu_enabled).parameters) == []
    source = inspect.getsource(_imate_gpu_enabled)
    assert "torch" not in source.split('"""')[2], "must not consult torch"
    assert "_cupy_gpu_available" not in source, "must not consult cupy"

    enabled = _imate_gpu_enabled()
    try:
        from imate.device import get_num_gpu_devices
        assert enabled == (int(get_num_gpu_devices()) > 0)
    except ImportError:                                   # pragma: no cover
        assert enabled is False


def test_missing_pytorch_is_reported_as_such(monkeypatch):
    """The 'not installed' branch, which a machine with pytorch cannot otherwise reach."""
    import importlib
    import fvgp.gp_lin_alg as la

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torch" else real_find_spec(name, *a, **k))
    assert la.gpu_engine_unavailable_reason("torch") == "pytorch is not installed"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert la.get_gpu_engine({"GPU_engine": "pytorch"}) is None
    assert any("pytorch is not installed" in str(w.message) for w in caught)


def test_gpu_engine_unavailable_reason_distinguishes_the_failure_modes():
    """A user needs to tell 'the package is missing' from 'the package is there but
    there is no device' -- the fix is different."""
    from fvgp.gp_lin_alg import gpu_engine_unavailable_reason
    import importlib

    for engine in ("torch", "pytorch"):
        reason = gpu_engine_unavailable_reason(engine)
        if importlib.util.find_spec("torch") is None:
            assert reason is not None and "not installed" in reason
        else:
            # installed here, so either it works or it has no device -- never "not installed"
            assert reason is None or "no usable CUDA or MPS device" in reason

    reason = gpu_engine_unavailable_reason("cupy")
    if importlib.util.find_spec("cupy") is None:
        assert reason == "cupy is not installed"

    unknown = gpu_engine_unavailable_reason("tensorflow")
    assert "not a supported GPU engine" in unknown and "tensorflow" in unknown


def test_requested_gpu_engine_warns_when_it_cannot_be_honored():
    """Silently computing on the CPU after the user asked for a GPU helps nobody."""
    from fvgp.gp_lin_alg import get_gpu_engine

    for requested in ("cupy", "torch", "pytorch", "tensorflow"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine = get_gpu_engine({"GPU_engine": requested})
        if engine is None:
            messages = [str(w.message) for w in caught]
            assert any(f"`{requested}` GPU engine" in m for m in messages), requested
            assert any("Falling back to the CPU" in m for m in messages), requested
        else:
            assert engine in ("torch", "cupy")


def test_gpu_request_without_an_engine_warns_when_nothing_is_usable():
    from fvgp.gp_lin_alg import get_gpu_engine

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = get_gpu_engine({})
    if engine is None:
        assert any("no usable GPU backend was found" in str(w.message) for w in caught)


def test_gpu_engine_aliases_resolve_to_the_same_backend():
    """'pytorch' is what the docs call it; 'torch' is what the arg took."""
    from fvgp.gp_lin_alg import _GPU_ENGINE_ALIASES

    assert _GPU_ENGINE_ALIASES["pytorch"] == "torch"
    assert _GPU_ENGINE_ALIASES["torch"] == "torch"
    assert _GPU_ENGINE_ALIASES["cupy"] == "cupy"


def test_gpu_kernels_take_args_so_the_engine_request_reaches_them():
    """The GPU Wendland kernels resolve their backend through the same helper, which
    means they honor args['GPU_engine'] -- and warn when they cannot."""
    import inspect
    from fvgp.kernels import (wendland_anisotropic_gp2Scale_gpu,
                              wendland_anisotropic_gp2Scale_gpu_sparse,
                              wendland_anisotropic_gp2Scale_cpu)

    for kernel in (wendland_anisotropic_gp2Scale_gpu, wendland_anisotropic_gp2Scale_gpu_sparse):
        params = list(inspect.signature(kernel).parameters)
        assert params == ["x1", "x2", "hps", "args"], kernel.__name__

    x = np.random.rand(6, 2)
    hps = np.array([1.0, 0.5, 0.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = wendland_anisotropic_gp2Scale_gpu(x, x, hps, {"GPU_engine": "cupy"})
    if not _gpu_engines_available():
        assert any("`cupy` GPU engine" in str(w.message) for w in caught)
        assert np.allclose(out, wendland_anisotropic_gp2Scale_cpu(x, x, hps))


def test_gpu_wendland_kernel_arity_is_seen_by_the_prior(client):
    """A four-argument kernel means GPprior passes `args` through, so the engine
    request survives all the way to the distributed workers."""
    import gc
    rng = np.random.default_rng(0)
    x = rng.random((30, 1))
    y = np.sin(np.linalg.norm(x, axis=1) * 5.0)
    hps = np.array([1.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GP(x, y, hps, gp2Scale=True, gp2Scale_batch_size=15, compute_device="gpu",
                dask_client=client, linalg_mode="Chol", args={"GPU_engine": "cupy"})
        assert gp.prior.k_n_params == 4, "args must reach the kernel"
        assert gp.prior.kernel is wendland_anisotropic_gp2Scale_gpu
        # no GPU here, so the kernel fell back to the CPU implementation -- correctly
        assert np.allclose(gp.prior.K.toarray(),
                           wendland_anisotropic_gp2Scale_cpu(x, x, hps))
    del gp
    gc.collect()
    client.run(lambda: None)


def test_imate_gpu_gate_does_not_consult_torch_or_cupy():
    """imate ships its own CUDA backend. Gating it on pytorch/cupy would switch off a
    perfectly usable GPU on a machine where neither package can see one."""
    import inspect
    from fvgp.gp_lin_alg import _imate_gpu_enabled

    assert list(inspect.signature(_imate_gpu_enabled).parameters) == []
    source = inspect.getsource(_imate_gpu_enabled)
    assert "torch" not in source.split('"""')[2], "must not consult torch"
    assert "_cupy_gpu_available" not in source, "must not consult cupy"

    enabled = _imate_gpu_enabled()
    try:
        from imate.device import get_num_gpu_devices
        assert enabled == (int(get_num_gpu_devices()) > 0)
    except ImportError:                                   # pragma: no cover
        assert enabled is False


def test_missing_pytorch_is_reported_as_such(monkeypatch):
    """The 'not installed' branch, which a machine with pytorch cannot otherwise reach."""
    import importlib
    import fvgp.gp_lin_alg as la

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torch" else real_find_spec(name, *a, **k))
    assert la.gpu_engine_unavailable_reason("torch") == "pytorch is not installed"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert la.get_gpu_engine({"GPU_engine": "pytorch"}) is None
    assert any("pytorch is not installed" in str(w.message) for w in caught)


def test_stochastic_logdet_says_when_imate_cannot_use_a_gpu(monkeypatch):
    import fvgp.gp_lin_alg as la

    monkeypatch.setattr(la, "_imate_gpu_enabled", lambda: False)
    KV = sparse.csr_matrix(_spd(20, seed=51))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logdet = la.calculate_random_logdet(KV, "gpu")
    assert np.isfinite(logdet)
    assert any("imate reports no usable GPU device" in str(w.message) for w in caught)


def test_imate_import_does_not_silence_every_later_warning():
    """imate calls logging.captureWarnings(True) on import, which sends every Python
    warning to a NullHandler -- silently, process-wide. fvGP imports imate on the first
    stochastic log-determinant, so without undoing that, every gp2Scale run would go
    quiet from then on, taking the GPU-request warnings with it."""
    import fvgp.gp_lin_alg as la

    before = warnings.showwarning
    imate_logdet = la._import_imate_logdet()
    assert callable(imate_logdet)
    assert warnings.showwarning is before, "imate's global warning capture must be undone"

    # a warning raised after the import must still be visible
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        la._import_imate_logdet()
        warnings.warn("still visible", UserWarning)
    assert any("still visible" in str(w.message) for w in caught)


def test_stochastic_logdet_warning_survives_the_imate_import():
    import fvgp.gp_lin_alg as la

    la._import_imate_logdet()          # flush imate's import side effects first
    original = la._imate_gpu_enabled
    la._imate_gpu_enabled = lambda: False
    try:
        KV = sparse.csr_matrix(_spd(20, seed=51))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logdet = la.calculate_random_logdet(KV, "gpu")
        assert np.isfinite(logdet)
        assert any("imate reports no usable GPU device" in str(w.message) for w in caught)
    finally:
        la._imate_gpu_enabled = original


###########################################################################
################ preconditioners are logged and timed #####################
###########################################################################
def _capture_fvgp_debug_log():
    """Collect fvGP's debug records; fvGP disables its logger at import."""
    from loguru import logger
    sink = []
    logger.enable("fvgp")
    handle = logger.add(lambda m: sink.append(str(m)), level="DEBUG", format="{message}")
    return logger, handle, sink


def test_every_preconditioner_logs_its_construction_time():
    """The solvers announce themselves and report a compute time; a preconditioner is
    just as expensive, so it reports one in the same shape."""
    from fvgp.gp_lin_alg import calculate_sparse_preconditioner

    KV = sparse.csr_matrix(_spd(40, seed=61))
    types = ["ilu", "block_jacobi", "additive_schwarz", "native_incomplete_cholesky"]
    if _importlib_for_tests.util.find_spec("pyamg") is not None:
        types.append("amg")
    if _importlib_for_tests.util.find_spec("ilupp") is not None:
        types.append("ichol0")

    logger, handle, sink = _capture_fvgp_debug_log()
    try:
        for kind in types:
            calculate_sparse_preconditioner(KV, args={"sparse_preconditioner_type": kind})
    finally:
        logger.remove(handle)
        logger.disable("fvgp")

    text = "\n".join(sink)
    for kind in types:
        assert f"{kind} preconditioner construction in progress" in text, kind
        assert f"{kind} preconditioner compute time:" in text, kind
    # the size of the problem is on the line, so a time can be judged against it
    assert "n = 40" in text and "K+V nnz =" in text


def test_a_reused_preconditioner_says_so_instead_of_going_quiet():
    """A cached preconditioner produces no construction time. Without a line saying it
    was reused, that gap reads as a preconditioner that never ran."""
    gp = _tiny_gp(linalg_mode="sparseCGpre",
                  args={"sparse_preconditioner_refresh_interval": 5})
    K, V, m = gp.kv._get_KVm()
    KV = sparse.csr_matrix(gp.kv.addKV(K, V))

    logger, handle, sink = _capture_fvgp_debug_log()
    try:
        gp.kv._get_or_refresh_preconditioner(KV, force_refresh=True)   # builds
        gp.kv._get_or_refresh_preconditioner(KV)                        # reuses
        gp.kv._get_or_refresh_preconditioner(KV)                        # reuses again
    finally:
        logger.remove(handle)
        logger.disable("fvgp")

    text = "\n".join(sink)
    assert "ilu preconditioner construction in progress" in text
    assert "ilu preconditioner reused (1 consecutive reuses)" in text
    assert "ilu preconditioner reused (2 consecutive reuses)" in text
    # build and reuse lines share the word, so one grep shows the whole story
    assert sum("preconditioner" in line for line in sink) >= 4


###########################################################################
########## task sizing: the RAM budget gp2Scale_batch_size implies ########
###########################################################################
def test_task_budget_is_per_distribution():
    """`B` is not the same physical quantity in the two modes: a B x B block against an
    N x B strip. At B=10000 and N=1e6 that is 0.8 GB against 80 GB."""
    from fvgp.gp2Scale_covariance import task_budget

    assert task_budget(1_000_000, 10000, "blockwise") == 10000 * 10000
    assert task_budget(1_000_000, 10000, "rowwise") == 1_000_000 * 10000
    # the row-wise budget scales with the dataset, the block-wise one does not
    assert task_budget(2_000_000, 10, "rowwise") == 2 * task_budget(1_000_000, 10, "rowwise")
    assert task_budget(2_000_000, 10, "blockwise") == task_budget(1_000_000, 10, "blockwise")
    assert task_budget(100, 0, "blockwise") == 1, "a zero batch size must not divide by zero"


def test_should_distribute_only_above_the_budget():
    from fvgp.gp2Scale_covariance import should_distribute, task_budget

    B, n1 = 10, 10000
    budget = task_budget(n1, B, "rowwise")                       # 100 000 entries
    assert should_distribute(n1, budget // n1, B, "rowwise") is False       # exactly at
    assert should_distribute(n1, budget // n1 + 1, B, "rowwise") is True    # just over

    # the reported cases: both are trivially small against the row-wise budget
    assert should_distribute(10000, 2, 10, "rowwise") is False    # posterior, 2 points
    assert should_distribute(10000, 5, 10, "rowwise") is False    # append, 5 points
    assert should_distribute(5, 5, 10, "rowwise") is False        # k(x_new, x_new)
    assert should_distribute(5, 5, 10, "blockwise") is False


def test_cross_covariance_strip_width_from_the_budget():
    """A strip spans the long axis, so its width is what the budget buys -- and it can
    never be narrower than 1, even when the budget cannot afford a single column."""
    from fvgp.gp2Scale_covariance import strip_width, task_budget, ranges

    # N=1e8, 8 prediction points, strip width 4 -> two tasks of 1e8 x 4
    n1, n2, B = 100_000_000, 8, 4
    w = strip_width(n1, n2, task_budget(n1, B, "rowwise"))
    assert w == 4
    assert [e - s for s, e in ranges(n2, w)] == [4, 4]

    # block-wise with a small B cannot afford even one full column; clamps to 1
    assert strip_width(10000, 2, task_budget(10000, 10, "blockwise")) == 1
    # a realistic block-wise setting lands exactly on budget
    assert strip_width(1_000_000, 500, task_budget(1_000_000, 10000, "blockwise")) == 100
    # never wider than the axis being split
    assert strip_width(10, 3, 10**9) == 3


def test_column_strip_worker_and_assembly():
    from fvgp.gp2Scale_covariance import col_strip_csc, assemble_col_strips

    x = np.sort(np.random.rand(40, 1), axis=0)
    xp = np.random.rand(6, 1)
    hps = np.array([1.0, 0.4])

    calls = []

    def recording_kernel(x1, x2, hps):
        calls.append((len(x1), len(x2)))
        return wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)

    start, strip = col_strip_csc((2, 5), x, xp, hps, recording_kernel, 3, None, 40, np.int32)
    assert calls == [(40, 3)], "a column strip spans every row in one call"
    assert start == 2 and strip.shape == (40, 3) and strip.format == "csc"
    assert np.allclose(strip.toarray(), wendland_anisotropic_gp2Scale_cpu(x, xp[2:5], hps))

    harvest = [col_strip_csc(r, x, xp, hps, wendland_anisotropic_gp2Scale_cpu,
                             3, None, 40, np.int32) for r in ((0, 3), (3, 6))]
    K = assemble_col_strips(iter(harvest), 40, 6)
    assert K.format == "csr"
    assert np.allclose(K.toarray(), wendland_anisotropic_gp2Scale_cpu(x, xp, hps))
    assert assemble_col_strips(iter([]), 4, 9).shape == (4, 9)


def test_empty_column_strip():
    from fvgp.gp2Scale_covariance import col_strip_csc

    x = np.random.rand(20, 1)
    far = np.random.rand(4, 1) + 50.0            # nothing within the Wendland support
    start, strip = col_strip_csc((0, 4), x, far, np.array([1.0, 0.1]),
                                 wendland_anisotropic_gp2Scale_cpu, 3, None, 20, np.int32)
    assert start == 0 and strip.shape == (20, 4) and strip.nnz == 0


def test_wide_cross_covariance_strips_along_rows(client):
    """When the short axis is the *rows* -- k(x_pred, x_data) rather than the usual
    orientation -- the strips run the other way."""
    from fvgp.gp2Scale_covariance import distributed_covariance

    rng = np.random.default_rng(17)
    xp = rng.random((9, 1))
    x = np.sort(rng.random((60, 1)), axis=0)
    hps = np.array([1.0, 0.3])
    f1 = client.scatter(xp, broadcast=True, direct=True, hash=False)
    f2 = client.scatter(x, broadcast=True, direct=True, hash=False)

    K = distributed_covariance(client, wendland_anisotropic_gp2Scale_cpu, hps,
                               x1_future=f1, n1=9, x2_future=f2, n2=60,
                               batch_size=3, symmetric=False, distribution="rowwise")
    assert K.shape == (9, 60)
    assert np.allclose(K.toarray(), wendland_anisotropic_gp2Scale_cpu(xp, x, hps))
    f1.release(); f2.release()


def test_cross_covariance_is_striped_under_both_distributions(client):
    """A cross-covariance is tall and thin, so it is always split into strips -- blocking
    it would multiply tasks without reducing per-task memory. Both settings must
    therefore agree exactly."""
    from fvgp.gp2Scale_covariance import distributed_covariance

    rng = np.random.default_rng(5)
    x = np.sort(rng.random((60, 1)), axis=0)
    xp = rng.random((9, 1))
    hps = np.array([1.0, 0.3])
    f1 = client.scatter(x, broadcast=True, direct=True, hash=False)
    f2 = client.scatter(xp, broadcast=True, direct=True, hash=False)

    reference = wendland_anisotropic_gp2Scale_cpu(x, xp, hps)
    results = {}
    for dist in ("blockwise", "rowwise"):
        K = distributed_covariance(client, wendland_anisotropic_gp2Scale_cpu, hps,
                                   x1_future=f1, n1=60, x2_future=f2, n2=9,
                                   batch_size=3, symmetric=False, distribution=dist)
        assert np.allclose(K.toarray(), reference), dist
        results[dist] = K
    assert abs(results["blockwise"] - results["rowwise"]).max() == 0.0
    f1.release(); f2.release()


def test_small_cross_covariances_never_reach_the_cluster(client, monkeypatch):
    """The reported pathology: a 10 000 x 2 posterior and a 5-point append used to become
    1000 dask tasks of 20 entries. Neither may touch the cluster now."""
    import gc
    import fvgp.gp_prior as prior_module

    def must_not_be_called(*a, **k):     # pragma: no cover - the point is that it is not
        raise AssertionError("this shape must not be distributed")

    rng = np.random.default_rng(9)
    x = rng.random((2000, 1))
    y = np.sin(np.linalg.norm(x, axis=1) * 5.0)
    gp = GP(x, y, np.array([1.0, 0.5]), gp2Scale=True, gp2Scale_batch_size=10,
            gp2Scale_distribution="rowwise", dask_client=client, linalg_mode="Chol")

    monkeypatch.setattr(prior_module, "distributed_covariance", must_not_be_called)

    k = gp.prior.compute_data_cross_covariance(rng.random((2, 1)), gp.hyperparameters)
    assert sparse.issparse(k) and k.shape == (2000, 2)

    x_add = rng.random((5, 1))
    gp.update_gp_data(x_add, np.sin(np.linalg.norm(x_add, axis=1) * 5.0), append=True)
    assert gp.prior.K.shape == (2005, 2005) and sparse.issparse(gp.prior.K)

    del gp
    gc.collect()
    client.run(lambda: None)
