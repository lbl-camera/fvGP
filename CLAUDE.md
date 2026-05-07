# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with test dependencies
pip install -e ".[tests]"

# Run the full test suite
pytest tests/

# Run a single test (tests are top-level functions, not class methods)
pytest tests/test_fvgp.py::test_single_task_init_basic

# Run tests with coverage
pytest tests --cov=./ --cov-report=xml

# Lint
flake8 fvgp tests

# Build docs
make docs
```

## Architecture

fvGP is a Gaussian Process library optimized for large-scale and multi-task settings. The two public-facing classes are:

- **`GP`** ([fvgp/gp.py](fvgp/gp.py)) — single-task GP; the primary entry point
- **`fvGP`** ([fvgp/fvgp.py](fvgp/fvgp.py)) — multi-task GP; inherits from `GP` and treats multi-task as a single-task over the Cartesian product of input × output space

Both classes are composed of internal specialist objects created at `__init__` time:

| Class | File | Responsibility |
|---|---|---|
| `GPdata` | [gp_data.py](fvgp/gp_data.py) | Data validation, shape tracking, Euclidean vs. non-Euclidean |
| `GPprior` | [gp_prior.py](fvgp/gp_prior.py) | Kernel and mean function; default is anisotropic Matérn with ARD |
| `GPlikelihood` | [gp_likelihood.py](fvgp/gp_likelihood.py) | Noise model (variances or callable) |
| `GPkv` | [gp_kv.py](fvgp/gp_kv.py) | Owns K+V matrix state and all factorizations; dispatches solves/logdets across linalg modes |
| `GPMarginalLikelihood` | [gp_marginal_likelihood.py](fvgp/gp_marginal_likelihood.py) | Log marginal likelihood and its gradient; delegates factorization to `GPkv` |
| `GPposterior` | [gp_posterior.py](fvgp/gp_posterior.py) | Posterior mean/covariance; information-theoretic quantities |
| `GPtraining` | [gp_training.py](fvgp/gp_training.py) | Hyperparameter optimization (scipy, hgdl async, MCMC, Adam) |

### Key supporting modules

- **[gp_lin_alg.py](fvgp/gp_lin_alg.py)** — CPU/GPU linear algebra primitives; Cholesky, LU, sparse solvers; defines `NonPositiveDefiniteError`
- **[gp_kv.py](fvgp/gp_kv.py)** — `GPkv` manages all K+V state across linalg modes: `"Chol"`, `"CholInv"`, `"Inv"`, `"sparseLU"`, `"sparseCG"`, `"sparseMINRES"`, and preconditioned variants. The mode is set at init and determines which factorization is updated when data or hyperparameters change. Custom solvers can be injected as a 3-tuple of callables.
- **[kernels.py](fvgp/kernels.py)** — 15+ built-in kernels including Matérn, squared exponential, Wendland (compactly supported)
- **[gp_mcmc.py](fvgp/gp_mcmc.py)** — Adaptive Metropolis–Hastings sampler used for Bayesian hyperparameter inference
- **[gp_actor.py](fvgp/gp_actor.py)** — `AsyncOptimizer` wraps `_MCMCActor` and `_AdamActor` for non-blocking background training; used by `GPtraining` for async MCMC and Adam modes

### Scaling to large datasets (`gp2Scale`)

When `gp2Scale=True`, `GP` switches to a Wendland (compactly supported) kernel producing sparse covariance matrices and uses Dask for distributed computation. This path requires a Dask client to be passed in and uses sparse linear solvers instead of dense Cholesky.

### Customization API

Kernels, mean functions, and noise models are all plain Python callables with standardized signatures. Users pass them as arguments to `GP`/`fvGP` constructors. The full hyperparameter vector is shared across kernel, mean, and noise callables, but each callable must only read its reserved index range. Kernel gradients can be user-supplied or computed via finite differences.

### Information-theoretic methods

`GP` exposes `gp_entropy()`, `gp_mutual_information()`, `gp_kl_div()`, and predictive metrics (`rmse`, `nlpd`, `crps`, `r2`, `picp`), all computed via `GPposterior`.

### Extension modules

- **[ggmp.py](fvgp/ggmp.py)** — `GGMP` (Gaussian GP for Gaussian Mixture data): fits K GMM components per station, each backed by its own `GP`; intended for distributional regression. Written by Vardaan Tekriwal. Excluded from the test suite (`# pragma: no cover`).
- **[deep_kernel_network.py](fvgp/deep_kernel_network.py)** — `Network` (PyTorch `nn.Module`): a 3-layer ReLU network used as a feature extractor for deep kernel learning. Excluded from the test suite (`# pragma: no cover`).

## Dependencies

Core: `numpy`, `scipy`, `dask`, `distributed`, `hgdl`, `loguru`

Optional GPU backend: `torch` or `cupy` (selected via `compute_device` parameter)
