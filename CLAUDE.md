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
| `GPdata` | [gp_data.py](fvgp/gp_data.py) | Data validation, shape tracking, Euclidean vs. non-Euclidean. Sole source of truth for `x_data`, `y_data`, `noise_variances`, plus the pre-append snapshot (`x_old`, `y_old`, `noise_variances_old`) and last-appended chunk (`x_new`, `y_new`, `noise_variances_new`) |
| `GPprior` | [gp_prior.py](fvgp/gp_prior.py) | Kernel and mean function (default: anisotropic Matérn with ARD). In gp2Scale mode also owns `x_data_scatter_future` (the persistent dask scatter of `x_data`) |
| `GPlikelihood` | [gp_likelihood.py](fvgp/gp_likelihood.py) | Noise model (variances or callable) |
| `GPkv` | [gp_kv.py](fvgp/gp_kv.py) | Owns K+V matrix state and all factorizations; dispatches solves/logdets across linalg modes |
| `GPMarginalLikelihood` | [gp_marginal_likelihood.py](fvgp/gp_marginal_likelihood.py) | Log marginal likelihood and its gradient; delegates factorization to `GPkv`. Maintains `_warm_start_KVinvY` for iterative training solves when `args["sparse_krylov_warm_start"]=True`. |
| `GPposterior` | [gp_posterior.py](fvgp/gp_posterior.py) | Posterior mean/covariance; information-theoretic quantities |
| `GPtraining` | [gp_training.py](fvgp/gp_training.py) | Hyperparameter optimization (scipy, hgdl async, MCMC, Adam) |

### State propagation

Sources of truth: `GPtraining.hyperparameters` and `GPdata.x_data` / `y_data` / `noise_variances`. Everywhere else reads these via `@property`. Cached state that must be invalidated on a change:

| Mutator | What's refreshed |
|---|---|
| `GP.set_hyperparameters(hps)` | `trainer.hyperparameters` → `prior.update_state_hyperparameters()` (recomputes `m`, `K`) → `likelihood.update_state()` (`V`) → `kv.update_state_hyperparameters()` (factorization + `KVinvY`) |
| `GP.update_gp_data(..., append=True)` | `data.update()` snapshots `x_old`/`y_old`/etc. → `prior.augment_state_data()` (rank-n update of `m`, `K`) → `likelihood.update_state()` → `kv.update_state_data(rank_n_update)` |
| `GP.update_gp_data(..., append=False)` | `data.update()` clears `_old`/`_new` slots → `prior.update_state_data()` (full recompute) → `likelihood.update_state()` → `kv.update_state_data(rank_n_update)` |
| `GP.train(...)` (sync) / `GP.update_hyperparameters(opt_obj)` (async) | both end with `set_hyperparameters(...)` |

`GPposterior` and `GPMarginalLikelihood` hold **no cached state** — every read goes through properties, so they're automatically consistent.

Gotchas:
- **`GP.set_args(new_args)` does NOT invalidate `K`, `m`, `V`, or factorizations.** If `args` flows into a user kernel/mean/noise callable, new args take effect only on the next `set_hyperparameters`, `update_gp_data(append=False)`, fresh `train`, or posterior call with explicit `hyperparameters=`. To force a flush: `set_hyperparameters(self.hyperparameters)`.
- **`update_gp_data(append=False, rank_n_update=True)`** is invalid (the previous factorization is for data that no longer exists); `GP.update_gp_data` emits a `UserWarning` and forces `rank_n_update=False`.
- **`kv.solve(b, x0=...)`** zero-pads `x0` along axis 0 when shapes don't match, so a pre-append `KVinvY` can warm-start the post-append solve in iterative modes (sparseCG/MINRES/preconditioned variants). See [gp_kv.py:333-342](fvgp/gp_kv.py#L333-L342).

### Key supporting modules

- **[gp_lin_alg.py](fvgp/gp_lin_alg.py)** — CPU/GPU linear algebra primitives; Cholesky, LU, sparse solvers; defines `NonPositiveDefiniteError`
- **[gp_kv.py](fvgp/gp_kv.py)** — `GPkv` manages all K+V state across linalg modes: `"Chol"`, `"CholInv"`, `"Inv"`, `"sparseLU"`, `"sparseCG"`, `"sparseMINRES"`, and preconditioned variants. The mode is set at init and determines which factorization is updated when data or hyperparameters change. Custom solvers can be injected as a 3-tuple of callables. For `sparseMINRESpre`/`sparseCGpre`, `GPkv` caches the preconditioner across `update_KV` / `compute_new_*` calls and rebuilds when `Preconditioner_reuse_counter` ≥ `args["sparse_preconditioner_refresh_interval"] - 1` or when the shape/`sparse_preconditioner_*` args fingerprint changes. `set_KV` always force-refreshes. Aliases like `"sparseCGpre_amg"` are resolved at `__init__` into the canonical mode plus `args["sparse_preconditioner_type"]`.
- **[kernels.py](fvgp/kernels.py)** — 15+ built-in kernels including Matérn, squared exponential, Wendland (compactly supported)
- **[gp_mcmc.py](fvgp/gp_mcmc.py)** — Adaptive Metropolis–Hastings sampler used for Bayesian hyperparameter inference
- **[gp_actor.py](fvgp/gp_actor.py)** — `AsyncOptimizer` wraps `_MCMCActor` and `_AdamActor` for non-blocking background training; used by `GPtraining` for async MCMC and Adam modes

### Scaling to large datasets (`gp2Scale`)

When `gp2Scale=True`, `GP` switches to a Wendland (compactly supported) kernel producing sparse covariance matrices and uses Dask for distributed computation. This path requires a Dask client to be passed in and uses sparse linear solvers instead of dense Cholesky.

**Scatter ownership and lifecycle:**

- `GPprior.x_data_scatter_future` is the single persistent dask scatter of the current `x_data`. Scattered once at `GPprior.__init__` (see [gp_prior.py:93-96](fvgp/gp_prior.py#L93-L96)).
- `GPdata` does NOT scatter — it's pure-Python data only.
- `_compute_prior_covariance_gp2Scale` reads `self.x_data_scatter_future` directly; **no scatter per call**, so training stays dask-quiet.
- On data changes, `augment_state_data` / `update_state_data` refresh the scatter by **overwriting** `self.x_data_scatter_future` (no explicit `release()`). The old future loses its only Python ref and is cleaned up via `__del__`. Calling `release()` explicitly schedules a `_dec_ref` that races against subsequent scatter `replicate` operations in the scheduler — don't do it.
- `_update_prior_covariance_gp2Scale` (the augment path) uses `self.x_data_scatter_future` for the `x_old` side (no content-hash collision since it shares the existing key) and scatters only `x_new` locally, releasing that local future at the end.

**Cross-instance race guard:** [gp.py:14-21](fvgp/gp.py#L14-L21) defines `_GP_INSTANCES_PER_CLIENT`, a `WeakValueDictionary` keyed by `dask_client.id`. `GP.__init__` ([gp.py:285-303](fvgp/gp.py#L285-L303)) raises with a descriptive remediation message if you try to construct a second gp2Scale `GP` on a client that already has a live one — that pattern reliably triggers `FutureCancelledError`/`KeyError` from the scheduler. To reuse a client for a sequence of GPs:

```python
import gc
del previous_gp
gc.collect()
client.run(lambda: None)  # flush pending releases
```

The `test_gp2Scale` test uses exactly this pattern between linalg-mode iterations.

### Iterative-solver acceleration (sparseCG / sparseMINRES / *pre modes)

For `sparseCG`, `sparseMINRES`, `sparseCGpre`, and `sparseMINRESpre`, the user can opt into two orthogonal accelerators via `args` on the `GP` constructor:

- **Preconditioner caching** (`sparseCGpre`/`sparseMINRESpre` only): `args["sparse_preconditioner_refresh_interval"] = N` reuses a single preconditioner for up to N consecutive `update_KV` / `compute_new_*` calls before rebuilding. Default `N=1` rebuilds on every call (same as no caching). `args["sparse_preconditioner_type"]` selects the kernel — `"ilu"` (default), `"ic"`/`"incomplete_cholesky"`, `"block_jacobi"`, `"schwarz"`/`"additive_schwarz"`, `"amg"` (requires pyamg). Mode aliases `"sparseCGpre_<type>"` / `"sparseMINRESpre_<type>"` set the type as a shortcut. Cache is invalidated automatically when `KV.shape` or any `sparse_preconditioner_*` arg changes.
- **Warm-start** (all iterative modes): `args["sparse_krylov_warm_start"] = True` makes `GPMarginalLikelihood` pass the previous training iteration's `KVinvY` as `x0` to the next iterative solve. Cuts iteration counts substantially when successive hyperparameter trials are close. Stored in `marginal_likelihood._warm_start_KVinvY`; reset to `None` on pickling.

Both default off so existing behavior is preserved.

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
