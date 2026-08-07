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

# Skip the slow distributed test while iterating
pytest tests/ -q --deselect tests/test_fvgp.py::test_gp2Scale

# Run tests with coverage (the package is at 100%; ggmp.py is omitted in pyproject)
pytest tests --cov=fvgp --cov-report=term-missing

# Optional compiled preconditioner backends (pyamg, ilupp). Not in [tests]: ilupp has no
# Linux wheels and pyamg none for 3.14, so both build from source. Their tests skip when
# absent, which costs coverage but not a green run.
pip install -e ".[preconditioners]"

# Lint
flake8 fvgp tests

# Build docs
make docs
```

There is no `setup.py` (build backend is hatchling + hatch-vcs, version written to [fvgp/_version.py](fvgp/_version.py)), so the `test`, `coverage`, `install`, and `dist` Makefile targets and `tox.ini` are stale — only `make docs` / `make lint` work. Use `pytest` and `hatch build` directly.

There is no `conftest.py`: [tests/test_fvgp.py](tests/test_fvgp.py) imports the `client` / `loop` / `cluster_fixture` fixtures from `distributed.utils_test` at module level, so any test taking a `client` argument spins up a real local Dask cluster.

**Coverage is 100% and worth keeping there.** `[tool.coverage]` in `pyproject.toml` omits `ggmp.py`. Two things to know before adding code:
- Anything that runs **on a dask worker** (the `gp_actor` classes, the `gp2Scale_covariance` worker functions) is not measured there. Test it in-process by calling it directly — they are all plain classes and plain functions over plain arrays.
- Use `--cov=fvgp`, not a dotted module target like `--cov=fvgp.gp2Scale_covariance`. The dotted form corrupts numpy's `_NoValue` sentinel on some setups and makes `ndarray.max()` raise `TypeError` inside scipy, which looks like a code bug and is not.

`# pragma: no cover` is used for what genuinely cannot run on a CPU-only runner — GPU backends, branches unreachable behind an earlier assert, numerical breakdowns that cannot be forced. Each one carries its reason on the same line.

**Python 3.10–3.14** are supported and all five are in CI. Dependencies are declared as ranges, not `~=` pins, because no single scipy/numpy release spans that range; pip resolves per interpreter (3.10 → scipy 1.13 / dask 2024.1 via hgdl 2.2.3; 3.11+ → the 2025-era stack). Note **hgdl `~=`-pins the whole scientific stack**, so it, not fvGP, decides which versions a given Python gets.

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
| `GPtraining` | [gp_training.py](fvgp/gp_training.py) | Hyperparameter optimization; owns both the sync (`train`) and async (`train_async`) dispatch over `global` / `local` / `hgdl` / `mcmc` / `adam` / `bo` |

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

### Hyperparameter training (`GP.train` / `train(asynchronous=True)`)

`GP.train` validates and normalizes everything (bounds, initial hyperparameters, objective + gradient + Hessian, gp2Scale and async restrictions) and then hands off to `GPtraining.train` or `GPtraining.train_async`. `method` is `"global"`, `"local"`, `"hgdl"`, `"mcmc"` (default), `"adam"`, `"bo"`, or a callable taking the `GP` and returning a hyperparameter vector.

- Async is supported for `hgdl`, `mcmc`, `adam`, `bo` and needs a `dask_client`; everything else warns and falls back to sync.
- gp2Scale trains synchronously and only allows `mcmc` or `bo` (other methods are silently switched to `mcmc`).
- `mcmc` ignores a user `objective_function` (it always maximizes the log marginal likelihood). A user objective with `local`/`hgdl` must come with a gradient.
- Diagnostics land on read-only `GP` properties: `mcmc_info` after `mcmc`, `bo_info` after a sync `bo` run.

**`method='bo'` ([gp_bo.py](fvgp/gp_bo.py))** — Bayesian optimization over the hyperparameters, for when the marginal likelihood is expensive, noisy, and effectively gradient-free (the gp2Scale/mBCG regime: stochastic-Lanczos log-determinant plus truncated CG). Points to know before touching it:

- **The surrogate is an `fvgp.GP`.** It deliberately uses fvGP and not gpCAM, which depends on fvGP — importing gpCAM here would be circular. The recursion bottoms out: the inner GP sees only the tens-to-hundreds of θ points evaluated, uses an ARD Matérn-5/2 kernel with analytic gradients, and trains with `method='local'` — never `'bo'`.
- **`max_iter` changes meaning**: for `bo` it is a *cap on objective-function evaluations*, not iterations. The run normally stops earlier on the `patience` / `f_rtol` / `x_tol` criteria in `bo_args`.
- **Observation noise is wired in automatically.** When the objective is the default, `GP.train` injects a `bo_args["noise_function"]` reading `GPMarginalLikelihood.log_likelihood_variance()` — the SLQ log-determinant's own precision (`0.25 * kv.last_logdet_variance`), or `None` in exact modes, where the surrogate instead learns a single homoscedastic noise level whose lower bound acts as a nugget.
- **The search space is log-transformed per dimension** (`_LogAffineTransform`): log where both bounds are strictly positive, linear otherwise, then rescaled to the unit cube. Positivity is only a *proxy* for being scale-like — a positive hyperparameter that enters additively (a center in a non-stationary/Gibbs kernel, a mixing weight) is hurt by it. Override with `bo_args['log_scale']`.
- `GP._warn_about_bo_suitability` warns before the run when the budget is too small for the initial design or the hyperparameter count is too high for BO.
- `bo_info` carries the payoff beyond the optimum: `sensitivity` (curvature-based ranking of which hyperparameters matter) and `posterior covariance` (Laplace approximation at the mode, in searched coordinates), both free of extra likelihood evaluations.

### Key supporting modules

- **[gp_lin_alg.py](fvgp/gp_lin_alg.py)** — CPU/GPU linear algebra primitives; Cholesky, LU, sparse solvers; defines `NonPositiveDefiniteError`
- **[gp_kv.py](fvgp/gp_kv.py)** — `GPkv` manages all K+V state across linalg modes: `"Chol"`, `"CholInv"`, `"Inv"`, `"sparseLU"`, `"sparseCG"`, `"sparseMINRES"`, and preconditioned variants. The mode is set at init and determines which factorization is updated when data or hyperparameters change. Custom solvers can be injected as a 3-tuple of callables. For `sparseMINRESpre`/`sparseCGpre`, `GPkv` caches the preconditioner across `update_KV` / `compute_new_*` calls and rebuilds when `Preconditioner_reuse_counter` ≥ `args["sparse_preconditioner_refresh_interval"] - 1` or when the shape/`sparse_preconditioner_*` args fingerprint changes. `set_KV` always force-refreshes. Aliases like `"sparseCGpre_amg"` are resolved at `__init__` into the canonical mode plus `args["sparse_preconditioner_type"]`.
- **[kernels.py](fvgp/kernels.py)** — 15+ built-in kernels including Matérn, squared exponential, Wendland (compactly supported)
- **[gp_mcmc.py](fvgp/gp_mcmc.py)** — Adaptive Metropolis–Hastings sampler used for Bayesian hyperparameter inference
- **[gp_actor.py](fvgp/gp_actor.py)** — `AsyncOptimizer` wraps `_MCMCActor`, `_AdamActor`, and `_BOActor` for non-blocking background training; used by `GPtraining` for the async MCMC, Adam, and BO modes
- **[gp_bo.py](fvgp/gp_bo.py)** — `bayesian_optimize` and the noisy-EI machinery behind `method='bo'`; see the training section above. Its module docstring is the design rationale and is worth reading before changing anything here.
- **[utils.py](fvgp/utils.py)** — `log_time` context manager for cumulative timing via loguru. Note fvGP calls `logger.disable('fvgp')` at import ([`__init__.py`](fvgp/__init__.py)); re-enable it to get the debug stream.

### Scaling to large datasets (`gp2Scale`)

When `gp2Scale=True`, `GP` switches to a Wendland (compactly supported) kernel producing sparse covariance matrices and uses Dask for distributed computation. This path requires a Dask client to be passed in and uses sparse linear solvers instead of dense Cholesky.

**One distributed covariance primitive.** Every kernel evaluation gp2Scale distributes goes through `distributed_covariance` in [gp2Scale_covariance.py](fvgp/gp2Scale_covariance.py) — the symmetric prior covariance, the rectangular `B` and symmetric `D` blocks of an append, and the posterior's `k(x_data, x_pred)`. They differ only in the `symmetric` flag. `GPprior._gp2Scale_covariance` is the sole caller; it owns scatter lifetime and nothing else. The module's docstring is the design rationale.

Two ways of cutting the work, via `GP(..., gp2Scale_distribution=...)`:

| | tasks | kernel evaluations | host assembly |
|---|---|---|---|
| `"blockwise"` (default) | (row block, col block) pairs, upper triangle only when symmetric | half | global COO + mirror, one preallocation |
| `"rowwise"` | row strips; workers return finished CSR | double (no symmetry) | `vstack` of strips — concatenation only |

Row-wise is the choice when host assembly, not kernel evaluation, is the bottleneck; it also caps host peak memory at the finished matrix plus one strip.

**Posterior at scale:** `GPposterior.cross_covariance` → `GPprior.compute_data_cross_covariance` returns a **sparse** `k`, so `posterior_mean` never materializes an `(N × n_pred)` array. `posterior_covariance` cannot avoid a dense solve (`KV⁻¹` is dense regardless), so `GPposterior._cross_solve_product` chunks over prediction points at `gp2Scale_batch_size` to cap the intermediate at `(N × chunk)`. `joint_gp_prior`, `joint_gp_prior_grad`, `gp_mutual_information` and `gp_total_correlation` build a joint `(N + n_pred)²` matrix and are dense-in-N by construction; they route `K` through `_dense_K()`, which warns under gp2Scale. `posterior_covariance_grad` is likewise dense-in-N via `d_kernel_dx` and was left on the direct path.

**Scatter ownership and lifecycle:**

- `GPprior.x_data_scatter_future` is the single persistent dask scatter of the current `x_data`. Scattered once at `GPprior.__init__`.
- `GPdata` does NOT scatter — it's pure-Python data only.
- The prior covariance reads `self.x_data_scatter_future` directly; **no scatter per call**, so training stays dask-quiet. The append path additionally scatters `x_new`, and the posterior path `x_pred`; both release what they created, in a `finally`.
- **All scatters go through `GPprior._scatter`, which does two non-obvious things.** It passes `hash=False`: dask otherwise keys scattered data by a content hash, so scattering the same array twice — a new GP on the same data, a second prediction at the same points — lands on one key, and the first copy's release races the second scatter inside the scheduler, returning `CancelledError`/`KeyError` from the tasks. And it wraps a **list** in a one-element list before scattering, taking `[0]` of the result: `client.scatter(a_list)` scatters *element-wise* and returns a list of futures, not one future for the point set. Without that, gp2Scale on a non-Euclidean input space silently hands the workers the wrong thing. `_harvest` in [gp2Scale_covariance.py](fvgp/gp2Scale_covariance.py) additionally raises on an exception result rather than letting it reach the assembly.
- On data changes, `augment_state_data` / `update_state_data` refresh the persistent scatter by **overwriting** it (no explicit `release()`); the old future loses its only Python ref and is cleaned up via `__del__`.

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

**Both are gated to `method='mcmc'` during training.** `GP.train` (sync path) wraps the call in `sequential_linalg_state(self.args, method)` from [gp_kv.py](fvgp/gp_kv.py#L30), which temporarily forces `sparse_krylov_warm_start=False` and `sparse_preconditioner_refresh_interval=1` for every other method, warning if that overrides an explicit user setting and restoring it afterwards. Rationale: both mechanisms carry state between likelihood evaluations and are only sound when successive evaluations are close. For a non-local method the leftover residual of a stale-seeded truncated solve makes the likelihood *order-dependent* — a bias, not zero-mean noise, which is exactly what a Bayesian optimizer's noise model cannot absorb. `_SEQUENTIAL_STATE_METHODS` / `_SEQUENTIAL_STATE_DEFAULTS` at the top of `gp_kv.py` hold the policy; the finer per-evaluation checks are `GPkv._validated_warm_start` and `GPkv._can_reuse_sparse_preconditioner`, which discard cached state whenever K+V has actually drifted, whatever the method.

### GPU backends and `compute_device="gpu"`

Not every GPU path goes through the same library, which is the thing to keep straight:

- **The dense linear algebra and the GPU Wendland kernels** use pytorch or cupy. Every one of them resolves its backend through `gp_lin_alg.get_gpu_engine(args)` — the single source of truth, honoring `args["GPU_engine"]` (`"torch"`/`"pytorch"`/`"cupy"`) and `args["GPU_device"]`. Reaching that function already means a GPU was asked for, so a `None` result is always an unmet request and always warns, naming the cause via `gpu_engine_unavailable_reason`: *not installed* versus *installed but no usable device*. `kernels.py` no longer keeps its own detector; it imports the same helper under a private alias, and the GPU Wendland kernels take an optional `args` (making them 4-argument, which is how `GPprior` knows to pass `args` through to the workers).
- **imate's stochastic log-determinant** has its own CUDA backend and reaches a GPU through neither pytorch nor cupy. `_imate_gpu_enabled()` asks imate (`imate.device.get_num_gpu_devices()`); gating it on torch/cupy disables a working GPU on any machine with a CPU-only pytorch build. Do not "unify" this with `get_gpu_engine`.

**Importing imate silences all warnings.** It calls `logging.captureWarnings(True)`, routing every Python warning to the `py.warnings` logger, which has a `NullHandler` — so warnings vanish process-wide, fvGP's and numpy's alike. Always import it through `gp_lin_alg._import_imate_logdet()`, which restores the previous `showwarning` and only if imate changed it.

Under gp2Scale the kernel runs on dask workers, so a warning raised inside it never reaches the client. The request is still honored; only the message is lost.

### Preconditioner logging

`calculate_sparse_preconditioner` logs `"<type> preconditioner construction in progress ..."` and a compute time carrying `n` and `K+V nnz`, matching the solvers it feeds so a gp2Scale debug log reads as one timeline. `GPkv._get_or_refresh_preconditioner` logs reuse in the same phrasing — without it, a cached preconditioner's missing construction time looks like one that never ran. Application cost is inside the CG/MINRES timings, not broken out.

### Customization API

Kernels, mean functions, and noise models are all plain Python callables with standardized signatures. Users pass them as arguments to `GP`/`fvGP` constructors. The full hyperparameter vector is shared across kernel, mean, and noise callables, but each callable must only read its reserved index range. Kernel gradients can be user-supplied or computed via finite differences.

### Non-Euclidean input spaces

`x_data` may be a **list of arbitrary objects** instead of an array — strings, dicts, custom classes, or lists. `GPdata` then sets `Euclidean=False` and `index_set_dim = input_set_dim = 1` regardless of what the objects are, and a user kernel is mandatory. Multi-task works too: `fvGP`'s index-set transform turns the list into `[point, task]` pairs, and the kernel sees those (`a[0]` is the object, `a[1]` the task).

The rule that keeps this working: **never ask numpy about the input set.** `np.ndim`/`np.shape` silently *reinterpret* a list of equal-length lists as a 2-d array and **raise** on a ragged one, so both are wrong for a space whose points are opaque objects. Use `len()`, or `isinstance(x, list)`. Three separate bugs came from breaking this rule — the update assertion, the gp2Scale scatter-reuse guard, and `_update_prior`'s `np.vstack`.

`posterior_mean_grad` / `d_kernel_dx` are the exception and are dense-in-N by construction: they do `np.array(x1)[:, direction] += eps`, which cannot work for object points. Derivatives with respect to a string are not meaningful, so that is correct-by-construction rather than a gap.

### Information-theoretic methods

`GP` exposes `gp_entropy()`, `gp_mutual_information()`, `gp_kl_div()`, and predictive metrics (`rmse`, `nlpd`, `crps`, `r2`, `picp`), all computed via `GPposterior`.

### Extension modules

- **[ggmp.py](fvgp/ggmp.py)** — `GGMP` (Gaussian GP for Gaussian Mixture data): fits K GMM components per station, each backed by its own `GP`; intended for distributional regression. Written by Vardaan Tekriwal. Excluded from the test suite (`# pragma: no cover`).
- **[deep_kernel_network.py](fvgp/deep_kernel_network.py)** — `Network` (PyTorch `nn.Module`): a 3-layer ReLU network used as a feature extractor for deep kernel learning. Excluded from the test suite (`# pragma: no cover`).

## Dependencies

Core: `numpy`, `scipy`, `dask`, `distributed`, `hgdl`, `loguru` — declared as **ranges**, not `~=` pins, so pip can resolve across Python 3.10–3.14 (see Commands).

`imate` is required for gp2Scale (the stochastic log-determinant) but is not a declared core dependency; `GP.__init__` raises a pointed message when it is missing. It lives in the `tests` extra.

Optional GPU backend: `torch` or `cupy` (selected via `compute_device`, refined by `args["GPU_engine"]`). Optional compiled preconditioners: `pyamg`, `ilupp` (the `preconditioners` extra).
