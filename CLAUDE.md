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

**Python 3.10–3.14** are supported and all five are in CI. Dependencies are declared as ranges, not `~=` pins, because no single scipy/numpy release spans that range; pip resolves per interpreter. hgdl (since 2.4) declares the same ranges, so it no longer decides the stack. fvGP pins **hgdl `~= 2.4.2`** (≥ 2.4.2, < 2.5) because `GPtraining` passes HGDL's `info` argument, which 2.4.2 introduced.

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

`GP.train` validates and normalizes everything (bounds, initial hyperparameters, objective + gradient + Hessian, gp2Scale and async restrictions) and then hands off to `GPtraining.train` or `GPtraining.train_async`. `method` is `"global"`, `"local"`, `"hgdl"`, `"mcmc"`, `"adam"`, `"bo"`, or a callable taking the `GP` and returning a hyperparameter vector.

**The default (`method=None`) is `local` with `n_restarts=5`**, resolved in `GP.train`, synchronous or asynchronous; `mcmc` under gp2Scale (no "Method switched" warning then). Measured on 9- and 11-hyperparameter GPs (analytic and finite-difference gradients): every non-failed optimum, local or MCMC or global, predicted held-out data identically (RMSE 0.0555–0.0563), restarts were 2–4× faster than MCMC, and the danger is a *single* local run, which stopped on a flat large-length-scale plateau in 17/20 runs from uniform starts (test error 6–10× worse). Restart starts are therefore drawn log-uniformly where `resolve_log_scale("auto", bounds)` says so (7/20 failures, so ~0.5% for all five). The first start is the current/init hyperparameters. BO's inner surrogate fit pins `n_restarts=1`: it is warm-started from the previous fit, and restarts would multiply every BO iteration.

- Async is supported for `local`, `hgdl`, `mcmc`, `adam`, `bo` and needs a `dask_client`; everything else warns and falls back to sync. Async `local` (`gp_actor._LocalActor`) runs the restarts one after another in a thread on **one** worker; `hgdl` spreads local runs over the cluster. Both sync and async `local` go through `GPtraining.local_optimize`, which takes `callback(x, f, start, iteration)` and `early_stop()` (raised into scipy as `StopIteration`, honored by L-BFGS-B and most methods; otherwise the stop lands between starts). The polled `x` is the best point evaluated so far over all starts. Its scipy callback must keep the single parameter `intermediate_result` — any extra parameter, even a default, makes scipy pass a bare array instead of an `OptimizeResult`.
- `hgdl` gets `info` passed through (needs hgdl ≥ 2.4.2). Synchronous `hgdl` without a `dask_client` lets HGDL start its own local cluster; `GPtraining.train` closes it (`kill_client()`) in a `finally`, never a user's client.
- Known open issue: `method='adam'` ignores the hyperparameter bounds (returned values outside them in the gpCAM example notebooks).
- gp2Scale trains synchronously and only allows `mcmc` or `bo` (other methods are silently switched to `mcmc`).
- `mcmc` ignores a user `objective_function` (it always maximizes the log marginal likelihood). A user objective with `local`/`hgdl` must come with a gradient.
- Diagnostics land on read-only `GP` properties: `mcmc_info` after `mcmc`, `bo_info` after a sync `bo` run.
- **`mcmc` ([gp_mcmc.py](fvgp/gp_mcmc.py)).** Shaby–Wells adaptive Metropolis: proposals are `N(x, sigma_m · prop_Sigma)`, and `_adapt` tunes `sigma_m` toward 23% acceptance and blends `prop_Sigma` toward the running covariance of the whole chain (a last-10-states window locked in a wrongly shaped initial proposal: 8500 steps instead of ~3000 for the notebook's 2-hyperparameter GP). The proposal once ignored `sigma_m`, which made the adaptation dead code (acceptance 0.68); `test_mcmc_recovers_a_correlated_gaussian` guards it. The trained hyperparameters are `median(x)` after `mcmc_burn_in` (0.5). `mcmc_log_scale="auto"` (`gp_mcmc.resolve_log_scale`) samples hyperparameters with positive bounds spanning ≥ a decade in log space — exact, since `_log_prior` adds the Jacobian and `_log_posterior` removes it again for `"MAP"`; the trace, summaries and user callables stay in θ, while `_adapt` learns from `_sampling_trace` (u). The decade threshold keeps narrow location-like ranges linear, where log helps nothing and can slow mixing; custom proposals turn it off. `mcmc_break_condition="default"` is a single-chain heuristic (Geweke z < 2 and ESS ≥ 50 on the post-burn-in chain, 2 passing checks 100 steps apart, from 400 on, never on a chain that accepted nothing). Real convergence checks need several chains and `gp_mcmc.split_rhat`/`ess`. All traces start at `x0`, so index j is the same state everywhere. `summarize()` builds `mcmc_info` once, after the loop, and for the async actor when polled; building it per iteration made runs quadratic.
- **`accept_only_if_improved=True` (default) rolls back a regression.** `GP._reject_if_not_improved` compares `log_likelihood()` before and after and, if the trained hyperparameters are worse, restores the incumbent, returns it, and warns. Active **only** for sync training with `local`, `hgdl`, `adam` or a callable — `local` accepts `OptimumEvaluation["x"]` regardless of `success`, so it can genuinely return worse. Exempt: `global` (elitist `differential_evolution` seeded with `x0`), `mcmc` (returns `median(x)`, a summary — guarding it would make repeat `train()` calls no-ops), `bo` (its recommendation can be worse in observed value by design), a user `objective_function` (wrong criterion), and all async paths. Tolerance comes from `log_likelihood_variance()` (`None` → strict) so a stochastic log-determinant doesn't cause coin-flip rejections. Both likelihood reads are free — no argument means "use the cached `KVinvY`/`logdet_KV`" — so only a rejection costs an extra `set_hyperparameters`.

**`method='bo'` ([gp_bo.py](fvgp/gp_bo.py))** — Bayesian optimization over the hyperparameters, for when the marginal likelihood is expensive, noisy, and effectively gradient-free (the gp2Scale/mBCG regime: stochastic-Lanczos log-determinant plus truncated CG). Points to know before touching it:

- **The surrogate is an `fvgp.GP`.** It deliberately uses fvGP and not gpCAM, which depends on fvGP — importing gpCAM here would be circular. The recursion bottoms out: the inner GP sees only the tens-to-hundreds of θ points evaluated, uses an ARD Matérn-5/2 kernel with analytic gradients, and trains with `method='local'` — never `'bo'`.
- **`max_iter` changes meaning**: for `bo` it is a *cap on objective-function evaluations*, not iterations. The run normally stops earlier on the `patience` / `f_rtol` / `x_tol` criteria in `bo_args`.
- **Observation noise is wired in automatically.** When the objective is the default, `GP.train` injects a `bo_args["noise_function"]` reading `GPMarginalLikelihood.log_likelihood_variance()` — the SLQ log-determinant's own precision (`0.25 * kv.last_logdet_variance`), or `None` in exact modes, where the surrogate instead learns a single homoscedastic noise level whose lower bound acts as a nugget.
- **The search space is log-transformed per dimension** (`_LogAffineTransform`): log where both bounds are strictly positive, linear otherwise, then rescaled to the unit cube. Positivity is only a *proxy* for being scale-like — a positive hyperparameter that enters additively (a center in a non-stationary/Gibbs kernel, a mixing weight) is hurt by it. Override with `bo_args['log_scale']`.
- `GP._warn_about_bo_suitability` warns before the run when the budget is too small for the initial design or the hyperparameter count is too high for BO.
- **`bayesian_optimize` warns after the run when it returns the warm start** (`best_idx == 0` — the warm start overwrites design point 0, so the test is exact rather than a float comparison). Measured cause: a marginal likelihood's variance/length-scale ridge is a 1-d curve in θ-space, which a Sobol design cannot land on and an axis-aligned ARD surrogate reads as "no coordinate matters" (13 of 14 length scales pinned at the 2.0 cap in the reproduction). Giving the surrogate mean a linear term fixes the ARD degeneracy completely and still does not find the ridge — being at the right radius is one condition, being *on* a 1-d curve is `d-1` more. Don't try to fix this by changing where BO samples; recommend `local`/`mcmc`.
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
| `"rowwise"` | row strips; **one kernel call per strip over all columns**; workers return finished CSR | double (no symmetry) | `vstack` of strips — concatenation only |

`gp2Scale_batch_size` sizes both, but means different things: for block-wise it is the side of a square block (`batch_size × batch_size` per worker call), for row-wise it is the **strip width** (`batch_size × N` per worker call). It is a **maximum with a remainder** — every chunk is exactly `batch_size` except the last along each axis. 100 000 at 15 000 → six chunks of 15 000 and one of 10 000.

**`batch_size` is a RAM budget, and it decides whether to distribute at all.** One task is meant to hold `B x B` entries block-wise, `N x B` row-wise. Anything that fits inside one task is computed in a single kernel call on the client instead — a 10 000 x 2 cross-covariance is 0.83 ms directly against 1.8 s over 1000 dask tasks. The two budgets are **not comparable numbers**: at `B=10 000`, N=1e6 that is 0.8 GB against 80 GB, so the same setting can distribute under one mode and stay local under the other. That is deliberate — each mode gets the memory its own user declared.

**Cross-covariances are always striped, whatever `gp2Scale_distribution` says.** `k(x_data, x_pred)` (posterior) and `k(x_old, x_new)` (append) are tall and thin; blocking them multiplies the task count without reducing per-task memory, since the long axis is traversed either way. The short axis is split into strips as wide as the budget allows (`budget // long_axis`, at least 1), each task spanning the long axis. `gp2Scale_distribution` therefore governs only the **symmetric** case — the prior and `k(x_new, x_new)` — which is exactly what the `symmetric` flag already passed to `distributed_covariance` distinguishes. Column strips are built as **CSC** and `hstack`ed (scipy's fast path), row strips as CSR and `vstack`ed.

Two consequences worth knowing before recommending row-wise:
- A **dense** kernel allocates `batch_size × N` per row-wise task — 8 GB at N=100 000 and a 10 000 strip. That allocation, not the block count, is what bounds worker memory.
- Row-wise makes `N/batch_size` tasks against block-wise's `(N/batch_size)²/2`, so it can starve a large cluster; it usually wants a *smaller* `batch_size` than block-wise.

Whether row-wise wins depends on the kernel, not the assembly. Measured at N=6000 on 2 workers: with a dense kernel block-wise is ahead (0.85 s vs 1.39 s — row-wise pays 2× evaluations for nothing), but with the **support-aware sparse** kernel row-wise wins (0.171 s vs 0.193 s), because a whole row lets it prune in a single neighbor search.

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

- **Preconditioner caching** (`sparseCGpre`/`sparseMINRESpre` only): `args["sparse_preconditioner_refresh_interval"] = N` caps consecutive reuses of a single preconditioner across `update_KV` / `compute_new_*` calls. Default `None` (no cap): reuse is decided by drift — the cached operator is kept while K+V stays within `sparse_preconditioner_max_matrix_drift` (0.1) of the matrix it was built for. `args["sparse_preconditioner_type"]` selects the kernel — `"ilu"` (default), `"ic"`/`"incomplete_cholesky"`, `"block_jacobi"`, `"schwarz"`/`"additive_schwarz"`, `"amg"` (requires pyamg). Mode aliases `"sparseCGpre_<type>"` / `"sparseMINRESpre_<type>"` set the type as a shortcut. Cache is invalidated automatically when `KV.shape` or any `sparse_preconditioner_*` arg changes.
- **Warm-start** (all iterative modes): `args["sparse_krylov_warm_start"] = True` makes `GPMarginalLikelihood` pass the previous training iteration's `KVinvY` as `x0` to the next iterative solve. Cuts iteration counts substantially when successive hyperparameter trials are close. Stored in `marginal_likelihood._warm_start_KVinvY`; reset to `None` on pickling.

Both default off so existing behavior is preserved.

**Both are gated to `method='mcmc'` during training.** `GP.train` (sync path) wraps the call in `sequential_linalg_state(self.args, method)` from [gp_kv.py](fvgp/gp_kv.py#L30), which temporarily forces `sparse_krylov_warm_start=False` and `sparse_preconditioner_refresh_interval=1` for every other method, warning if that overrides an explicit user setting and restoring it afterwards. Rationale: both mechanisms carry state between likelihood evaluations and are only sound when successive evaluations are close. For a non-local method the leftover residual of a stale-seeded truncated solve makes the likelihood *order-dependent* — a bias, not zero-mean noise, which is exactly what a Bayesian optimizer's noise model cannot absorb. `_SEQUENTIAL_STATE_METHODS` / `_SEQUENTIAL_STATE_DEFAULTS` at the top of `gp_kv.py` hold the policy; the finer per-evaluation checks are `GPkv._validated_warm_start` and `GPkv._can_reuse_sparse_preconditioner`, which discard cached state whenever K+V has actually drifted, whatever the method.

**During training a preconditioner is built lazily** (`GPkv._training_preconditioner`, on by default, `args["sparse_preconditioner_lazy"]=False` to disable). Rebuilt for every evaluation, even the best preconditioner leaves CG slower than plain MINRES, so a build only pays once it is reused. When the cache is stale, the `compute_new_*` solves use **unpreconditioned MINRES** (for `sparseCGpre` too) unless the last `args["sparse_preconditioner_settle_steps"]` (default 3) evaluations each moved K+V less than the drift threshold *from the previous evaluation* (`Settled_steps`, counted in `_record_evaluation` from the fingerprint the warm-start check already takes). Two drift checks, two jobs: the existing one compares with the matrix the cache was **built for** (still valid?), the new one compares **consecutive evaluations** (will a new one be reused?). Under `refresh_interval=1` — every method but MCMC, via `sequential_linalg_state` — nothing is built during training at all. `set_KV`, `update_KV` and `solve()` are unchanged: `set_hyperparameters` still builds one preconditioner, which the posterior's many solves then reuse.

### Low-rank mode (`linalg_mode="low_rank_<r>"`)

For kernels of known rank. K is **computed as everywhere else** (dense, or sparse under gp2Scale); the mode starts after K exists. `gp_lin_alg.calculate_low_rank_factorization(K, V, rank, args)` runs a rank-r **pivoted Cholesky of the stored K** (`_pivoted_cholesky`: diagonal + r columns, O(N r²). K is symmetric, so column j is read as **row j** — straight from the CSR arrays gp2Scale produces, no `tocsc()` copy of every nonzero, and contiguous for dense K. Blocked pivoting (b candidates per round, one GEMM) was measured at only 1.5–2× on a loaded workstation and not adopted; worth re-measuring on an idle many-core node) and returns `LowRankPlusDiagonal`: `solve` by Woodbury with C = I + LᵀD⁻¹L (accumulated in row chunks; D⁻¹L is applied on the fly, never stored — it would be a second N×r array, 8 GB at N=1e6, r=1000), `logdet` by the determinant lemma — exact, so `last_logdet_variance` stays None (no SLQ, no BO noise). It stops early and exactly when K's rank is lower; when higher, it warns with the trace left out (`low_rank_residual_tol`) and the result is a Nyström-type approximation. `GPkv` needs **K and the noise vector apart**: `_refresh` factors (K, V) without `addKV`, `compute_new_KVinvY` takes `K=`/`V=` (callers in `GPMarginalLikelihood` and `GPposterior._new_KVinvY` pass them), `compute_new_KVlogdet_KVinvY` already had them. The gradient applies the factor to each dense `dK/dθ` (O(N² r)) instead of the batched dense solve. Noise must be a positive vector; matrix noise is deprecated everywhere (`gp_likelihood._warn_matrix_noise`, a `FutureWarning` because `DeprecationWarning` is hidden outside `__main__`) and rejected here.

### Multi-task modes (`fvGP(..., multi_task_mode=...)`)

Three ways to model several outputs, all taking `y_data` as (V, No) and stored behind `GP` with `data.multi_task` set (fvGP's `GP._multi_task` hook):

- `"cartesian"` (default): `_transform_index_set2` turns every (input, output) into a point `[x, task]` of a single-task GP; NaN outputs skipped; any kernel. `y_data` (V·No, 1).
- `"independent"`: GP's 2-d `y_data` path — No right-hand sides sharing K (one V×V factorization), noise one variance per input, no NaN. fvGP passes the data through untransformed and sets `data.x_out = None` (so the dense posterior does not build a Cartesian product); `GPposterior.posterior_mean`/`posterior_covariance` run the shared multi-column path (`_posterior_mean`/`_posterior_covariance`) and `_select_outputs` keeps the `x_out` columns. All outputs share one `S`; there is no cross-output covariance. `GPdata.set_fvgp_data` must not subtract a task dimension here. Plain `GP(x, y_2d)` remains the same thing and is not deprecated.
- `"kronecker"`: below.

Multi-column likelihoods average over the columns, and since the fix in this release so does the gradient: `neg_log_likelihood_gradient(component=None)` (was `component=0`, column 0 alone — training several GPs followed the wrong gradient).

### Kronecker mode (`fvGP(..., multi_task_mode="kronecker")`)

For multi-task data on a full grid with a separable kernel K = K_x ⊗ F_1 ⊗ … ⊗ F_m (`gp_lin_alg.KroneckerFactorization`). K is **never formed**; `GPprior.K` is the list of factors the kernel returns, `y_data` stays one `(n_x, n_out)` tensor (fvGP skips `_transform_index_set2`), and the mean has that shape too, or is 1-d and shared by all outputs (see Prior mean shapes). The noise keeps the contract of a GP with several right-hand sides: one variance per input, the same for every output (`noise_variances` (V), a noise function returning a vector), so `GPlikelihood` has no Kronecker branch. Everything else is a `data.kronecker` branch in the existing components, `GPkv` mode `"kronecker"` (set by the flag, never by `linalg_mode`), the pattern of `low_rank`. The fvGP flag reaches `GPdata` through the class attribute `GP._multi_task` (None, `"cartesian"` or `"kronecker"`), which fvGP sets before `GP.__init__` and pickles; `GPdata` stores it as `multi_task` and derives `kronecker` from it.

- **Whitening is what makes non-constant noise exact.** For noise D = ⊗ D_f, K + D = D^½ (⊗ F̃_f + I) D^½ with F̃_f = D_f^-½ F_f D_f^-½, so the eigendecompositions of the whitened factors give solve, log-determinant, `diag((K+V)⁻¹)` and every trace. fvGP's per-input noise is D = D_x ⊗ I, so only K_x is whitened (`calculate_kronecker_factorization` passes ones for the output axes; matrix or per-output noise raises). `KroneckerFactorization` itself takes any per-axis noise: per-output noise (e.g. per-pixel detector noise) would be an additive change. A separable-noise version (log-axis-means factoring with a tolerance) was implemented and dropped in favor of this simpler contract.
- **2-D `y_data` means something else here.** Plain GP treats it as several right-hand sides (`m[:, None]`, `l1 / y.shape[1]`); every Kronecker branch must sit before that code.
- **Gradient:** product rule over factors, through factor gradients. `KroneckerFactorization.factor_gradients` computes G_f = dL/dF_f for every factor once — ½(M_f − T_f), M_f the data term (α contracted with the other factors), T_f = `trace_matrix(f)` the trace term in the whitened eigenbasis — and `GPprior.kronecker_hyperparameter_gradient` turns them into dL/dh: a kernel object with `chain_rule(x, hps, G)` (the coregionalization kernel) maps them exactly itself; otherwise Σ_f ⟨G_f, dF_f/dh⟩ per hyperparameter, O(Σ n_f²) instead of the O(N Σ n_f) of the earlier per-hyperparameter mode products (N = 1e7 three-factor benchmark: 8.5 s → 2.0 s with finite differences, 1.7 s analytic). `GPprior.kronecker_dk_dh` supplies the factor derivatives: from a user `kernel_function_grad` returning `[dK_x, dF_1, …]` per hyperparameter (None for an unaffected factor; all hyperparameters in one call, or one per call with `gradient_per_direction`, `GPprior.user_kernel_grad` tells it apart from the built-in gradients), else central differences, two kernel calls per hyperparameter (an unaffected factor comes out exactly zero). Finite differences are always taken one hyperparameter at a time, so memory stays at one factor list. Verified against the Cartesian fvGP with the `np.kron` kernel: likelihood and posterior to 1e-12, gradient to the finite-difference level.
- **Posterior:** the mean is mode products with `[K_x(x*, X)ᵀ, F_1, …]`; variances are one contraction of the inverse spectrum with the squared projections `factor.project`; the full covariance builds those projections explicitly (N × pairs). Output order of `v(x)`, `S`, `*_flat` matches the Cartesian fvGP (task-major flat index). Measured noise at new inputs is the scalar mean, as in the Cartesian path, so `add_noise` agrees with it.
- **Default kernel: `kernels.CoregionalizationKernel` (ICM)**, K_x ⊗ B with K_x a unit-variance anisotropic Matérn and B = W Wᵀ + diag(κ), W (n_out × rank); hyperparameters `[ℓ (D), W row-major, κ (n_out)]`. fvGP builds it in `__init__` (`_default_coregionalization_kernel`, Euclidean inputs only) and passes it as the kernel, filling `init_hyperparameters` from `default_hyperparameters(x, y)` when kernel, mean and noise are all default; `fvGP._get_default_hyperparameter_bounds` asks the kernel. W must not start at 0 (dB/dW = 0 there, a stationary point) and, for rank > 1, its columns start as distinct DCT cosines (identical columns stay identical under the gradient). B carries all scale; a signal variance on K_x would be a redundant ridge. Rank from `args["kronecker_coregionalization_rank"]` (1). Above `args["kronecker_coregionalization_max_outputs"]` (`kernels.COREGIONALIZATION_MAX_OUTPUTS` = 500) fvGP passes no kernel and `GPprior._default_kronecker_kernel` applies: Matérn on the flat output index, D + 2 hyperparameters (that layout is why `GPdata` makes `index_set_dim` = D + 1 in Kronecker mode), with a warning. Measured, n_x = 500, rank 1, likelihood + gradient: n_out 100 0.1 s, 500 0.36 s, 1000 1.2 s, 2000 6.2 s; `train(n_restarts=1)` 3 s at 100, 10.5 s at 500 — the threshold is a judgment on those numbers, not a hard limit.

### GPU backends and `compute_device="gpu"`

Not every GPU path goes through the same library, which is the thing to keep straight:

- **The dense linear algebra and the GPU Wendland kernels** use pytorch or cupy. Every one of them resolves its backend through `gp_lin_alg.get_gpu_engine(args)` — the single source of truth, honoring `args["GPU_engine"]` (`"torch"`/`"pytorch"`/`"cupy"`) and `args["GPU_device"]`. Reaching that function already means a GPU was asked for, so a `None` result is always an unmet request and always warns, naming the cause via `gpu_engine_unavailable_reason`: *not installed* versus *installed but no usable device*. `kernels.py` no longer keeps its own detector; it imports the same helper under a private alias, and the GPU Wendland kernels take an optional `args` (making them 4-argument, which is how `GPprior` knows to pass `args` through to the workers).
- **imate's stochastic log-determinant** has its own CUDA backend and reaches a GPU through neither pytorch nor cupy. `_imate_gpu_enabled()` asks imate (`imate.device.get_num_gpu_devices()`); gating it on torch/cupy disables a working GPU on any machine with a CPU-only pytorch build. Do not "unify" this with `get_gpu_engine`.

**Never pass imate a relative tolerance.** Its SLQ stops sampling once `error < max(error_atol, error_rtol * mean)` — the signed `mean`, not `|mean|` (still so in 0.29.11, `convergence_tools.cpp:204`) — so for a negative log-determinant, the common case, a relative target is never met and every call runs to `max_num_samples`. `calculate_random_logdet` therefore passes only `error_atol`, converting the relative target (`random_logdet_error_rtol`, default 0.01) into an absolute one from a 10-probe pilot run. The default is relative because |log|K+V||, and with it the likelihood, varies too strongly across problems for one absolute target. A tight absolute target (±2) was tried and reverted: at N≈8000 it costs about the 5000-probe cap for every evaluation. The log-determinant, not the solve, dominates a gp2Scale likelihood evaluation (measured: ~10–30 s at the cap vs. <0.1 s for MINRES).

**Importing imate silences all warnings.** It calls `logging.captureWarnings(True)`, routing every Python warning to the `py.warnings` logger, which has a `NullHandler` — so warnings vanish process-wide, fvGP's and numpy's alike. Always import it through `gp_lin_alg._import_imate_logdet()`, which restores the previous `showwarning` and only if imate changed it.

Under gp2Scale the kernel runs on dask workers, so a warning raised inside it never reaches the client. The request is still honored; only the message is lost.

### Broken preconditioners and CG failures

A preconditioner is never trusted blindly. `calculate_sparse_preconditioner` rejects a factor with any NaN/inf (`_preconditioner_is_finite` inspects every factor type: ilupp `factors()`, SuperLU `L`/`U`, the builders' dicts, pyamg levels) — ilupp's IC(0) breaks down *silently* on smooth, non-diagonally-dominant K+V in random point order. A rejected or failed build means unpreconditioned **MINRES**, in training (`_preconditioned_training_solve`) and in `solve()`. `calculate_sparse_conj_grad` caps CG at `_DEFAULT_CG_MAXITER` (5000), or `_DEFAULT_PRECONDITIONED_CG_MAXITER` (500) with a preconditioner — measured on a random-order 1-D replica, working preconditioners converge in 1–20 iterations (ilu 1, ichol 4, Schwarz/1024 7) and plain CG in ~150–250 — unless an explicit `sparse_cg_maxiter`/`sparse_krylov_maxiter` is set (`None` = scipy's 10 N), aborts at the first non-finite iterate via the callback, and redoes any failed solve — not converged or non-finite, preconditioned or not, single or block — with unpreconditioned MINRES (`_cg_with_minres_fallback`). A failed build is remembered like a successful one is cached (`GPkv.Preconditioner_failure`, checked in `_build_preconditioner`, which every path goes through): no new attempt while K+V stays within the drift threshold of the failed matrix with the same `sparse_preconditioner_*` settings — otherwise `set_KV` and the `solve()` right after it each spent minutes on the same doomed IC(0). Reordering, or probing preconditioners with test vectors, was considered and rejected as the fix.

### Preconditioner logging

The CG and MINRES lines say how each solve starts: `"CG solve in progress (warm start) ..."` or `"(cold start)"`, judged by whether an `x0` reached the solver — so a `sparse_krylov_warm_start` run shows in the log whether the guess was used, and a drift-dropped one is preceded by `"Warm start dropped: ..."` from `GPkv._validated_warm_start`.

`calculate_sparse_preconditioner` logs `"<type> preconditioner construction in progress ..."` and a compute time carrying `n` and `K+V nnz`, matching the solvers it feeds so a gp2Scale debug log reads as one timeline. `GPkv._get_or_refresh_preconditioner` logs reuse in the same phrasing — without it, a cached preconditioner's missing construction time looks like one that never ran. Application cost is inside the CG/MINRES timings, not broken out.

### Customization API

Kernels, mean functions, and noise models are all plain Python callables with standardized signatures. Users pass them as arguments to `GP`/`fvGP` constructors. The full hyperparameter vector is shared across kernel, mean, and noise callables, but each callable must only read its reserved index range. Kernel gradients can be user-supplied or computed via finite differences.

`gradient_per_direction` (constructor; named `ram_economy` before 4.9 — the old name is a `TypeError`, gpCAM maps it in old pickles) sets the contract for `kernel_function_grad`/`noise_function_grad`: True → `f(x1, x2, hps, direction)` returning one hyperparameter's `(N1 x N2)` slice; False → the full `(H x N1 x N2)` stack. Internally it is still stored as `GPdata.ram_economy`.

### Non-Euclidean input spaces

`x_data` may be a **list of arbitrary objects** instead of an array — strings, dicts, custom classes, or lists. `GPdata` then sets `Euclidean=False` and `index_set_dim = input_set_dim = 1` regardless of what the objects are, and a user kernel is mandatory. Multi-task works too: `fvGP`'s index-set transform turns the list into `[point, task]` pairs, and the kernel sees those (`a[0]` is the object, `a[1]` the task).

The rule that keeps this working: **never ask numpy about the input set.** `np.ndim`/`np.shape` silently *reinterpret* a list of equal-length lists as a 2-d array and **raise** on a ragged one, so both are wrong for a space whose points are opaque objects. Use `len()`, or `isinstance(x, list)`. Three separate bugs came from breaking this rule — the update assertion, the gp2Scale scatter-reuse guard, and `_update_prior`'s `np.vstack`.

`posterior_mean_grad` / `d_kernel_dx` are the exception and are dense-in-N by construction: they do `np.array(x1)[:, direction] += eps`, which cannot work for object points. Derivatives with respect to a string are not meaningful, so that is correct-by-construction rather than a gap.

### Prior mean shapes

The default mean (`GPprior._default_mean_function`) is each task's average: per column, (N, k), whenever `y_data` has k > 1 columns (several GPs, or Kronecker outputs); for the Cartesian fvGP (`data.multi_task == "cartesian"`), whose transformed `y_data` is one column, each (point, task) pair gets its task's average (`_task_means`, task read per point from the last coordinate or the pair's second element; overall average for a task without observations), so the Cartesian and Kronecker defaults are the same model; otherwise a 1-d constant. `y_data` is always 2-d in storage: (N, 1) for a single-task GP and the Cartesian fvGP (after the transform), (N, k) for k right-hand sides, (n_x, n_out) in Kronecker mode. The mean function may return **(N,)**, shared by all columns, or **(N, k)**, one per column; `GPprior._check_mean` enforces exactly that in `_compute_prior` and on append (`ValueError` naming the mode). `GPprior.mean_columns(m)` is the single place where a mean meets 2-d data: a read-only `np.broadcast_to` view to (len(m), k). Every `y - m`, the posterior's `prior_mean + A`, and the mean terms of the gradient (`[:, component]`) and of `posterior_mean_grad` go through it; `self.m` itself keeps whatever shape the function returned (`_update_mean` concatenates along the points). The joint-prior and information-theoretic methods still flatten `m` and are single-output by construction.

### Information-theoretic methods

`GP` exposes `gp_entropy()`, `gp_mutual_information()`, `gp_kl_div()`, and predictive metrics (`rmse`, `nlpd`, `crps`, `r2`, `picp`), all computed via `GPposterior`.

`add_noise=True` (and the metrics built on it) uses `GPlikelihood.calculate_V(x_pred)`. With measured `noise_variances` and no noise function, `_measured_noise_function` returns the per-point variances only when `x_pred` *is* the data (`_is_x_data`: same object, equal array, or identical objects for non-Euclidean lists — never by length, which once handed a same-size test set the data's variances positionally); elsewhere `mean(noise_variances)`, with a warning if the measured noise is heteroscedastic. A malformed noise array raises in `GPposterior.add_noise`.

### Extension modules

- **[ggmp.py](fvgp/ggmp.py)** — `GGMP` (Gaussian GP for Gaussian Mixture data): fits K GMM components per station, each backed by its own `GP`; intended for distributional regression. Written by Vardaan Tekriwal. Excluded from the test suite (`# pragma: no cover`).
- **[deep_kernel_network.py](fvgp/deep_kernel_network.py)** — `Network` (PyTorch `nn.Module`): a 3-layer ReLU network used as a feature extractor for deep kernel learning. Excluded from the test suite (`# pragma: no cover`).

## Dependencies

Core: `numpy`, `scipy`, `dask`, `distributed`, `loguru` — declared as **ranges**, not `~=` pins, so pip can resolve across Python 3.10–3.14 (see Commands) — and `hgdl ~= 2.4.2` (≥ 2.4.2 for HGDL's `info`; hgdl 2.4 itself declares the stack as ranges, so the pin does not decide scipy/numpy).

`imate` is required for gp2Scale (the stochastic log-determinant) but is not a declared core dependency; `GP.__init__` raises a pointed message when it is missing. It lives in the `tests` extra.

Optional GPU backend: `torch` or `cupy` (selected via `compute_device`, refined by `args["GPU_engine"]`). Optional compiled preconditioners: `pyamg`, `ilupp` (the `preconditioners` extra).
