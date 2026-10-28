# Roadmap: fvGP 4.10 → fvGP 5.0.0 = gpNOVA 1.0.0

**Status:** Planning document, started 2026-10-26. Nothing below is decided unless marked so.
**Companion to:** [fvgp_to_gpnova_migration.md](../../gpNova/fvgp_to_gpnova_migration.md) (in the gpNova folder), which covers the mechanics of the rename (repository, PyPI, docs, gpCAM switch).

---

## 1. Decisions so far

- **fvGP becomes gpNOVA.** With function-to-function operator maps (function-valued inputs through custom kernels; function-valued outputs through the multi-task modes, Kronecker mode for outputs such as images), "function-valued GP" undersells the package. The migration guide gives the rationale and the backronym.
- **The last fvGP release is 5.0.0, and it becomes gpNOVA 1.0.0.** The rename itself changes only the import path, as the migration guide (§7) promises users. So **every breaking change has to be in fvGP 5.0.0**, a major version, where semantic versioning allows it. Deprecations should land in 4.x releases first wherever possible, so 5.0.0 only removes what users were already warned about.
- **Open:** the timing; the capitalization (gpNOVA vs. gpNova; the import and PyPI name is `gpnova` either way, so this only affects branding and docs).
- **fvGP ends at 5.0.0, with no shim.** fvGP 5.0.0 is the full code, identical to gpNOVA 1.0.0 at the moment of the rename; there will never be another fvGP release. All development continues in gpNOVA. Consequences:
  - The migration guide is updated to match (§4.2, §9–§11).
  - It works because almost all users reach fvGP through gpCAM: once gpCAM depends on gpNOVA, they move without noticing.
  - fvGP 5.0.0 can still carry an import-time note ("this is the last fvgp release; development continues in gpnova").
  - **Pickles:** gpNOVA will not load objects pickled with fvGP; users use one package or the other. No compatibility aliases (decided). gpCAM's changelog should say so at the switch.

---

## 2. Where fvGP stands (4.10.0, released 2026-10-26)

Capabilities relevant to the gpNOVA framing:

- **Three multi-task modes** through `fvGP(multi_task_mode=...)`:
  - `"cartesian"` (default): full multi-task learning over the input × output space, any kernel, missing outputs allowed.
  - `"independent"`: several GPs sharing one set of hyperparameters (one input covariance, several right-hand sides).
  - `"kronecker"`: full-grid data with a separable kernel K = K_x ⊗ F_1 ⊗ … ⊗ F_m, never formed; N = 10⁷ (1000 inputs × a 100 × 100 image) trains on one node.
- **Kronecker default kernel:** intrinsic coregionalization, K_x ⊗ (W Wᵀ + diag(κ)), with an exact gradient, up to 500 outputs; flat-index Matérn above that.
- **Low-rank mode** `linalg_mode="low_rank_<r>"`: exact likelihood for kernels of known rank, via Woodbury and the determinant lemma.
- **gp2Scale:** distributed, sparse, compactly supported kernels; stochastic log-determinant; Bayesian optimization of hyperparameters for the noisy regime.
- **Training:** local (multi-start, the default, also asynchronous), global, HGDL, MCMC, Adam, BO.
- **Non-Euclidean inputs:** arbitrary objects as inputs with user kernels, in every multi-task mode.

---

## 3. Candidate changes before the rename

### 3.1 Breaking: must be in 5.0.0 (deprecate in 4.x first where possible)

1. **One class instead of `GP` and `fvGP`. Decided 2026-10-27, implemented for 4.11.0.** `GP` takes `multi_task_mode` (default `"independent"`, so `GP(x, y_2d)` is unchanged); the shape of `y_data` decides single- vs. multi-task; one data representation per mode, the `fvgp_*` copies removed. `fvGP` is a deprecated wrapper (`FutureWarning`) to delete in 5.0.0. gpCAM's side: [gpCAMupdate.md](gpCAMupdate.md).
2. **Remove matrix-valued noise.** A noise function or gradient returning a matrix has been deprecated (FutureWarning) since 4.10. Correlated noise belongs in the kernel; the low-rank and Kronecker modes already require vector noise.
3. **Make the multi-task setting an explicit argument. Done in 4.11.0** with item 1: `GP(..., multi_task_mode=...)`, the `GP._multi_task` hook removed, and `ram_economy` renamed `gradient_per_direction` internally (old pickles mapped).
4. **A consistent posterior result contract.** The returned dicts are uneven: `"m(x)_flat"` is 2-d for several GPs but flat elsewhere, and `"m(x)"`, `"v(x)"`, `"S"` and `"S_flat"` mean slightly different things per mode (e.g. one shared `S` for independent outputs, a block covariance for Cartesian ones). A small result object, or a strict, documented shape contract, is hard to introduce without breaking.
5. **Typed options instead of the `args` dict.** `args` has grown to dozens of string keys (preconditioners, imate, low rank, Kronecker, GPU). A misspelled key is silently ignored. Typed option groups would catch that.
6. **The extension modules** `ggmp.py` and `deep_kernel_network.py` are excluded from the tests and from coverage. Drop them, move them to a separate package, or bring them under test.

### 3.2 Non-breaking: any 4.x release, ideally before 5.0.0

1. **Bug: `method='adam'` ignores the hyperparameter bounds.** It returned values outside them in gpCAM's example notebooks.
2. **Matrix-free low-rank mode.** The low-rank mode still forms the full N × N covariance matrix first, so it stops near the usual dense limit (N ≈ 2–5·10⁴). The pivoted Cholesky needs only K's diagonal and r rows, which the kernel could compute on demand: O(N r) memory, N = 10⁶ and beyond. The gradient needs reworking too (it forms each dK/dθ densely). The biggest scaling gain still on the table.
3. **Reuse stochastic-Lanczos probes across MCMC steps** (common random numbers). Measured at N = 8000: log-determinant *differences* to ±1.6 with 10 fixed probes, against ±47 with independent ones. Blocker: multithreaded imate is not reproducible with a fixed seed. Options: fvGP's own block Lanczos, or deterministic per-probe seeding upstream in imate.
4. **Per-output noise in Kronecker mode**, e.g. per-pixel detector noise. `KroneckerFactorization` already takes per-axis noise, so this is additive if wanted.
5. **Report the imate bug upstream:** `convergence_tools.cpp:204` compares against `error_rtol * mean` instead of `|mean|`; fvGP works around it.
6. **Unmeasured cases:** the lazy preconditioner above N = 10⁵ (its intended range); preconditioner choice in 2-D/3-D; HGDL's BLAS oversubscription on non-nanny Dask workers (a per-walker thread cap was proposed).

### 3.3 Known scaling limits, for the docs

| Mode | Binding limit | Rough ceiling on one node |
|---|---|---|
| Dense (Cholesky) | forming and factoring the N × N matrix | N ≈ 2–5·10⁴ |
| Low rank | forming K (the linear algebra is O(N r²)) | same as dense until matrix-free (3.2.2) |
| Kronecker | the dense eigendecomposition of each factor; the N-sized tensors in RAM | inputs n_x ≈ 10⁴; N ≈ 10⁹ |
| Kronecker, coregionalization default | the n_out × n_out output factor and its n_out·(rank + 1) hyperparameters | 500 outputs by default (configurable) |
| gp2Scale | sparsity of K; the stochastic log-determinant dominates training | 10⁶–10⁷ points |

---

## 4. Ecosystem

- **gpCAM, next step (not yet done):** forward `multi_task_mode` through `GPOptimizerBase` / `fvGPOptimizer` (including its pickled state); sync gpCAM's constructor docstrings with fvGP 4.10's (low-rank mode, matrix-noise deprecation, `multi_task_mode` and Kronecker, prior-mean shapes, asynchronous local training, the gradient's `component`); pin `fvgp ~= 4.10.0`, which is on PyPI.
- **At the rename:** gpCAM switches its dependency to `gpnova` (migration guide §6.1). gpAtlas and gp2Scale follow; HGDL is unaffected.

---

## 5. Suggested order

1. gpCAM sync to fvGP 4.10 (§4).
2. Fix the ADAM bounds bug (§3.2.1).
3. ~~Decide the class structure (§3.1.1)~~ — done: one class, in 4.11.0.
4. Release 4.x versions carrying the deprecations for everything in §3.1.
5. Non-breaking features as wanted, matrix-free low rank first (§3.2).
6. Release fvGP 5.0.0: the deprecated pieces removed, the final API.
7. Rename, following the migration guide: gpNOVA 1.0.0 from the 5.0.0 code, then the fvGP shim release.
