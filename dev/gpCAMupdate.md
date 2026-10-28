# gpCAM update for fvGP 4.11.x

**Status:** Done in gpCAM 8.7.0 (2026-10-28, unreleased), which pins `fvgp ~= 4.11.0`:
`fvGPOptimizer` is a deprecated `GPOptimizer(multi_task_mode="cartesian")`, `get_data()`
returns the stored form in every mode, and the EI incumbent is the best task-summed
posterior mean. Not done, by decision: old multi-task pickles. fvGP 4.11.0's
`GPdata.__setstate__` restores a pre-4.10 Cartesian fvGP, or a 4.10 GP with 2-d `y_data`,
as single-task; gpCAM's HISTORY says to rebuild such optimizers from data. The list below
is what was planned.

**What changes in fvGP 4.11:** `GP` absorbs `fvGP`. The shape of `y_data` decides: (N,) or
(N, 1) is a scalar GP, (N, Ni) with Ni > 1 is multi-task, modeled by
`GP(..., multi_task_mode=...)` with `"independent"` (the default, today's `GP(x, y_2d)`),
`"cartesian"` (today's `fvGP` default) or `"kronecker"`. `fvGP` is a deprecated wrapper
(`FutureWarning`) and is deleted in 5.0.0. There is one data representation per mode, and the
`fvgp_*` copies are gone.

## To do

1. **Pin** `fvgp ~= 4.11.0`, only once 4.11.0 is on PyPI.

2. **`fvGPOptimizer` stops subclassing `fvGP`.** Build it on `GP` and pass
   `multi_task_mode`, defaulting to `"cartesian"` to keep gpCAM's current behavior. Forward
   the argument through `GPOptimizerBase` and its pickled state (still open from 4.10).
   Consider merging `GPOptimizer` and `fvGPOptimizer` the way fvGP merged `GP` and `fvGP`:
   `GP` now decides from `y_data`'s shape, and `GPOptimizerBase.multi_task` already exists.

3. **`fvgp_x_data`, `fvgp_y_data` and `fvgp_noise_variances` no longer exist.** `x_data`,
   `y_data` and `noise_variances` are the stored form of each mode:

   | mode | `x_data` | `y_data` | `noise_variances` |
   |---|---|---|---|
   | scalar | (N, D) | (N, 1) | (N,) |
   | cartesian | (N·Ni, D+1), NaN outputs dropped | (N·Ni, 1) | (N·Ni,) |
   | independent, kronecker | (N, D) | (N, Ni) | (N,) |

   The readers in gpCAM:
   - **The info dict**, `gp_optimizer_base.py` l. 162–163: display only. Show the stored
     form, or keep gpCAM's own copy of what it passed in if the (N, Ni) view is wanted.
   - **The multi-task EI incumbent**, `surrogate_model._observed_task_sums` (l. 405–415).
     Today `np.sum` over NaN makes the incumbent NaN whenever an output is missing at a
     point, which is an existing bug. Replace it with the best posterior mean (summed over
     `x_out`) at the observed points, which handles missing outputs and noise, or with
     `y_data.reshape(Ni, N).T` if only full grids matter.
   - `tests/test_gpCAM.py` l. 270.

   fvGP has no inverse of the Cartesian transform, on purpose. If one is ever needed,
   record each row's point index at transform time; regrouping rows by their coordinates
   would merge duplicate x points.

4. **`x_out` per mode.** Cartesian takes output coordinates; kronecker and independent take
   output indices. Check the acquisition functions and `self.x_out = np.arange(...)`
   (`gp_optimizer_base.py` l. 115) against all three modes.

5. **Docstrings.** Still open from 4.10:
   - low-rank mode;
   - the matrix-noise deprecation;
   - `multi_task_mode` and Kronecker;
   - prior mean shapes;
   - async `local` (now the async default);
   - `component=None` in the gradient.

   New in 4.11:
   - one class;
   - `y_data`'s shape decides scalar vs. multi-task;
   - default `"independent"`;
   - noise shapes per mode;
   - `fvGP` deprecated;
   - `ram_economy` gone internally.

6. **Pickles.** fvGP 4.11 maps GPdata's old `ram_economy` key. An `fvGPOptimizer` pickled
   with the previous gpCAM may still not load, because its class bases change. Say so in
   gpCAM's HISTORY.

7. **The posterior result contract** (fvGP 4.11, `GP` class docstring, Notes). Every
   posterior/information result has the same keys and shapes in every mode: single-task
   without an output axis, multi-task (V, n), (V, V, n, n), flat arrays task-major. What gpCAM
   reads (`m(x)`, `v(x)`, `S_flat`, `dm/dx`, `RIE`, `total correlation`) keeps its shape for
   single-task and Cartesian GPs, so nothing breaks; but:
   - `S_flat` is now the joint over the requested pairs in **every** mode (block-diagonal for
     independent outputs), so `_cross_cov` in `surrogate_model.py` (l. 528 ff.) and the
     defensive `reshape(len(x), len(x_out))` in the multi-task acquisition functions work for
     all three modes without per-mode branches.
   - `variance_only=True` returns `S = S_flat = None` everywhere; gpCAM only reads `v(x)`
     there, so fine.
   - The `x_pred` key is gone; `x_out` is new in every result. gpCAM reads neither.
   - `gp_entropy(_grad)` return dicts; `posterior_mean_grad`'s `component` is deprecated.
     gpCAM uses neither.

8. **Run gpCAM's tests and example notebooks** against fvGP 4.11.
