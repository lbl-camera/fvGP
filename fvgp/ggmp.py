from fvgp.gp import GP
from fvgp.gp_training import GPtraining
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.stats import norm, multivariate_normal, wasserstein_distance
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import LinAlgError
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import warnings
import inspect
import re
import time
from typing import Iterable, Sequence


################################################################################################
###CREDITS: This code was written by Vardaan Tekriwal (UC Berkeley, vtekriwal@berkeley.edu)#####
############It was adapted from the original ggmp code by Marcus Noack##########################
################################################################################################


# ============================================================================
# PICKLABLE MEAN FUNCTIONS
# These are module-level functions that can be pickled (avoid lambdas if you
# need multiprocessing elsewhere).
# ============================================================================


def constant_mean(x, hyperparameters):  # pragma: no cover
    """Constant mean function: mean = hyperparameters[-1]"""
    return np.ones(len(x)) * hyperparameters[-1]


#def zero_mean(x, hyperparameters):  # pragma: no cover
#    """Zero mean function."""
#    return np.zeros(len(x))


#def linear_mean(x, hyperparameters):  # pragma: no cover
#    """Linear mean function: mean = hyperparameters[-1] (intercept only)."""
#    return np.ones(len(x)) * hyperparameters[-1]


#def integral(f, domain):  # pragma: no cover
#    # Robust to decreasing/unsorted domains.
#    dx = np.abs(np.gradient(domain))
#    return float(np.sum(f * dx))


#def gaussian(mean, std, x):  # pragma: no cover
#    x = np.asarray(x, dtype=float)
#    std = np.maximum(std, 1e-12)
#    g = (1.0 / (np.sqrt(2.0 * np.pi) * std)) * np.exp(-np.power(x - mean, 2.0) / (2.0 * np.power(std, 2.0)))
#    if np.all(g < 1e-6): g[:] = 1e-6
#    inte = integral(g, x)
#    if not np.isfinite(inte) or inte < 1e-300:
#        inte = 1e-300
#    gn = g / inte
#    if np.any(np.isnan(gn)): print("NaN in Gaussian normalized")
#    return gn


class GGMP:  # pragma: no cover
    def __init__(
        self,
        x_data,
        y_data,
        *,
        hps_obj,
        gp_kernel_functions=None,
        gp_mean_functions=None,
        likelihood_terms=5,
        gp_init_kwargs=None,
        gp_device_ids=None,
        gp_eval_parallel=False,
    ):
        """
        The constructor for the GGMP class.

        GGMP uses K GMM components per station. Each component k gets its own GP
        that is trained on the component's mean vector (across all stations) with
        the component's variance as noise. Components are independent and equally
        weighted (1/K).

        Parameters:
            x_data: (N, D) array of station locations
            y_data: list of N (domain, density) tuples
            hps_obj: hyperparameters object with K sets of hyperparameters
            likelihood_terms: K, the number of GMM components (and GPs)
        """
        assert len(x_data) == len(y_data)
        self.likelihood_terms = likelihood_terms
        # One GP per component: number of GPs equals K
        self.number_of_GPs = likelihood_terms
        self.len_data = len(x_data)
        self.hps_obj = hps_obj
        # Fixed uniform weights: 1/K for each component
        self.init_weights = np.ones((self.number_of_GPs)) / float(self.number_of_GPs)
        self.x_data = x_data
        self.y_data = y_data
        self.gp_kernel_functions = gp_kernel_functions
        self.gp_mean_functions = gp_mean_functions
        if gp_kernel_functions is None: self.gp_kernel_functions = [None] * self.number_of_GPs
        # Default to constant_mean function so mean hyperparameter gradient is computed correctly
        if gp_mean_functions is None: self.gp_mean_functions = [constant_mean] * self.number_of_GPs
        self.gp_init_kwargs = {} if gp_init_kwargs is None else dict(gp_init_kwargs)
        self.gp_device_ids = gp_device_ids
        self.gp_eval_parallel = bool(gp_eval_parallel)

    def __getattr__(self, name):
        def not_implemented(*args, **kwargs):
            print(f"method {name} is not implemented in GGMPs")
        return not_implemented

    def build_pairwise_data_generating_normals(self, idx_a, idx_b):
        """
        Construct the K 2D Gaussians for a pair of datapoints by pairing the
        component mean/variance entries at each index.
        """
        if not hasattr(self, "likelihoods") or len(self.likelihoods) == 0:
            raise ValueError("Call initLikelihoods first.")
        if idx_a < 0 or idx_a >= self.len_data or idx_b < 0 or idx_b >= self.len_data:
            raise IndexError("Datapoint index out of range.")

        joints = []
        for comp in range(self.likelihood_terms):
            mean_vec = np.array([self.likelihoods[comp].mean[idx_a], self.likelihoods[comp].mean[idx_b]])
            cov_mat = np.diag([self.likelihoods[comp].variance[idx_a], self.likelihoods[comp].variance[idx_b]])
            joints.append({"mean": mean_vec, "cov": cov_mat, "weight": self.likelihoods[comp].weight})
        return joints

    def initLikelihoods(self, init_mean=None, init_std=None, weights=None):
        assert init_mean is None or isinstance(init_mean, list)
        assert init_std is None or isinstance(init_std, list)
        if isinstance(init_mean, list): assert len(init_mean) == self.likelihood_terms
        if isinstance(init_std, list): assert len(init_std) == self.likelihood_terms

        # If not provided, initialize from per-station first/second moments of the empirical PDFs in self.y_data.
        # This keeps `NormalLikelihood.mean` as a numeric vector (len = number of stations), which GGMP expects.
        # Always calculate station observational moments for later weight fitting (Phase 2)
        # We need these regardless of whether init_mean was passed.
        station_means = np.zeros(self.len_data, dtype=float)
        station_vars = np.zeros(self.len_data, dtype=float)

        for i, (domain, density) in enumerate(self.y_data):
            domain = np.asarray(domain, dtype=float)
            density = np.asarray(density, dtype=float)
            dx = np.abs(np.gradient(domain))
            mass = np.maximum(density, 0.0) * dx
            z = float(np.sum(mass))
            if z <= 0:
                mu = float(np.mean(domain))
                var = float(np.var(domain))
            else:
                p = mass / z
                mu = float(np.sum(p * domain))
                var = float(np.sum(p * (domain - mu) ** 2))
            station_means[i] = mu
            station_vars[i] = max(var, 1e-6)

        self.station_means = station_means
        self.station_vars = station_vars

        # If not provided, initialize from per-station first/second moments of the empirical PDFs in self.y_data.
        # This keeps `NormalLikelihood.mean` as a numeric vector (len = number of stations), which GGMP expects.
        if init_mean is None or init_std is None:
            # Already calculated station_means/vars above
            pass

        if init_mean is None:
            # Spread initial component means slightly so components aren't identical
            base = self.station_means
            offsets = np.linspace(-1.0, 1.0, self.likelihood_terms)
            init_mean = [base + offsets[k] for k in range(self.likelihood_terms)]
        if init_std is None:
            init_std = [np.sqrt(station_vars) for _ in range(self.likelihood_terms)]

        #var_bounds = std_bounds**2
        if weights is None:
            weights = np.ones((self.likelihood_terms))
            weights = weights / np.sum(weights)

        self.likelihoods = []
        for i in range(self.likelihood_terms):
            self.likelihoods.append(NormalLikelihood(init_mean[i], init_std[i] ** 2, weights[i]))

        return self.likelihoods

    def initGPs(self):
        gp_kwargs = getattr(self, "gp_init_kwargs", {}) or {}

        def _normalize_gp_kwargs(kwargs):
            """
            fvGP exposes advanced compute backends via the `args` dict (e.g. GPU_engine,
            Chol_*_compute_device). For convenience, we also accept a top-level
            `compute_device` and expand it into the relevant args keys.
            """
            if not kwargs:
                return {}
            kwargs = dict(kwargs)
            args = dict(kwargs.get("args") or {})
            compute_device = kwargs.pop("compute_device", None)
            if compute_device is not None:
                # Expand into fvGP's recognized args keys.
                dev = str(compute_device)
                # Also pass through as a first-class kwarg for fvGP versions that accept it.
                kwargs.setdefault("compute_device", dev)
                for k in (
                    "random_logdet_lanczos_compute_device",
                    "Chol_factor_compute_device",
                    "update_Chol_factor_compute_device",
                    "Chol_solve_compute_device",
                    "Chol_logdet_compute_device",
                ):
                    args.setdefault(k, dev)
            if args:
                kwargs["args"] = args
            return kwargs

        gp_kwargs = _normalize_gp_kwargs(gp_kwargs)
        # fvGP's default gradient path (`ram_economy=False`) can allocate extremely large
        # intermediate arrays (O(d * N^2)) during neg_log_likelihood_gradient for large N.
        # Default to `ram_economy=True` unless the caller explicitly overrides it.
        gp_kwargs.setdefault("ram_economy", True)
        self._gp_kwargs_supported = getattr(self, "_gp_kwargs_supported", None)
        self._gpu_engine = None
        self._effective_gp_device_ids = None

        args = dict(gp_kwargs.get("args") or {})
        gpu_engine = args.get("GPU_engine")
        if gpu_engine is not None:
            self._gpu_engine = str(gpu_engine).lower()

        # Determine if fvGP is instructed to use GPU for any core linear-algebra paths.
        using_gpu = False
        for k in (
            "random_logdet_lanczos_compute_device",
            "Chol_factor_compute_device",
            "update_Chol_factor_compute_device",
            "Chol_solve_compute_device",
            "Chol_logdet_compute_device",
        ):
            if str(args.get(k, "")).lower() == "gpu":
                using_gpu = True
                break

        def _gpu_count():
            eng = self._gpu_engine
            if eng == "torch" or eng is None:
                try:
                    import torch  # type: ignore
                    return int(torch.cuda.device_count())
                except Exception:
                    pass
            if eng == "cupy" or eng is None:
                try:
                    import cupy  # type: ignore
                    return int(cupy.cuda.runtime.getDeviceCount())
                except Exception:
                    pass
            return 0

        if using_gpu:
            n_gpu = _gpu_count()
            if n_gpu > 0:
                req = getattr(self, "gp_device_ids", None)
                if req is None:
                    # Default: one GPU.
                    self._effective_gp_device_ids = [0]
                elif isinstance(req, str) and req.lower() == "auto":
                    self._effective_gp_device_ids = list(range(n_gpu))
                else:
                    ids = [int(x) for x in (req if isinstance(req, (list, tuple, np.ndarray)) else [req])]
                    ids = [i for i in ids if 0 <= i < n_gpu]
                    self._effective_gp_device_ids = ids or [0]
            else:
                # GPU was requested but no backend/device is available: force a true CPU fallback
                # so fvGP doesn't attempt GPU linear algebra with an invalid backend.
                print(
                    "[initGPs] GPU compute was requested via gp_init_kwargs, but no GPU backend was detected; falling back to CPU.")
                using_gpu = False
                self._gpu_engine = None
                self._effective_gp_device_ids = None
                gp_kwargs["compute_device"] = "cpu"
                args = dict(gp_kwargs.get("args") or {})
                for k in (
                    "random_logdet_lanczos_compute_device",
                    "Chol_factor_compute_device",
                    "update_Chol_factor_compute_device",
                    "Chol_solve_compute_device",
                    "Chol_logdet_compute_device",
                ):
                    if k in args and str(args[k]).lower() == "gpu":
                        args[k] = "cpu"
                if args:
                    gp_kwargs["args"] = args
        else:
            self._effective_gp_device_ids = None

        gp_init_params = None
        try:
            gp_init_params = set(inspect.signature(GP.__init__).parameters.keys())
        except Exception:
            gp_init_params = None

        dropped_keys_total = set()

        def _safe_gp_init(y, init_hps, kwargs, *, required_keys=()):
            """
            Best-effort GP constructor across fvGP versions:
            - If fvGP rejects an *optional* kwarg (TypeError: unexpected keyword), drop it and retry.
            - If fvGP rejects a kwarg we are *specifically* trying (kernel/mean alias), return None so
              the caller can try the next alias instead of silently dropping the feature.
            """
            kwargs = dict(kwargs)
            removed = []
            while True:
                try:
                    return GP(self.x_data, y, init_hps, **kwargs), removed
                except TypeError as e:
                    m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
                    if m:
                        bad = m.group(1)
                        if bad in required_keys:
                            removed.append(bad)
                            return None, removed
                        if bad in kwargs:
                            removed.append(bad)
                            kwargs.pop(bad)
                            continue
                    raise

        if not hasattr(self, "likelihoods") or not self.likelihoods:
            raise ValueError("Initialize likelihoods first (call initLikelihoods).")

        def _make_component_gp(k):
            """
            Create a GP for component k, trained on likelihoods[k].mean with
            likelihoods[k].variance as noise.
            """
            # Each GP gets its dedicated component's mean/variance
            y = np.asarray(self.likelihoods[k].mean, dtype=float).ravel()
            nv = np.asarray(self.likelihoods[k].variance, dtype=float).ravel()
            init_hps = np.asarray(self.hps_obj.hps[k], dtype=float).copy()

            # CRITICAL FIX: Initialize mean hyperparameter to component mean
            # The last hyperparameter is the prior mean when using constant_mean function
            # This ensures each GP starts with a unique prior mean matching its data
            init_hps[-1] = float(y.mean())

            base_kwargs = {"noise_variances": nv}

            # Kernel kwarg name differs across fvGP versions; try aliases.
            kf = self.gp_kernel_functions[k] if k < len(self.gp_kernel_functions) else None
            kernel_keys = ["kernel_function", "gp_kernel_function", "kernel"]
            if gp_init_params is not None:
                kernel_keys = [key for key in kernel_keys if key in gp_init_params] + [key for key in kernel_keys if
                                                                                       key not in gp_init_params]
            kernel_kwargs_list = [{}] if kf is None else [{key: kf} for key in kernel_keys] + [{}]

            # Prior mean kwarg name differs across fvGP versions; try aliases.
            pm = self.gp_mean_functions[k] if k < len(self.gp_mean_functions) else None
            mean_keys = ["prior_mean_function", "prior_mean", "mean_function"]
            if gp_init_params is not None:
                mean_keys = [key for key in mean_keys if key in gp_init_params] + [key for key in mean_keys if
                                                                                   key not in gp_init_params]
            mean_kwargs_list = [{}] if pm is None else [{key: pm} for key in mean_keys] + [{}]

            # Filter gp_kwargs by signature, if known; otherwise let _safe_gp_init drop unsupported keys.
            filtered_gp_kwargs = gp_kwargs
            if gp_init_params is not None and gp_kwargs:
                filtered_gp_kwargs = {key: val for key, val in gp_kwargs.items() if key in gp_init_params}

            for k_kwargs in kernel_kwargs_list:
                for m_kwargs in mean_kwargs_list:
                    required = tuple(k_kwargs.keys()) + tuple(m_kwargs.keys())
                    kwargs = {**base_kwargs, **k_kwargs, **m_kwargs, **filtered_gp_kwargs}
                    gp_obj, removed = _safe_gp_init(y, init_hps, kwargs, required_keys=required)
                    dropped_keys_total.update(removed)
                    if gp_obj is not None:
                        return gp_obj

            # Final fallback: no kernel/mean kwargs, and drop any remaining unsupported gp_kwargs.
            gp_obj, removed = _safe_gp_init(y, init_hps, {**base_kwargs, **filtered_gp_kwargs}, required_keys=())
            dropped_keys_total.update(removed)
            if gp_obj is None:
                raise RuntimeError("Failed to initialize fvGP GP object with the provided configuration.")
            return gp_obj

        # Create K GPs, one per component
        self._component_GPs = [_make_component_gp(k) for k in range(self.likelihood_terms)]
        # Legacy alias for backward compatibility
        self._expert_GPs = self._component_GPs
        self.gps = self._component_GPs

        # CRITICAL: Sync hps_obj with actual GP hyperparameters (including corrected mean values)
        # Without this, training would reset GPs to the original hps_obj values (with mean=0)
        # We must use hps_obj.set() to properly update the cached vectorized_hps!
        synced_hps = []
        for k in range(self.likelihood_terms):
            gp = self._component_GPs[k]
            actual_hps = np.asarray(gp.hyperparameters, dtype=float).copy()
            synced_hps.append(actual_hps)
        self.hps_obj.set(self.hps_obj.weights, synced_hps)
        print(f"[initGPs] Synced hps_obj: mean values = {[float(h[-1]) for h in synced_hps]}")

        if dropped_keys_total:
            print(f"[initGPs] Note: GP.__init__ rejected kwargs: {sorted(dropped_keys_total)}")

        if using_gpu and self._effective_gp_device_ids is None:
            print(
                "[initGPs] GPU compute was requested via gp_init_kwargs, but no GPU backend was detected; falling back to CPU.")
        if self._effective_gp_device_ids is not None and len(self._effective_gp_device_ids) > 1:
            print(
                f"[initGPs] Multi-GPU enabled: device_ids={self._effective_gp_device_ids} (engine={self._gpu_engine or 'auto'})")
        elif self._effective_gp_device_ids is not None:
            print(
                f"[initGPs] GPU enabled: device_id={self._effective_gp_device_ids[0]} (engine={self._gpu_engine or 'auto'})")

    @contextmanager
    def _gpu_device(self, device_id):
        """
        Best-effort GPU device context (torch/cupy). If anything fails, falls back silently.
        """
        if device_id is None:
            yield
            return
        eng = getattr(self, "_gpu_engine", None)
        if eng == "torch" or eng is None:
            try:
                import torch  # type: ignore
                prev = torch.cuda.current_device()
                torch.cuda.set_device(int(device_id))
                try:
                    yield
                finally:
                    torch.cuda.set_device(int(prev))
                return
            except Exception:
                pass
        if eng == "cupy" or eng is None:
            try:
                import cupy  # type: ignore
                prev = int(cupy.cuda.runtime.getDevice())
                cupy.cuda.Device(int(device_id)).use()
                try:
                    yield
                finally:
                    cupy.cuda.Device(int(prev)).use()
                return
            except Exception:
                pass
        yield

    def _pair_device_id(self, j, i):
        ids = getattr(self, "_effective_gp_device_ids", None)
        if not ids:
            return None
        return ids[(int(j) * int(self.number_of_GPs) + int(i)) % len(ids)]

    @staticmethod
    def _as_float(x, *, reduce="sum"):
        """
        Best-effort conversion of fvGP return values to a scalar float.

        Some fvGP versions return a scalar numpy.float64 for (neg_)log_likelihood,
        while others may return a 1D/2D array (e.g. per-output or per-datum).
        For GGMP we need a single scalar; by default we sum all entries.
        """
        if x is None:
            raise TypeError("Expected a numeric value, got None.")

        # Common fast path
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)

        arr = np.asarray(x)
        if arr.shape == ():
            return float(arr)
        flat = arr.ravel()
        if flat.size == 1:
            return float(flat[0])
        red = str(reduce).lower()
        if red == "sum":
            return float(np.sum(flat))
        if red == "mean":
            return float(np.mean(flat))
        raise ValueError("reduce must be 'sum' or 'mean'.")

    def _gp_log_likelihood(self, gp):
        """
        Compute scalar log-likelihood for a configured GP object.

        Prefers marginal_density.neg_log_likelihood (most consistent with gradients),
        and falls back to marginal_density.log_likelihood / gp.log_likelihood.
        """
        md = getattr(gp, "marginal_density", None)
        if md is not None:
            if hasattr(md, "neg_log_likelihood"):
                return -self._as_float(md.neg_log_likelihood(hyperparameters=None), reduce="sum")
            if hasattr(md, "log_likelihood"):
                return self._as_float(md.log_likelihood(hyperparameters=None), reduce="sum")
        if hasattr(gp, "log_likelihood"):
            hps = getattr(getattr(gp, "trainer", None), "hyperparameters", None)
            return self._as_float(gp.log_likelihood(hps), reduce="sum")
        raise AttributeError("GP object has no (neg_)log_likelihood method available.")

    def _gp_neg_log_likelihood_gradient(self, gp):
        """
        Compute gradient of the negative log-likelihood for a configured GP object.

        Uses fvGP's analytical gradient (fixed in v4.7.8), falling back to numerical
        gradient for gp2Scale/GPU mode where analytical is not supported.
        """
        md = getattr(gp, "marginal_density", None)
        if md is None:
            raise AttributeError("GP has no marginal_density object")

        hps = np.asarray(gp.hyperparameters, dtype=float)

        # Check if gp2Scale is enabled (GPU mode) - analytical gradient not supported
        is_gp2scale = getattr(getattr(gp, "data", None), "gp2Scale", False)

        if not is_gp2scale and hasattr(md, "neg_log_likelihood_gradient"):
            # Use fvGP's analytical gradient (fixed in v4.7.8)
            try:
                return md.neg_log_likelihood_gradient(hyperparameters=hps, component=0)
            except Exception:
                pass  # Fall through to numerical

        # Fallback to numerical gradient for gp2Scale/GPU mode
        n_hps = len(hps)
        epsilon = 1e-6
        nll_base = float(md.neg_log_likelihood(hyperparameters=hps))

        grad = np.zeros(n_hps, dtype=float)
        for i in range(n_hps):
            hps_plus = hps.copy()
            hps_plus[i] += epsilon
            nll_plus = float(md.neg_log_likelihood(hyperparameters=hps_plus))
            grad[i] = (nll_plus - nll_base) / epsilon

        return grad

    def _set_expert_component(self, expert_idx, component_idx):
        """
        DEPRECATED: In the new architecture, each GP is dedicated to one component.
        This method is kept for legacy compatibility but just returns the GP.
        """
        if not hasattr(self, "_component_GPs"):
            self.initGPs()
        # In new architecture, expert_idx should equal component_idx
        k = int(component_idx)
        if k < 0 or k >= self.likelihood_terms:
            raise IndexError("component_idx out of range.")
        return self._component_GPs[k]

    def _safe_set_hyperparameters(self, gp, hps_new):
        """
        Update GP hyperparameters and trigger the necessary state rebuild.

        We use gp.set_hyperparameters() directly but ensure y_data has the
        correct shape to avoid reshape errors.
        """
        hps_new = np.asarray(hps_new, dtype=float)

        # Check if hyperparameters actually changed
        try:
            current_hps = np.asarray(gp.trainer.hyperparameters, dtype=float)
            if np.array_equal(current_hps, hps_new):
                return  # No change needed
        except Exception:
            pass

        # Ensure y_data has the correct shape (N, 1) for fvGP's log_likelihood
        if hasattr(gp, 'data') and hasattr(gp.data, 'y_data'):
            y_data = gp.data.y_data
            if y_data is not None and y_data.ndim == 1:
                gp.data.y_data = y_data.reshape(-1, 1)

        # Use the standard set_hyperparameters to properly update state
        gp.set_hyperparameters(hps_new)

    def _eval_ll_vector(self, hps):
        """
        Compute ll[k] = log_likelihood for component k's dedicated GP.
        Returns (ll, failures).

        Each GP is trained on its own component's data, so we just
        evaluate K log-likelihoods (one per component).
        """
        t0 = time.time()
        K = int(self.likelihood_terms)
        ll = np.empty(K, dtype=float)
        failures = 0

        if not hasattr(self, "_component_GPs"):
            self.initGPs()

        for k in range(K):
            gp = self._component_GPs[k]
            hps_k = np.asarray(hps[k], dtype=float)

            # Update hyperparameters
            self._safe_set_hyperparameters(gp, hps_k)

            try:
                ll[k] = float(self._gp_log_likelihood(gp))
            except (LinAlgError, ValueError, RuntimeError) as e:
                failures += 1
                ll[k] = -1e20

        self._last_ll_vector_time = float(time.time() - t0)
        self._last_ll_vector_failures = int(failures)
        return ll, failures

    def _eval_ll_matrix(self, hps):
        """
        Legacy wrapper: returns (K, K) matrix for backward compatibility,
        but with simplified diagonal structure (each GP only contributes to its component).
        """
        K = int(self.likelihood_terms)
        ll_vec, failures = self._eval_ll_vector(hps)
        # Create a "diagonal" matrix where ll[k,k] = ll_vec[k], others = -inf
        ll = np.full((K, K), -1e20, dtype=float)
        np.fill_diagonal(ll, ll_vec)
        self._last_ll_matrix_failures = failures
        return ll, failures


class hyperparameters: # pragma: no cover
    """
    Parameters:
        * weights: 1d numpy array
        * weights_bounds: 2d numpy array
        * hps: list of 1d numpy arrays
        * hps_bounds: list of 2d numpy arrays
    """

    def __init__(self, weights, weights_bounds, hps, hps_bounds):
        self.hps_bounds = hps_bounds
        self.weights_bounds = weights_bounds
        self.weights = weights
        self.hps = hps
        self.number_of_weights = len(weights)
        self.number_of_hps_sets = len(hps)
        self.number_of_hps = [len(hps[i]) for i in range(len(hps))]
        if len(hps) != len(hps_bounds): raise Exception("hps and hps_bounds have to be lists of equal length")
        if len(weights) != len(weights_bounds):
            raise Exception("weights (1d) and weights_bounds (2d) have to be numpy arrays of equal length")

        self.vectorized_hps = self.vectorize_hps(weights, hps)
        self.vectorized_bounds = self.vectorize_bounds(weights_bounds, hps_bounds)

    def set(self, weights, hps):
        if len(hps) != len(self.hps_bounds): raise Exception("hps and hps_bounds have to be lists of equal length")
        if len(weights) != len(self.weights_bounds):
            raise Exception("weights (1d) and weights_bounds (2d) have to be numpy arrays of equal length")

        self.weights = weights
        self.hps = hps
        self.vectorized_hps = self.vectorize_hps(weights, hps)

    def vectorize_hps(self, weights, hps):
        v = [weights[i] for i in range(self.number_of_weights)]
        for i in range(self.number_of_hps_sets):
            for j in range(self.number_of_hps[i]):
                v.append(hps[i][j])
        return np.asarray(v)

    def devectorize_hps(self, v):
        weights = v[0:self.number_of_weights]
        index = self.number_of_weights
        hps = []
        for i in range(self.number_of_hps_sets):
            hps.append(v[index:index + self.number_of_hps[i]])
            index += self.number_of_hps[i]
        return weights, hps

    def vectorize_bounds(self, weights_bounds, hps_bounds):
        b = [weights_bounds[i] for i in range(self.number_of_weights)]
        for i in range(self.number_of_hps_sets):
            for j in range(self.number_of_hps[i]):
                b.append(hps_bounds[i][j])
        return np.asarray(b)

    def devectorize_bounds(self, b):
        weights_bounds = b[0:self.number_of_weights]
        index = self.number_of_weights
        hps_bounds = []
        for i in range(self.number_of_hps_sets):
            hps_bounds.append(b[index:index + self.number_of_hps[i]])
            index += self.number_of_hps[i]
        return weights_bounds, hps_bounds


class NormalLikelihood: # pragma: no cover
    def __init__(self, mean, variance, weight):
        self.mean = mean
        self.variance = variance
        self.dim = len(mean)
        self.weight = weight
        self.weight_bounds = np.array([0, 1])

    def set_moments(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def set_weight(self, weight):
        self.weight = weight

    def unravel(self):
        return np.concatenate([self.mean, self.variance])

    def ravel(self, vec):
        return vec[0:self.dim], vec[self.dim:]

#    def marginalize(self, domain, direction):
#        return gaussian(self.mean[direction], np.sqrt(self.variance[direction]), domain)


# ============================================================================
# Eq. 11 EM end-to-end helpers (CURRENT)
# Used by `climate_ggmp_eq11_em_end_to_end.ipynb`.
# ============================================================================


def _get_key(d, keys: Iterable[str]): # pragma: no cover
    """
    Helper for fvGP return dicts that may use different key names across versions.
    """
    if not isinstance(d, dict):
        return d
    for k in keys:
        if k in d:
            return d[k]
    return next(iter(d.values()))


def gaussian_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray: # pragma: no cover
    """
    1D Gaussian PDF evaluated at x (variance parameterization).
    """
    x = np.asarray(x, dtype=float)
    var = float(max(var, 1e-12))
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def _normalize_pdf(domain: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # pragma: no cover
    """
    Normalize an (unnormalized) density on a grid to integrate to 1.

    Returns (domain, p, dx) where sum(p * dx) == 1 (up to fp error).
    """
    domain = np.asarray(domain, dtype=float).reshape(-1)
    density = np.asarray(density, dtype=float).reshape(-1)
    if domain.shape != density.shape:
        raise ValueError("domain and density must have the same shape")
    dx = np.abs(np.gradient(domain))
    mass = np.maximum(density, 0.0) * dx
    z = float(np.sum(mass))
    if not np.isfinite(z) or z <= 0.0:
        p = np.ones_like(domain, dtype=float) / float(domain.size)
        dx = np.ones_like(domain, dtype=float) * (1.0 / float(domain.size))
        return domain, p, dx
    p = np.maximum(density, 0.0) / z
    return domain, p, dx


def empirical_pdf_from_samples(y: np.ndarray, *, bins: int = 120) -> tuple[np.ndarray, np.ndarray]: # pragma: no cover
    """
    Empirical 1D PDF from samples via a normalized histogram.
    Returns (domain_centers, density) with unit integral.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    hist, edges = np.histogram(y, bins=int(bins), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = np.maximum(hist, 1e-12)
    centers, density, _dx = _normalize_pdf(centers, density)
    return centers, density


def fit_gmm_fixed_weights(
    y: np.ndarray,
    K: int,
    w_fixed: np.ndarray,
    *,
    means_init: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]: # pragma: no cover
    """
    Fit a K-component 1D Gaussian mixture to samples y with FIXED weights w_fixed.
    Only means/variances are updated (weighted EM).
    Returns ordered (means, variances) by increasing mean.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = int(y.shape[0])
    if n == 0:
        raise ValueError("Empty station series")

    K = int(K)
    w_fixed = np.asarray(w_fixed, dtype=float).reshape(-1)
    if w_fixed.size != K:
        raise ValueError("w_fixed must have length K")
    w_fixed = np.maximum(w_fixed, 1e-12)
    w_fixed = w_fixed / float(np.sum(w_fixed))

    if means_init is None:
        # Prefer a stable init (k-means), but fall back to quantiles if sklearn isn't available.
        try:
            from sklearn.cluster import KMeans  # type: ignore

            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            km.fit(y)
            means = np.sort(km.cluster_centers_.reshape(-1))
        except Exception:
            y_flat = y.reshape(-1)
            means = np.quantile(y_flat, np.linspace(0.1, 0.9, K))
            means = np.asarray(means, dtype=float).reshape(-1)
    else:
        means = np.asarray(means_init, dtype=float).reshape(-1)
        if means.size != K:
            raise ValueError("means_init must have length K")

    vars_ = np.ones(K, dtype=float) * float(np.var(y)) / float(max(K, 1))
    y_flat = y.reshape(-1)

    for _ in range(int(max_iter)):
        old_means = means.copy()

        # E-step with fixed weights
        resp = np.zeros((n, K), dtype=float)
        for k in range(K):
            resp[:, k] = w_fixed[k] * gaussian_pdf(y_flat, float(means[k]), float(vars_[k]))
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-12)

        # M-step: update means/vars (not weights)
        for k in range(K):
            r_k = resp[:, k]
            N_k = float(np.sum(r_k) + 1e-12)
            means[k] = float(np.sum(r_k * y_flat) / N_k)
            vars_[k] = float(np.sum(r_k * (y_flat - means[k]) ** 2) / N_k + 1e-6)

        if float(np.max(np.abs(means - old_means))) < float(tol):
            break

    order = np.argsort(means)
    return means[order], vars_[order]


def _as_2d(y: np.ndarray) -> np.ndarray: # pragma: no cover
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError("Expected array with shape (n_samples, n_dims).")
    if y.shape[0] == 0:
        raise ValueError("Empty sample array.")
    return y


def _covariances_to_full(
    covariances: np.ndarray,
    *,
    covariance_type: str,
    K: int,
    d: int,
) -> np.ndarray: # pragma: no cover
    cov_type = str(covariance_type).lower()
    cov = np.asarray(covariances, dtype=float)
    if cov_type == "full":
        out = cov
    elif cov_type == "diag":
        out = np.array([np.diag(cov[k]) for k in range(int(K))], dtype=float)
    elif cov_type == "spherical":
        out = np.array([np.eye(int(d), dtype=float) * float(cov[k]) for k in range(int(K))], dtype=float)
    elif cov_type == "tied":
        out = np.repeat(cov.reshape(1, int(d), int(d)), int(K), axis=0)
    else:
        raise ValueError(f"Unsupported covariance_type={covariance_type!r}")
    return np.asarray(out, dtype=float)


def fit_gmm_free_weights_multivariate(
    y: np.ndarray,
    K: int,
    *,
    covariance_type: str = "diag",
    reg_covar: float = 1e-6,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int | None = 42,
    init_params: str = "kmeans",
    weight_floor: float = 1e-9,
    sort_if_1d: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]: # pragma: no cover
    """
    Fit K-component multivariate GMM with free weights.

    Returns:
      weights: (K,)
      means:   (K, d)
      covs:    (K, d, d) full covariance form
      info:    dict with diagnostics
    """
    y = _as_2d(y)
    K = int(K)
    if y.shape[0] < K:
        raise ValueError(f"Need at least K samples. n={y.shape[0]}, K={K}")

    try:
        from sklearn.mixture import GaussianMixture  # type: ignore
    except Exception as exc:
        raise ImportError("scikit-learn is required for multivariate GMM fitting.") from exc

    gm = GaussianMixture(
        n_components=K,
        covariance_type=str(covariance_type),
        reg_covar=float(reg_covar),
        n_init=int(max(1, n_init)),
        max_iter=int(max(1, max_iter)),
        tol=float(tol),
        random_state=random_state,
        init_params=str(init_params),
    )
    gm.fit(y)

    weights = np.asarray(gm.weights_, dtype=float).reshape(-1)
    means = np.asarray(gm.means_, dtype=float)
    covs = _covariances_to_full(
        gm.covariances_,
        covariance_type=str(covariance_type),
        K=int(K),
        d=int(y.shape[1]),
    )

    weights = np.maximum(weights, float(weight_floor))
    weights = weights / float(np.sum(weights))

    if bool(sort_if_1d) and means.shape[1] == 1:
        order = np.argsort(means[:, 0])
        weights = weights[order]
        means = means[order]
        covs = covs[order]

    info = {
        "converged": bool(getattr(gm, "converged_", True)),
        "n_iter": int(getattr(gm, "n_iter_", 0)),
        "lower_bound": float(getattr(gm, "lower_bound_", np.nan)),
        "aic": float(gm.aic(y)),
        "bic": float(gm.bic(y)),
        "covariance_type": str(covariance_type),
        "reg_covar": float(reg_covar),
        "n_init": int(max(1, n_init)),
    }
    return weights, means, covs, info


def fit_local_gmms_multivariate(
    y_list: Sequence[np.ndarray],
    K: int,
    *,
    covariance_type: str = "diag",
    reg_covar: float = 1e-6,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int | None = 42,
    init_params: str = "kmeans",
    weight_floor: float = 1e-9,
) -> dict: # pragma: no cover
    """
    Fit one free-weight multivariate GMM per input.
    """
    weights_list = []
    means_list = []
    covs_list = []
    info_list = []

    for n, y in enumerate(y_list):
        rs = None if random_state is None else int(random_state) + int(n)
        w, m, c, info = fit_gmm_free_weights_multivariate(
            y,
            K,
            covariance_type=str(covariance_type),
            reg_covar=float(reg_covar),
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=rs,
            init_params=str(init_params),
            weight_floor=float(weight_floor),
            sort_if_1d=False,
        )
        weights_list.append(w)
        means_list.append(m)
        covs_list.append(c)
        info_list.append(info)

    return {
        "weights": weights_list,
        "means": means_list,
        "covs": covs_list,
        "fit_info": info_list,
        "K": int(K),
        "d": int(np.asarray(means_list[0]).shape[1]),
    }


def _sym_psd(a: np.ndarray) -> np.ndarray: # pragma: no cover
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


def _sqrtm_psd(a: np.ndarray, *, eps: float = 1e-12) -> np.ndarray: # pragma: no cover
    a = _sym_psd(a)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, float(eps), None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def gaussian_w2_squared(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> float: # pragma: no cover
    """
    Squared W2 distance between two Gaussians.
    """
    mean_a = np.asarray(mean_a, dtype=float).reshape(-1)
    mean_b = np.asarray(mean_b, dtype=float).reshape(-1)
    cov_a = _sym_psd(np.asarray(cov_a, dtype=float))
    cov_b = _sym_psd(np.asarray(cov_b, dtype=float))

    diff = mean_a - mean_b
    term_mean = float(diff @ diff)
    sqrt_a = _sqrtm_psd(cov_a)
    mid = _sym_psd(sqrt_a @ cov_b @ sqrt_a)
    sqrt_mid = _sqrtm_psd(mid)
    term_cov = float(np.trace(cov_a + cov_b - 2.0 * sqrt_mid))
    return float(term_mean + max(term_cov, 0.0))


def align_gmm_components_hungarian(
    means_ref: np.ndarray,
    covs_ref: np.ndarray,
    means_cur: np.ndarray,
    covs_cur: np.ndarray,
    *,
    metric: str = "w2",
    return_cost: bool = False,
): # pragma: no cover
    """
    Align current components to reference components by linear assignment.
    """
    means_ref = np.asarray(means_ref, dtype=float)
    means_cur = np.asarray(means_cur, dtype=float)
    covs_ref = np.asarray(covs_ref, dtype=float)
    covs_cur = np.asarray(covs_cur, dtype=float)

    if means_ref.shape != means_cur.shape:
        raise ValueError("means_ref and means_cur must have same shape")
    if covs_ref.shape != covs_cur.shape:
        raise ValueError("covs_ref and covs_cur must have same shape")

    K = int(means_ref.shape[0])
    cost = np.zeros((K, K), dtype=float)
    metric_key = str(metric).lower()
    if metric_key != "w2":
        raise ValueError("Currently supported metric is 'w2'.")

    for i in range(K):
        for j in range(K):
            cost[i, j] = gaussian_w2_squared(means_ref[i], covs_ref[i], means_cur[j], covs_cur[j])

    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(K, dtype=int)
    perm[rows] = cols
    if bool(return_cost):
        return perm, cost
    return perm


def align_local_gmms_sequence(
    weights_list: Sequence[np.ndarray],
    means_list: Sequence[np.ndarray],
    covs_list: Sequence[np.ndarray],
    *,
    metric: str = "w2",
    reference: str = "previous",
) -> dict: # pragma: no cover
    """
    Align local GMM components across inputs to a consistent global labeling.
    """
    if not (len(weights_list) == len(means_list) == len(covs_list)):
        raise ValueError("weights_list, means_list, covs_list must have equal length")
    N = int(len(means_list))
    if N == 0:
        raise ValueError("Empty sequence")

    ref_mode = str(reference).lower()
    if ref_mode not in {"previous", "first"}:
        raise ValueError("reference must be 'previous' or 'first'")

    aligned_w = [np.asarray(weights_list[0], dtype=float).copy()]
    aligned_m = [np.asarray(means_list[0], dtype=float).copy()]
    aligned_c = [np.asarray(covs_list[0], dtype=float).copy()]
    perms = [np.arange(aligned_m[0].shape[0], dtype=int)]
    costs = [None]

    m_first = aligned_m[0]
    c_first = aligned_c[0]

    for n in range(1, N):
        m_cur = np.asarray(means_list[n], dtype=float)
        c_cur = np.asarray(covs_list[n], dtype=float)
        w_cur = np.asarray(weights_list[n], dtype=float).reshape(-1)

        if ref_mode == "previous":
            m_ref = aligned_m[-1]
            c_ref = aligned_c[-1]
        else:
            m_ref = m_first
            c_ref = c_first

        perm, cost = align_gmm_components_hungarian(
            m_ref,
            c_ref,
            m_cur,
            c_cur,
            metric=str(metric),
            return_cost=True,
        )
        aligned_m.append(m_cur[perm].copy())
        aligned_c.append(c_cur[perm].copy())
        aligned_w.append(w_cur[perm].copy())
        perms.append(perm.copy())
        costs.append(cost.copy())

    return {
        "weights": aligned_w,
        "means": aligned_m,
        "covs": aligned_c,
        "perms": perms,
        "costs": costs,
        "metric": str(metric),
        "reference": ref_mode,
    }


def _log_mvn_density(
    y: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    reg: float = 1e-9,
) -> np.ndarray: # pragma: no cover
    y = _as_2d(y)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = _sym_psd(np.asarray(cov, dtype=float))
    d = int(mean.size)
    cov = cov + float(reg) * np.eye(d, dtype=float)
    return multivariate_normal.logpdf(y, mean=mean, cov=cov, allow_singular=False)


def optimize_weights_em_multivariate_samples(
    y_list: Sequence[np.ndarray],
    means_list: Sequence[np.ndarray],
    covs_list: Sequence[np.ndarray],
    *,
    K: int,
    weight_floor: float = 1e-9,
    max_iter: int = 200,
    tol_l1: float = 1e-10,
    log_every: int = 10,
    w0: np.ndarray | None = None,
    cov_reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # pragma: no cover
    """
    EM optimization of shared mixture weights on multivariate sample objective:
      sum_n sum_t log sum_k w_k N(y_nt | mu_nk, Sigma_nk)
    """
    K = int(K)
    if w0 is None:
        w = np.ones(K, dtype=float) / float(K)
    else:
        w = np.asarray(w0, dtype=float).reshape(-1)
        w = np.maximum(w, 1e-12)
        w = w / float(np.sum(w))

    w_hist = []
    obj_hist = []

    for it in range(int(max_iter)):
        log_w = np.log(np.maximum(w, 1e-300))
        counts = np.zeros(K, dtype=float)
        obj = 0.0

        for y_n, means_n, covs_n in zip(y_list, means_list, covs_list):
            y_n = _as_2d(y_n)
            means_n = np.asarray(means_n, dtype=float)
            covs_n = np.asarray(covs_n, dtype=float)

            log_pdf = np.zeros((y_n.shape[0], K), dtype=float)
            for k in range(K):
                log_pdf[:, k] = _log_mvn_density(y_n, means_n[k], covs_n[k], reg=float(cov_reg))

            log_num = log_pdf + log_w.reshape(1, -1)
            log_den = logsumexp(log_num, axis=1, keepdims=True)
            r = np.exp(log_num - log_den)
            counts += np.sum(r, axis=0)
            obj += float(np.sum(log_den))

        w_new = counts / float(np.sum(counts) + 1e-300)
        w_new = np.maximum(w_new, float(weight_floor))
        w_new = w_new / float(np.sum(w_new))

        w_hist.append(w_new.copy())
        obj_hist.append(float(obj))
        delta = float(np.linalg.norm(w_new - w, ord=1))
        if int(log_every) > 0 and ((it % int(log_every) == 0) or it == int(max_iter) - 1):
            logging.info("[EM-MV %03d] obj=%.6f | L1_delta=%.3e | w=%s", it, obj, delta,
                         np.array2string(w_new, precision=6))

        w = w_new
        if delta < float(tol_l1):
            break

    return w, np.asarray(w_hist), np.asarray(obj_hist)


def loglik_multivariate_mixture_samples(
    y: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    *,
    cov_reg: float = 1e-9,
) -> np.ndarray: # pragma: no cover
    """
    Per-sample log-likelihood under multivariate Gaussian mixture.
    """
    y = _as_2d(y)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    K = int(weights.size)

    log_w = np.log(np.maximum(weights, 1e-300))
    log_pdf = np.zeros((y.shape[0], K), dtype=float)
    for k in range(K):
        log_pdf[:, k] = _log_mvn_density(y, means[k], covs[k], reg=float(cov_reg))
    return logsumexp(log_pdf + log_w.reshape(1, -1), axis=1)


def sample_gmm_multivariate(
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    n_samples: int,
    *,
    random_state: int | None = None,
    cov_reg: float = 1e-9,
) -> np.ndarray: # pragma: no cover
    """
    Draw samples from a multivariate Gaussian mixture.
    """
    rng = np.random.default_rng(random_state)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    K = int(weights.size)

    comp = rng.choice(K, size=int(n_samples), p=weights / float(np.sum(weights)))
    out = np.zeros((int(n_samples), int(means.shape[1])), dtype=float)
    for k in range(K):
        idx = np.where(comp == k)[0]
        if idx.size == 0:
            continue
        cov_k = _sym_psd(covs[k]) + float(cov_reg) * np.eye(means.shape[1], dtype=float)
        out[idx] = rng.multivariate_normal(mean=means[k], cov=cov_k, size=idx.size)
    return out


def energy_distance_multivariate(
    a: np.ndarray,
    b: np.ndarray,
) -> float: # pragma: no cover
    """
    Energy distance between two multivariate empirical samples.
    """
    a = _as_2d(a)
    b = _as_2d(b)
    aa = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    bb = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=2)
    ab = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    val = 2.0 * np.mean(ab) - np.mean(aa) - np.mean(bb)
    return float(max(val, 0.0))


def sliced_wasserstein_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_projections: int = 64,
    random_state: int | None = 42,
) -> float: # pragma: no cover
    """
    Sliced Wasserstein distance via random 1D projections.
    """
    a = _as_2d(a)
    b = _as_2d(b)
    d = int(a.shape[1])
    rng = np.random.default_rng(random_state)
    acc = []
    for _ in range(int(max(1, n_projections))):
        v = rng.normal(size=d)
        v = v / max(np.linalg.norm(v), 1e-12)
        a_proj = a @ v
        b_proj = b @ v
        acc.append(float(wasserstein_distance(a_proj, b_proj)))
    return float(np.mean(acc))


def mmd_rbf(
    a: np.ndarray,
    b: np.ndarray,
    *,
    gamma: float | None = None,
) -> float: # pragma: no cover
    """
    Unbiased MMD^2 with RBF kernel.
    """
    a = _as_2d(a)
    b = _as_2d(b)
    n, m = int(a.shape[0]), int(b.shape[0])
    if n < 2 or m < 2:
        return float("nan")

    def _sqdist(x, y):
        return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)

    d_aa = _sqdist(a, a)
    d_bb = _sqdist(b, b)
    d_ab = _sqdist(a, b)

    if gamma is None:
        med = np.median(d_ab)
        gamma = 1.0 / max(2.0 * med, 1e-12)

    k_aa = np.exp(-float(gamma) * d_aa)
    k_bb = np.exp(-float(gamma) * d_bb)
    k_ab = np.exp(-float(gamma) * d_ab)

    np.fill_diagonal(k_aa, 0.0)
    np.fill_diagonal(k_bb, 0.0)
    term_aa = np.sum(k_aa) / float(n * (n - 1))
    term_bb = np.sum(k_bb) / float(m * (m - 1))
    term_ab = np.sum(k_ab) / float(n * m)
    return float(term_aa + term_bb - 2.0 * term_ab)


def _gmm_cache_path(
    *,
    cache_dir: Path,
    data_path: Path,
    K: int,
    max_iter: int,
    tol: float,
) -> tuple[Path, dict]: # pragma: no cover
    data_path = Path(data_path)
    st = data_path.stat()
    meta = {
        "data_file": str(data_path.resolve()),
        "data_size": int(st.st_size),
        "data_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "K": int(K),
        "fit_method": "fixed_weight_em_quantile_init",
        "weights": "uniform",
        "gmm_max_iter": int(max_iter),
        "gmm_tol": float(tol),
    }
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gmm_fits_{key}_K{int(K)}.npz", meta


def _load_gmm_cache(path: Path) -> dict | None: # pragma: no cover
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=False) as z:
            station_ids = np.asarray(z["station_ids"], dtype=int).reshape(-1)
            means = np.asarray(z["means"], dtype=float)
            vars_ = np.asarray(z["vars"], dtype=float)
            meta_json = str(z["meta_json"].tolist())
        meta = json.loads(meta_json)
        if means.shape != vars_.shape:
            return None
        if station_ids.shape[0] != means.shape[0]:
            return None
        return {"station_ids": station_ids, "means": means, "vars": vars_, "meta": meta}
    except Exception:
        return None


def _save_gmm_cache(
    path: Path,
    *,
    station_ids: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
    meta: dict,
) -> None: # pragma: no cover
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        str(tmp),
        station_ids=np.asarray(station_ids, dtype=int),
        means=np.asarray(means, dtype=float),
        vars=np.asarray(vars_, dtype=float),
        meta_json=json.dumps(meta, sort_keys=True),
    )
    tmp.replace(path)


def fit_station_gmms_fixed_weights_cached(
    series_list: list[np.ndarray],
    station_ids: np.ndarray,
    *,
    data_path: Path,
    K: int,
    gmm_max_iter: int = 100,
    gmm_tol: float = 1e-4,
    cache: bool = True,
    cache_dir: Path | None = None,
    log_every: int = 100,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, Path | None]: # pragma: no cover
    """
    Fit per-station ordered (means, vars) for a fixed-weight K-component 1D GMM.

    - Uses quantile init by default (stable ordering).
    - Optionally caches fits to disk keyed by data file stat + K + fit params.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    station_ids = np.asarray(station_ids, dtype=int).reshape(-1)
    if len(series_list) != station_ids.size:
        raise ValueError("series_list and station_ids must have the same length")

    K = int(K)
    w_init = np.ones(K, dtype=float) / float(K)
    means_mat = np.zeros((station_ids.size, K), dtype=float)
    vars_mat = np.zeros((station_ids.size, K), dtype=float)

    cache_path = None
    cache_meta = None
    cache_map: dict[int, int] = {}
    means_c = None
    vars_c = None
    cached = None

    if cache:
        cache_dir = Path(cache_dir) if cache_dir is not None else Path("ggmp_cache")
        cache_path, cache_meta = _gmm_cache_path(
            cache_dir=cache_dir,
            data_path=Path(data_path),
            K=K,
            max_iter=int(gmm_max_iter),
            tol=float(gmm_tol),
        )
        cached = _load_gmm_cache(cache_path)
        if cached is not None and cached.get("meta") == cache_meta:
            ids = np.asarray(cached["station_ids"], dtype=int).reshape(-1)
            means_c = np.asarray(cached["means"], dtype=float)
            vars_c = np.asarray(cached["vars"], dtype=float)
            if means_c.shape == (ids.size, K) and vars_c.shape == (ids.size, K):
                cache_map = {int(sid): j for j, sid in enumerate(ids)}
                logger.info("GMM cache hit: %s (stations=%d)", cache_path, ids.size)
            else:
                cached = None

    t0 = time.time()
    n_failed = 0
    added_station_ids: list[int] = []
    added_means: list[np.ndarray] = []
    added_vars: list[np.ndarray] = []

    for i, y in enumerate(series_list, start=1):
        sid = int(station_ids[i - 1])
        if sid in cache_map:
            j = cache_map[sid]
            means_mat[i - 1] = means_c[j]
            vars_mat[i - 1] = vars_c[j]
        else:
            try:
                y = np.asarray(y, dtype=float).reshape(-1)
                means_init = np.quantile(y, np.linspace(0.1, 0.9, K))
                m, v = fit_gmm_fixed_weights(
                    y,
                    K,
                    w_init,
                    means_init=means_init,
                    max_iter=int(gmm_max_iter),
                    tol=float(gmm_tol),
                )
                means_mat[i - 1] = m
                vars_mat[i - 1] = v
                if cache:
                    added_station_ids.append(sid)
                    added_means.append(m.copy())
                    added_vars.append(v.copy())
            except Exception as e:
                n_failed += 1
                logger.exception("Station GMM fit failed (station_id=%d): %s", sid, e)

        if log_every and ((i % int(log_every) == 0) or (i == len(series_list))):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-12)
            eta = (len(series_list) - i) / max(rate, 1e-12)
            logger.info(
                "  fitted %d/%d stations (%.1f%%) | failed=%d | elapsed=%.1fs | rate=%.2f st/s | eta=%.1fs",
                i,
                len(series_list),
                100.0 * i / max(len(series_list), 1),
                n_failed,
                elapsed,
                rate,
                eta,
            )

    logger.info("Component fits done in %.2fs (failed=%d)", time.time() - t0, n_failed)

    if cache and cache_path is not None and cache_meta is not None and (added_station_ids or cached is not None):
        if cached is None:
            all_ids = np.asarray(added_station_ids, dtype=int)
            all_means = np.asarray(added_means, dtype=float)
            all_vars = np.asarray(added_vars, dtype=float)
        else:
            old_ids = np.asarray(cached["station_ids"], dtype=int).reshape(-1)
            old_means = np.asarray(cached["means"], dtype=float)
            old_vars = np.asarray(cached["vars"], dtype=float)
            if added_station_ids:
                new_ids = np.asarray(added_station_ids, dtype=int)
                new_means = np.asarray(added_means, dtype=float)
                new_vars = np.asarray(added_vars, dtype=float)
                all_ids = np.concatenate([old_ids, new_ids])
                all_means = np.vstack([old_means, new_means])
                all_vars = np.vstack([old_vars, new_vars])
            else:
                all_ids, all_means, all_vars = old_ids, old_means, old_vars

        order = np.argsort(all_ids)
        all_ids = all_ids[order]
        all_means = all_means[order]
        all_vars = all_vars[order]
        uniq_ids, uniq_idx = np.unique(all_ids, return_index=True)
        all_ids = uniq_ids
        all_means = all_means[uniq_idx]
        all_vars = all_vars[uniq_idx]

        _save_gmm_cache(cache_path, station_ids=all_ids, means=all_means, vars_=all_vars, meta=cache_meta)
        logger.info("Wrote GMM cache: %s (stations=%d, added=%d)", cache_path, all_ids.size, len(added_station_ids))

    return means_mat, vars_mat, cache_path


def _parse_device_ids(s: str | None): # pragma: no cover
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.lower() == "auto":
        return "auto"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def build_gp_init_kwargs(*, use_gpu: bool = False, gpu_engine: str = "torch") -> tuple[dict, str | None]: # pragma: no cover
    """
    Convenience wrapper for GGMP(gp_init_kwargs=..., gp_device_ids=...).
    """
    if not use_gpu:
        return {"compute_device": "cpu"}, None
    return ({"compute_device": "gpu", "args": {"GPU_engine": str(gpu_engine)}}, None)


def _threadpool_limits_ctx(n_threads: int | None): # pragma: no cover
    """
    Best-effort runtime BLAS/OpenMP thread limiter using threadpoolctl (if available).
    """
    if n_threads is None:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        return threadpool_limits(limits=int(max(1, n_threads)))
    except Exception:
        return nullcontext()


def _atomic_savez(path: Path, **arrays) -> None: # pragma: no cover
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), **arrays)
    tmp.replace(path)


def _save_gp_mcmc_info(
    *,
    run_dir: Path,
    k: int,
    gp,
    thin: int = 1,
    tag: str = "",
    extra_meta: dict | None = None,
) -> None: # pragma: no cover
    """
    Save fvGP MCMC trace to disk.

    fvGP exposes trace information in `gp.mcmc_info` for `method='mcmc'`.
    """
    info = getattr(gp, "mcmc_info", None)
    if not isinstance(info, dict) or "x" not in info:
        return

    thin = int(max(1, thin))
    x = np.asarray(info.get("x"))
    x = x[::thin]
    f = np.asarray(info.get("f(x)"))[::thin] if "f(x)" in info else None
    t = np.asarray(info.get("time stamps"))[::thin] if "time stamps" in info else None

    meta = {
        "k": int(k),
        "thin": int(thin),
        "tag": str(tag),
        "keys": sorted(list(info.keys())),
    }
    if extra_meta:
        meta.update({str(kk): extra_meta[kk] for kk in extra_meta})

    payload = {
        "x": np.asarray(x),
        "meta_json": json.dumps(meta, sort_keys=True),
    }
    for key, out_key in (
        ("median(x)", "median_x"),
        ("mean(x)", "mean_x"),
        ("var(x)", "var_x"),
        ("MAP", "map"),
        ("max x", "max_x"),
        ("max f(x)", "max_f"),
    ):
        if key in info:
            payload[out_key] = np.asarray(info[key])
    if f is not None:
        payload["f"] = np.asarray(f)
    if t is not None:
        payload["time"] = np.asarray(t)

    suffix = f"_{tag}" if tag else ""
    path = Path(run_dir) / f"gp{k:02d}_mcmc_trace{suffix}.npz"
    _atomic_savez(path, **payload)


def train_gp_mcmc_until_converged(
    model,
    gp,
    *,
    bounds: np.ndarray,
    init_hps: np.ndarray,
    chunk: int,
    max_total: int,
    tol_rel: float,
    patience: int,
    verbose_prefix: str,
    trace_hook=None,
) -> tuple[np.ndarray, list[dict]]: # pragma: no cover
    """
    Heuristic "until convergence" loop for fvGP's `method='mcmc'`.

    We train in `chunk`-sized batches, reusing the last hyperparameters as the
    next init. We stop when relative hyperparameter change is below `tol_rel`
    for `patience` consecutive chunks, or when `max_total` iterations are reached.
    """
    chunk = int(max(chunk, 1))
    max_total = int(max(max_total, chunk))
    patience = int(max(patience, 1))
    tol_rel = float(tol_rel)

    hps = np.asarray(init_hps, dtype=float).copy()
    hist: list[dict] = []
    stable = 0
    total = 0

    while total < max_total:
        model._safe_set_hyperparameters(gp, hps)
        gp.train(
            hyperparameter_bounds=bounds,
            init_hyperparameters=hps,
            method="mcmc",
            max_iter=int(chunk),
            info=False,
        )
        new_hps = np.asarray(gp.hyperparameters, dtype=float).copy()
        denom = float(np.linalg.norm(hps) + 1e-12)
        rel = float(np.linalg.norm(new_hps - hps) / denom)

        total += int(chunk)
        stable = stable + 1 if rel < tol_rel else 0

        hist.append({"iters": int(total), "rel_hps_change": rel, "hps": new_hps.copy()})
        logging.info("%s iters=%d | rel_hps_change=%.3e | stable=%d/%d", verbose_prefix, total, rel, stable, patience)

        if trace_hook is not None:
            try:
                trace_hook(int(total), gp)
            except Exception:
                pass

        hps = new_hps
        if stable >= patience:
            break

    return hps, hist


def _suppress_fvgp_mcmc_overflow(): # pragma: no cover
    """
    fvGP's MCMC uses `np.exp(log_alpha)` for Metropolis ratios, which can overflow
    for large log_alpha. This is usually benign (ratio -> inf), but noisy.
    """
    return warnings.catch_warnings()


def train_component_gps_mcmc(
    model,
    hps_obj,
    *,
    n_updates_gp: int = 500,
    mcmc_until_converged: bool = False,
    mcmc_chunk: int = 100,
    mcmc_max_total: int = 5000,
    mcmc_tol_rel: float = 1e-3,
    mcmc_patience: int = 3,
    gp_parallel: bool = False,
    gp_workers: int | None = None,
    blas_threads_per_gp: int | None = None,
    run_dir: Path | None = None,
    save_gp_mcmc: bool = False,
    gp_mcmc_thin: int = 1,
    save_gp_mcmc_chunks: bool = True,
) -> list[np.ndarray]: # pragma: no cover
    """
    Train each component GP independently using fvGP MCMC.

    Returns a list of trained hyperparameters (length K).
    """
    K = int(model.likelihood_terms)
    trained_hps = [np.asarray(hps_obj.hps[k], dtype=float).copy() for k in range(K)]

    if mcmc_until_converged:
        logging.info(
            "Training %d GPs with MCMC until convergence (chunk=%d, max_total=%d, tol_rel=%g, patience=%d)...",
            K,
            int(mcmc_chunk),
            int(mcmc_max_total),
            float(mcmc_tol_rel),
            int(mcmc_patience),
        )
    else:
        logging.info("Training %d GPs with MCMC (iters=%d)...", K, int(n_updates_gp))

    def _train_one_gp(k: int) -> tuple[int, np.ndarray]: # pragma: no cover
        gp = model.gps[k]
        bounds = np.asarray(hps_obj.hps_bounds[k], dtype=float)

        # Round-robin device assignment for multi-GPU.
        device_id = None
        devs = getattr(model, "_effective_gp_device_ids", None)
        if devs is not None and len(devs) > 0:
            device_id = devs[int(k) % int(len(devs))]

        t_gp = time.time()
        with _threadpool_limits_ctx(blas_threads_per_gp), model._gpu_device(device_id):
            with _suppress_fvgp_mcmc_overflow():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*overflow encountered in exp.*",
                    category=RuntimeWarning,
                    module=r".*fvgp\\.gp_mcmc.*",
                )
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    if mcmc_until_converged:

                        def _hook(total_iters: int, _gp):
                            if not (save_gp_mcmc and save_gp_mcmc_chunks and run_dir is not None):
                                return
                            _save_gp_mcmc_info(
                                run_dir=Path(run_dir),
                                k=int(k),
                                gp=_gp,
                                thin=int(gp_mcmc_thin),
                                tag=f"chunk{int(total_iters):06d}",
                                extra_meta={"bounds": np.asarray(bounds, dtype=float).tolist()},
                            )

                        hps_k, _hist = train_gp_mcmc_until_converged(
                            model,
                            gp,
                            bounds=bounds,
                            init_hps=trained_hps[k],
                            chunk=int(mcmc_chunk),
                            max_total=int(mcmc_max_total),
                            tol_rel=float(mcmc_tol_rel),
                            patience=int(mcmc_patience),
                            verbose_prefix=f"  GP[{k}]",
                            trace_hook=_hook,
                        )
                        out = np.asarray(hps_k, dtype=float).copy()
                    else:
                        model._safe_set_hyperparameters(gp, trained_hps[k])
                        gp.train(
                            hyperparameter_bounds=bounds,
                            init_hyperparameters=trained_hps[k],
                            method="mcmc",
                            max_iter=int(n_updates_gp),
                            info=False,
                        )
                        out = np.asarray(gp.hyperparameters, dtype=float).copy()

        logging.info("  GP[%d] done in %.2fs | hps=%s", k, time.time() - t_gp, np.array2string(out, precision=4))

        if save_gp_mcmc and run_dir is not None:
            _save_gp_mcmc_info(
                run_dir=Path(run_dir),
                k=int(k),
                gp=gp,
                thin=int(gp_mcmc_thin),
                extra_meta={"bounds": np.asarray(bounds, dtype=float).tolist()},
            )

        return int(k), out

    if gp_parallel and K > 1:
        workers = int(gp_workers) if gp_workers is not None else int(K)
        workers = max(1, min(int(K), workers))
        if blas_threads_per_gp is None:
            logging.info(
                "Note: gp_parallel is enabled but blas_threads_per_gp is not set. "
                "Consider setting it (e.g. 1-4) to avoid BLAS oversubscription.",
            )
        logging.info("Parallel GP training enabled: workers=%d | BLAS_THREADS_PER_GP=%s", workers, blas_threads_per_gp)

        results = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_train_one_gp, k) for k in range(K)]
            for fut in as_completed(futs):
                k, hps_k = fut.result()
                results[int(k)] = hps_k
        for k in range(K):
            trained_hps[k] = np.asarray(results[int(k)], dtype=float).copy()
    else:
        for k in range(K):
            kk, out = _train_one_gp(k)
            trained_hps[kk] = out

    if run_dir is not None:
        try:
            np.save(Path(run_dir) / "trained_hps.npy", np.asarray(trained_hps, dtype=float))
            logging.info("Saved %s", Path(run_dir) / "trained_hps.npy")
        except Exception:
            pass

    return trained_hps


def prepare_station_terms_density(model, hps_list):  # pragma: no cover
    """
    For each station i, build (p_obs, dx, log_pdf_grid) where:
      log_pdf_grid[j,k] = log N(domain[j] | mu_ik, var_ik)
    with var_ik = GP predictive var + within-component variance (from fitted likelihoods).
    """
    K = int(model.likelihood_terms)
    N = int(model.len_data)

    mu = np.empty((N, K), dtype=float)
    var_total = np.empty((N, K), dtype=float)

    # Precompute GP predictive mean/var at training stations
    for k in range(K):
        gp = model.gps[k]
        model._safe_set_hyperparameters(gp, hps_list[k])

        pm = gp.posterior_mean(model.x_data)
        pc = gp.posterior_covariance(model.x_data, variance_only=True)
        mu[:, k] = np.asarray(_get_key(pm, ("m(x)", "f(x)")), dtype=float).reshape(-1)
        var_gp = np.asarray(_get_key(pc, ("v(x)", "v", "variance", "cov")), dtype=float).reshape(-1)
        var_gp = np.maximum(var_gp, 0.0)

        var_comp = np.asarray(model.likelihoods[k].variance, dtype=float).reshape(-1)
        var_comp = np.maximum(var_comp, 1e-9)

        var_total[:, k] = var_gp + var_comp

    terms = []
    ll_comp = np.zeros(K, dtype=float)

    for i, (domain, density) in enumerate(model.y_data):
        domain, p_obs, dx = _normalize_pdf(domain, density)
        log_pdf = np.empty((domain.size, K), dtype=float)
        for k in range(K):
            sigma = float(np.sqrt(max(var_total[i, k], 1e-12)))
            log_pdf[:, k] = norm.logpdf(domain, loc=float(mu[i, k]), scale=sigma)

        ll_comp += np.sum((p_obs[:, None] * log_pdf) * dx[:, None], axis=0)
        terms.append((p_obs, dx, log_pdf))

    return terms, ll_comp


def weight_objective_density(w: np.ndarray, terms) -> float: # pragma: no cover
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.maximum(w, 1e-300)
    w = w / float(np.sum(w))
    log_w = np.log(w)
    total = 0.0
    for p_obs, dx, log_pdf in terms:
        log_mix = logsumexp(log_pdf + log_w.reshape(1, -1), axis=1)
        total += float(np.sum(p_obs * dx * log_mix))
    return float(total)


def optimize_weights_em_density(
    terms,
    *,
    K: int,
    weight_floor: float,
    max_iter: int,
    tol_l1: float,
    log_every: int,
    w0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # pragma: no cover
    """
    EM for weights on the density-based objective.
    Returns (w, w_hist, obj_hist).
    """
    K = int(K)
    if w0 is None:
        w = np.ones(K, dtype=float) / float(K)
    else:
        w = np.asarray(w0, dtype=float).reshape(-1)
        w = np.maximum(w, 1e-12)
        w = w / float(np.sum(w))

    w_hist = []
    obj_hist = []

    for t in range(int(max_iter)):
        log_w = np.log(np.maximum(w, 1e-300))
        counts = np.zeros(K, dtype=float)
        obj = 0.0

        for p_obs, dx, log_pdf in terms:
            log_num = log_pdf + log_w.reshape(1, -1)
            log_den = logsumexp(log_num, axis=1, keepdims=True)
            r = np.exp(log_num - log_den)
            weights = (p_obs * dx).reshape(-1, 1)
            counts += np.sum(weights * r, axis=0)
            obj += float(np.sum((p_obs * dx) * log_den.reshape(-1)))

        w_new = counts / float(np.sum(counts))
        w_new = np.maximum(w_new, float(weight_floor))
        w_new = w_new / float(np.sum(w_new))

        w_hist.append(w_new.copy())
        obj_hist.append(float(obj))

        delta = float(np.linalg.norm(w_new - w, ord=1))
        if log_every and ((t % int(log_every) == 0) or t == int(max_iter) - 1):
            logging.info("[EM %03d] obj=%.6f | L1_delta=%.3e | w=%s", t, obj, delta,
                         np.array2string(w_new, precision=6))

        w = w_new
        if delta < float(tol_l1):
            break

    return w, np.asarray(w_hist), np.asarray(obj_hist)


def bhattacharyya_distance(domain: np.ndarray, p: np.ndarray, q: np.ndarray) -> float: # pragma: no cover
    domain = np.asarray(domain, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    if not (domain.shape == p.shape == q.shape):
        raise ValueError("domain, p, q must have same shape")
    dx = np.abs(np.gradient(domain))
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p = p / float(np.sum(p * dx) + 1e-300)
    q = q / float(np.sum(q * dx) + 1e-300)
    bc = float(np.sum(np.sqrt(np.maximum(p * q, 0.0)) * dx))
    return float(-np.log(max(bc, 1e-300)))


def kl_divergence(domain: np.ndarray, p: np.ndarray, q: np.ndarray) -> float: # pragma: no cover
    domain = np.asarray(domain, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    dx = np.abs(np.gradient(domain))
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p = p / float(np.sum(p * dx) + 1e-300)
    q = q / float(np.sum(q * dx) + 1e-300)
    eps = 1e-300
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps)) * dx))


def wasserstein_1d(domain: np.ndarray, p: np.ndarray, q: np.ndarray) -> float: # pragma: no cover
    domain = np.asarray(domain, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    dx = np.abs(np.gradient(domain))
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p = p / float(np.sum(p * dx) + 1e-300)
    q = q / float(np.sum(q * dx) + 1e-300)
    F = np.cumsum(p * dx)
    G = np.cumsum(q * dx)
    return float(np.sum(np.abs(F - G) * dx))
