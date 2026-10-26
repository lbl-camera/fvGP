import numpy as np
import warnings
from loguru import logger
from scipy.sparse import issparse
from .gp_lin_alg import *
from .gp_lin_alg import _outer


class GPposterior:
    def __init__(self,
                 data,
                 prior,
                 trainer,
                 kv,
                 likelihood):

        self.kv = kv
        self.prior = prior
        self.likelihood = likelihood
        self.data = data
        self.trainer = trainer

    def compute_covariances(self, x1, x2, hps):
        """Direct, dense kernel evaluation. For the small (n_pred x n_pred) blocks."""
        return self.prior.compute_covariances(x1, x2, hps)

    def cross_covariance(self, x_pred, hps):
        """k(x_data, x_pred). Distributed and sparse under gp2Scale, dense otherwise."""
        return self.prior.compute_data_cross_covariance(x_pred, hps)

    def compute_mean(self, x, hps):
        return self.prior.compute_mean(x, hps)

    def d_kernel_dx(self, x_pred, x_data, direction, hyperparameters):
        return self.prior.d_kernel_dx(x_pred, x_data, direction, hyperparameters)

    def KVsolve(self, v):
        return self.kv.solve(v)

    def compute_new_KVinvY(self, KV, m, K=None, V=None):
        return self.kv.compute_new_KVinvY(KV, m, K=K, V=V)

    def _new_KVinvY(self, K, V, m):
        """KVinvY for explicit hyperparameters; ``"low_rank"`` keeps K and the noise vector apart."""
        if self.kv.mode in ("low_rank", "kronecker"): return self.compute_new_KVinvY(None, m, K=K, V=V)
        if np.ndim(V) == 1: V = np.diag(V)
        return self.compute_new_KVinvY(self.addKV(K, V), m)

    def compute_prior_covariance_matrix(self, x_data, hyperparameters):
        return self.prior.compute_prior_covariance_matrix(x_data, hyperparameters)

    def calculate_V(self, x_data, hyperparameters):
        return self.likelihood.calculate_V(x_data, hyperparameters)

    def noise_function(self, x_pred, hyperparameters):
        return self.likelihood.calculate_V(x_pred, hyperparameters)

    def addKV(self, K, V):
        return self.kv.addKV(K, V)

    #####################################################
    @property
    def args(self):
        return self.data.args

    @property
    def hyperparameters(self):
        return self.trainer.hyperparameters

    @property
    def x_data(self):
        return self.data.x_data

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def x_out(self):
        return self.data.x_out

    @property
    def KVinvY(self):
        return self.kv.KVinvY

    @property
    def KVinv(self):
        return self.kv.KVinv

    @property
    def input_set_dim(self):
        return self.data.input_set_dim

    @property
    def K(self):
        return self.prior.K

    @property
    def m(self):
        return self.prior.m

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    @property
    def kronecker(self):
        return self.data.kronecker

    @property
    def independent(self):
        return self.data.multi_task == "independent"

    def _reject_kronecker(self, method):
        if self.kronecker:
            raise NotImplementedError(f"{method} is not supported in Kronecker mode (fvGP(multi_task_mode='kronecker')).")

    @staticmethod
    def _dense(matrix):
        """A dense view of a covariance block that may have arrived sparse from gp2Scale."""
        return matrix.toarray() if issparse(matrix) else matrix

    def _dense_K(self):
        """K as a dense array, for the methods that build a joint (N + n_pred)^2 matrix.

        Those methods are dense in N by construction, so under gp2Scale they are only
        usable on small problems and the user should know that the sparsity they asked
        for is being thrown away here.
        """
        if not issparse(self.K): return self.K
        warnings.warn(
            "This method assembles a joint covariance over data and prediction points, "
            "which is dense in the number of data points. Under gp2Scale that discards "
            "the sparse representation and costs O(N^2) memory; it is only usable on "
            "small problems. Consider posterior_covariance instead.")
        return self.K.toarray()

    def _cross_solve_product(self, k, chunk_size=None):
        """``k.T @ KV^-1 @ k``, evaluated in column chunks.

        The solve is dense whatever ``k`` is -- ``KV^-1`` is dense even when ``KV`` is not
        -- so chunking over prediction points is what keeps the intermediate at
        (N x chunk) rather than (N x n_pred). ``k`` itself stays sparse throughout under
        gp2Scale; only the chunk handed to the solver is densified.
        """
        n_pred = k.shape[1]
        if chunk_size is None: chunk_size = n_pred if not self.gp2Scale else self.prior.batch_size
        chunk_size = max(1, min(int(chunk_size), n_pred))
        product = np.empty((n_pred, n_pred))
        for start in range(0, n_pred, chunk_size):
            end = min(start + chunk_size, n_pred)
            solved = self.KVsolve(self._dense(k[:, start:end]))
            product[:, start:end] = np.asarray(k.T @ solved)
        return product
    ##########################################################

    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        if self.kronecker: return self._kronecker_posterior_mean(x_pred, hyperparameters, x_out)
        if self.independent:
            return self._select_outputs(self._posterior_mean(x_pred, hyperparameters, None), x_out,
                                        ("m(x)", "m(x)_flat"))
        return self._posterior_mean(x_pred, hyperparameters, x_out)

    def _posterior_mean(self, x_pred, hyperparameters, x_out):
        x_data, KVinvY = self.x_data.copy(), self.KVinvY.copy()
        if hyperparameters is not None:
            K = self.compute_prior_covariance_matrix(x_data, hyperparameters)
            V = self.calculate_V(x_data, hyperparameters)
            m = self.compute_mean(x_data, hyperparameters)
            KVinvY = self._new_KVinvY(K, V, m)
        else:
            hyperparameters = self.hyperparameters

        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        # Sparse under gp2Scale, and it stays sparse: the product with KVinvY is the whole
        # use of k here, so the posterior mean never materializes an (N x n_pred) array.
        k = self.cross_covariance(x_pred, hyperparameters)
        A = np.asarray(k.T @ KVinvY)
        prior_mean = self.compute_mean(x_pred, hyperparameters)
        posterior_mean = self.prior.mean_columns(prior_mean) + A
        if isinstance(x_out, np.ndarray): posterior_mean_re = posterior_mean.reshape(len(x_orig), len(x_out), order='F')
        else: posterior_mean_re = posterior_mean

        if KVinvY.shape[1] == 1 and not isinstance(x_out, np.ndarray):
            return {"x": x_orig,
                    "m(x)": np.squeeze(posterior_mean_re),
                    "m(x)_flat": np.squeeze(posterior_mean),
                    "x_pred": x_pred}
        elif KVinvY.shape[1] == 1 and isinstance(x_out, np.ndarray):
            return {"x": x_orig,
                    "m(x)": posterior_mean_re,
                    "m(x)_flat": np.squeeze(posterior_mean),
                    "x_pred": x_pred}
        # unreachable: the order='F' reshape above already fails for a multi-column
        # KVinvY combined with x_out, since the flat product space has one column
        elif KVinvY.shape[1] > 1 and isinstance(x_out, np.ndarray):  # pragma: no cover - defensive
            raise Exception("KVinvY.shape[1] > 1 and isinstance(x_out, np.ndarray)=True")
        else:
            return {"x": x_orig,
                    "m(x)": posterior_mean_re,
                    "m(x)_flat": posterior_mean,
                    "x_pred": x_pred}

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None, component=0):
        self._reject_kronecker("posterior_mean_grad")
        x_data, KVinvY = self.x_data.copy(), self.KVinvY.copy()[:, component]

        if hyperparameters is not None:
            K = self.compute_prior_covariance_matrix(x_data, hyperparameters)
            V = self.calculate_V(x_data, hyperparameters)
            m = self.compute_mean(x_data, hyperparameters)
            KVinvY = self._new_KVinvY(K, V, m)[:, component]
            assert np.ndim(KVinvY) == 1, "KVinvY must be 1-d after column selection"
        else:
            hyperparameters = self.hyperparameters

        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        f = self.compute_mean(x_pred, hyperparameters)
        eps = 1e-6
        if direction is not None:
            x1 = np.array(x_pred)
            x1[:, direction] = x1[:, direction] + eps
            mean_der = self.prior.mean_columns((self.compute_mean(x1, hyperparameters) - f) / eps)[:, component]
            k_g = self.d_kernel_dx(x_pred, x_data, direction, hyperparameters)
            posterior_mean_grad = mean_der + (k_g @ KVinvY)
            if isinstance(x_out, np.ndarray):
                posterior_mean_grad = posterior_mean_grad.reshape(len(x_orig), len(x_out), order='F')
        else:
            posterior_mean_grad = np.zeros((len(x_pred), x_orig.shape[1]))
            for direction in range(len(x_orig[0])):
                x1 = np.array(x_pred)
                x1[:, direction] = x1[:, direction] + eps
                mean_der = self.prior.mean_columns((self.compute_mean(x1, hyperparameters) - f) / eps)[:, component]
                k_g = self.d_kernel_dx(x_pred, x_data, direction, hyperparameters)
                posterior_mean_grad[:, direction] = mean_der + (k_g @ KVinvY)
            direction = "ALL"
            if isinstance(x_out, np.ndarray):
                posterior_mean_grad = posterior_mean_grad.reshape(len(x_orig), len(x_orig[0]), len(x_out), order='F')

        return {"x": x_orig,
                "direction": direction,
                "dm/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        if self.kronecker: return self._kronecker_posterior_covariance(x_pred, x_out, variance_only, add_noise)
        if self.independent:
            # the tasks share one GP: S, over the inputs, is the same for all of them
            return self._select_outputs(self._posterior_covariance(x_pred, None, variance_only, add_noise), x_out,
                                        ("v(x)", "v_flat"))
        return self._posterior_covariance(x_pred, x_out, variance_only, add_noise)

    def _select_outputs(self, result, x_out, keys):
        """Keep the columns ``x_out`` (all by default) of the per-task entries of a posterior result."""
        columns = self._output_indices(x_out)
        for key in keys: result[key] = result[key][:, columns]
        return result

    def _posterior_covariance(self, x_pred, x_out, variance_only, add_noise):
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.cross_covariance(x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters)

        if self.KVinv is not None and variance_only and self.y_data.shape[1] == 1:
            # The explicit-inverse modes are dense in N anyway, so densifying k costs
            # nothing here, and the einsum gets the variances without ever forming S.
            k_dense = self._dense(k)
            S = None
            v = np.diag(kk) - np.einsum('ij,jk,ki->i', k_dense.T,
                                        self.KVinv, k_dense, optimize=True)
        else:
            S = kk - self._cross_solve_product(k)
            v = np.array(np.diag(S))
        if np.any(v < -0.0001):
            warnings.warn(
                "Negative variances encountered. That normally means that the model is unstable. "
                "Rethink the kernel definition, add more noise to the data, "
                "or double check the hyperparameter optimization bounds. This will not "
                "terminate the algorithm, but expect anomalies.")
            logger.debug("Negative variances encountered.")
        # Always clip tiny negatives (numerical roundoff in iterative solvers
        # leaves the diagonal slightly below zero); downstream sqrt would NaN.
        if np.any(v < 0.0):
            v[v < 0.0] = 0.0
            if not variance_only: np.fill_diagonal(S, v)

        if add_noise: v, S = self.add_noise(x_pred, v, S)

        if isinstance(x_out, np.ndarray):
            v_re = v.reshape(len(x_orig), len(x_out), order='F')
            if S is not None:
                # The flat product-space index is task-major (k = point + Npts*task),
                # matching cartesian_product and the order='F' reshape of v above. A
                # direct reshape to (Npts, Npts, No, No) would therefore interleave the
                # point and task axes and silently scramble S. Reshape to the layout the
                # flat index actually has -- (point, task, point, task) -- and then move
                # the axes into the documented (Npts, Npts, No, No) arrangement, so that
                # S_re[i, j, t, u] == Cov(f(x_i, task_t), f(x_j, task_u)).
                S_re = S.reshape(len(x_orig), len(x_out), len(x_orig), len(x_out),
                                 order='F').transpose(0, 2, 1, 3)
            else: S_re = None
        else:
            v_re = v
            S_re = S
            if self.y_data.shape[1] > 1:
                v = np.tile(v[:, None], (1, self.y_data.shape[1]))
                v_re = np.tile(v_re[:, None], (1, self.y_data.shape[1]))

        return {"x": x_orig,
                "x_pred": x_pred,
                "v(x)": v_re,
                "S": S_re,
                "S_flat": S,
                "v_flat": v}

    def _output_indices(self, x_out):
        """x_out as integer output indices, the only output coordinates the Kronecker and
        independent modes have."""
        n_out = self.y_data.shape[1]
        if x_out is None: return np.arange(n_out)
        indices = np.asarray(x_out)
        if (np.ndim(indices) != 1 or not np.all(np.mod(indices, 1) == 0)
                or np.any(indices < 0) or np.any(indices >= n_out)):
            raise ValueError(f"With multi_task_mode={self.data.multi_task!r} x_out selects outputs by index: "
                             f"a 1-d array of integers in 0 ... {n_out - 1}.")
        return indices.astype(int)

    def _kronecker_posterior_mean(self, x_pred, hyperparameters, x_out):
        """Mode products of the tensor (K+V)^-1 (y - m) with [K_x(x_pred, X), F_1, ..., F_m]."""
        KVinvY = self.KVinvY
        if hyperparameters is not None:
            K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
            V = self.calculate_V(self.x_data, hyperparameters)
            m = self.compute_mean(self.x_data, hyperparameters)
            KVinvY = self._new_KVinvY(K, V, m)
        else:
            hyperparameters = self.hyperparameters
        x_out = self._output_indices(x_out)
        self._perform_input_checks(x_pred, None)
        k = self.cross_covariance(x_pred, hyperparameters)
        alpha = KVinvY.reshape((len(self.x_data),) + tuple(len(F) for F in k[1:]))
        A = kronecker_mode_product(alpha, [np.asarray(k[0]).T] + list(k[1:])).reshape(len(x_pred), -1)
        prior_mean = self.compute_mean(x_pred, hyperparameters)
        posterior_mean = (self.prior.mean_columns(prior_mean) + A)[:, x_out]
        return {"x": x_pred,
                "m(x)": posterior_mean,
                "m(x)_flat": posterior_mean.ravel(order='F'),
                "x_pred": x_pred}

    def _kronecker_posterior_covariance(self, x_pred, x_out, variance_only, add_noise):
        """Posterior (co)variances at every (x_pred, output) pair, without forming K.

        Every k(x, o) is a Kronecker product of per-axis columns, so its projection into the
        whitened eigenbasis of K + V is too, and k^T (K+V)^-1 k is one contraction of the
        inverse spectrum: O(N) per pair, all pairs at once by mode products. The full
        covariance builds those projections explicitly, (N x pairs).
        """
        hps = self.hyperparameters
        factor = self.kv.kronecker_factor
        x_out = self._output_indices(x_out)
        self._perform_input_checks(x_pred, None)
        n_pred = len(x_pred)
        k = self.cross_covariance(x_pred, hps)
        kk = self.compute_covariances(x_pred, x_pred, hps)
        P = [factor.project(f, M) for f, M in enumerate(k)]
        reduction = kronecker_mode_product(factor.inv_spectrum, [(p ** 2).T for p in P]).reshape(n_pred, -1)
        prior = _outer([np.diag(F) for F in kk]).reshape(n_pred, -1)
        v = (prior - reduction)[:, x_out]
        S = None
        if not variance_only:
            # pair (point j, output t) at j + n_pred * t, task-major like the Cartesian fvGP
            points = np.tile(np.arange(n_pred), len(x_out))
            outputs = np.unravel_index(np.repeat(x_out, n_pred), [len(F) for F in kk[1:]])
            B = P[0][:, points]
            S = np.asarray(kk[0])[np.ix_(points, points)]
            for p, F, o in zip(P[1:], kk[1:], outputs):
                B = (B[:, None, :] * p[:, o][None, :, :]).reshape(-1, len(points))
                S = S * np.asarray(F)[np.ix_(o, o)]
            S = S - (B * factor.inv_spectrum.reshape(-1, 1)).T @ B
            v = np.diag(S).reshape(len(x_out), n_pred).T
        if np.any(v < -0.0001):
            warnings.warn(
                "Negative variances encountered. That normally means that the model is unstable. "
                "Rethink the kernel definition, add more noise to the data, "
                "or double check the hyperparameter optimization bounds. This will not "
                "terminate the algorithm, but expect anomalies.")
        v = np.maximum(v, 0.0)
        if S is not None: np.fill_diagonal(S, v.ravel(order='F'))
        if add_noise:
            # one variance per input, the same for every output
            noise = np.tile(np.reshape(self.noise_function(x_pred, hps), (n_pred, 1)), (1, len(x_out)))
            v = v + noise
            if S is not None: S = S + np.diag(noise.ravel(order='F'))
        S_re = None if S is None else S.reshape(n_pred, len(x_out), n_pred, len(x_out), order='F').transpose(0, 2, 1, 3)
        return {"x": x_pred,
                "x_pred": x_pred,
                "v(x)": v,
                "S": S_re,
                "S_flat": S,
                "v_flat": v.ravel(order='F')}

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        self._reject_kronecker("posterior_covariance_grad")
        x_data = self.x_data.copy()
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        k_covariance_prod = self.KVsolve(k)
        if direction is not None:
            k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
            x1 = np.array(x_pred)
            x2 = np.array(x_pred)
            eps = 1e-6
            x1[:, direction] = x1[:, direction] + eps
            kk_g = (self.compute_covariances(x1, x1, self.hyperparameters) -
                    self.compute_covariances(x2, x2, self.hyperparameters)) / eps
            dSdx = kk_g - (2.0 * k_g.T @ k_covariance_prod)
            a = np.diag(dSdx)
            if isinstance(x_out, np.ndarray):
                a = a.reshape(len(x_orig), len(x_out), order='F')
                dSdx = dSdx.reshape(len(x_orig), len(x_orig), len(x_out), len(x_out), order='F')
            return {"x": x_orig,
                    "dv/dx": a,
                    "dS/dx": dSdx}
        else:
            grad_v = np.zeros((len(x_pred), len(x_orig[0])))
            for direction in range(len(x_orig[0])):
                k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
                x1 = np.array(x_pred)
                x2 = np.array(x_pred)
                eps = 1e-6
                x1[:, direction] = x1[:, direction] + eps
                kk_g = (self.compute_covariances(x1, x1, self.hyperparameters) -
                        self.compute_covariances(x2, x2, self.hyperparameters)) / eps
                grad_v[:, direction] = np.diag(kk_g - (2.0 * k_g.T @ k_covariance_prod))

            if isinstance(x_out, np.ndarray):
                grad_v = grad_v.reshape(len(x_orig), len(x_orig[0]), len(x_out), order='F')

            return {"x": x_orig,
                    "dv/dx": grad_v}

    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out=None):
        self._reject_kronecker("joint_gp_prior")
        x_data, K, prior_mean_vec = (self.x_data.copy(),
                                     self._dense_K() + (np.identity(len(self.x_data)) * 1e-9),
                                     self.m.copy())
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters)
        post_mean = self.compute_mean(x_pred, self.hyperparameters)
        joint_gp_prior_mean = np.append(prior_mean_vec, post_mean)
        joint_gp_prior_cov = np.block([[K, k], [k.T, kk]])

        return {"x": x_pred,
                "K": K,
                "k": k,
                "kappa": kk,
                "prior mean": joint_gp_prior_mean,
                "S": joint_gp_prior_cov + np.identity(len(joint_gp_prior_cov)) * 1e-9}

    ###########################################################################
    def joint_gp_prior_grad(self, x_pred, direction, x_out=None):
        self._reject_kronecker("joint_gp_prior_grad")
        x_data, K, prior_mean_vec = (self.x_data.copy(),
                                     self._dense_K() + (np.identity(len(self.x_data)) * 1e-9),
                                     self.m.copy())
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        eps = 1e-6
        x1[:, direction] = x1[:, direction] + eps
        x2[:, direction] = x2[:, direction] - eps
        kk_g = (self.compute_covariances(x1, x1, self.hyperparameters) -
                self.compute_covariances(x2, x2, self.hyperparameters)) / (2.0 * eps)

        mean_der = ((self.compute_mean(x1, self.hyperparameters) -
                     self.compute_mean(x2, self.hyperparameters)) /
                    (2.0 * eps))
        full_gp_prior_mean_grad = np.append(np.zeros(prior_mean_vec.shape), mean_der)
        prior_cov_grad = np.zeros(K.shape)
        return {"x": x_pred,
                "K": K,
                "dk/dx": k_g,
                "d kappa/dx": kk_g,
                "d prior mean/x": full_gp_prior_mean_grad,
                "dS/dx": np.block([[prior_cov_grad, k_g], [k_g.T, kk_g]])}

    ###########################################################################
    @staticmethod
    def entropy(S):
        dim = len(S[0])
        ldet = calculate_logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * ldet)

    ###########################################################################
    def gp_entropy(self, x_pred, x_out=None):
        """
        Function to compute the entropy of the gp prior probability distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multi-task GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Entropy : float
        """
        self._reject_kronecker("gp_entropy")

        priors = self.joint_gp_prior(x_pred, x_out=x_out)
        S = priors["S"]
        dim = len(S[0])
        ldet = calculate_logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * ldet)

    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out=None):
        self._reject_kronecker("gp_entropy_grad")
        priors1 = self.joint_gp_prior(x_pred, x_out=x_out)
        priors2 = self.joint_gp_prior_grad(x_pred, direction, x_out=x_out)
        S1 = priors1["S"]
        S2 = priors2["dS/dx"]
        return 0.5 * np.trace(calculate_inv(S1) @ S2)

    ###########################################################################
    @staticmethod
    def kl_div(mu1, mu2, S1, S2):
        logdet1 = calculate_logdet(S1)
        logdet2 = calculate_logdet(S2)
        x1 = solve(S2, S1)
        mu = np.subtract(mu2, mu1)
        x2 = solve(S2, mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu)[0] - float(dim) + (logdet2 - logdet1))
        if kld < -1e-4:
            warnings.warn("Negative KL divergence encountered. That happens when "
                          "one of the covariance matrices is close to positive semi definite "
                          "and therefore the logdet() calculation becomes unstable. "
                          "Returning abs(KLD)")
            logger.debug("Negative KL divergence encountered")
        return abs(kld)

    ###########################################################################
    def gp_kl_div(self, x_pred, comp_mean, comp_cov, x_out=None):
        if x_out is None: x_out = self.x_out

        res = self.posterior_mean(x_pred, x_out=x_out)
        gp_mean = res["m(x)_flat"]
        gp_cov = self.posterior_covariance(x_pred, x_out=x_out)["S_flat"]
        gp_cov = gp_cov + np.identity(len(gp_cov)) * 1e-9
        comp_cov = comp_cov + np.identity(len(comp_cov)) * 1e-9
        return {"x": x_pred,
                "gp posterior mean": gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}

    ###########################################################################
    def mutual_information(self, joint, m1, m2):
        return self.entropy(m1) + self.entropy(m2) - self.entropy(joint)

    ###########################################################################
    def gp_mutual_information(self, x_pred, x_out=None, add_noise=False):
        self._reject_kronecker("gp_mutual_information")
        x_data, K = self.x_data.copy(), self._dense_K() + (np.identity(len(self.x_data)) * 1e-9)
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters) + (np.identity(len(x_pred)) * 1e-9)
        if add_noise: v, kk = self.add_noise(x_pred, np.diag(kk), kk)

        joint_covariance = np.block([[K, k], [k.T, kk]])
        return {"x": x_orig,
                "mutual information": self.mutual_information(joint_covariance, kk, K)}

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None, add_noise=False):
        self._reject_kronecker("gp_total_correlation")
        x_data, K = self.x_data.copy(), self._dense_K() + (np.identity(len(self.x_data)) * 1e-9)
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        k = self.compute_covariances(x_data, x_pred, self.hyperparameters)
        kk = self.compute_covariances(x_pred, x_pred, self.hyperparameters) + (np.identity(len(x_pred)) * 1e-9)
        if add_noise: v, kk = self.add_noise(x_pred, np.diag(kk), kk)
        joint_covariance = np.block([[K, k], [k.T, kk]])

        prod_covariance = np.block([[K, k * 0.], [k.T * 0., kk * np.identity(len(kk))]])

        return {"x": x_orig,
                "total correlation": self.kl_div(np.zeros((len(joint_covariance))), np.zeros((len(joint_covariance))),
                                                 joint_covariance, prod_covariance)}

    ###########################################################################
    def gp_relative_information_entropy(self, x_pred, x_out=None, add_noise=False):
        self._reject_kronecker("gp_relative_information_entropy")
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        if isinstance(x_out, np.ndarray): x_pred_aux = self.cartesian_product(x_pred, x_out)
        else: x_pred_aux = x_pred
        kk = self.compute_covariances(x_pred_aux, x_pred_aux, self.hyperparameters) + (np.identity(len(x_pred_aux)) * 1e-9)
        post_cov = self.posterior_covariance(x_pred, x_out=x_out, add_noise=add_noise)["S_flat"]
        post_cov = post_cov + (np.identity(len(post_cov)) * 1e-9)
        hyperparameters = self.hyperparameters
        post_mean = self.posterior_mean(x_pred, x_out=x_out)["m(x)_flat"]
        prio_mean = self.compute_mean(x_pred_aux, hyperparameters)
        return {"x": x_orig,
                "RIE": self.kl_div(prio_mean, post_mean, kk, post_cov)}

    ###########################################################################
    def gp_relative_information_entropy_set(self, x_pred, x_out=None, add_noise=False):
        if x_out is None: x_out = self.x_out
        #self._perform_input_checks(x_pred, x_out)
        x_orig = x_pred.copy()
        #if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)
        RIE = np.zeros((len(x_pred)))
        for i in range(len(x_pred)):
            RIE[i] = self.gp_relative_information_entropy(
                x_pred[i].reshape(1, len(x_pred[i])), x_out=x_out, add_noise=add_noise)["RIE"]

        return {"x": x_orig,
                "RIE": RIE}

    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out=None):
        if x_out is None: x_out = self.x_out
        self._perform_input_checks(x_pred, x_out)
        #if isinstance(x_out, np.ndarray): x_pred = self.cartesian_product(x_pred, x_out)

        gp_mean = self.posterior_mean(x_pred, x_out=x_out)["m(x)_flat"]
        gp_cov = self.posterior_covariance(x_pred, x_out=x_out, add_noise=True)["S_flat"]
        gp_cov_inv = calculate_inv(gp_cov)
        comp_cov_inv = calculate_inv(comp_cov)
        cov = calculate_inv(gp_cov_inv + comp_cov_inv)
        mu = cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        logdet1 = calculate_logdet(cov)
        logdet2 = calculate_logdet(gp_cov)
        logdet3 = calculate_logdet(comp_cov)
        dim = len(mu)
        C = 0.5 * (((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T
                    @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean))
                   - (gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0 * np.pi) ** (dim / 2.0)) + (0.5 * (logdet2 + logdet3)))
        return {"mu": mu,
                "covariance": cov,
                "probability":
                    np.exp(ln_p)
                }

    def add_noise(self, x_pred, v, S):
        noise = self.noise_function(x_pred, self.hyperparameters)
        assert isinstance(noise, np.ndarray), "noise function must return np.ndarray"
        if np.ndim(noise) == 1:
            v = v + noise
            if S is not None: S = S + np.diag(noise)
        elif np.ndim(noise) == 2:
            v = v + np.diag(noise)
            if S is not None: S = S + noise
        else:
            raise Exception("Wrong noise format: the noise function must return a 1-d or 2-d array")
        return v, S

    ###########################################################################
    @staticmethod
    def _int_gauss(S):
        return ((2.0 * np.pi) ** (len(S) / 2.0)) * np.sqrt(np.linalg.det(S))

    def _perform_input_checks(self, x_pred, x_out):
        assert isinstance(x_pred, np.ndarray) or isinstance(x_pred, list), "wrong format in x_pred"
        if isinstance(x_pred, np.ndarray):
            assert np.ndim(x_pred) == 2, "wrong dim in x_pred, has to be 2-d"
            assert x_pred.shape[1] == self.input_set_dim, "wrong number of columns in x_pred"

        assert isinstance(x_out, np.ndarray) or x_out is None or isinstance(x_out, list), "wrong format in x_out"
        if isinstance(x_out, np.ndarray): assert np.ndim(x_out) == 1, "wrong dim in x_out, has to be 1-d"

    @staticmethod
    def cartesian_product(x, y):
        """
        Input x,y have to be 2d numpy arrays
        The return is the cartesian product of the two sets
        """
        assert isinstance(y, np.ndarray), "x_out must be np.ndarray for cartesian product"
        assert np.ndim(y) == 1, "x_out must be 1-d for cartesian product"

        res = []
        if isinstance(x, list):
            for j in range(len(y)):
                for i in range(len(x)):
                    res.append([x[i], y[j]])
            return res
        elif isinstance(x, np.ndarray):
            for j in range(len(y)):
                for i in range(len(x)):
                    res.append(np.append(x[i], y[j]))
            return np.asarray(res)
        else:
            raise Exception("Cartesian product out of options")

    def __getstate__(self):
        state = dict(
            prior=self.prior,
            likelihood=self.likelihood,
            data=self.data,
            trainer=self.trainer,
            kv=self.kv
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
