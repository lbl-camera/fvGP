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

    def mean_columns(self, m):
        return self.prior.mean_columns(m)

    def d_kernel_dx(self, x_pred, x_data, direction, hyperparameters):
        return self.prior.d_kernel_dx(x_pred, x_data, direction, hyperparameters)

    def KVsolve(self, v):
        return self.kv.solve(v)

    def compute_new_KVinvY(self, KV, m, K=None, V=None):
        return self.kv.compute_new_KVinvY(KV, m, K=K, V=V)

    def _new_KVinvY(self, K, V, m):
        """KVinvY for explicit hyperparameters; ``"low_rank"`` keeps K and the noise vector apart."""
        if self.kv_mode in ("low_rank", "kronecker"): return self.compute_new_KVinvY(None, m, K=K, V=V)
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
    def kv_mode(self):
        return self.kv.mode

    @property
    def kronecker_factor(self):
        return self.kv.kronecker_factor

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
    def batch_size(self):
        return self.prior.batch_size

    @property
    def kronecker(self):
        return self.data.kronecker

    @property
    def multi_task(self):
        return self.data.multi_task

    @property
    def independent(self):
        return self.multi_task == "independent"

    def _reject_kronecker(self, method):
        if self.kronecker:
            raise NotImplementedError(f"{method} is not supported in Kronecker mode (GP(multi_task_mode='kronecker')).")

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
        if chunk_size is None: chunk_size = n_pred if not self.gp2Scale else self.batch_size
        chunk_size = max(1, min(int(chunk_size), n_pred))
        product = np.empty((n_pred, n_pred))
        for start in range(0, n_pred, chunk_size):
            end = min(start + chunk_size, n_pred)
            solved = self.KVsolve(self._dense(k[:, start:end]))
            product[:, start:end] = np.asarray(k.T @ solved)
        return product

    def _cross_solve_diagonal(self, k, chunk_size=None):
        """``diag(k.T @ KV^-1 @ k)`` in the chunks of :py:meth:`_cross_solve_product`, without
        forming the (n_pred x n_pred) product."""
        if self.KVinv is not None:
            # The explicit-inverse modes are dense in N anyway, so densifying k costs
            # nothing here, and the einsum gets the variances without ever forming S.
            k_dense = self._dense(k)
            return np.einsum('ij,jk,ki->i', k_dense.T, self.KVinv, k_dense, optimize=True)
        n_pred = k.shape[1]
        if chunk_size is None: chunk_size = n_pred if not self.gp2Scale else self.batch_size
        chunk_size = max(1, min(int(chunk_size), n_pred))
        diagonal = np.empty(n_pred)
        for start in range(0, n_pred, chunk_size):
            end = min(start + chunk_size, n_pred)
            k_chunk = self._dense(k[:, start:end])
            diagonal[start:end] = np.einsum('ij,ij->j', k_chunk, np.asarray(self.KVsolve(k_chunk)))
        return diagonal

    ##########################################################
    # The posterior result contract. Every mode computes flat, task-major arrays -- the pair
    # (point j, output t) at j + V * t -- and the helpers below build every returned dict from
    # them; they are the one place result shapes are decided. A single-task result has no
    # output axis; a multi-task result has it, the same in every mode.
    def _outputs(self, x_out):
        """The requested outputs: None for a single-task GP; output indices for independent and
        Kronecker outputs; output coordinates, all of ``self.x_out`` by default, for Cartesian ones."""
        if self.multi_task is None:
            if x_out is not None:
                raise ValueError("x_out selects the outputs of a multi-task GP; this GP has a single output.")
            return None
        if self.multi_task != "cartesian": return self._output_indices(x_out)
        if x_out is None: return self.x_out
        x_out = np.asarray(x_out)
        if np.ndim(x_out) != 1: raise ValueError("x_out must be a 1-d array of output coordinates.")
        return x_out

    def _index_points(self, x_pred, x_out):
        """The points the kernel sees: in Cartesian mode the (point, output) pairs, task-major."""
        if self.multi_task == "cartesian": return self.cartesian_product(x_pred, x_out)
        return x_pred

    def _per_output(self, values, x_out):
        """A (points, columns) array raveled task-major: the requested columns when the outputs
        are columns (independent, Kronecker), the single column otherwise."""
        if self.multi_task in ("independent", "kronecker"): values = np.asarray(values)[:, x_out]
        return np.asarray(values).ravel(order='F')

    def _shared_by_outputs(self, x_out, shared, pairs=False):
        """What independent outputs share, once per requested output, task-major: stacked, or
        block-diagonal for a quantity over pairs. Unchanged in every other mode."""
        if shared is None or not self.independent: return shared
        if pairs: return np.kron(np.identity(len(x_out)), shared)
        return np.tile(shared, (len(x_out),) + (1,) * (np.ndim(shared) - 1))

    def _independent_count(self, x_out):
        """How many times a quantity of one independent output adds up over the requested ones."""
        return len(x_out) if self.independent else 1

    @staticmethod
    def _by_output(flat, n_pred, x_out, pairs=False):
        """The structured view of a flat, task-major array: (V n) -> (V, n), (V n, D) -> (V, D, n),
        and with ``pairs`` (V n, V n) -> (V, V, n, n). Unchanged for a single-task GP."""
        if x_out is None or flat is None: return flat
        n = len(x_out)
        # The flat index is task-major, so the order='F' split gives (point, output) axes; a
        # direct reshape to the documented layout would interleave the point and output axes.
        if pairs: return flat.reshape(n_pred, n, n_pred, n, order='F').transpose(0, 2, 1, 3)
        if np.ndim(flat) == 1: return flat.reshape(n_pred, n, order='F')
        return flat.reshape(n_pred, n, flat.shape[1], order='F').transpose(0, 2, 1)

    def _mean_result(self, x_pred, x_out, m):
        return {"x": x_pred,
                "x_out": x_out,
                "m(x)": self._by_output(m, len(x_pred), x_out),
                "m(x)_flat": m}

    def _covariance_result(self, x_pred, x_out, v, S):
        return {"x": x_pred,
                "x_out": x_out,
                "v(x)": self._by_output(v, len(x_pred), x_out),
                "S": self._by_output(S, len(x_pred), x_out, pairs=True),
                "S_flat": S,
                "v_flat": v}

    def _gradient_result(self, x_pred, x_out, direction, gradients):
        result = {"x": x_pred, "x_out": x_out, "direction": direction}
        for key, value in gradients.items():
            result[key] = self._by_output(value, len(x_pred), x_out, pairs=(key == "dS/dx"))
        return result

    def _KVinvY_and_hyperparameters(self, hyperparameters):
        """(K+V)^-1 (y - m) for the given hyperparameters, the stored one by default."""
        if hyperparameters is None: return self.KVinvY, self.hyperparameters
        K = self.compute_prior_covariance_matrix(self.x_data, hyperparameters)
        V = self.calculate_V(self.x_data, hyperparameters)
        m = self.compute_mean(self.x_data, hyperparameters)
        return self._new_KVinvY(K, V, m), hyperparameters

    ##########################################################
    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        KVinvY, hyperparameters = self._KVinvY_and_hyperparameters(hyperparameters)
        if self.kronecker: m = self._kronecker_posterior_mean(x_pred, hyperparameters, KVinvY)
        else:
            x_in = self._index_points(x_pred, x_out)
            # Sparse under gp2Scale, and it stays sparse: the product with KVinvY is the whole
            # use of k here, so the posterior mean never materializes an (N x n_pred) array.
            k = self.cross_covariance(x_in, hyperparameters)
            m = self.mean_columns(self.compute_mean(x_in, hyperparameters)) + np.asarray(k.T @ KVinvY)
        return self._mean_result(x_pred, x_out, self._per_output(m, x_out))

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None, component=None):
        self._reject_kronecker("posterior_mean_grad")
        if component is not None:
            warnings.warn("component is deprecated and will be removed in fvgp 5.0.0: the gradient covers "
                          "every requested output; select one with x_out=[component].", FutureWarning, stacklevel=3)
            if self.independent: x_out = [component]
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        KVinvY, hyperparameters = self._KVinvY_and_hyperparameters(hyperparameters)
        x_in = self._index_points(x_pred, x_out)
        f = self.compute_mean(x_in, hyperparameters)
        eps = 1e-6

        def mean_gradient(d):
            x1 = np.array(x_in)
            x1[:, d] = x1[:, d] + eps
            mean_der = self.mean_columns((self.compute_mean(x1, hyperparameters) - f) / eps)
            k_g = self.d_kernel_dx(x_in, self.x_data, d, hyperparameters)
            return self._per_output(mean_der + k_g @ KVinvY, x_out)

        if direction is not None: dm = mean_gradient(direction)
        else: dm = np.stack([mean_gradient(d) for d in range(self.input_set_dim)], axis=1)
        return self._gradient_result(x_pred, x_out, direction, {"dm/dx": dm})

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        if self.kronecker: v, S = self._kronecker_posterior_covariance(x_pred, x_out, variance_only, add_noise)
        else:
            # independent outputs share one GP: v and S over the inputs, the same for all of them
            v, S = self._posterior_covariance(self._index_points(x_pred, x_out), variance_only, add_noise)
            v, S = self._shared_by_outputs(x_out, v), self._shared_by_outputs(x_out, S, pairs=True)
        return self._covariance_result(x_pred, x_out, v, S)

    def _posterior_covariance(self, x_in, variance_only, add_noise):
        """Posterior variances, and unless ``variance_only`` the covariance, at the index points ``x_in``."""
        k = self.cross_covariance(x_in, self.hyperparameters)
        kk = self.compute_covariances(x_in, x_in, self.hyperparameters)
        if variance_only:
            S = None
            v = np.diag(kk) - self._cross_solve_diagonal(k)
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
            if S is not None: np.fill_diagonal(S, v)
        if add_noise: v, S = self.add_noise(x_in, v, S)
        return v, S

    def _output_indices(self, x_out):
        """x_out as integer output indices, the only output coordinates the Kronecker and
        independent modes have."""
        n_out = self.y_data.shape[1]
        if x_out is None: return np.arange(n_out)
        indices = np.asarray(x_out)
        if (np.ndim(indices) != 1 or not np.all(np.mod(indices, 1) == 0)
                or np.any(indices < 0) or np.any(indices >= n_out)):
            raise ValueError(f"With multi_task_mode={self.multi_task!r} x_out selects outputs by index: "
                             f"a 1-d array of integers in 0 ... {n_out - 1}.")
        return indices.astype(int)

    def _kronecker_posterior_mean(self, x_pred, hyperparameters, KVinvY):
        """Mode products of the tensor (K+V)^-1 (y - m) with [K_x(x_pred, X), F_1, ..., F_m],
        for every output: (n_pred, n_out)."""
        k = self.cross_covariance(x_pred, hyperparameters)
        alpha = KVinvY.reshape((len(self.x_data),) + tuple(len(F) for F in k[1:]))
        A = kronecker_mode_product(alpha, [np.asarray(k[0]).T] + list(k[1:])).reshape(len(x_pred), -1)
        return self.mean_columns(self.compute_mean(x_pred, hyperparameters)) + A

    @staticmethod
    def _kronecker_pairs(n_pred, x_out, factors):
        """The point and per-factor output indices of the requested (point, output) pairs, task-major."""
        points = np.tile(np.arange(n_pred), len(x_out))
        outputs = np.unravel_index(np.repeat(x_out, n_pred), [len(F) for F in factors[1:]])
        return points, outputs

    @staticmethod
    def _kronecker_pair_covariance(factors, points, outputs):
        """The prior covariance between pairs: the product of the factors' entries."""
        S = np.asarray(factors[0])[np.ix_(points, points)]
        for F, o in zip(factors[1:], outputs): S = S * np.asarray(F)[np.ix_(o, o)]
        return S

    def _kronecker_posterior_covariance(self, x_pred, x_out, variance_only, add_noise):
        """Posterior (co)variances at every (x_pred, output) pair, without forming K.

        Every k(x, o) is a Kronecker product of per-axis columns, so its projection into the
        whitened eigenbasis of K + V is too, and k^T (K+V)^-1 k is one contraction of the
        inverse spectrum: O(N) per pair, all pairs at once by mode products. The full
        covariance builds those projections explicitly, (N x pairs).
        """
        hps = self.hyperparameters
        factor = self.kronecker_factor
        n_pred = len(x_pred)
        k = self.cross_covariance(x_pred, hps)
        kk = self.compute_covariances(x_pred, x_pred, hps)
        P = [factor.project(f, M) for f, M in enumerate(k)]
        reduction = kronecker_mode_product(factor.inv_spectrum, [(p ** 2).T for p in P]).reshape(n_pred, -1)
        prior = _outer([np.diag(F) for F in kk]).reshape(n_pred, -1)
        v = (prior - reduction)[:, x_out]
        S = None
        if not variance_only:
            points, outputs = self._kronecker_pairs(n_pred, x_out, kk)
            B = P[0][:, points]
            for p, o in zip(P[1:], outputs):
                B = (B[:, None, :] * p[:, o][None, :, :]).reshape(-1, len(points))
            S = self._kronecker_pair_covariance(kk, points, outputs) - (B * factor.inv_spectrum.reshape(-1, 1)).T @ B
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
        return v.ravel(order='F'), S

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        self._reject_kronecker("posterior_covariance_grad")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        x_in = self._index_points(x_pred, x_out)
        hps = self.hyperparameters
        kk = self.compute_covariances(x_in, x_in, hps)
        k_covariance_prod = self.KVsolve(self.compute_covariances(self.x_data, x_in, hps))
        eps = 1e-6

        def covariance_gradient(d):
            k_g = self.d_kernel_dx(x_in, self.x_data, d, hps).T
            x1 = np.array(x_in)
            x1[:, d] = x1[:, d] + eps
            kk_g = (self.compute_covariances(x1, x1, hps) - kk) / eps
            # d(k^T KV^-1 k) = M + M^T: 2 M would be right on the diagonal only
            M = k_g.T @ k_covariance_prod
            return kk_g - (M + M.T)

        if direction is not None:
            dS = covariance_gradient(direction)
            dv = np.array(np.diag(dS))
        else:
            dS = None
            dv = np.stack([np.diag(covariance_gradient(d)) for d in range(self.input_set_dim)], axis=1)
        dv, dS = self._shared_by_outputs(x_out, dv), self._shared_by_outputs(x_out, dS, pairs=True)
        return self._gradient_result(x_pred, x_out, direction, {"dv/dx": dv, "dS/dx": dS})

    ###########################################################################
    def _joint_prior_covariances(self, x_pred, x_out):
        """K (dense, jittered), k and kk of the joint prior of the data and the requested pairs;
        for independent outputs the blocks of one output, which all of them share."""
        x_in = self._index_points(x_pred, x_out)
        K = self._dense_K() + np.identity(len(self.x_data)) * 1e-9
        k = self.compute_covariances(self.x_data, x_in, self.hyperparameters)
        kk = self.compute_covariances(x_in, x_in, self.hyperparameters)
        return K, k, kk

    def _joint_prior_gradients(self, x_pred, x_out, direction):
        """dk/dx and d kk/dx of the joint prior, and the mean's derivative at the requested pairs,
        by central differences; for independent outputs those of one output (the mean's per output)."""
        x_in = self._index_points(x_pred, x_out)
        hps = self.hyperparameters
        k_g = self.d_kernel_dx(x_in, self.x_data, direction, hps).T
        x1 = np.array(x_in)
        x2 = np.array(x_in)
        eps = 1e-6
        x1[:, direction] = x1[:, direction] + eps
        x2[:, direction] = x2[:, direction] - eps
        kk_g = (self.compute_covariances(x1, x1, hps) - self.compute_covariances(x2, x2, hps)) / (2.0 * eps)
        mean_der = self.mean_columns((self.compute_mean(x1, hps) - self.compute_mean(x2, hps)) / (2.0 * eps))
        return k_g, kk_g, self._per_output(mean_der, x_out)

    def _independent_joint(self, x_out, K, k, kk):
        """The joint over every output's data and the requested outputs' predictions, for
        independent outputs sharing K: block-diagonal, data then predictions, both task-major."""
        n_out = self.y_data.shape[1]
        feeds = np.identity(n_out)[:, x_out]  # data of output t meets prediction u only if t == x_out[u]
        return np.kron(np.identity(n_out), K), np.kron(feeds, k), np.kron(np.identity(len(x_out)), kk)

    def joint_gp_prior(self, x_pred, x_out=None):
        self._reject_kronecker("joint_gp_prior")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        K, k, kk = self._joint_prior_covariances(x_pred, x_out)
        data_mean = self.mean_columns(self.m).ravel(order='F')
        x_in = self._index_points(x_pred, x_out)
        prediction_mean = self._per_output(self.mean_columns(self.compute_mean(x_in, self.hyperparameters)), x_out)
        if self.independent: K, k, kk = self._independent_joint(x_out, K, k, kk)
        joint_gp_prior_cov = np.block([[K, k], [k.T, kk]])
        return {"x": x_pred,
                "x_out": x_out,
                "K": K,
                "k": k,
                "kappa": kk,
                "prior mean": np.append(data_mean, prediction_mean),
                "S": joint_gp_prior_cov + np.identity(len(joint_gp_prior_cov)) * 1e-9}

    ###########################################################################
    def joint_gp_prior_grad(self, x_pred, direction, x_out=None):
        self._reject_kronecker("joint_gp_prior_grad")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        K = self._dense_K() + np.identity(len(self.x_data)) * 1e-9
        k_g, kk_g, mean_der = self._joint_prior_gradients(x_pred, x_out, direction)
        if self.independent: K, k_g, kk_g = self._independent_joint(x_out, K, k_g, kk_g)
        return {"x": x_pred,
                "x_out": x_out,
                "K": K,
                "dk/dx": k_g,
                "d kappa/dx": kk_g,
                "d prior mean/x": np.append(np.zeros(len(K)), mean_der),
                "dS/dx": np.block([[np.zeros(K.shape), k_g], [k_g.T, kk_g]])}

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
            The requested outputs of a multi-task GP.

        Return
        ------
        Entropy : dict
        """
        self._reject_kronecker("gp_entropy")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        K, k, kk = self._joint_prior_covariances(x_pred, x_out)
        S = np.block([[K, k], [k.T, kk]])
        S = S + np.identity(len(S)) * 1e-9
        entropy = self.entropy(S)
        if self.independent:
            # independent blocks add up: each requested output its data and predictions,
            # every other output its data alone
            unrequested = self.y_data.shape[1] - len(x_out)
            entropy = len(x_out) * entropy + unrequested * self.entropy(S[:len(K), :len(K)])
        return {"x": x_pred, "x_out": x_out, "entropy": entropy}

    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out=None):
        self._reject_kronecker("gp_entropy_grad")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        K, k, kk = self._joint_prior_covariances(x_pred, x_out)
        S1 = np.block([[K, k], [k.T, kk]])
        S1 = S1 + np.identity(len(S1)) * 1e-9
        k_g, kk_g, _ = self._joint_prior_gradients(x_pred, x_out, direction)
        S2 = np.block([[np.zeros(K.shape), k_g], [k_g.T, kk_g]])
        # independent outputs: the data blocks of the unrequested ones do not move
        gradient = self._independent_count(x_out) * 0.5 * np.trace(calculate_inv(S1) @ S2)
        return {"x": x_pred, "x_out": x_out, "direction": direction, "dH/dx": gradient}

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
        x_out = self._outputs(x_out)
        gp_mean = self.posterior_mean(x_pred, x_out=x_out)["m(x)_flat"]
        gp_cov = self.posterior_covariance(x_pred, x_out=x_out)["S_flat"]
        gp_cov = gp_cov + np.identity(len(gp_cov)) * 1e-9
        comp_cov = comp_cov + np.identity(len(comp_cov)) * 1e-9
        return {"x": x_pred,
                "x_out": x_out,
                "gp posterior mean": gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}

    ###########################################################################
    def mutual_information(self, joint, m1, m2):
        return self.entropy(m1) + self.entropy(m2) - self.entropy(joint)

    def _noisy_joint_prior(self, x_pred, x_out, add_noise):
        """K, kk and the joint covariance of the data and the requested pairs, kk jittered and with
        the noise if ``add_noise``; for independent outputs those of one output."""
        K, k, kk = self._joint_prior_covariances(x_pred, x_out)
        kk = kk + np.identity(len(kk)) * 1e-9
        if add_noise: v, kk = self.add_noise(self._index_points(x_pred, x_out), np.diag(kk), kk)
        return K, kk, np.block([[K, k], [k.T, kk]])

    ###########################################################################
    def gp_mutual_information(self, x_pred, x_out=None, add_noise=False):
        self._reject_kronecker("gp_mutual_information")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        K, kk, joint_covariance = self._noisy_joint_prior(x_pred, x_out, add_noise)
        # independent outputs: each requested one shares information with its own data only
        return {"x": x_pred,
                "x_out": x_out,
                "mutual information":
                    self._independent_count(x_out) * self.mutual_information(joint_covariance, kk, K)}

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None, add_noise=False):
        self._reject_kronecker("gp_total_correlation")
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        K, kk, joint_covariance = self._noisy_joint_prior(x_pred, x_out, add_noise)
        k = joint_covariance[:len(K), len(K):]
        prod_covariance = np.block([[K, k * 0.], [k.T * 0., kk * np.identity(len(kk))]])
        zeros = np.zeros((len(joint_covariance)))
        return {"x": x_pred,
                "x_out": x_out,
                "total correlation":
                    self._independent_count(x_out) * self.kl_div(zeros, zeros, joint_covariance, prod_covariance)}

    ###########################################################################
    def _prior_at(self, x_pred, x_out):
        """The prior mean and covariance at the requested (point, output) pairs, flat."""
        hps = self.hyperparameters
        x_in = self._index_points(x_pred, x_out)
        mean = self._per_output(self.mean_columns(self.compute_mean(x_in, hps)), x_out)
        kk = self.compute_covariances(x_in, x_in, hps)
        if self.kronecker: kk = self._kronecker_pair_covariance(kk, *self._kronecker_pairs(len(x_pred), x_out, kk))
        return mean, self._shared_by_outputs(x_out, kk, pairs=True)

    def gp_relative_information_entropy(self, x_pred, x_out=None, add_noise=False):
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
        prio_mean, kk = self._prior_at(x_pred, x_out)
        kk = kk + (np.identity(len(kk)) * 1e-9)
        post_cov = self.posterior_covariance(x_pred, x_out=x_out, add_noise=add_noise)["S_flat"]
        post_cov = post_cov + (np.identity(len(post_cov)) * 1e-9)
        post_mean = self.posterior_mean(x_pred, x_out=x_out)["m(x)_flat"]
        return {"x": x_pred,
                "x_out": x_out,
                "RIE": self.kl_div(prio_mean, post_mean, kk, post_cov)}

    ###########################################################################
    def gp_relative_information_entropy_set(self, x_pred, x_out=None, add_noise=False):
        x_out = self._outputs(x_out)
        RIE = np.array([self.gp_relative_information_entropy(x_pred[i:i + 1], x_out=x_out, add_noise=add_noise)["RIE"]
                        for i in range(len(x_pred))])
        return {"x": x_pred,
                "x_out": x_out,
                "RIE": RIE}

    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out=None):
        x_out = self._outputs(x_out)
        self._perform_input_checks(x_pred)
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
        return {"x": x_pred,
                "x_out": x_out,
                "mu": mu,
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

    def _perform_input_checks(self, x_pred):
        assert isinstance(x_pred, np.ndarray) or isinstance(x_pred, list), "wrong format in x_pred"
        if isinstance(x_pred, np.ndarray):
            assert np.ndim(x_pred) == 2, "wrong dim in x_pred, has to be 2-d"
            assert x_pred.shape[1] == self.input_set_dim, "wrong number of columns in x_pred"

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
