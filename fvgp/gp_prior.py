import numpy as np
import inspect
import warnings
from .kernels import *
from .gp2Scale_covariance import distributed_covariance, stack_augmented_covariance
from loguru import logger
warnings.simplefilter("once", UserWarning)


class GPprior:
    def __init__(self,
                 data,
                 trainer,
                 kernel=None,
                 prior_mean_function=None,
                 kernel_grad=None,
                 prior_mean_function_grad=None,
                 gp2Scale_batch_size=10000,
                 gp2Scale_distribution="blockwise",
                 ):

        self.kernel_function = kernel
        self.prior_mean_function = prior_mean_function
        self.batch_size = gp2Scale_batch_size
        self.gp2Scale_distribution = gp2Scale_distribution
        self.data = data
        self.trainer = trainer

        assert gp2Scale_distribution in ("blockwise", "rowwise"), \
            "gp2Scale_distribution must be `blockwise` or `rowwise`"

        assert callable(kernel) or kernel is None, "kernel must be callable or None"
        assert callable(prior_mean_function) or prior_mean_function is None, \
            "prior_mean_function must be callable or None"
        assert isinstance(self.hyperparameters, np.ndarray), "hyperparameters must be np.ndarray"
        assert np.ndim(self.hyperparameters) == 1, "hyperparameters must be 1-d"

        if not self.Euclidean and not callable(kernel):
            raise Exception(
                "For GPs on non-Euclidean input spaces you need a user-defined kernel and initial hyperparameters.")

        if self.gp2Scale:
            if not callable(kernel):
                warnings.warn("You have chosen to activate gp2Scale. A powerful tool! "
                              "But you have not supplied a kernel that is compactly supported. "
                              "I will use an anisotropic Wendland kernel for now.",
                              stacklevel=2)
                if self.compute_device == "cpu":
                    kernel = wendland_anisotropic_gp2Scale_cpu
                elif self.compute_device == "gpu":
                    kernel = wendland_anisotropic_gp2Scale_gpu
            if not self.compute_workers:  # pragma: no cover - needs a worker-less client
                logger.debug("No workers available")


        # kernel
        self.k_n_params = 3
        if callable(kernel):
            self.kernel = kernel
            self.k_n_params = len(inspect.signature(kernel).parameters)
        elif kernel is None:
            self.kernel = self._default_kernel
        else:  # pragma: no cover - GP/fvGP assert a callable-or-None kernel first
            raise Exception("No valid kernel function specified")
        self.d_kernel_dx = self._d_kernel_dx
        if callable(kernel_grad):
            self._dk_dh = kernel_grad
        elif not callable(kernel):
            self._dk_dh = self._default_kernel_analytical_gradient
            self.ram_economy = False
        else:
            if self.ram_economy is True:
                self._dk_dh = self._kernel_derivative
            else:
                self._dk_dh = self._kernel_gradient

        # prior-mean
        self.m_n_params = 2
        if callable(prior_mean_function):
            self._default_mean = False
            self.mean_function = prior_mean_function
            self.m_n_params = len(inspect.signature(prior_mean_function).parameters)
        else:
            self.mean_function = self._default_mean_function
            self._default_mean = True

        if callable(prior_mean_function_grad):
            self._dm_dh = prior_mean_function_grad
        elif callable(prior_mean_function):
            self._dm_dh = self._finitediff_dm_dh
        else:
            self._dm_dh = self._default_dm_dh

        self.x_data_scatter_future = None
        if self.gp2Scale and self.client is not None:
            self.x_data_scatter_future = self._scatter(self.x_data)

        self.m, self.K = self._compute_prior(self.x_data, self.hyperparameters)
        logger.debug("Prior successfully initialized.")

    ##############################################################
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
    def x_old(self):
        return self.data.x_old
    
    @property
    def x_new(self):
        return self.data.x_new

    @property
    def y_data(self):
        return self.data.y_data

    @property
    def ram_economy(self):
        return self.data.ram_economy

    @ram_economy.setter
    def ram_economy(self, value):
        self.data.ram_economy = value

    @property
    def gp2Scale(self):
        return self.data.gp2Scale

    @property
    def Euclidean(self):
        return self.data.Euclidean

    @property
    def compute_device(self):
        return self.data.compute_device

    @property
    def client(self):
        return self.data.dask_client

    @property
    def compute_workers(self):
        return self.data.compute_workers

    ################################################################
    #START: FUNCTIONS THAT ALLOW INTERACTING WITH THE CLASS#########
    ################################################################
    def augment_state_data(self):
        self.m, self.K = self._update_prior(self.x_old, self.x_new, self.hyperparameters)
        if self.gp2Scale and self.client is not None:
            # Refresh the persistent x_data scatter so it reflects the full post-append
            # dataset.  Overwrite (no explicit release): the old future loses its only
            # Python ref and is cleaned up via __del__.  This is race-free within a
            # single GP's lifetime; do NOT churn many GP instances back-to-back without
            # a `del gp; gc.collect(); client.run(lambda: None)` between them.
            self.x_data_scatter_future = self._scatter(self.x_data)
        logger.debug("Prior mean and covariance updated after data augmentation.")

    def update_state_data(self):
        """
        This is for the case that the data has changed, but not just been augmented. 
        For example, in an online learning setting where old data points are replaced by new ones.
        """
        if self.gp2Scale and self.client is not None:
            # Full data change: refresh the persistent scatter before rebuilding K.
            # Overwrite (no explicit release); the old future is GC'd at a quiet moment.
            self.x_data_scatter_future = self._scatter(self.x_data)
        self.m, self.K = self._compute_prior(self.x_data, self.hyperparameters)
        logger.debug("Prior mean and covariance updated after data change.")

    def update_state_hyperparameters(self):
        self.m, self.K = self._compute_prior(self.x_data, self.hyperparameters)
        logger.debug("Prior mean and covariance updated after hyperparameter change.")

    def compute_prior_covariance_matrix(self, x, hyperparameters):
        """computes the prior covariance matrix from the kernel"""
        if self.gp2Scale:
            # Every caller of this method hands over the current training set, so the
            # persistent scatter is the right one to slice -- but only reuse it when the
            # lengths agree, in case a caller ever does otherwise. Compare with len() and
            # not np.shape(): a non-Euclidean point is an arbitrary object, and asking
            # numpy for the shape of a list of ragged objects raises.
            future = self.x_data_scatter_future
            if future is not None and len(x) != len(self.x_data): future = None
            K = self._gp2Scale_covariance(x, x, hyperparameters, symmetric=True, x1_future=future)
        else:
            K = self.compute_covariances(x, x, hyperparameters)
        return K

    def compute_data_cross_covariance(self, x_pred, hyperparameters):
        """computes k(x_data, x_pred), the cross-covariance the posterior needs.

        Under gp2Scale this is the one covariance the posterior cannot simply evaluate on
        the client: it has as many rows as there are data points.  It goes through the
        same distributed assembler as the prior and comes back sparse, so a posterior
        mean never materializes an (N x n_pred) dense array.  Below one batch of data
        there is nothing to gain from the cluster, so the kernel is called directly.
        """
        if self.gp2Scale and self.client is not None and len(self.x_data) > self.batch_size:
            return self._gp2Scale_covariance(self.x_data, x_pred, hyperparameters,
                                             x1_future=self.x_data_scatter_future)
        return self.compute_covariances(self.x_data, x_pred, hyperparameters)

    def compute_covariances(self, x1, x2, hps):
        """computes the covariances via k(x,x')"""
        if self.k_n_params == 3:
            return self.kernel(x1, x2, hps)
        elif self.k_n_params == 4:
            return self.kernel(x1, x2, hps, self.args)
        else:
            raise Exception("No valid kernel function signature")

    def compute_mean(self, x, hyperparameters):
        """computes the mean from some x"""
        if self.m_n_params == 2:
            m = self.mean_function(x, hyperparameters)
        elif self.m_n_params == 3:
            m = self.mean_function(x, hyperparameters, self.args)
        else:
            raise Exception("No valid mean function signature")
        return m

    def dk_dh(self, x1, x2, hyperparameters, direction=None):
        if self.ram_economy:
            return self._dk_dh(x1, x2, hyperparameters, direction)
        else:
            return self._dk_dh(x1, x2, hyperparameters)

    def dm_dh(self, x_data, hyperparameters):
        return self._dm_dh(x_data, hyperparameters)
    #################################################################
    #END: FUNCTIONS THAT ALLOW INTERACTING WITH THE CLASS############
    #################################################################

    def _compute_prior(self, x_data, hyperparameters):
        m = self.compute_mean(x_data, hyperparameters)
        K = self.compute_prior_covariance_matrix(x_data, hyperparameters)
        assert np.ndim(m) == 1, "mean function returned non-1-d result: " + str(m)
        assert np.ndim(K) == 2, "prior covariance K must be 2-d"
        logger.debug("Prior mean and covariance matrix successfully computed.")
        return m, K

    def _update_prior(self, x_old, x_new, hyperparameters):
        # self.x_data is already the appended dataset here -- GPdata.update runs before
        # augment_state_data -- so use it rather than rebuilding it. np.vstack cannot
        # concatenate the plain lists a non-Euclidean input space is made of.
        if self._default_mean: m = self.compute_mean(self.x_data, hyperparameters)
        else: m = self._update_mean(x_new, hyperparameters)
        K = self._update_prior_covariance_matrix(x_old, x_new, hyperparameters)
        assert np.ndim(m) == 1, "updated mean must be 1-d"
        assert np.ndim(K) == 2, "updated covariance K must be 2-d"
        return m, K

    def _update_prior_covariance_matrix(self, x_old, x_new, hyperparameters):
        """This updated K based on new data"""
        if self.gp2Scale:
            # self.x_data_scatter_future still holds x_old at this point; augment_state_data
            # refreshes it to the full dataset only after this call returns.  The x_new
            # scatter is ours, so it is ours to release.
            x_new_future = self._scatter(x_new)
            try:
                B = self._gp2Scale_covariance(x_old, x_new, hyperparameters,
                                              x1_future=self.x_data_scatter_future,
                                              x2_future=x_new_future)
                D = self._gp2Scale_covariance(x_new, x_new, hyperparameters,
                                              symmetric=True, x1_future=x_new_future)
            finally:
                x_new_future.release()
            K = stack_augmented_covariance(self.K, B, D)
        else:
            k = self.compute_covariances(x_old, x_new, hyperparameters)
            kk = self.compute_covariances(x_new, x_new, hyperparameters)
            K = np.block([
                [self.K, k],
                [k.T, kk]
            ])
        return K

    def _update_mean(self, x_new, hyperparameters):
        if np.ndim(self.m) == 1:
            m = np.append(self.m, self.compute_mean(x_new, hyperparameters))
        elif np.ndim(self.m) == 2:
            raise Exception("prior mean has to be a vector")
        else:
            raise Exception("Prior mean in wrong format")
        return m

    def _scatter(self, x):
        """Broadcast a point set to the compute workers, under a key of its own.

        ``hash=False`` is what makes the key unique.  By default dask keys scattered data
        by a hash of its content, so scattering the same array twice -- a new GP on the
        same data, a second prediction at the same points, a re-scatter after an append
        that did not change x -- lands on one key.  The first copy's release then
        schedules a ``_dec_ref`` that races the second scatter inside the scheduler, and
        the tasks depending on it come back as ``KeyError`` or ``CancelledError``.  A
        unique key per scatter removes the collision at its source; the only thing given
        up is a de-duplication we never wanted, since each copy is released with the
        object that made it.
        """
        # A non-Euclidean point set is a plain list, and dask scatters a list
        # *element-wise*, handing back a list of futures rather than one future for the
        # whole set. Wrapping it in a single-element list and taking that one future
        # keeps the point set together, which is what the workers slice.
        if isinstance(x, list):
            return self.client.scatter([x], workers=self.compute_workers, broadcast=True,
                                       direct=True, hash=False)[0]
        return self.client.scatter(x, workers=self.compute_workers, broadcast=True,
                                   direct=True, hash=False)

    def _gp2Scale_covariance(self, x1, x2, hyperparameters, symmetric=False,
                             x1_future=None, x2_future=None):
        """The single distributed kernel evaluation, shared by prior, append and posterior.

        Owns nothing but scatter lifetime: a future passed in belongs to the caller and is
        left alone (this is how the persistent ``x_data_scatter_future`` survives), while
        any future created here is released before returning.  The scheduling and assembly
        live in :py:mod:`fvgp.gp2Scale_covariance`.
        """
        if self.client is None:
            raise Exception("gp2Scale needs a dask client to compute covariances.")

        own1 = x1_future is None
        if own1: x1_future = self._scatter(x1)
        if symmetric:
            # One broadcast copy, sliced on both axes -- the assembler relies on this to
            # schedule only the upper triangle.
            # len(), not np.shape(): non-Euclidean points are arbitrary objects
            assert x2 is x1 or len(x2) == len(x1), "symmetric requires x1 == x2"
            x2_future, own2 = x1_future, False
        else:
            own2 = x2_future is None
            if own2: x2_future = self._scatter(x2)

        try:
            return distributed_covariance(
                self.client, self.kernel, hyperparameters,
                x1_future=x1_future, n1=len(x1),
                x2_future=x2_future, n2=len(x2),
                batch_size=self.batch_size,
                symmetric=symmetric,
                distribution=self.gp2Scale_distribution,
                k_n_params=self.k_n_params,
                args=self.args)
        finally:
            if own1: x1_future.release()
            if own2: x2_future.release()

    ####################################################
    ####################################################
    ####################################################
    ####################################################
    @staticmethod
    def _default_kernel(x1, x2, hyperparameters):
        """
        Function for the default kernel, a Matern kernel of first-order differentiability.

        Parameters
        ----------
        x1 : np.ndarray
            Numpy array of shape (U x D).
        x2 : np.ndarray
            Numpy array of shape (V x D).
        hyperparameters : np.ndarray
            Array of hyperparameters. For this kernel we need D + 1 hyperparameters.

        Return
        ------
        Covariance matrix : np.ndarray
        """
        logger.debug("Default kernel in use.")
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])):
            distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        distance_matrix = np.sqrt(distance_matrix)
        return hps[0] * matern_kernel_diff1(distance_matrix, 1)

    def _d_kernel_dx(self, x1, x2, direction, hyperparameters):
        new_points = np.array(x1)
        epsilon = 1e-8
        new_points[:, direction] += epsilon
        a = self.compute_covariances(new_points, x2, hyperparameters)
        b = self.compute_covariances(x1, x2, hyperparameters)
        derivative = (a - b) / epsilon
        return derivative

    def _kernel_gradient(self, x1, x2, hyperparameters):
        gradient = np.empty((len(hyperparameters), len(x1), len(x2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self._dkernel_dh(x1, x2, direction, hyperparameters)
        return gradient

    def _kernel_derivative(self, x1, x2, hyperparameters, direction):
        derivative = self._dkernel_dh(x1, x2, direction, hyperparameters)
        return derivative

    @staticmethod
    def _default_kernel_analytical_gradient(x1, x2, hyperparameters):
        gradient = np.zeros((len(hyperparameters), len(x1), len(x2)))
        hps = hyperparameters
        dm = np.zeros((len(x1), len(x2)))
        for i in range(len(x1[0])): dm += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
        dm = np.sqrt(dm)

        non_zero_ind = np.where(dm != 0.0)
        for direction in range(len(x1[0])):
            dddh = np.zeros(dm.shape)
            dddh[non_zero_ind] = -abs(np.subtract.outer(x1[:, direction], x2[:, direction]))[non_zero_ind] ** 2 / (
                    hps[direction + 1] ** 3 * dm[non_zero_ind])
            gradient[direction + 1] = hps[0] * matern_kernel_diff1_grad(dm, dddh)
        gradient[0] = matern_kernel_diff1(dm, 1)
        return gradient

    def _dkernel_dh(self, x1, x2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-8
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.compute_covariances(x1, x2, new_hyperparameters1)
        b = self.compute_covariances(x1, x2, new_hyperparameters2)
        derivative = (a - b) / (2.0 * epsilon)
        return derivative

    def _default_mean_function(self, x, hyperparameters):
        """evaluates the gp mean function at the data points """
        if np.ndim(self.y_data) == 1:
            raise Exception("y_data wrong format")
        elif np.ndim(self.y_data) == 2:
            mean = np.zeros((len(x)))
            mean[:] = np.mean(self.y_data)
        else:
            raise Exception("Wrong dim in default mean function")
        return mean

    def _finitediff_dm_dh(self, x, hps):
        gr = np.empty((len(hps), len(x)))
        for i in range(len(hps)):
            temp_hps1 = np.array(hps)
            temp_hps1[i] = temp_hps1[i] + 1e-6
            temp_hps2 = np.array(hps)
            temp_hps2[i] = temp_hps2[i] - 1e-6
            a = self.compute_mean(x, temp_hps1)
            b = self.compute_mean(x, temp_hps2)
            gr[i] = (a - b) / 2e-6
        return gr

    @staticmethod
    def _default_dm_dh(x, hps):
        gr = np.zeros((len(hps), len(x)))
        return gr

    def __getstate__(self):
        state = dict(
            kernel_function=self.kernel_function,
            prior_mean_function=self.prior_mean_function,
            m_n_params=self.m_n_params,
            k_n_params=self.k_n_params,
            batch_size=self.batch_size,
            gp2Scale_distribution=self.gp2Scale_distribution,
            data=self.data,
            trainer=self.trainer,
            kernel=self.kernel,
            d_kernel_dx=self.d_kernel_dx,
            _dk_dh=self._dk_dh,
            mean_function=self.mean_function,
            _dm_dh=self._dm_dh,
            m=self.m,
            K=self.K,
            x_data_scatter_future=None,
            _default_mean=self._default_mean
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
