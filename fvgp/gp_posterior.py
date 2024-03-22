class GPosterior:  # pragma: no cover
    def __init__(self, marginal_density_obj, data_obj):
        assert isinstance(marginal_density_obj.KVinvY, np.ndarray)

        self.KV = prior_obj.KV
        self.factorization_obj = prior_obj.factorization_obj
        self.prior_obj = prior_obj
        self.data_obj = data_obj

    def _KVsolve(self, b):
        if self.factorization_obj[0] == "LU":
            LU = self.factorization_obj[1]
            return LU.solve(b)
        elif self.factorization_obj[0] == "Chol":
            c, l = self.factorization_obj[1], self.factorization_obj[2]
            return cho_solve((c, l), b)
        elif self.factorization_obj[0] == "gp2Scale":
            res = np.empty((len(b), b.shape[1]))
            if b.shape[1] > 100: warnings.warn(
                "You want to predict at >100 points. \n When using gp2Scale, this takes a while. \n \
                Better predict at only a handful of points.")
            for i in range(b.shape[1]):
                res[:, i], exit_status = cg(self.KV, b[:, i])
            return res
        elif self.factorization_obj[0] == "Inv":
            return self.KVinv @ b
        else:
            raise Exception("Non-permitted factorization object encountered.")

    def posterior_mean(self, x_pred, hyperparameters=None, x_out=None):
        """
        This function calculates the posterior mean for a set of input points.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        hyperparameters : np.ndarray, optional
            A numpy array of the correct size depending on the kernel. This is optional in case the posterior mean
            has to be computed with given hyperparameters, which is, for instance, the case if the posterior mean is
            a constraint during training. The default is None which means the initialized or trained hyperparameters
            are used.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution points and function values : dict
        """
        x_data, y_data, KVinvY = self.x_data.copy(), self.y_data.copy(), self.KVinvY.copy()
        if hyperparameters is not None:
            hps = hyperparameters
            K, KV, KVinvY, logdet, FO, KVinv, mean, cov = self._compute_GPpriorV(x_data, y_data, hps, calc_inv=False)
        else:
            hps = self.hyperparameters

        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, hps, self)
        A = k.T @ KVinvY
        posterior_mean = self.mean_function(x_pred, hps, self) + A

        return {"x": x_pred,
                "f(x)": posterior_mean}

    def posterior_mean_grad(self, x_pred, hyperparameters=None, x_out=None, direction=None):
        """
        This function calculates the gradient of the posterior mean for a set of input points.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        hyperparameters : np.ndarray, optional
            A numpy array of the correct size depending on the kernel. This is optional in case the posterior mean
            has to be computed with given hyperparameters, which is, for instance, the case if the posterior mean is
            a constraint during training. The default is None which means the initialized or trained hyperparameters
            are used.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        direction : int, optional
            Direction of derivative, If None (default) the whole gradient will be computed.

        Return
        ------
        Solution : dict
        """
        x_data, y_data, KVinvY = self.x_data.copy(), self.y_data.copy(), self.KVinvY.copy()
        if hyperparameters is not None:
            hps = hyperparameters
            K, KV, KVinvY, logdet, FO, KVinv, mean, cov = self._compute_GPpriorV(x_data, y_data, hps, calc_inv=False)
        else:
            hps = self.hyperparameters

        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, hps, self)
        f = self.mean_function(x_pred, hps, self)
        eps = 1e-6
        if direction is not None:
            x1 = np.array(x_pred)
            x1[:, direction] = x1[:, direction] + eps
            mean_der = (self.mean_function(x1, hps, self) - f) / eps
            k = self.kernel(x_data, x_pred, hps, self)
            k_g = self.d_kernel_dx(x_pred, x_data, direction, hps)
            posterior_mean_grad = mean_der + (k_g @ KVinvY)
        else:
            posterior_mean_grad = np.zeros((x_pred.shape))
            for direction in range(len(x_pred[0])):
                x1 = np.array(x_pred)
                x1[:, direction] = x1[:, direction] + eps
                mean_der = (self.mean_function(x1, hps, self) - f) / eps
                k = self.kernel(x_data, x_pred, hps, self)
                k_g = self.d_kernel_dx(x_pred, x_data, direction, hps)
                posterior_mean_grad[:, direction] = mean_der + (k_g @ KVinvY)
            direction = "ALL"

        return {"x": x_pred,
                "direction": direction,
                "df/dx": posterior_mean_grad}

    ###########################################################################
    def posterior_covariance(self, x_pred, x_out=None, variance_only=False, add_noise=False):
        """
        Function to compute the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        variance_only : bool, optional
            If True the computation of the posterior covariance matrix is avoided which can save compute time.
            In that case the return will only provide the variance at the input points.
            Default = False.
        add_noise : bool, optional
            If True the noise variances will be added to the posterior variances. Default = False.

        Return
        ------
        Solution : dict
        """

        x_data = self.x_data.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
        if self.KVinv is not None:
            if variance_only:
                S = None
                v = np.diag(kk) - np.einsum('ij,jk,ki->i', k.T, self.KVinv, k)
            else:
                S = kk - (k.T @ self.KVinv @ k)
                v = np.array(np.diag(S))
        else:
            k_cov_prod = self._KVsolve(k)
            S = kk - (k_cov_prod.T @ k)
            v = np.array(np.diag(S))
        if np.any(v < -0.001):
            logger.warning(inspect.cleandoc("""#
            Negative variances encountered. That normally means that the model is unstable.
            Rethink the kernel definitions, add more noise to the data,
            or double check the hyperparameter optimization bounds. This will not
            terminate the algorithm, but expect anomalies."""))
            v[v < 0.0] = 0.0
            if not variance_only:
                np.fill_diagonal(S, v)

        if add_noise and callable(self.noise_function):
            noise = self.noise_function(x_pred, self.hyperparameters, self)
            if scipy.sparse.issparse(noise): noise = noise.toarray()
            v = v + np.diag(noise)
            if S is not None: S = S + noise

        return {"x": x_pred,
                "v(x)": v,
                "S": S}

    def posterior_covariance_grad(self, x_pred, x_out=None, direction=None):
        """
        Function to compute the gradient of the posterior covariance.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.
        direction : int, optional
            Direction of derivative, If None (default) the whole gradient will be computed.

        Return
        ------
        Solution : dict
        """
        x_data = self.x_data.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        k_covariance_prod = self._KVsolve(k)
        if direction is not None:
            k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
            kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
            x1 = np.array(x_pred)
            x2 = np.array(x_pred)
            eps = 1e-6
            x1[:, direction] = x1[:, direction] + eps
            kk_g = (self.kernel(x1, x1, self.hyperparameters, self) - \
                    self.kernel(x2, x2, self.hyperparameters, self)) / eps
            a = kk_g - (2.0 * k_g.T @ k_covariance_prod)
            return {"x": x_pred,
                    "dv/dx": np.diag(a),
                    "dS/dx": a}
        else:
            grad_v = np.zeros((len(x_pred), len(x_pred[0])))
            for direction in range(len(x_pred[0])):
                k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
                kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
                x1 = np.array(x_pred)
                x2 = np.array(x_pred)
                eps = 1e-6
                x1[:, direction] = x1[:, direction] + eps
                kk_g = (self.kernel(x1, x1, self.hyperparameters, self) - \
                        self.kernel(x2, x2, self.hyperparameters, self)) / eps
                grad_v[:, direction] = np.diag(kk_g - (2.0 * k_g.T @ k_covariance_prod))
            return {"x": x_pred,
                    "dv/dx": grad_v}

    ###########################################################################
    def joint_gp_prior(self, x_pred, x_out=None):
        """
        Function to compute the joint prior over f (at measured locations) and f_pred at x_pred.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution : dict
        """

        x_data, K, prior_mean_vec = self.x_data.copy(), self.K.copy(), self.prior_mean_vec.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
        post_mean = self.mean_function(x_pred, self.hyperparameters, self)
        joint_gp_prior_mean = np.append(prior_mean_vec, post_mean)
        return {"x": x_pred,
                "K": K,
                "k": k,
                "kappa": kk,
                "prior mean": joint_gp_prior_mean,
                "S": np.block([[K, k], [k.T, kk]])}

    ###########################################################################
    def joint_gp_prior_grad(self, x_pred, direction, x_out=None):
        """
        Function to compute the gradient of the data-informed prior.

        Parameters
        ------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        direction : int
            Direction of derivative.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution : dict
        """
        x_data, K, prior_mean_vec = self.x_data.copy(), self.K.copy(), self.prior_mean_vec.copy()
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self)
        k_g = self.d_kernel_dx(x_pred, x_data, direction, self.hyperparameters).T
        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        eps = 1e-6
        x1[:, direction] = x1[:, direction] + eps
        x2[:, direction] = x2[:, direction] - eps
        kk_g = (self.kernel(x1, x1, self.hyperparameters, self) - self.kernel(x2, x2, self.hyperparameters, self)) / (
            2.0 * eps)
        post_mean = self.mean_function(x_pred, self.hyperparameters, self)
        mean_der = (self.mean_function(x1, self.hyperparameters, self) - self.mean_function(x2, self.hyperparameters,
                                                                                            self)) / (2.0 * eps)
        full_gp_prior_mean_grad = np.append(np.zeros((prior_mean_vec.shape)), mean_der)
        prior_cov_grad = np.zeros(K.shape)
        return {"x": x_pred,
                "K": K,
                "dk/dx": k_g,
                "d kappa/dx": kk_g,
                "d prior mean/x": full_gp_prior_mean_grad,
                "dS/dx": np.block([[prior_cov_grad, k_g], [k_g.T, kk_g]])}

    ###########################################################################
    def entropy(self, S):
        """
        Function computing the entropy of a normal distribution
        res = entropy(S); S is a 2d np.ndarray array, a covariance matrix which is non-singular.

        Parameters
        ----------
        S : np.ndarray
            A covariance matrix.

        Return
        ------
        Entropy : float
        """
        dim = len(S[0])
        logdet = self._logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * logdet)

    ###########################################################################
    def gp_entropy(self, x_pred, x_out=None):
        """
        Function to compute the entropy of the gp prior probability distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Entropy : float
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        priors = self.joint_gp_prior(x_pred, x_out=None)
        S = priors["S"]
        dim = len(S[0])
        logdet = self._logdet(S)
        return (float(dim) / 2.0) + ((float(dim) / 2.0) * np.log(2.0 * np.pi)) + (0.5 * logdet)

    ###########################################################################
    def gp_entropy_grad(self, x_pred, direction, x_out=None):
        """
        Function to compute the gradient of entropy of the prior in a given direction.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        direction : int
            Direction of derivative.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Entropy gradient in given direction : float
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        priors1 = self.joint_gp_prior(x_pred, x_out=None)
        priors2 = self.joint_gp_prior_grad(x_pred, direction, x_out=None)
        S1 = priors1["S"]
        S2 = priors2["dS/dx"]
        return 0.5 * np.trace(self._inv(S1) @ S2)

    ###########################################################################
    def _kl_div_grad(self, mu1, dmu1dx, mu2, S1, dS1dx, S2):
        """
        This function computes the gradient of the KL divergence between two normal distributions
        when the gradients of the mean and covariance are given.
        a = kl_div(mu1, dmudx,mu2, S1, dS1dx, S2); S1, S2 are 2d numpy arrays, matrices have to be non-singular,
        mu1, mu2 are mean vectors, given as 2d arrays
        """
        logdet1 = self._logdet(S1)
        logdet2 = self._logdet(S2)
        x1 = self._solve(S2, dS1dx)
        mu = np.subtract(mu2, mu1)
        x2 = self._solve(S2, mu)
        x3 = self._solve(S2, -dmu1dx)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + ((x3.T @ mu) + (x2.T @ -dmu1dx)) - np.trace(np.linalg.inv(S1) @ dS1dx))
        return kld

    ###########################################################################
    def kl_div(self, mu1, mu2, S1, S2):
        """
        Function to compute the KL divergence between two Gaussian distributions.

        Parameters
        ----------
        mu1 : np.ndarray
            Mean vector of distribution 1.
        mu1 : np.ndarray
            Mean vector of distribution 2.
        S1 : np.ndarray
            Covariance matrix of distribution 1.
        S2 : np.ndarray
            Covariance matrix of distribution 2.

        Return
        ------
        KL divergence : float
        """
        logdet1 = self._logdet(S1)
        logdet2 = self._logdet(S2)
        x1 = self._solve(S2, S1)
        mu = np.subtract(mu2, mu1)
        x2 = self._solve(S2, mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu)[0] - float(dim) + (logdet2 - logdet1))
        if kld < -1e-4:
            warnings.warn("Negative KL divergence encountered. That happens when \n \
                    one of the covariance matrices is close to positive semi definite \n\
                    and therefore the logdet() calculation becomes unstable.\n \
                    Returning abs(KLD)")
            logger.debug("Negative KL divergence encountered")
        return abs(kld)

    ###########################################################################
    def gp_kl_div(self, x_pred, comp_mean, comp_cov, x_out=None):
        """
        Function to compute the kl divergence of a posterior at given points
        and a given normal distribution.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        comp_mean : np.ndarray
            Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
        comp_cov : np.ndarray
            Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        res = self.posterior_mean(x_pred, x_out=None)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_pred, x_out=None)["S"] + np.identity(len(x_pred)) * 1e-9
        comp_cov = comp_cov + np.identity(len(comp_cov)) * 1e-9
        return {"x": x_pred,
                "gp posterior mean": gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}

    ###########################################################################
    def gp_kl_div_grad(self, x_pred, comp_mean, comp_cov, direction, x_out=None):
        """
        Function to compute the gradient of the kl divergence of a posterior at given points.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        comp_mean : np.ndarray
            Comparison mean vector for KL divergence. len(comp_mean) = len(x_pred)
        comp_cov : np.ndarray
            Comparison covariance matrix for KL divergence. shape(comp_cov) = (len(x_pred),len(x_pred))
        direction: int
            The direction in which the gradient will be computed.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        ------
        Solution : dict
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        gp_mean = self.posterior_mean(x_pred, x_out=None)["f(x)"]
        gp_mean_grad = self.posterior_mean_grad(x_pred, direction=direction, x_out=None)["df/dx"]
        gp_cov = self.posterior_covariance(x_pred, x_out=None)["S"] + np.identity(len(x_pred)) * 1e-9
        gp_cov_grad = self.posterior_covariance_grad(x_pred, direction=direction, x_out=None)["dS/dx"]
        comp_cov = comp_cov + np.identity(len(comp_cov)) * 1e-9
        return {"x": x_pred,
                "gp posterior mean": gp_mean,
                "gp posterior mean grad": gp_mean_grad,
                "gp posterior covariance": gp_cov,
                "gp posterior covariance grad": gp_cov_grad,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div grad": self._kl_div_grad(gp_mean, gp_mean_grad, comp_mean, gp_cov, gp_cov_grad, comp_cov)}

    ###########################################################################
    def mutual_information(self, joint, m1, m2):
        """
        Function to calculate the mutual information between two normal distributions, which is
        equivalent to the KL divergence(joint, marginal1 * marginal1).

        Parameters
        ----------
        joint : np.ndarray
            The joint covariance matrix.
        m1 : np.ndarray
            The first marginal distribution
        m2 : np.ndarray
            The second marginal distribution

        Return
        ------
        Mutual information : float
        """
        return self.entropy(m1) + self.entropy(m2) - self.entropy(joint)

    ###########################################################################
    def gp_mutual_information(self, x_pred, x_out=None):
        """
        Function to calculate the mutual information between
        the random variables f(x_data) and f(x_pred).
        The mutual information is always positive, as it is a KL divergence, and is bounded
        from below by 0. The maxima are expected at the data points. Zero is expected far from the
        data support.
        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
        """
        x_data, K = self.x_data.copy(), self.K.copy() + (np.identity(len(self.K)) * 1e-9)
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self) + (np.identity(len(x_pred)) * 1e-9)

        joint_covariance = \
            np.asarray(np.block([[K, k], \
                                 [k.T, kk]]))
        return {"x": x_pred,
                "mutual information": self.mutual_information(joint_covariance, kk, K)}

    ###########################################################################
    def gp_total_correlation(self, x_pred, x_out=None):
        """
        Function to calculate the interaction information between
        the random variables f(x_data) and f(x_pred). This is the mutual information
        of each f(x_pred) with f(x_data). It is also called the Multiinformation.
        It is best used when several prediction points are supposed to be mutually aware.
        The total correlation is always positive, as it is a KL divergence, and is bounded
        from below by 0. The maxima are expected at the data points. Zero is expected far from the
        data support.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            Total correlation between prediction points, as a collective.
        """
        x_data, K = self.x_data.copy(), self.K.copy() + (np.identity(len(self.K)) * 1e-9)
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        k = self.kernel(x_data, x_pred, self.hyperparameters, self)
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self) + (np.identity(len(x_pred)) * 1e-9)
        joint_covariance = np.asarray(np.block([[K, k],
                                                [k.T, kk]]))

        prod_covariance = np.asarray(np.block([[K, k * 0.],
                                               [k.T * 0., kk * np.identity(len(kk))]]))

        return {"x": x_pred,
                "total correlation": self.kl_div(np.zeros((len(joint_covariance))), np.zeros((len(joint_covariance))),
                                                 joint_covariance, prod_covariance)}

    ###########################################################################
    def gp_relative_information_entropy(self, x_pred, x_out=None):
        """
        Function to compute the KL divergence and therefore the relative information entropy
        of the prior distribution over predicted function values and the posterior distribution.
        The value is a reflection of how much information is predicted to be gained
        through observing a set of data points at x_pred.

        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            Relative information entropy of prediction points, as a collective.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")
        kk = self.kernel(x_pred, x_pred, self.hyperparameters, self) + (np.identity(len(x_pred)) * 1e-9)
        post_cov = self.posterior_covariance(x_pred, x_out=None)["S"] + (np.identity(len(x_pred)) * 1e-9)
        return {"x": x_pred,
                "RIE": self.kl_div(np.zeros((len(x_pred))), np.zeros((len(x_pred))), kk, post_cov)}

    ###########################################################################
    def gp_relative_information_entropy_set(self, x_pred, x_out=None):
        """
        Function to compute the KL divergence and therefore the relative information entropy
        of the prior distribution over predicted function values and the posterior distribution.
        The value is a reflection of how much information is predicted to be gained
        through observing each data point in x_pred separately, not all
        at once as in `gp_relative_information_entrop`.


        Parameters
        ----------
        x_pred : np.ndarray
            A numpy array of shape (V x D), interpreted as  an array of input point positions.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            Relative information entropy of prediction points, but not as a collective.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")
        RIE = np.zeros((len(x_pred)))
        for i in range(len(x_pred)):
            RIE[i] = self.gp_relative_information_entropy(x_pred[i].reshape(1, len(x_pred[i])), x_out=None)["RIE"]

        return {"x": x_pred,
                "RIE": RIE}

    ###########################################################################
    def posterior_probability(self, x_pred, comp_mean, comp_cov, x_out=None):
        """
        Function to compute probability of a probabilistic quantity of interest,
        given the GP posterior at a given point.

        Parameters
        ----------
        x_pred: 1d or 2d numpy array of points, note, these are elements of the
                index set which results from a cartesian product of input and output space
        comp_mean: a vector of mean values, same length as x_pred
        comp_cov: covarianve matrix, in R^{len(x_pred)xlen(x_pred)}
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            The probability of a probabilistic quantity of interest, given the GP posterior at a given point.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        res = self.posterior_mean(x_pred, x_out=None)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_pred, x_out=None)["S"]
        gp_cov_inv = self._inv(gp_cov)
        comp_cov_inv = self._inv(comp_cov)
        cov = self._inv(gp_cov_inv + comp_cov_inv)
        mu = cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        logdet1 = self._logdet(cov)
        logdet2 = self._logdet(gp_cov)
        logdet3 = self._logdet(comp_cov)
        dim = len(mu)
        C = 0.5 * (((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T \
                    @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean)) \
                   - (gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0 * np.pi) ** (dim / 2.0)) + (0.5 * (logdet2 + logdet3)))
        return {"mu": mu,
                "covariance": cov,
                "probability":
                    np.exp(ln_p)
                }

    def posterior_probability_grad(self, x_pred, comp_mean, comp_cov, direction, x_out=None):
        """
        Function to compute the gradient of the probability of a probabilistic quantity of interest,
        given the GP posterior at a given point.

        Parameters
        ----------
        x_pred: 1d or 2d numpy array of points, note, these are elements of the
                index set which results from a cartesian product of input and output space
        comp_mean: a vector of mean values, same length as x_pred
        comp_cov: covarianve matrix, in R^{len(x_pred)xlen(x_pred)}
        direction : int
            The direction to compute the gradient in.
        x_out : np.ndarray, optional
            Output coordinates in case of multitask GP use; a numpy array of size (N x L),
            where N is the number of output points,
            and L is the dimensionality of the output space.

        Return
        -------
        Solution : dict
            The gradient of the probability of a probabilistic quantity of interest,
            given the GP posterior at a given point.
        """
        if not self.non_Euclidean:
            if np.ndim(x_pred) == 1: raise Exception("x_pred has to be a 2d numpy array, not 1d")
            if x_out is not None: x_pred = self._cartesian_product_euclid(x_pred, x_out)
            if len(x_pred[0]) != self.input_space_dim: raise Exception(
                "Wrong dimensionality of the input points x_pred.")
        elif x_out is not None:
            raise Exception("Multi-task GPs on non-Euclidean spaces not implemented yet.")

        x1 = np.array(x_pred)
        x2 = np.array(x_pred)
        x1[:, direction] = x1[:, direction] + 1e-6
        x2[:, direction] = x2[:, direction] - 1e-6

        probability_grad = (self.posterior_probability(x1, comp_mean, comp_cov, x_out=None)["probability"] - \
                            self.posterior_probability(x2, comp_mean, comp_cov, x_out=None)["probability"]) / 2e-6
        return {"probability grad": probability_grad}

    ###########################################################################
    def _int_gauss(self, S):
        return ((2.0 * np.pi) ** (len(S) / 2.0)) * np.sqrt(np.linalg.det(S))
