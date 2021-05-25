"""
Software: FVGP, version: 2.3.4
File containing the gp class
use help() to find information about usage
Author: Marcus Noack
Institution: CAMERA, Lawrence Berkeley National Laboratory
email: MarcusNoack@lbl.gov
This file contains the FVGP class which trains a Gaussian process and predicts
function values.

License:
Copyright (C) 2020 Marcus Michael Noack

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact: MarcusNoack@lbl.gov
"""

from fvgp.gp import GP



class EnsembleGP():
    """
    Ensemble GP class: Provides all tool for a single-task ensemble GP.

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space

    Attributes:
        input_space_dim (int):         dim1
        points (N x dim1 numpy array): 2d numpy array of points
        values (N x n numpy array):    2d numpy array of values
        init_hyperparameters:          list of (1d numpy array (>0)), one entry per GP

    Optional Attributes:
        variances (N x n numpy array):                  variances of the values, default = array of shape of points
                                                        with 1 % of the values
        compute_device:                                 cpu/gpu, default = cpu
        gp_kernel_function(func):                       None/list of functions defining the 
                                                        kernel def name(x1,x2,hyperparameters,self), default = None
        gp_mean_function(func):                         None/list of functions def name(x, self), default = None
        sparse (bool):                                  default = False
        normalize_y:                                    default = False, normalizes the values \in [0,1]

    Example:
        obj = fvGP(3,np.array([[1,2,3],[4,5,6]]),
                         np.array([2,4]),
                         [np.array([2,3,4,5])],
                         variances = np.array([0.01,0.02]),
                         gp_kernel_function = [kernel_function],
                         gp_mean_function = [some_mean_function]
        )
    """
    def __init__(
        self,
        input_space_dim,
        points,
        values,
        number_of_GPs,
        init_hyperparameters,
        variances = None,
        compute_device = "cpu",
        gp_kernel_function = [None],
        gp_mean_function = [None],
        sparse = False,
        normalize_y = False
        ):
        """
        The constructor for the ensemblegp class.
        type help(GP) for more information about attributes, methods and their parameters
        """
        self.number_of_GPs = number_of_GPs
        self.EnsembleGPs = [GP(input_space_dim,points,values,init_hyperparameters[i],
                               variances = variances,compute_device = compute_device,
                               gp_kernel_function = gp_kernel_function[i],
                               gp_mean_function = gp_mean_function[i],
                               sparse = sparse, normalize_y = normalize_y)
                               for i in range(number_of_GPs)]

    def update_EnsembleGP_data(self, points, values, variances = None):
        for i in range(len(self.number_of_GPs)):
            self.EnsembleGPs[i].update_gp_data(points,values, variances = variances)
    ############################################################################
    def stop_training(self):
        print("Ensemble fvGP is cancelling the asynchronous training...")
        try: self.opt.cancel_tasks(); print("Ensemble fvGP successfully cancelled the current training.")
        except: print("No asynchronous training to be cancelled in Ensemble fvGP, no training is running.")

    def kill_training(self):
        print("fvGP is killing asynchronous training....")
        try: self.opt.kill(); print("fvGP successfully killed the training.")
        except: print("No asynchronous training to be killed, no training is running.")

    def train_async(self,
        hps_bounds,
        init_hyperparameters = None,
        optimization_dict = None,
        pop_size = 20,
        tolerance = 0.1,
        max_iter = 120,
        dask_client = None
        ):
        #hps_vector = 
        self.optimize_log_likelihood_async(
            init_hyperparameters,
            hps_bounds,
            max_iter,
            pop_size,
            tolerance,
            local_optimizer,
            global_optimizer,
            deflation_radius,
            dask_client
            )

    def update_hyperparameters(self, n = 1):
        try:
            res = self.opt.get_latest(n)
            self.hyperparameters = res["x"][0]
            self.compute_prior_pdf()
            print("Ensemble fvGP async hyperparameter update successful")
            print("Latest hyperparameters: ", self.hyperparameters)
        except:
            print("Async Hyper-parameter update not successful in Ensemble fvGP. I am keeping the old ones.")
            print("That probbaly means you are not optimizing them asynchronously")
            print("hyperparameters: ", self.hyperparameters)
        return self.hyperparameters

    def optimize_log_likelihood_async(self):
        print("Ensemble fvGP submitted to HGDL optimization")
        print('bounds are',hp_bounds)
        from hgdl.hgdl import HGDL
        try:
            res = self.opt.get_latest(10)
            x0 = res["x"][0:min(len(res["x"])-1,likelihood_pop_size)]
            print("fvGP hybrid HGDL training is starting with points from the last iteration")
        except Exception as err:
            print("fvGP hybrid HGDL training is starting with random points because")
            print(str(err))
            print("This is nothing to worry about, especially in the first iteration")
            x0 = None

        self.opt = HGDL(self.log_likelihood,
                self.log_likelihood_gradient,
                #hess = self.log_likelihood_hessian,
                bounds = hp_bounds,
                num_epochs = max_iter)

        self.opt.optimize(dask_client = dask_client, x0 = x0)

    def ensemble_log_likelihood(self,hyperparameters):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        L = 1

        for i in range(self.number_of_GPs):
            L *= np.ln(weight[i]) + self.EnsembleGPs[i].log_likelihood(hyperparameters[i])
        return L

    def ensemble_log_likelihood_gradient(self,hyperparameters):
        return 0

    def ensemble_log_likelihood_hessian(self,hyperparameters):
        return 0
    ##########################################################
    def compute_prior_pdf(self):
        for i in range(len(self.number_of_GPs)):
            self.EnsembleGPs[i].compute_prior_fvGP_pdf()

    ##########################################################
    def posterior(self,x_iset, res = 100):
        means = [self.EnsembleGPs[i].posterior_mean(x_iset)["f(x)"] for i in range(self.number_of_GPs)]
        covs  = [self.EnsembleGPs[i].posterior_covariance(x_iset)["v(x)"] for i in range(self.number_of_GPs)]
        lower_bounds = [min(means[:][i]) for i in range(len(x_iset))]
        upper_bounds = [max(means[:][i]) for i in range(len(x_iset))]
        pdfs = []
        for i in range(len(x_iset)):
            pdf = np.zeros((res))
            for j in range(self.number_of_GPs):
                pdf += Gaussian(means[i],covs[i],lower_bounds[i],upper_bounds[i], res)
            pdfs.append(pdf)
        return {"f(x)": means, "v(x)":covs, "pdf": pdfs}
    ##########################################################
    def _Gaussian(self,mean,var,lower,upper,res):
        x = np.linspace(lower,upper,res)
        return np.exp(-np.power(x - mean, 2.) / (2. * np.power(var, 2.)))

    def _vectorize(self, d):
        return 0
