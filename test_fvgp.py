import matplotlib.pyplot as plt
import numpy as np
from fvgp import FVGP

def kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])**2
    d = np.sqrt(d)
    return hps[0] * np.exp(-d)

points = np.sort(np.random.uniform(low = 0.0, high = 1.0, size = (500,1)), axis = 0)
values = 3.0*points + 0.3*np.sin(10.0*points)
my_gp = FVGP(1,1,1,points,values,)
my_gp.train([[0.001,10.1],[0.001,10.0]],[[.99,1.0]],
        likelihood_pop_size = 10,
        likelihood_optimization_tolerance = 0.01,
        likelihood_optimization_max_iter = 20)

mean = np.mean(values)
x_input = np.sort(np.random.rand(1000,1), axis = 0)
hps = my_gp.hyper_parameters
pred2 = my_gp.compute_posterior_fvGP_pdf(x_input , mode='cartesian product', 
        compute_entropies=False, compute_prior_covariances=False,
        compute_posterior_covariances=True, compute_means=True)

m = np.ndarray.flatten(pred2["posterior means"])
s = np.ndarray.flatten(pred2["posterior covariances"])
plt.plot(x_input,m)
plt.fill_between(x_input[:,0],m-3.0*np.sqrt(s),m+3.0*np.sqrt(s), alpha = 0.5)
plt.scatter(points,values)
plt.show()
