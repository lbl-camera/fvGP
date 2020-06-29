import matplotlib.pyplot as plt
import numpy as np
from fvgp import FVGP
import sys
np.random.seed(42)
def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])**2
    d = np.sqrt(d)
    s = abs(np.add.outer(x1[:,0],x2[:,0]))
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)

def non_stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])**2
    d = np.sqrt(d)
    s = abs(np.add.outer(x1[:,0],x2[:,0]))
    return hps[0] * np.outer(x1[:,0],x2[:,0]) * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)


def func(points):
    return 3.0*points + 0.3*np.sin(10.0*points)
    #return 0.3*np.sin((points)*10.0*points)

points = np.sort(np.random.uniform(low = 0.0, high = 2.0, size = (50,1)), axis = 0)
values = func(points)

#my_gp = FVGP(1,1,1,points,values,gp_kernel_function = stationary_kernel, compute_device = "multi-gpu")
my_gp = FVGP(1,1,1,points,values,gp_kernel_function = stationary_kernel, compute_device = "cpu")

my_gp.train([[0.00001,1000.1],[0.0001,1.0],[.99,1.0]],
        optimization_method = 'global',
        likelihood_pop_size = 10,
        likelihood_optimization_tolerance = 0.01,
        likelihood_optimization_max_iter = 20)

mean = np.mean(values)
x_input = np.empty((1000,1))
x_input[:,0] = np.linspace(0,2.0,1000)
y = func(x_input)
hps = my_gp.hyper_parameters
pred2 = my_gp.compute_posterior_fvGP_pdf(x_input , mode='cartesian product', 
        compute_entropies=False, compute_prior_covariances=False,
        compute_posterior_covariances=True, compute_means=True)

plt.figure(figsize = (10,4))
m = np.ndarray.flatten(pred2["posterior means"])
s = np.ndarray.flatten(pred2["posterior covariances"])
plt.plot(x_input,m, label = "posterior mean",linewidth = 3.0)
plt.plot(x_input,y, label = "ground truth",linewidth = 3.0)
plt.fill_between(x_input[:,0],m-3.0*np.sqrt(s),m+3.0*np.sqrt(s), alpha = 0.5, label = "95% confidence interval")
plt.scatter(points,values, label = "data",linewidth = 3.0)
plt.plot(x_input[:,0], np.abs(m-y[:,0]), label = "error",linewidth = 3.0)
plt.legend()
plt.savefig('plot.png')
plt.show()

