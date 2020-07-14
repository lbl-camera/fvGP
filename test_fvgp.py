import matplotlib.pyplot as plt
import numpy as np
from fvgp import FVGP
import sys
np.random.seed(42)
def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)

def non_stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])**2
    d = np.sqrt(d)
    return hps[0] * np.outer(x1[:,0],x2[:,0]) * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)

def mt_kernel(x1,x2,hps, obj = None):
    d1 = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])**2
    d2 = abs(np.subtract.outer(x1[:,1],x2[:,1]))**2
    d = np.sqrt(d1+d2)
    return hps[0] * (1.0+(np.sqrt(3.0)*d)) * np.exp(-np.sqrt(3.0)*d)

def func(points):
    return 3.0*points + 0.3*np.sin(10.0*points)
    #return 0.3*np.sin((points)*10.0*points)
#######################################################
#######################################################
#######################################################
#######################################################


N = 20 ### how many training points
points = np.empty((N,1))
points[:,0] = np.linspace(0,2,N) + np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
####change here for multi-task case
values = func(points)
#values = np.empty((20,3))
#values[:,0] = func(points)[0]
#values[:,1] = func(points)[0]*2.44
#values[:,2] = func(points)[0]*3.55
#########################################
my_gp = FVGP(1,1,1,points,values,gp_kernel_function = stationary_kernel, compute_device = "cpu")
#my_gp = FVGP(1,1,3,points,values,gp_kernel_function = mt_kernel, compute_device = "cpu")
#my_gp = FVGP(1,1,1,points,values,gp_kernel_function = non_stationary_kernel, compute_device = "cpu")

my_gp.train([[0.0001,50000.0],[0.001,400.0],[.9,1.1]],
        init_hyper_parameters = [10.0,10.0,1.0],
        optimization_method = 'global',
        likelihood_optimization_pop_size = 20,
        likelihood_optimization_tolerance = 0.1,
        likelihood_optimization_max_iter = 200)


x_input = np.empty((1000,1))
x_input[:,0] = np.linspace(0,2.0,1000)
y = func(x_input)
#hps = my_gp.hyper_parameters

pred1_mean = my_gp.posterior_mean(x_input)
pred1_cov = my_gp.posterior_covariance(x_input)
sig = np.empty((len(x_input)))
for i in range(len(x_input)):
    sig[i] = my_gp.shannon_information_gain(np.array([x_input[i]]))["sig"]
plt.figure(figsize = (10,4))
plt.plot(x_input,pred1_mean["f(x)"], label = "posterior mean",linewidth = 3.0)
plt.plot(x_input,y, label = "ground truth",linewidth = 3.0)
plt.plot(x_input,sig + 10.0, label = "shannon ig", linewidth = 3.0)
m = pred1_mean["f(x)"]
s = np.diag(pred1_cov["S"])
plt.plot(x_input, 1000.0*s, label = "std", linewidth = 3.0)
plt.fill_between(x_input[:,0], m-3.0*np.sqrt(s), m+3.0*np.sqrt(s), alpha = 0.5, label = "95% confidence interval")
plt.scatter(points,values, label = "data",linewidth = 3.0)
plt.legend()

comp_mean_vec = np.array(m) + np.random.rand()
comp_var = np.zeros((len(m), len(m)))
np.fill_diagonal(comp_var,np.random.rand(len(comp_var)))
s = my_gp.posterior_probability(x_input, comp_mean_vec, comp_var)
print("s: ",s)


plt.savefig('plot.png')
plt.show()
exit()
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################

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
plt.legend()
plt.savefig('plot.png')
plt.show()

