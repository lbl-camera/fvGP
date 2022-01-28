import numpy as np
import matplotlib.pyplot as plt

def out_of_bounds(x,bounds):
    for i in range(len(x)):
        if x[i] < bounds[i,0] or x[i] > bounds[i,1]:
            return True
    return False


def mcmc(func,bounds, x0 = None, distr = None, max_iter = 100, ):
    ######IMPORTANT: This is for minimization!!!!
    x = []
    f = []
    if x0 is None: x.append(np.random.uniform(low = bounds[:,0],high = bounds[:,1],size = len(bounds)))
    else: x.append(x0)
    bc = False
    if distr == None: l = np.diag(np.abs(np.subtract(bounds[:,0],bounds[:,1]))/10.0)
    counter = 0
    new_func = 1e16
    run = True
    while run:
        x_new = np.random.multivariate_normal(x[counter], l)
        while out_of_bounds(x_new,bounds):
            x_new = np.random.multivariate_normal(x[counter], l)
        old_func = new_func
        new_func = func(x_new) ####call function
        max_f = max(abs(new_func),abs(old_func))
        min_f = min(abs(new_func),abs(old_func))
        acceptance_prob = 1./(1. + np.exp(- 20.0 * (old_func-new_func)/min_f))
        u = np.random.rand()
        if u <= acceptance_prob: x.append(x_new);f.append(func(x_new))
        else: x.append(x[counter]);f.append(func(x[counter]))
        print("mcmc res f(x): ",f[-1])
        counter += 1
        if counter >= max_iter: run = False
    arg_min = np.argmin(f)
    print("mcmc res: ",f[arg_min]," at ",x[arg_min])
    return {"f(x)": f[arg_min],"x":x[arg_min],"distribution": f}

