import numpy as np
import matplotlib.pyplot as plt

def out_of_bounds(x,bounds):
    for i in range(len(x)):
        if x[i] < bounds[i,0] or x[i] > bounds[i,1]:
            return True
    return False


def mcmc(func,bounds, start = None, l = None, max_iter = 100):
    ######IMPORTANT: This is for minimization!!!!
    x = []
    f = []
    if start is None: x.append(np.random.uniform(low = bounds[:,0],high = bounds[:,1],size = len(bounds)))
    else: x.append(start)
    bc = False
    if l == None: l = np.diag(np.abs(np.subtract(bounds[:,0],bounds[:,1]))/10.0)
    counter = 0
    new_func = 1e16
    run = True
    while run:
        oob = True
        while oob:
            x_new = np.random.multivariate_normal(x[counter], l)
            oob = out_of_bounds(x_new,bounds)
        old_func = new_func
        new_func = func(x_new) ####call function
        max_f = max(abs(new_func),abs(old_func))
        min_f = min(abs(new_func),abs(old_func))
        #new_func = (new_func - min_f) + 0.0001
        #old_func = (old_func - min_f) + 0.0001
        acceptance_prob = 1./(1. + np.exp(- 20.0 * (old_func-new_func)/min_f))
        #xx = np.linspace(-20000,20000,100)
        #y = 1./(1. + np.exp(- 20.0*xx/min_f))
        #plt.plot(xx,y)
        #plt.show()
        #print("old: ", old_func," new: ", new_func,old_func-new_func,max_f)
        #print("old: ", old_func," new: ", new_func," acceptance prob: ", acceptance_prob)
        u = np.random.rand()
        if u <= acceptance_prob: x.append(x_new);f.append(func(x_new))
        else: x.append(x[counter]);f.append(func(x[counter]))
        print("mcmc f(x): ",f[-1])
        counter += 1
        #if len(f) > 200 and np.sqrt(np.var(f[-100:])) < np.mean(f[-100])/1000.0: success = True;break
        #if len(f) > 10000: success = False; break
        if counter >= max_iter: run = False
    #plt.plot(f)
    #plt.show()
    arg_min = np.argmin(f)
    print("mcmc res: ",f[arg_min]," at ",x[arg_min])
    return {"f(x)": f[arg_min],"x":x[arg_min],"success":success,"distribution": f}

