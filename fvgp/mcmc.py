import numpy as np
import matplotlib.pyplot as plt

def out_of_bounds(x,bounds):
    for i in range(len(x)):
        if x[i] < bounds[i,0] or x[i] > bounds[i,1]:
            return True
    return False


def mcmc(func,bounds, l = None):
    ######IMPORTANT: This is for minimization!!!!
    x = []
    f = []
    x.append(np.random.uniform(low = bounds[:,0],high = bounds[:,1],size = len(bounds)))
    bc = False
    if l == None: l = np.diag(np.abs(np.subtract(bounds[:,0],bounds[:,1]))/1000.0)
    counter = 0
    while bc == False:
        oob = True
        while oob:
            x_new = np.random.multivariate_normal(x[counter], l)
            oob = out_of_bounds(x_new,bounds)
        new_func = func(x_new)
        old_func = func(x[counter])
        m = min(new_func,old_func)
        new_func = (new_func - m) + 0.0001
        old_func = (old_func - m) + 0.0001
        a = old_func/new_func
        u = np.random.rand()
        if u <= a: x.append(x_new);f.append(func(x_new))
        else: x.append(x[counter]);f.append(func(x[counter]))
        counter += 1
        if len(f) > 200 and np.sqrt(np.var(f[-100:])) < np.mean(f[-100])/1000.0: success = True;break
        if len(f) > 10000: success = False; break
    #plt.plot(f)
    #plt.show()
    arg_min = np.argmin(f)
    return {"f(x)": f[arg_min],"x":x[arg_min],"success":success,"conv number": len(f)}

