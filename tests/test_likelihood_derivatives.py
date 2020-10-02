import matplotlib.pyplot as plt
import numpy as np
from fvgp.fvgp import FVGP
import sys
import time

def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    return hps[0] * obj.matern_kernel_diff1(d,1)

def func(points):
    return 3.0*points + 0.3*np.sin(10.0*points)



def main():

    N = 100 ### how many training points
    points = np.empty((N,1))
    points[:,0] = np.linspace(0,2,N)+ np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
    ####change here for multi-task case
    values = func(points)
    variances = np.ones((values.shape)) * 0.1
    #########################################
    my_gp = FVGP(1,1,1,points,values,variances = variances,gp_kernel_function = stationary_kernel, compute_device = "cpu")


    training_method = 'global'

    my_gp.train([[100.0,200.0],[5.0,10.0]],
            init_hyperparameters = [10.0,10.0],
            optimization_method = training_method,
            optimization_pop_size = 20,
            optimization_tolerance = 0.0001,
            optimization_max_iter = 200,
            dask_client = False)
    
    print(my_gp.log_likelihood(my_gp.hyperparameters))
    x = np.linspace(100,105,100)
    y = np.linspace(5,10,100)
    dx = 5.0/100.0
    dy = 5.0/100.0
    X,Y = np.meshgrid(x,y)
    L = np.empty((X.shape))
    gx = np.empty((X.shape))
    gy = np.empty((X.shape))
    t = time.time()
    for i in range(len(x)):
        print("progress: ", i, " time passed: ", time.time()-t)
        for j in range(len(y)):
            L[i,j] = my_gp.log_likelihood(np.array([X[i,j],Y[i,j]]))
            g = my_gp.log_likelihood_gradient(np.array([X[i,j],Y[i,j]]))
            gx[i,j]= g[0]
            gy[i,j]= g[1]

    plt.figure(0)
    a = plt.pcolormesh(X,Y,L)
    plt.title("marginal log-likelihood")
    plt.colorbar(a)
    gxfd,gyfd = np.gradient(L,dx,dy)

    plt.figure(1)
    a = plt.pcolormesh(X,Y,gxfd)
    plt.title("finite diff grad x")
    plt.colorbar(a)
    plt.figure(2)
    a = plt.pcolormesh(X,Y,gyfd)
    plt.title("finite diff grad y")
    plt.colorbar(a)
    plt.figure(3)
    a = plt.pcolormesh(X,Y,gx)
    plt.title("analytical grad x")
    plt.colorbar(a)
    plt.figure(4)
    a = plt.pcolormesh(X,Y,gy)
    plt.title("analytical grad y")
    plt.colorbar(a)

    plt.show()



if __name__ == "__main__":
    main()

