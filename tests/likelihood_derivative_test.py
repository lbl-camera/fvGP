import matplotlib.pyplot as plt
import numpy as np
from fvgp.fvgp import FVGP
import sys
import time

def stationary_kernel(x1,x2,hps, obj = None):
    d = abs(np.subtract.outer(x1[:,0],x2[:,0])/hps[1])
    #return hps[0] * obj.matern_kernel_diff1(d,1)
    #return hps[0] * obj.matern_kernel_diff2(d,1)
    return hps[0] * obj.exponential_kernel(d,1)

def func(points):
    return 3.0*points + 0.3*np.sin(10.0*points)



def main():

    N = 20 ### how many training points
    points = np.empty((N,1))
    points[:,0] = np.linspace(0,2,N)+ np.random.uniform(low = -0.05, high = 0.05, size = points[:,0].shape)
    ####change here for multi-task case
    values = func(points)
    variances = np.ones((values.shape)) * 0.2
    #########################################
    my_gp = FVGP(1,1,1,points,values,np.array([10,10]),variances = variances,gp_kernel_function = stationary_kernel, compute_device = "cpu")


    training_method = 'global'

    my_gp.train([[100.0,200.0],[5.0,10.0]],
            init_hyperparameters = [10.0,10.0],
            optimization_method = training_method,
            optimization_pop_size = 20,
            optimization_tolerance = 0.0001,
            optimization_max_iter = 200,
            dask_client = False)
    
    print(my_gp.log_likelihood(my_gp.hyperparameters))
    res = 100
    x = np.linspace(100,110,res)
    y = np.linspace(5,10,res)
    dx = 10.0/float(res)
    dy = 5.0/float(res)
    X,Y = np.meshgrid(x,y)
    L = np.empty((X.shape))
    gx = np.empty((X.shape))
    gy = np.empty((X.shape))
    h21a = np.empty((X.shape))
    t = time.time()

    for i in range(len(x)):
        print("progress: ", ((i+0.001)/float(res))*100.0, " time passed: ", time.time()-t)
        for j in range(len(y)):
            L[i,j] = my_gp.log_likelihood(np.array([X[i,j],Y[i,j]]))
            g = my_gp.log_likelihood_gradient(np.array([X[i,j],Y[i,j]]))
            h = my_gp.log_likelihood_hessian(np.array([X[i,j],Y[i,j]]))
            gx[i,j]= g[0]
            gy[i,j]= g[1]
            h21a[i,j] = h[1,0]


    
    #a = plt.pcolormesh(X,Y,L)
    #plt.title("marginal log-likelihood")
    #plt.colorbar(a)
    
    fig,axs = plt.subplots(3,2)
    gxfd,gyfd = np.gradient(L,dx,dy)

    a = axs[0,0].pcolormesh(X,Y,gxfd)
    axs[0,0].set_title("finite diff grad x")
    fig.colorbar(a, ax = axs[0,0])

    a = axs[1,0].pcolormesh(X,Y,gyfd)
    axs[1,0].set_title("finite diff grad y")
    fig.colorbar(a, ax = axs[1,0])

    a = axs[1,1].pcolormesh(X,Y,gx)
    axs[1,1].set_title("analytical grad x")
    fig.colorbar(a, ax = axs[1,1])
    
    a = axs[0,1].pcolormesh(X,Y,gy)
    axs[0,1].set_title("analytical grad y")
    fig.colorbar(a, ax = axs[0,1])

    a = axs[2,0].pcolormesh(X,Y,h21a)
    axs[2,0].set_title("analytical H")
    fig.colorbar(a,ax = axs[2,0])


    h11n,h12n = np.gradient(gxfd,dx,dy)
    a = axs[2,1].pcolormesh(X,Y,h12n)
    axs[2,1].set_title("numerical H")
    fig.colorbar(a,ax = axs[2,1])

    plt.show()


if __name__ == "__main__":
    main()

