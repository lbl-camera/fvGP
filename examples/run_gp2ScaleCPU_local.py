from dask.distributed import Client
import socket
import time
import numpy
from fvgp import gp
import numpy as np
import time
from fvgp.gp2Scale import gp2Scale
import argparse
import datetime
import time
import sys
from dask.distributed import performance_report
from fvgp.advanced_kernels import kernel_cpu


def main():
    client = Client()
    input_dim = 3
    N = 10000
    x_data = np.random.rand(N,input_dim)
    y_data = np.sin(np.linalg.norm(x_data,axis = 1) * 5.0)
    hps_n = 42

    hps_bounds = np.array([
                            [0.,1.],   ##pos bump 1 f comp 1
                            [0.,1.],    ##pos bump 1 f comp 2
                            [0.,1.],  ##pos bump 1 f comp 3
                            #
                            [0.,1.],   ##pos bump 2 f
                            [0.,1.],    ##pos bump 2 f
                            [0.,1.],    ##pos bump 2 f
                            #
                            [0.,1.],   ##pos bump 3 f
                            [0.,1.],    ##pos bump 3 f
                            [0.,1.],    ##pos bump 3 f
                            #
                            [0.,1.],   ##pos bump 4 f
                            [0.,1.],    ##pos bump 4 f
                            [0.,1.],    ##pos bump 4 f
                            #
                            [0.01,0.1],    ##radius bump 1 f
                            [0.01,0.1],    ##...2
                            [0.01,0.1],    ##...3
                            [0.01,0.1],    ##...4
                            [0.1,1.],    ##ampl bump 1 f
                            [0.1,1.],    ##...2
                            [0.1,1.],    ##...3
                            [0.1,1.],    ##...4
                            #
                            [0.,1.],    ##pos bump 1 g comp 1
                            [0.,1.],     ##pos bump 1 g comp 2
                            [0.,1.],   ##pos bump 1 g comp 3
                            #
                            [0.,1.],    ##pos bump 2 g comp 1
                            [0.,1.],     ##pos bump 2 g comp 2
                            [0.,1.],   ##pos bump 2 g comp 3
                            #
                            [0.,1.],    ##pos bump 3 g comp 1
                            [0.,1.],     ##pos bump 3 g comp 2
                            [0.,1.],   ##pos bump 3 g comp 3
                            #
                            [0.,1.],    ##pos bump 4 g comp 1
                            [0.,1.],     ##pos bump 4 g comp 2
                            [0.,1.],   ##pos bump 4 g comp 3
                            #
                            [0.01,0.1],    ##radius bump 1 g
                            [0.01,0.1],    ##...2
                            [0.01,0.1],    ##...3
                            [0.01,0.1],    ##...4
                            [0.1,1.],    ##ampl bump 1 g
                            [0.1,1.],    ##...2
                            [0.1,1.],    ##...3
                            [0.1,1.],    ##...4
                            [0.1,10.],    ##signal var of stat kernel
                            [0.001,0.02]     ##length scale for stat kernel
                            ])


    init_hps = np.random.uniform(size = len(hps_bounds), low = hps_bounds[:,0], high = hps_bounds[:,1])

    print(init_hps)
    print(hps_bounds)
    print("INITIALIZED")
    st = time.time()

    my_gp = gp2Scale(input_dim, x_data, y_data, init_hps, 1000,
                        gp_kernel_function = kernel_cpu, info = True,
                        covariance_dask_client = client)
    print("initialization done after: ",time.time() - st," seconds")
    print("===============")
    print("Log Likelihood: ", my_gp.log_likelihood(my_gp.hyperparameters))
    print("all done after: ",time.time() - st," seconds")

    my_gp.train(hps_bounds, max_iter = 5, init_hyperparameters = init_hps)


if __name__ == '__main__':
    main()

