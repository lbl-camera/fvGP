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



def client_is_ready(ip, n_workers):
    ready = False
    print("checking the client", flush = True)
    counter = 0
    while ready is False:
        try:
            c = Client(ip)
            n = len(list(c.scheduler_info()["workers"]))
            if n >= n_workers:
                ready=True
                return ready
            else:
                print("only ",n," of desired",n_workers," workers available", flush = True)
        except: pass
        time.sleep(5)
        counter += 1
        if counter > 20: print("getting the client is taking a long time: ", counter * 5, "seconds", flush = True)



def normalize(v):
    v = v - np.min(v)
    v = v/np.max(v)
    return v



def main():
    print("inputs to the run script: ",sys.argv, flush = True)
    print("port: ", str(sys.argv[1]), flush = True)
    if client_is_ready(str(sys.argv[1]),int(sys.argv[2])/2):
        client = Client(str(sys.argv[1]))
        print("Client is ready", flush = True)
        print(datetime.datetime.now().isoformat())
        print("client received: ", client, flush = True)

    print("Everything is ready to call gp2Scale", flush = True)

    with performance_report(filename="dask-report.html"):
        input_dim = 3
        target_worker_number = int(sys.argv[2])



        #station_locations = np.load("station_coord.npy")
        #temperatures = np.load("data.npy")
        #N = len(station_locations) * len(temperatures)
        #x_data = np.zeros((N,3))
        #y_data = np.zeros((N))
        #count  = 0
        #for i in range(len(temperatures)):
        #    for j in range(len(temperatures[0])):
        #        x_data[count] = np.array([station_locations[j,0],station_locations[j,1],float(i)])
        #        y_data[count] = temperatures[i,j]
        #        count += 1

        #non_nan_indices = np.where(y_data == y_data)  ###nans in data
        #x_data = x_data[non_nan_indices]
        #y_data = y_data[non_nan_indices]
        #x_data = x_data[::10]  ##1000: about 50 000 points; 100: 500 000; 10: 5 million
        #y_data = y_data[::10]
        #x_data[:,0] = normalize(x_data[:,0])
        #x_data[:,1] = normalize(x_data[:,1])
        #x_data[:,2] = normalize(x_data[:,2])
        #print(np.min(x_data[:,0]),np.max(x_data[:,0]))
        #print(np.min(x_data[:,1]),np.max(x_data[:,1]))
        #print(np.min(x_data[:,2]),np.max(x_data[:,2]))
        #N = 3200000
        #N = 1000000

        x_data = np.random.rand(N,input_dim)
        y_data = np.sin(np.linalg.norm(x_data,axis = 1) * 5.0)

        #N = len(x_data)
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
        #print(client.get_versions())
        st = time.time()

        my_gp = gp2Scale(input_dim, x_data, y_data, init_hps, 10000, target_worker_number,
                         gp_kernel_function = kernel, ram_limit = 400e9, info = True,
                         covariance_dask_client = client)
        print("initialization done after: ",time.time() - st," seconds")
        print("===============")
        print("Log Likelihood: ", my_gp.log_likelihood(my_gp.hyperparameters, recompute_xK = False))
        print("all done after: ",time.time() - st," seconds")

        #my_gp.train(hps_bounds, max_iter = 10, init_hyperparameters = init_hps)


if __name__ == '__main__':
    main()

