from fvgp.fvgp import FVGP as gp
import numpy as np

a = np.load("us_topo.npy")
points = a[::8,0:2]
values = a[::8,2:3]
print("length of data set: ", len(points))
my_gp = gp(2,1,1,points,values, sparse = False)

#train(hyper_parameter_bounds, init_hyper_parameters=None, optimization_method='global', optimization_pop_size=20, optimization_tolerance=0.1, optimization_max_iter=120, dask_client=False) method of fvgp.fvgp.FVGP instance

bounds = np.array([[10,10000000],[1,10000],[1,10000]])
my_gp.train(bounds, optimization_method = 'global', optimization_max_iter = 20,optimization_pop_size = 20, dask_client = False)




