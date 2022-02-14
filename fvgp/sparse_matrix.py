import time
import scipy.sparse as sparse
import scipy.sparse.linalg as solve
import numpy as np
import dask.distributed as distributed
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
from scipy.optimize import differential_evolution
from scipy.sparse import coo_matrix
import gc
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spilu
from .mcmc import mcmc
import torch
from dask.distributed import Variable


class gp2ScaleSparseMatrix:
    def __init__(self,n,workers):
        self.sparse_covariance = sparse.coo_matrix((n,n))

    def get_result(self):
        return self.sparse_covariance

    def insert(self, sm, i ,j):
        bg = self.sparse_covariance
        if i != j:
            row = np.concatenate([bg.row,sm.row + i, sm.col + j])
            col = np.concatenate([bg.col,sm.col + j, sm.row + i])
            res = coo_matrix((np.concatenate([bg.data,sm.data,sm.data]),(row,col)), shape = bg.shape )
        else:
            row = np.concatenate([bg.row,sm.row + i])
            col = np.concatenate([bg.col,sm.col + j])
            res = coo_matrix((np.concatenate([bg.data,sm.data]),(row,col)), shape = bg.shape)
        self.sparse_covariance = res
        return res

    def insert_many(self, list_of_3_tuples):
        for entry in list_of_3_tuples:
            res = self.insert(entry[0],entry[1],entry[2])
        return res

    def get_future_results(self, futures, info = False):
        res = []
        for future in futures:
            SparseCov_sub, ranges, ketime, worker = future.result()
            if info: print("Future ", future.key, " has finished its work in", ketime," seconds.", flush = True)
            res.append((SparseCov_sub,ranges[0],ranges[1]))
        self.insert_many(res)
        return 0
