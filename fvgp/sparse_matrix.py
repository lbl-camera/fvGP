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
        self.st = time.time()
        self.counter = 0

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
        #return res

    def imsert_many(self, list_of_3_tuples):
        l = list_of_3_tuples
        bg = self.sparse_covariance
        row_list = [bg.row]
        col_list = [bg.col]
        data = [bg.data]

        for entry in l:
            row_list.append(entry[0].row + entry[1])
            col_list.append(entry[0].col + entry[2])
            data.append(entry[0].data)
            if entry[1] != entry[2]:
                row_list.append(entry[0].col + entry[2])
                col_list.append(entry[0].row + entry[1])
                data.append(entry[0].data)

        res = sparse.coo_matrix((np.concatenate(data),(np.concatenate(row_list),np.concatenate(col_list))), shape = bg.shape)
        self.sparse_covariance = res
        return res


    def insert_many(self, list_of_3_tuples):
        for entry in list_of_3_tuples:
            res = self.insert(entry[0],entry[1],entry[2])
        return res

    def get_future_results(self, futures, info = False):
        res = []
        ##is gather better?
        #print("Starting loop at ",time.time() - self.st, "with ",len(futures)," to be collected", flush = True)
        for future in futures:
            SparseCov_sub, ranges, ketime, worker = future.result()
            #print("Collected Future ", future.key, " has finished its work in", ketime," seconds. time stamp: ",time.time() - self.st, flush = True)
            res.append((SparseCov_sub,ranges[0],ranges[1]))
            #print("I have read ", self.counter, "matrices", flush = True)
            self.counter += 1

        #print("Loop Done", time.time() - self.st, flush = True)
        self.imsert_many(res)
        #print("Done inserting", time.time() - self.st, flush = True)
        return 0
