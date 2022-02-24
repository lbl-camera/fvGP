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
        self.n = n
        self.sparse_covariance = sparse.coo_matrix((n,n))
        self.st = time.time()
        self.counter = 0

    def get_result(self):
        return self.sparse_covariance

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

        rows = np.concatenate(row_list).astype(int)
        columns = np.concatenate(col_list).astype(int)

        res = sparse.coo_matrix((np.concatenate(data),(rows,columns)), shape = bg.shape)
        self.sparse_covariance = res
        return res

    def get_future_results(self, futures, info = False):
        res = []
        info = True
        if info: print("Starting loop at ",time.time() - self.st, "with ",len(futures)," to be collected")
        for future in futures:
            SparseCov_sub, ranges, ketime, worker = future.result()
            if info: print("Collected Future ", future.key, " has finished its work in", ketime," seconds. time stamp: ",time.time() - self.st)
            res.append((SparseCov_sub,ranges[0],ranges[1]))
            if info: print("I have read ", self.counter, "matrices")
            self.counter += 1

        if info: print("Loop Done", time.time() - self.st)
        self.imsert_many(res)
        if info: print("Done inserting", time.time() - self.st)
        return 0

    def add_to_diag(self,vector):
        diag = sparse.eye(self.n, format="coo") ##make variance
        diag.setdiag(vector) ##make variance
        self.sparse_covariance = self.sparse_covariance + diag  ##add variance
        return 0


    def compute_LU(self):
        A = self.sparse_covariance
        self.LU = splu(A.tocsc())
        return 0

    def solve(self,x):
        return self.LU.solve(x)

    def logdet(self):
        upper_diag = abs(self.LU.U.diagonal())
        res = np.sum(np.log(upper_diag))
        return res



