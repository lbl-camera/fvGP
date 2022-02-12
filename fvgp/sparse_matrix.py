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
        self.thread_blocked = False
        self.idle_workers = set(workers)
        self.future_worker_assignments = {}

    def get_result(self):
        return self.sparse_covariance

    def thread_is_blocked(self):
        return self.thread_blocked

    def get_idle_worker(self):
        return self.idle_workers.pop()

    def get_idle_workers(self):
        return self.idle_workers

    def free_worker(self, future_key):
        self.idle_workers.add(self.future_worker_assignments[future_key])

    def free_workers(self, futures):
        for future in futures:
            if future.status == "finished": self.free_worker(future.key)

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
        self.thread_blocked = True
        for entry in list_of_3_tuples:
            res = self.insert(entry[0],entry[1],entry[2])
        self.thread_blocked = False
        return res

    def get_future_results(self,futures):
        self.thread_blocked = True
        res = []
        for future in futures:
            SparseCov_sub, ranges,ketime, worker = future.result()
            res.append((SparseCov_sub, ranges[0], ranges[1]))
            del self.future_worker_assignments[future.key]
        self.insert_many(res)
        self.thread_blocked = False
        return 0

    def assign_future_2_worker(self, future_key, worker_address):
        self.future_worker_assignments[future_key] = worker_address

