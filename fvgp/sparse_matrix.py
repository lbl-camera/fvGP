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
#from scikits import umfpack
#from scikits.umfpack import spsolve, splu


class gp2ScaleSparseMatrix:
    def __init__(self,n):
        self.n = n
        self.sparse_covariance = sparse.coo_matrix((n,n))
        self.st = time.time()

    def get_result(self):
        return self.sparse_covariance

    def reset_prior(self):
        self.sparse_covariance = sparse.coo_matrix((self.n,self.n))
        return 0

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
        return 0

    def get_future_results(self, futures, info = False):
        res = []
        info = False
        for future in futures:
            SparseCov_sub, ranges, ketime, worker = future.result()
            if info: print("Collected Future ", future.key, " has finished its work in", ketime," seconds. time stamp: ",time.time() - self.st)
            res.append((SparseCov_sub,ranges[0],ranges[1]))

        self.imsert_many(res)
        return 0

    def add_to_diag(self,vector):
        diag = sparse.eye(self.n, format="coo") ##make variance
        diag.setdiag(vector) ##make variance
        self.sparse_covariance = self.sparse_covariance + diag  ##add variance
        return 0


    def compute_LU(self):
        A = self.sparse_covariance.tocsc()
        A_new = A.__class__(A.shape)
        A_new.data = A.data
        A_new.indptr = np.array(A.indptr, copy=False, dtype=np.intc)
        A_new.indices = np.array(A.indices, copy=False, dtype=np.intc)
        self.LU = splu(A_new, diag_pivot_thresh = 0.01, options = dict(SymmetricMode = True), permc_spec = "MMD_AT_PLUS_A")
        return True


    def solve(self,x):
        success = True
        try: r = self.LU.solve(x)
        except: success = False
        if success is False:
            try:
                r,info = sparse.linalg.cg(self.sparse_covariance,x)
                success = True
            except: pass
        if success is False:
            try:
                r, info = sparse.linalg.cgs(self.sparse_covariance,x)
                success = True
            except: pass
        if success is False:
            try:
                r,info = sparse.linalg.minres(self.sparse_covariance,x)
                success = True
            except: raise Exception("No solve method was successful. EXIT")

        return r

    def logdet(self):
        success = True
        try:
            upper_diag = abs(self.LU.U.diagonal())
            r = np.sum(np.log(upper_diag))
        except: success = False
        if success is False:
            try: r = self.random_logdet(self.sparse_covariance)
            except: raise Exception("No logdet() method was successful, EXIT")
        return r

    def random_logdet(self, A,eps = 10.0,delta = 0.1,m = 100):
        #from: https://www.boutsidis.org/Boutsidis_LAA2017.pdf
        A = sparse.csc_matrix(A)
        N = A.shape[0]
        alpha = 7.0 * sparse.linalg.eigsh(A,k=1,tol=0.1, return_eigenvectors=False)[0]
        A.data = A.data / alpha
        diag = sparse.eye(N, format="csc")
        C = sparse.csc_matrix(diag - A)
        p = int(20.0 * np.log(2./delta)/eps**2)+1
        gamma = np.zeros((p,m))
        for i in range(p):
            print(i," of ",p)
            g = np.random.normal(0, 1., size = N)
            v = C @ g
            gamma[i,1] = g.T @ v
            for k in range(2,m):
                v = C @ v
                gamma[i,k] = g.T @ v
        s = np.sum(gamma, axis = 0) / float(p)
        s = s[1:] / np.arange(1,len(s))
        return N * np.log(alpha) - np.sum(s)

