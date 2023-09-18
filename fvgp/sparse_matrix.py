
import scipy.sparse as sparse
import time
import numpy as np
class gp2ScaleSparseMatrix:
    def __init__(self,n):
        self.n = n
        self.V = sparse.coo_matrix((n,n))
        self.K = sparse.coo_matrix((n,n))
        self.st = time.time()

    def get_result(self):
        return self. K


    def reset_prior(self):
        self.V = sparse.coo_matrix((self.n,self.n))
        self.K = sparse.coo_matrix((self.n,self.n))
        return 0

    def insert_many(self, list_of_3_tuples):
        l = list_of_3_tuples
        bg = self.K
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
        self.K = res
        print("    Size of the current covariance matrix: ", self.K.count_nonzero(), flush = True)
        return 0

    def get_future_results(self, futures, info = False):
        res = []
        info = False
        for future in futures:
            SparseCov_sub, ranges, ketime, worker = future.result()
            if info: print("Collected Future ", future.key, " has finished its work in", ketime," seconds. time stamp: ",time.time() - self.st)
            res.append((SparseCov_sub,ranges[0],ranges[1]))

        self.insert_many(res)
        return 0

    """
    def compute_LU_KV(self):
        A = self.KV.tocsc()
        A_new = A.__class__(A.shape)
        A_new.data = A.data
        A_new.indptr = np.array(A.indptr, copy=False, dtype=np.intc)
        A_new.indices = np.array(A.indices, copy=False, dtype=np.intc)
        self.LU = splu(A_new, diag_pivot_thresh = 0.01, options = dict(SymmetricMode = True), permc_spec = "MMD_AT_PLUS_A")
        return True


    def solveKV(self,x):
        success = True
        try: r = self.LU.solve(x)
        except: success = False
        if success is False:
            try:
                r,info = sparse.linalg.cg(self.KV,x)
                if info == 0: success = True
            except: pass
        if success is False:
            try:
                r, info = sparse.linalg.cgs(self.KV,x)
                if info == 0: success = True
            except: pass
        if success is False:
            try:
                r,info = sparse.linalg.minres(self.KV,x)
                success = True
            except: raise Exception("No solve method was successful. EXIT")
        return r

    def logdetKV(self):
        success = True
        try:
            upper_diag = abs(self.LU.U.diagonal())
            r = np.sum(np.log(upper_diag))
        except: success = False
        if success is False:
            try: r = self.random_logdet(self.KV)
            except: raise Exception("No logdet() method was successful, EXIT")
        return r

    def random_logdet(self, A,eps = 0.50, delta = 0.01,m = 10):
        #from: https://www.boutsidis.org/Boutsidis_LAA2017.pdf
        A = sparse.csc_matrix(A)
        N = A.shape[0]
        alpha = 7.0 * sparse.linalg.eigsh(A,k=1,tol=0.0001, return_eigenvectors=False)[0]
        A.data = A.data / alpha
        diag = sparse.eye(N, format="csc")
        C = sparse.csc_matrix(diag - A)
        p = math.ceil(20.0 * np.log(2./delta)/eps**2)
        gamma = np.zeros((p,m))
        for i in range(p):
            g = np.random.normal(0, 1., size = N)
            v = C @ g
            gamma[i,0] = g.T @ v
            for k in range(1,m):
                v = C @ v
                gamma[i,k] = g.T @ v
        s = np.sum(gamma, axis = 0) / float(p)
        s = s / np.arange(1,len(s)+1)
        return N * np.log(alpha) - np.sum(s)
    """
    def traceKXX(self,X): # pragma: no cover
        res = np.empty(X.shape)
        for i in range(X.shape[1]): res[:,i] = self.solve(X[:,i])
        tr = np.sum(X * res)
        return tr
