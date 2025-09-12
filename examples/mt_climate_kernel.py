import numpy as np
import time
from scipy.interpolate import griddata
import scipy.sparse as sparse
from fvgp import GP
import gc



#length scales
def lambda_func(x, hps):
    #returns a scalar at each x
    res = np.zeros((len(x)))
    return res


#anisotropy angle
def gamma_func(x, hps):
    #returns a scalar at each x
    res = np.zeros((len(x))) + hps[0]
    return res


#anisotropy axis
def gamma_func(x, hps):
    #has to return a 3-vector at each x
    res = np.zeros((len(x))) + hps[0]
    return res



#signal variance
def sigma_func(x, hps):
    #returns a scalar at each x
    res = np.zeros((len(x))) + hps[0]
    return res



#######################################
#######################################
#######################################
#######################################
#######################################

def kernel(x1,x2,hps):
    t1_start = time.perf_counter()
    k = core_kernel(x1,x2, hps)
    t1_stop = time.perf_counter()
    print("Elapsed kernel time:", t1_stop - t1_start, flush = True)
    return k

def _get_distance_matrix_gpu(x1, x2, hps):  # pragma: no cover
    d = np.zeros((len(x1), len(x2)))
    for i in range(2):
        d += ((x1[:, i].reshape(-1, 1) - x2[:, i]) / hps[i]) ** 2
    return np.sqrt(d)

def _get_an_distance_matrix_gpu(x1, x2, hps):  # pragma: no cover
    d = np.zeros((len(x1), len(x2)))
    for i in range(2):
        d += ((x1[:, i].reshape(-1, 1) - x2[:, i]) / hps[i]) ** 2
    return np.sqrt(d)


def Lambda(x, hps):
    l = lambda_func(x, hps)
    return l

def gamma(x, hps):
    ee = gamma_func(x, hps)
    return ee

def Q(x1,x2,outer_sum):
    ost = outer_sum/2.0
    M = np.linalg.inv(ost)
    diff_matrix = np.zeros((len(x1),len(x2),2))
    for i in range(3): diff_matrix[:,:,i] = (x1[:, i].reshape(-1, 1) - x2[:, i])
    Qres = np.einsum('ijk,ijkl,ijl->ij', diff_matrix,M,diff_matrix)
    return Qres


def S(x, gamma_hps, lambda_hps1, lambda_hps2):
    GG = G(x, gamma_hps)
    return GG @ L(x, lambda_hps1, lambda_hps2) @ GG.permute(0,2,1)

def L(x, hps1, hps2):
    res = np.zeros((len(x),len(x[0]),len(x[0])))
    res[:,0,0] = Lambda(x, hps1)
    res[:,1,1] = Lambda(x, hps2)
    res[:,2,2] = Lambda(x, hps3)
    return res


def rotation_matrix(axis, theta):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    axis: 3-element array-like (must be a unit vector)
    theta: angle in radians
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    ct = np.cos(theta)
    st = np.sin(theta)
    vt = 1 - ct
    return np.array([
        [ct + ux*ux*vt, ux*uy*vt - uz*st, ux*uz*vt + uy*st],
        [uy*ux*vt + uz*st, ct + uy*uy*vt, uy*uz*vt - ux*st],
        [uz*ux*vt - uy*st, uz*uy*vt + ux*st, ct + uz*uz*vt]
    ])


def G(x, hps):
    res = np.zeros((len(x),len(x[0]),len(x[0])))    
    res[:,0,0] = np.cos(gamma(x, hps))
    res[:,0,1] =-np.sin(gamma(x, hps))
    res[:,1,0] = np.sin(gamma(x, hps))
    res[:,1,1] = np.cos(gamma(x, hps))
    axis = 
    res[i] = rotation_matrix(axis_func(x,hps), gamma_func(x, hps))
    return res



def outer_sum_S(S1,S2):
    ###S1,S2 \in R^(len(x),2,2)
    S1list = [S1 for i in range(len(S2))]
    S2list = [S2 for i in range(len(S1))]
    res = (np.stack(S1list, dim=0) + np.stack(S2list, dim=0).permute(1,0,2,3)).permute(1,0,2,3)
    del S1list
    del S2list
    gc.collect()
    return res


def outer(S1,S2):
    return np.einsum('ijk,ljk->iljk',S1,S2)



def matern_kernel_diffGPU(distance):
    kernel = (1.0 + ((np.sqrt(3.0) * distance))) * np.exp(-(np.sqrt(3.0) * distance))
    return kernel


def exp_kernel_diffGPU(distance):
    kernel = np.exp(-distance)
    return kernel


def core_kernel(x1,x2,hps):
    #The kernel follows the mathematical definition of a kernel. This
    #means there is no limit to the variety of kernels you can define.
    #st = time.time()
    
    lambda1_hps = hps[0:2]
    lambda2_hps = hps[2:4]
    gamma_hps  = hps[4:5]
    hps_signal = hps[5:7]


    S1 = S(x1,gamma_hps,lambda1_hps,lambda2_hps)
    S2 = S(x2,gamma_hps,lambda1_hps,lambda2_hps)

    signal_variance1 = sigma_func(x1, hps_signal)
    signal_variance2 = sigma_func(x2, hps_signal)


    A = np.outer(signal_variance1,signal_variance2)
    outer_sum = outer_sum_S(S1,S2)
    
    det1 = np.linalg.det(S1)**(1/4)
    det2 = np.linalg.det(S2)**(1/4)
    det3 = np.linalg.det(outer_sum/2.0)
    B = np.outer(det1,det2)/np.sqrt(det3)
    C = wendland_q(np.sqrt(Q(x1,x2,outer_sum)))

    k = A * B * C
    del A
    del B
    del C
    del S1
    del S2
    gc.collect()
    return k
