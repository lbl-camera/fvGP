import time
import numpy as np
import torch


###################################
######gp2Scale GPU kernels#########
###################################
def sparse_stat_kernel(x1,x2, hps):
    d = 0
    for i in range(len(x1[0])): d += abs(np.subtract.outer(x1[:,i],x2[:,i]))**2
    d = np.sqrt(d)
    d[d == 0.0] = 1e-16
    d[d > hps[1]] = hps[1]
    kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
    ((3.0*(d/hps[1])**2*np.log((d/hps[1])/(1+np.sqrt(1.0 - (d/hps[1])**2))))+\
    ((2.0*(d/hps[1])**2 + 1.0) * np.sqrt(1.0-(d/hps[1])**2)))
    return hps[0] * kernel

def sparse_stat_kernel_robust(x1,x2, hps):
    d = 0
    for i in range(len(x1[0])): d += abs(np.subtract.outer(x1[:,i],x2[:,i]))**2
    d = np.sqrt(d)
    d[d == 0.0] = 1e-16
    d[d > 1./hps[1]**2] = 1./hps[1]**2
    kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
    ((3.0*(d*hps[1]**2)**2*np.log((d*hps[1]**2)/(1+np.sqrt(1.0 - (d*hps[1]**2)**2))))+\
    ((2.0*(d*hps[1]**2)**2 + 1.0) * np.sqrt(1.0-(d*hps[1]**2)**2)))
    return (hps[0]**2) * kernel


def f_gpu(x,x0, radii, amplts, device):
    b1 = b_gpu(x, x0[0:3], radii[0], amplts[0], device)  ###x0[0] ... D-dim location of bump func 1
    b2 = b_gpu(x, x0[3:6], radii[1], amplts[1], device)  ###x0[1] ... D-dim location of bump func 2
    b3 = b_gpu(x, x0[6:9], radii[2], amplts[2], device)  ###x0[1] ... D-dim location of bump func 2
    b4 = b_gpu(x, x0[9:12],radii[3], amplts[3], device)  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4

def g_gpu(x,x0, radii, amplts,device):
    b1 = b_gpu(x, x0[0:3], radii[0], amplts[0], device)  ###x0[0] ... D-dim location of bump func 1
    b2 = b_gpu(x, x0[3:6], radii[1], amplts[1], device)  ###x0[1] ... D-dim location of bump func 2
    b3 = b_gpu(x, x0[6:9], radii[2], amplts[2], device)  ###x0[1] ... D-dim location of bump func 2
    b4 = b_gpu(x, x0[9:12],radii[3], amplts[3], device)  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4


def b_gpu(x,x0, r, ampl, device):
    """
    evaluates the bump function
    x ... a point (1d numpy array)
    x0 ... 1d numpy array of location of bump function
    returns the bump function b(x,x0) with radius r
    """
    x_new = x - x0
    d = torch.linalg.norm(x_new, axis = 1)
    a = torch.zeros(d.shape).to(device, dtype = torch.float32)
    a = 1.0 - (d**2/r**2)
    i = torch.where(a > 0.0)
    bump = torch.zeros(a.shape).to(device, dtype = torch.float32)
    e = torch.exp((-1.0/a[i])+1).to(device, dtype = torch.float32)
    bump[i] = ampl * e
    return bump


def get_distance_matrix_gpu(x1,x2,device):
    d = torch.zeros((len(x1),len(x2))).to(device, dtype = torch.float32)
    for i in range(x1.shape[1]):
        d += ((x1[:,i].reshape(-1, 1) - x2[:,i]))**2
    return torch.sqrt(d)

def sparse_stat_kernel_gpu(x1,x2, hps,device):
    d = get_distance_matrix_gpu(x1,x2,device)
    d[d == 0.0] = 1e-6
    d[d > hps] = hps

    d_hps = d/hps
    d_hpss= d_hps**2
    sq = torch.sqrt(1.0 - d_hpss)


    kernel = (torch.sqrt(torch.tensor(2.0))/(3.0*torch.sqrt(torch.tensor(3.141592653))))*\
    ((3.0*d_hpss * torch.log((d_hps)/(1+sq)))+\
    ((2.0*d_hpss + 1.0)*sq))
    return kernel


def ks_gpu(x1,x2,hps,cuda_device):
    k1 = torch.outer(f_gpu(x1,hps[0:12],hps[12:16],hps[16:20],cuda_device),
                     f_gpu(x2,hps[0:12],hps[12:16],hps[16:20],cuda_device)) + \
         torch.outer(g_gpu(x1,hps[20:32],hps[32:36],hps[36:40],cuda_device),
                     g_gpu(x2,hps[20:32],hps[32:36],hps[36:40],cuda_device))
    k2 = sparse_stat_kernel_gpu(x1,x2, hps[41],cuda_device)
    return k1 + hps[40]*k2

def kernel_gpu(x1,x2, hps):
    cuda_device = torch.device("cuda:0")
    x1_dev = torch.from_numpy(x1).to(cuda_device, dtype = torch.float32)
    x2_dev = torch.from_numpy(x2).to(cuda_device, dtype = torch.float32)
    hps_dev = torch.from_numpy(hps).to(cuda_device, dtype = torch.float32)
    ksparse = ks_gpu(x1_dev,x2_dev,hps_dev,cuda_device).cpu().numpy()
    return ksparse
###################################
######gp2Scale CPU kernels#########
###################################
def b_cpu(x,x0,r = 0.1, ampl = 1.0):
    """
    evaluates the bump function
    x ... a point (1d numpy array)
    x0 ... 1d numpy array of location of bump function
    returns the bump function b(x,x0) with radius r
    """
    x_new = x - x0
    d = np.linalg.norm(x_new, axis = 1)
    a = np.zeros(d.shape)
    a = 1.0 - (d**2/r**2)
    i = np.where(a > 0.0)
    bump = np.zeros(a.shape)
    bump[i] = ampl * np.exp((-1.0/a[i])+1)
    return bump

def f_cpu(x,x0, radii, amplts):
    b1 = b_cpu(x, x0[0:3],r = radii[0], ampl = amplts[0])  ###x0[0] ... D-dim location of bump func 1
    b2 = b_cpu(x, x0[3:6],r = radii[1], ampl = amplts[1])  ###x0[1] ... D-dim location of bump func 2
    b3 = b_cpu(x, x0[6:9],r = radii[2], ampl = amplts[2])  ###x0[1] ... D-dim location of bump func 2
    b4 = b_cpu(x, x0[9:12],r = radii[3], ampl = amplts[3])  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4

def g_cpu(x,x0, radii, amplts):
    b1 = b_cpu(x, x0[0:3],r = radii[0], ampl = amplts[0])  ###x0[0] ... D-dim location of bump func 1
    b2 = b_cpu(x, x0[3:6],r = radii[1], ampl = amplts[1])  ###x0[1] ... D-dim location of bump func 2
    b3 = b_cpu(x, x0[6:9],r = radii[2], ampl = amplts[2])  ###x0[1] ... D-dim location of bump func 2
    b4 = b_cpu(x, x0[9:12],r = radii[3], ampl = amplts[3])  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4

def sparse_stat_kernel_cpu(x1,x2, hps):
    d = 0
    for i in range(len(x1[0])): d += np.subtract.outer(x1[:,i],x2[:,i])**2
    d = np.sqrt(d)
    d[d == 0.0] = 1e-6
    d[d > hps] = hps
    kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
    ((3.0*(d/hps)**2*np.log((d/hps)/(1+np.sqrt(1.0 - (d/hps)**2))))+\
    ((2.0*(d/hps)**2+1.0)*np.sqrt(1.0-(d/hps)**2)))
    return kernel


def kernel_cpu(x1,x2, hps):
    k = np.outer(f_cpu(x1,hps[0:12],hps[12:16],hps[16:20]),
                 f_cpu(x2,hps[0:12],hps[12:16],hps[16:20])) + \
                 np.outer(g_cpu(x1,hps[20:32],hps[32:36],hps[36:40]),
                 g_cpu(x2,hps[20:32],hps[32:36],hps[36:40]))
    return k + hps[40] * sparse_stat_kernel_cpu(x1,x2, hps[41])


############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################


