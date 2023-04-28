import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def out_of_bounds(x,bounds):
    for i in range(len(x)):
        if x[i] < bounds[i,0] or x[i] > bounds[i,1]:
            return True
    return False


def project_onto_bounds(x,bounds):
    for i in range(len(x)):
        if x[i] < bounds[i,0]: x[i] = bounds[i,0]
        if x[i] > bounds[i,1]: x[i] = bounds[i,1]
    return x


def mcmc(func,bounds, x0 = None, distr = None, max_iter = 1000, ):
    x = []
    f = []
    if x0 is None: x.append(np.random.uniform(low = bounds[:,0],high = bounds[:,1],size = len(bounds)))
    else: x.append(x0)
    if distr is None: l = np.diag((np.abs(np.subtract(bounds[:,0],bounds[:,1])))/100.0)**2
    counter = 0
    current_func = func(x[0])
    f.append(current_func)
    run = True
    while run:
        x_proposal = np.random.multivariate_normal(x[-1], l)
        x_proposal = project_onto_bounds(x_proposal,bounds)
        proposal_func = func(x_proposal) ####call function
        acceptance_prob = proposal_func - current_func ##these are already logs
        uu = np.random.rand()
        u = np.log(uu)
        if u < acceptance_prob:
            x.append(x_proposal)
            f.append(proposal_func)
            current_func = proposal_func
        else:
            x.append(x[-1])
            f.append(current_func)
        logger.debug("mcmc f(x):{}",f[-1])
        counter += 1
        if counter >= max_iter: run = False
        if len(x)>201 and np.linalg.norm(np.mean(x[-100:],axis = 0)-np.mean(x[-200:-100],axis = 0)) < 0.01 * np.linalg.norm(np.mean(x[-100:],axis = 0)): run = False
    arg_max = np.argmax(f)
    x = np.asarray(x)
    logger.debug(f"mcmc res: {f[arg_max]} at {x[arg_max]}") 
    return {"f(x)": f[arg_max],"x":x[arg_max],"distribution": x, "distribution mean": np.mean(x,axis = 0),"distribution var": np.var(x,axis = 0)}
