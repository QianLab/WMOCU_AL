
from scipy.special import expit
import numpy as np

def pz_theta_model(x, theta_r):
    # x can be np.array of size 2 or a list of two np.array x1 and x2
    w = theta_r[0:-1]
    b = theta_r[-1]
    pzx_theta = expit(x.dot(w)+b)
    return pzx_theta

def py_eq_z(x):
    pt = 0
    l = 0.5
    if x.ndim == 1:
        if np.abs(x[0])<l and np.abs(x[1])<l:
            return 1-pt
        return 1
    pyeqz = 1 - np.logical_and(np.abs(x[:, 0]) <l, np.abs(x[:, 1])<l)*pt
    return pyeqz

def Initialization(xnum = 100, thetanum = 100):
    xspace = np.random.uniform(-4, 4, (xnum, 2))
    yspace = None # None means the data is generated from Bayesian model
    w1list = np.random.uniform(0.3, 0.8, (thetanum, 1))
    w2list = np.random.uniform(-0.02, 0.02, (thetanum, 1))
    blist = np.random.uniform(-0.25, 0.25, (thetanum, 1))
    thetalist = np.concatenate((w1list,w2list, blist), axis = 1)
    return xspace, yspace, thetalist

classnum = 2
multiclass = False