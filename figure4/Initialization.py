
from scipy.special import expit
import numpy as np
from numpy.linalg import norm

def pz_theta_model(x, theta_r):
    
    xnum = x.shape[0]
    
    Sigma = theta_r
#    m1 = np.array([0, 0])
#    m2 = np.array([0, 1])
#    m3 = np.array([1, 0])
    m = np.array([[0, 0], [0, 1], [1, 0]])
    
    pzx_theta = np.ones((classnum, xnum))
    
#    if isinstance(x, np.ndarray):
#        x = np.reshape(x, (1, -1))
    
    for k in range(classnum):
        pzx_theta[k, :] = np.exp(-norm(x-m[k], axis = 1)**2/Sigma[k])
#    pz0x_theta = np.exp(-norm(x-m1, axis = 1)**2/Sigma1)
#    pz1x_theta = np.exp(-norm(x-m2, axis = 1)**2/Sigma2)
#    pz2x_theta = np.exp(-norm(x-m3, axis = 1)**2/Sigma3)    
    
#    pzx_theta = np.concatenate([pz0x_theta, pz1x_theta, pz2x_theta], axis = 0)# size should be 2*N
    pzx_theta = pzx_theta/np.sum(pzx_theta, axis = 0)
    return pzx_theta

def Initialization(xnum = 100, thetanum = 100):
    xspace = np.random.uniform(-1, 2, (xnum, 2))
#    xspace = np.reshape(xspace, (-1, 1))
    yspace = None # None means the data is generated from Bayesian model
    sigma1list = np.random.uniform(3, 10, (thetanum, 1))
#    w2list = np.random.uniform(-0.02, 0.02, (thetanum, 1))
    sigma2list = np.random.uniform(3, 10, (thetanum, 1))
    sigma3list = np.random.uniform(3, 10, (thetanum, 1))
    thetalist = np.concatenate((sigma1list,sigma2list, sigma3list), axis = 1)
    return xspace, yspace, thetalist

classnum = 3
multiclass = True
py_eq_z = None