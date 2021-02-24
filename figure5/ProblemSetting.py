

import numpy as np
#import matplotlib.pyplot as plt
#from numpy.random import choice
from time import time
#import json
#import copy
#import pickle 
import MethodsTrueData as Methods
import MethodsTrueData_M as Methods_M
#import MethodsFlipError as Methods
#from numpy.random import default_rng, SeedSequence
from Initialization import pz_theta_model, py_eq_z, Initialization, classnum, multiclass
from sklearn.linear_model import LogisticRegression
    
def SingleIteration(k, T, rglist, methodlist, xnum, thetanum):
    print(k)
    
    #Initialize
    np.random.seed(rglist[k])
    xspace, yspace, hyperlist = Initialization(xnum, thetanum)
#    errormat  = np.ones([20, T])
    ymatpile = np.zeros([len(xspace), T])
    dataidx = []
#    thetaridx = np.random.randint(len(thetalist))
    
#    thetalist = np.random.beta(alist, blist, size = (thetanum, bins))
    #Problem Setting
    
#    thetalist = np.zeros
    
    if multiclass:
        problem = Methods_M.Problem(xspace, yspace, hyperlist, pz_theta_model, py_eq_z, classnum = classnum, dataidx = dataidx)
#    else:
#        problem = Methods.Problem(xspace, yspace, thetalist, pz_theta_model, py_eq_z)
        
        
    if yspace is None:
        #that is synthetic data
        thetar = thetalist[thetaridx]
        bayesian_error = problem.BayesianError(thetar)
        for t in range(T):
            ymatpile[:, t] = problem.fr(xspace, thetar)
    else: 
        bins = np.max(xspace)+1
        count0 = np.zeros(bins)
        count1 = np.zeros(bins)
        
        for i, x in enumerate(xspace):
            if yspace[i] == 0:
                count0[x] +=1
            else:
                count1[x] += 1
                
        error_list = np.minimum(count0, count1)
        
#        bayesian_error = np.sum(error_list)/len(xspace)
        bayesian_error = 0           
    
    
    #Active Learning
    for i in methodlist:
#    for i in [0, 1, 2]:
        dataidx = []
#        pi_theta = np.ones(len(thetalist))
#        pi_theta /= pi_theta.sum()
        problem.Initialize(xspace, yspace, hyperlist, dataidx)
        if i == 0:
            str_label = 'random'
        elif i == 1:
            str_label = 'MES'
        elif i == 2:
            str_label = 'BALD'
        elif i == 3:
            str_label = 'ELR'
        elif i == 5:
            str_label = 'Weighted_MOCU'
        error_txt = open(str_label+'error.txt', 'a')
        data_txt = open(str_label+'data.txt', 'a')
        
        for t in range(T):
            start_time = time()
            if i == 0:
                xidx = np.random.randint(len(problem.xspace))
                xstar = problem.xspace[xidx]
            elif i == 1:
                xstar, _, xidx = problem.Selector(problem.UncertaintyWhole)
            elif i == 2:
                xstar, _, xidx = problem.Selector(problem.EntropyWhole)
            elif i == 3:
                xstar, _, xidx = problem.Selector(problem.MinIbrResidualWhole)
            elif i == 5:
                if multiclass:
                    xstar, _, xidx = problem.Selector(problem.SoftMOCUWhole(softtype = 3))
                else:
                    xstar, _, xidx = problem.Selector(problem.Weighted_MOCUWhole2)
            if yspace is None:
                ystar = ymatpile[xidx, t]
            else:
                problem.dataidx.append(xidx)
                ystar = problem.yspace[xidx]
            problem.Update(xstar, ystar, xidx)
            
#            errormat[i, t] = problem.ClassifierError(thetar, problem.pi_theta) - bayesian_error
            errortemp = problem.ClassifierError(xspace, yspace) - bayesian_error
            error_txt.write(str(errortemp)+'\t')
            data_txt.write(str([xstar, ystar])+'\t')
        error_txt.write('\n')
        data_txt.write('\n')
        error_txt.close()
        data_txt.close()
    
    
#%%
# setting:
#  xspace, yspace, thetalist, pz_theta_model, py_eq_z        
        

