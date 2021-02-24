

import numpy as np
import copy
from scipy.stats import entropy
from scipy.special import logsumexp
from numpy.random import choice
from scipy.special import softmax
#from scipy.stats import entropy
#import LinkedList as LL


#class LinkedList(object):
#    def __init__(self, head=None):
#        self.head = head
        


# remove xspace in each iteration 
class Problem():
    
    def __init__(self, xspace, yspace, hyperlist, pz_theta_model, py_eq_z, pi_theta = None, classnum = 2, dataidx = None):
        self.xspace = xspace
        self.yspace = yspace
        self.hyperlist = hyperlist
        self.PzGivenXTheta = pz_theta_model
        self.PYeqZ = py_eq_z#if PYeqZ is None, means that there is no flip error 
#        if pi_theta is None:
#            pi_theta = np.ones(len(thetalist))
#            pi_theta /= pi_theta.sum()
#        self.pi_theta = pi_theta
        
        #change pzmat_Theta
        self.cnum = classnum
        self.pzmat_Theta = self.PzGivenData(self.hyperlist)
        self.pymat_Theta = self.PyGivenData(self.hyperlist)
        self.thetanum  = 1000
        self.binnum = 16
        self.dataidx = dataidx
#        if self.PYeqZ is None:
#            self.pymat_Theta = self.pzmat_Theta
#        elif self.PYeqZ.__name__ == 'py_eq_z':
#            self.pymat_Theta = self.PYeqZ(self.xspace, self.pzmat_Theta)
#        elif self.PYeqZ.__name__ == 'py_eq_z_vari':
#            self.pymat_Theta = self.PyGivenData(self.pi_theta)
        

    def Initialize(self, xspace, yspace, hyperlist, dataidx = None):
        self.hyperlist = hyperlist
        self.xspace = xspace
        self.yspace = yspace
        self.dataidx = dataidx


#    def fr(self, x , thetar):
#        
#        
#                
#        pz = self.PzGivenXTheta(x, thetar)
#        if self.PYeqZ is None:
#            py = pz
#        elif self.PYeqZ.__name__ == 'py_eq_z':
#            py = self.PYeqZ(x, pz)
#        elif self.PYeqZ.__name__ == 'py_eq_z_vari':
#            py = self.PYeqZ(x, pz, thetar)
#            
#        if x is self.xspace:
#            ymat = np.zeros(len(self.xspace))
#            for i, xx in enumerate(self.xspace):
#                ymat[i] = choice(range(self.cnum), p=py[:, i] )
#        else:
#            ymat = choice(range(self.cnum), p = py)
#
#        return ymat


    def ParameterUpdate(self, x, y):#update this for error   #############################
        #Here the input of x can only be single input: x = np.array([x1[i][0],x2[j]])
        # the posterior distribution is deterministic here, we only need to update it
        # to probabilistic case    
        hyperlist2 = copy.copy(self.hyperlist) # hyperlist = [blist; alist] <=> p(y = 0|x), p(y = 1|x)
        y = int(y)
        x = np.reshape(x, (1, -1))
            
        hyperlist2[y, x]+=1
        
        return hyperlist2


    def PzGivenData(self, hyperlist):  ################closed form
        
        pTheta_in_bins = hyperlist/np.sum(hyperlist, axis = 0)
        
        pzmat = pTheta_in_bins[:, self.xspace]
        
#        pzmat = np.zeros([self.cnum, len(self.xspace)])
#        
#        for i in range(len(pi_theta)):
#            pzmat += self.PzGivenXTheta(self.xspace, self.thetalist[i])*pi_theta[i]
        return pzmat
    
    def PyGivenData(self, pi_theta):
        if self.PYeqZ is None:
            return self.pzmat_Theta
        if self.PYeqZ.__name__ == 'py_eq_z':
            return self.PYeqZ(self.xspace, self.pzmat_Theta)
        if self.PYeqZ.__name__ == 'py_eq_z_vari':
#            self.pymat_Theta = self.PyGivenData(self.pi_theta)
        #if PYeqZ.__name__ == 'py_eq_z_vari'
            pymat = np.zeros([self.cnum, len(self.xspace)])            
            for i in range(len(pi_theta)):
                pztemp = self.PzGivenXTheta(self.xspace, self.thetalist[i])
                pytemp = self.PYeqZ(self.xspace, pztemp, self.thetalist[i])
                pymat += pytemp*pi_theta[i]
            return pymat


    def ObcError(self, hyperlist):
        
        pzmat_Theta = self.PzGivenData(hyperlist)
        errormat = 1 - np.amax(pzmat_Theta, axis = 0)
        error = np.mean(errormat)#assume x is uniform distributed
        return error
    
    def MinIbrResidual(self, x, py_x):
        sumresidual = 0
        for i in range(self.cnum):
            p = py_x[i]
            y = i
#        for i in range(2):
#            if i == 0:
#                p = py_x
#                y = 1
#            else:
#                p = 1-py_x
#                y = 0
            hyperlist2 = self.ParameterUpdate(x, y)
            sumresidual += self.ObcError(hyperlist2)*p
        return -sumresidual
    
    def MinIbrResidualWhole(self):
        #the IbrResidual for the whole space
        utilitymat = np.zeros(len(self.xspace))
#        if self.PYeqZ is None:
#            pymat = self.pzmat_Theta
#        elif self.PYeqZ.__name__ == 'py_eq_z':
#            pymat = self.PYeqZ(self.xspace, self.pzmat_Theta)
#        elif self.PYeqZ.__name__ == 'py_eq_z_vari':
#            pymat = self.PyGivenData(self.pi_theta)
        pymat = self.pymat_Theta
        for i, x in enumerate(self.xspace):
            py_x = pymat[:, i]
            utilitymat[i] = self.MinIbrResidual(x, py_x)
        return self.ObcError(self.hyperlist)+utilitymat          
            
    
    def SMOCU(self, hyperlist, k, softtype):
#        smocu = np.zeros(len(self.xspace))
        pzmat = self.PzGivenData(hyperlist)
        if softtype == 1:
#            obc_correct = (pzmat*np.exp(pzmat*k) + (1-pzmat)*np.exp(k-pzmat*k))/(np.exp(pzmat*k)+np.exp(k-pzmat*k))
            obc_correct = np.sum(softmax(pzmat*k, axis = 0)*pzmat, axis=0)
            ############ softmax()
            smocu = np.mean( - obc_correct)
        elif softtype == 2:
#            pzmat_array = np.array([pzmat, 1-pzmat])
            obc_correct = logsumexp(k*pzmat, axis = 0)/k
            smocu = np.mean( - obc_correct)
        elif softtype == 3: #this is weighted-MOCU with weight 1-K
            bayesian_precision = np.zeros(len(self.xspace))
            thetalist = np.random.beta(hyperlist[1, :], hyperlist[0, :], size = (self.thetanum, self.binnum))
            pi_theta = np.ones(len(thetalist))
            pi_theta /= pi_theta.sum()
            for i, theta in enumerate(thetalist):
                bayesian_precision += np.amax(self.PzGivenXTheta(self.xspace, theta), axis=0 )*pi_theta[i]
            average_loss = bayesian_precision - np.amax(pzmat, axis=0) 
            weight = 1 - average_loss
            smocu = np.mean(weight*average_loss)
#        elif softtype == 4:
#            bayesian_precision = np.zeros(len(self.xspace))
#            for i, theta in enumerate(self.thetalist):
#                bayesian_precision += np.amax(self.PzGivenXTheta(self.xspace, theta), axis=0 )*pi_theta[i]
#            average_loss = bayesian_precision - np.amax(pzmat, axis=0) 
##            zhat = np.argmax(pzmat, axis = 0)
##            pzmat_r[zhat, range(self.xspace.shape[0])]
#            weight = np.exp(np.amax(pzmat, axis=0))/np.sum(np.exp(pzmat), axis = 0)
#            smocu = np.mean(weight*average_loss)
        return smocu
    
    def D_SMOCU(self, x, py_x, k, softtype):
#        x = self.xspace[xidx]
#        pz_x = self.pzmat_Theta[:, xidx]
#        if self.PYeqZ is None:
#            py_x = pz_x
#        elif self.PYeqZ.__name__ == 'py_eq_z':
#            py_x = self.PYeqZ(x, pz_x)
#        elif self.PYeqZ.__name__ == 'py_eq_z_vari':
#            py_x = self.pymat_Theta[xidx]
        smocu2 = 0
        for i in range(self.cnum):
            p = py_x[i]
            y = i
            hyperlist2 = self.ParameterUpdate(x, y)
            smocu2 += p*self.SMOCU(hyperlist2, k, softtype)
        return smocu2
        
    def SoftMOCU_K(self, k, softtype):##########################
#        smocu = self.SMOCU(self.pi_theta, k)
        utilitymat = np.zeros(len(self.xspace))
        utilitybin = np.zeros(self.binnum)
        pzTheta_in_bins = self.hyperlist/np.sum(self.hyperlist, axis = 0)
        for j in range(self.binnum):
            pz_x_bin = pzTheta_in_bins[:, j]
            utilitybin[j] = - self.D_SMOCU(j, pz_x_bin, k, softtype)
        for i, x in enumerate(self.xspace):
            utilitymat[i] = utilitybin[x]
#            utilitymat[i] = smocu - self.D_SMOCU(i, k)
#            pz_x = self.pzmat_Theta[:, i]
#            if self.PYeqZ is None:
#                py_x = pz_x
#            elif self.PYeqZ.__name__ == 'py_eq_z':
#                py_x = self.PYeqZ(x, pz_x)
#            elif self.PYeqZ.__name__ == 'py_eq_z_vari':
#                py_x = self.pymat_Theta[:, i]
#            utilitymat[i] = - self.D_SMOCU(x, py_x, k, softtype)
        return utilitymat
    
    def SoftMOCUWhole(self, k = 1, softtype = 1):######################
        return lambda: self.SoftMOCU_K(k, softtype)
    
#    def SMOCU2(self, pi_theta, k):
#        pzmat = self.PzGivenData(pi_theta)
##        pzmat1 = 1-pzmat
#        pzmat_array = np.array(pzmat, 1-pzmat)
#        obc_correct = logsumexp(k*pzmat, axis = 1)/k
#        smocu = np.mean(-obc_correct)
#        return smocu


    def EntropyWhole(self):
        entropymat = np.zeros(len(self.xspace))
#        self.pzmat_Theta = self.PzGivenData( self.pi_theta)
#        pymat = self.PzGivenData( self.pi_theta)
#        if self.PYeqZ is None:
#            pymat = self.pzmat_Theta
#        else:
#            pymat = self.PYeqZ(self.xspace, self.pzmat_Theta)
        pymat = self.pymat_Theta
        pytheta_entropy_mat = np.zeros(len(self.xspace))
#        posterior_entropy_mat2 = posterior_entropy_mat
        thetalist = np.random.beta(self.hyperlist[1, :], self.hyperlist[0, :], size = (self.thetanum, self.binnum))
        pi_theta = np.ones(len(thetalist))
        pi_theta /= pi_theta.sum()
        for i in range(len(thetalist)):
            theta = thetalist[i]
            pz_theta_mat = self.PzGivenXTheta(self.xspace, theta)
            if self.PYeqZ is None:
                py_theta_mat = pz_theta_mat
            elif self.PYeqZ.__name__ == 'py_eq_z':
                py_theta_mat = self.PYeqZ(self.xspace, pz_theta_mat)
            elif self.PYeqZ.__name__ == 'py_eq_z_vari':
                py_theta_mat = self.PYeqZ(self.xspace, pz_theta_mat, theta)
#                pz_theta_mat*self.PYeqZ(self.xspace) +\
#            (1-pz_theta_mat)*(1-self.PYeqZ(self.xspace))
#            posterior_entropy_mat += self.pi_theta[i]*bientropy(py_theta_mat)
            
            pytheta_entropy_mat += pi_theta[i]*entropy(py_theta_mat)##########################samples
#            posterior_entropy_mat += self.pi_theta[i]*entropy([py_theta_mat, 1-py_theta_mat])
#        entropymat = bientropy(self.pzmat_Theta) - posterior_entropy_mat
        entropymat = entropy(pymat) - pytheta_entropy_mat
        return entropymat
    
    
    def UncertaintyWhole(self):
#        pymat = self.PYeqZ(self.xspace)*self.pzmat_Theta+(1-self.pzmat_Theta)*(1-self.PYeqZ(self.xspace))
        objmat = entropy(self.pzmat_Theta)
        return objmat
#    def EntropyPoint(self, x, py_x):
#        bientropy = lambda x: -x*np.log(x)-(1-x)*np.log(1-x)
        

    def Selector(self, func):
        utilitymat = np.zeros(len(self.xspace))
        utilitymat = func()
        if self.yspace is not None:
            utilitymat[self.dataidx] = float('-Inf')
        max_index = np.argmax(utilitymat, axis = None)
        x = self.xspace[max_index]
        if self.yspace is not None:
            y = self.yspace[max_index]
        else:
            y = None
        
        return x, y, max_index
    
    
    def Update(self, xstar, ystar, xidx):
#        for i, pi in enumerate(self.pi_theta):
#            pz1_xtheta = self.PzGivenXTheta(xstar, self.thetalist[i])
#            if self.PYeqZ is None:
#                py1_xtheta = pz1_xtheta
#            else:
#                py1_xtheta = self.PYeqZ(xstar, pz1_xtheta)
#            
#            if ystar == 1:
#                py_xtheta = py1_xtheta
#            else:
#                py_xtheta = (1 - py1_xtheta)
#            self.pi_theta[i] *= py_xtheta
#            
#        self.pi_theta /= self.pi_theta.sum()
        self.hyperlist = self.ParameterUpdate(xstar, ystar)
        self.pzmat_Theta = self.PzGivenData(self.hyperlist)
        self.pymat_Theta = self.PyGivenData(self.hyperlist)
        return


    def ObcEstimate(self, pzmat_Theta):
    #    py = PyGivenTheta(xspace, pi_theta)
        zhat = np.argmax(pzmat_Theta, axis = 0)
#        zhat = (pzmat_Theta>= 0.5)
        zhat = zhat.astype(int)
        return zhat

    def ClassifierError(self, xspace, yspace):
    #    pymat is the prediction distribution of y given D
#        pzmat_Theta = self.PzGivenData(pi_theta)
        zhat = self.ObcEstimate(self.pzmat_Theta)
#        zhat = zhat.astype(int)
#        pzmat_r = self.PzGivenXTheta(self.xspace, thetar)
        error = np.mean((zhat != yspace).astype(float))
#        error = np.mean(np.abs(zhat - pzmat_r))
    #    z = fc(xspace, thetar)
    #    error = np.mean(zhat^z)
        return error
        
    def BayesianError(self, thetar):
        pzmat = self.PzGivenXTheta(self.xspace, thetar)
        errormat = np.amin(1-pzmat, axis = 0)
        error = np.mean(errormat)#assume x is uniform distributed
        return error