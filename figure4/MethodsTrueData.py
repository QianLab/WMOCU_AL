
import numpy as np
import copy
from scipy.stats import entropy
from scipy.special import logsumexp
#from scipy.stats import entropy
#import LinkedList as LL


#class LinkedList(object):
#    def __init__(self, head=None):
#        self.head = head
        


# remove xspace in each iteration 
class Problem():
    
    def __init__(self, xspace, yspace, thetalist, pz_theta_model, py_eq_z, pi_theta = None, dataidx = None):
        print('utils')
        self.xspace = xspace
        self.yspace = yspace
        self.thetalist = thetalist
        self.PzGivenXTheta = pz_theta_model
        self.PYeqZ = py_eq_z#if PYeqZ is None, means that there is no flip error 
        if pi_theta is None:
            pi_theta = np.ones(len(thetalist))
            pi_theta /= pi_theta.sum()
        self.pi_theta = pi_theta
        self.pzmat_Theta = self.PzGivenData(self.pi_theta)
        self.dataidx = dataidx
        

    def Initialize(self, xspace, yspace, pi_theta = None, dataidx = None):
        if pi_theta is not None:
            self.pi_theta = pi_theta
        
#        self.xspace = xspace
#        self.yspace = yspace
            self.pzmat_Theta = self.PzGivenData(self.pi_theta)
            self.dataidx = dataidx


    def fc(self, x, theta):
        # x can be np.array of size 2 or a list of two np.array x1 and x2
        a = theta[0]
        b = theta[1]
        c = theta[2]
        x1 = x[0]
        x2 = x[1]
        z = (x2 - (a*x1**2+b*x1+c)>=0)
        # class 1 return 1 and class 0 return 0
        return z


    def fr(self, x, thetar):
        zmat = np.random.random(len(x)) <=self.PzGivenXTheta(x, thetar)
#        zmat = zmat.astype(int)
        if self.PYeqZ is  None:
            ymat = zmat
        else:
            flipmat = np.random.random(zmat.shape)>self.PYeqZ(x)
            ymat = np.logical_xor(zmat, flipmat)
        ymat = ymat.astype(int)
        
        return ymat


    def MatTraceBack(self, x, zmat):
        #zmat is the given observation mat, we trace x back to xspace to find z
        x1 = self.xspace[0]
        x2 = self.xspace[1]
        idx1 = next(i for i, _ in enumerate(x1) if np.isclose(_, x[0]))
        idx2 = next(i for i, _ in enumerate(x2) if np.isclose(_, x[1]))
        return zmat[idx1, idx2]


    def ParameterUpdate(self, x, y):#update this for error
        #Here the input of x can only be single input: x = np.array([x1[i][0],x2[j]])
        # the posterior distribution is deterministic here, we only need to update it
        # to probabilistic case    
        pi_theta2 = copy.copy(self.pi_theta) # not to change the value in pi_theta
        for i in range(len(pi_theta2)):
            
            pz1_xtheta = self.PzGivenXTheta(x, self.thetalist[i]) 
            if self.PYeqZ is None:
                py1_xtheta = pz1_xtheta
            else:
                py1_xtheta = self.PYeqZ(x)*pz1_xtheta+(1-pz1_xtheta)*(1-self.PYeqZ(x))
            #update to posterior pi(theta|x, y) \prop pi(theta)p(y|x, theta)
            if y == 1:
                py_xtheta = py1_xtheta
            else:
                py_xtheta = (1 - py1_xtheta)
            pi_theta2[i] *= py_xtheta
        
        pi_theta2 /= pi_theta2.sum()
        
        return pi_theta2


    def PzGivenData(self, pi_theta):
#        x1 = self.xspace[0]
#        x2 = self.xspace[1]
#        pzmat = np.zeros([x1.size, x2.size])
        
        pzmat = np.zeros(len(self.xspace))
        
        for i in range(len(pi_theta)):
            pzmat += self.PzGivenXTheta(self.xspace, self.thetalist[i])*pi_theta[i]
        return pzmat


    def ObcError(self, pi_theta):
        
        pzmat_Theta = self.PzGivenData(pi_theta)
        errormat = np.minimum(1 - pzmat_Theta, pzmat_Theta)
        error = np.mean(errormat)#assume x is uniform distributed
        return error
    
    def MinIbrResidual(self, x, py_x):
        sumresidual = 0
        for i in range(2):
            if i == 0:
                p = py_x
                y = 1
            else:
                p = 1-py_x
                y = 0
            pi_theta2 = self.ParameterUpdate(x, y)
#            pymat = self.PYeqZ(self.xspace)*self.pzmat_Theta+(1-self.pzmat_Theta)*(1-self.PYeqZ(self.xspace))
            sumresidual += self.ObcError(pi_theta2)*p
        return -sumresidual
    
    def MinIbrResidualWhole(self):
        #the IbrResidual for the whole space
        utilitymat = np.zeros(len(self.xspace))
        if self.PYeqZ is None:
            pymat = self.pzmat_Theta
        else:
            pymat = self.PYeqZ(self.xspace)*self.pzmat_Theta+(1-self.pzmat_Theta)*(1-self.PYeqZ(self.xspace))
        for i, x in enumerate(self.xspace):
            py_x = pymat[i]
            utilitymat[i] = self.MinIbrResidual(x, py_x)
        return self.ObcError(self.pi_theta)+utilitymat          #self.ObcError(self.pi_theta)
            
    
    def WMOCU2(self, pi_theta):
        #it's just a x array of G(x, pi_theta)
        wmocu = np.zeros(len(self.xspace))
        pzmat = self.PzGivenData(pi_theta)
        
        bayesian_error = np.zeros(len(self.xspace))
        for i, theta in enumerate(self.thetalist):
            bayesian_error += np.minimum(self.PzGivenXTheta(self.xspace, theta),
                                  1-self.PzGivenXTheta(self.xspace, theta))*pi_theta[i]
        average_error = np.minimum(pzmat, 1 - pzmat) - bayesian_error#this term is not correct for 
        weight = 1 - average_error
        wmocu = np.mean(weight*average_error)
        
        return wmocu

    def DWeighted_MOCU2(self, xidx):
        #search acquisition function based on weighted mocu
        x = self.xspace[xidx]
        pz_x = self.pzmat_Theta[xidx]
        if self.PYeqZ is None:
            py_x = pz_x
        else:
            py_x = self.PYeqZ(x)*pz_x+(1 - self.PYeqZ(x))*(1 - pz_x)
           
        wmocu2 = 0
        for i in range(2):
            if i == 0:
                p = py_x
                y = 1
            else:
                p = 1-py_x
                y = 0
            pi_theta2 = self.ParameterUpdate(x, y)
            wmocu2 += p*self.WMOCU2(pi_theta2)
        return wmocu2
    
    def Weighted_MOCUWhole2(self):
#        wmocu = self.WMOCU2(self.pi_theta)
        utilitymat = np.zeros(len(self.xspace))
        for i, x in enumerate(self.xspace):
#            utilitymat[i] = wmocu - self.DWeighted_MOCU2(i)
            utilitymat[i] = - self.DWeighted_MOCU2(i)
        return utilitymat  


    def SMOCU(self, pi_theta, k = 1, softtype = 1):
#        smocu = np.zeros(len(self.xspace))
        pzmat = self.PzGivenData(pi_theta)
        if softtype == 1:
            obc_correct = (pzmat*np.exp(pzmat*k) + (1-pzmat)*np.exp(k-pzmat*k))/(np.exp(pzmat*k)+np.exp(k-pzmat*k))
    #        smocu = np.mean(bayesian_correct - obc_correct)
        elif softtype == 2:
            pzmat_array = np.array([pzmat, 1-pzmat])
            obc_correct = logsumexp(k*pzmat_array, axis = 0)/k
        smocu = np.mean( - obc_correct)
        return smocu
    
    def D_SMOCU(self, xidx, k, softtype):
        x = self.xspace[xidx]
        pz_x = self.pzmat_Theta[xidx]
        if self.PYeqZ is None:
            py_x = pz_x
        else:
            py_x = self.PYeqZ(x)*pz_x+(1 - self.PYeqZ(x))*(1 - pz_x)
        
        smocu2 = 0
        for i in range(2):
            if i == 0:
                p = py_x
                y = 1
            else:
                p = 1-py_x
                y = 0
            pi_theta2 = self.ParameterUpdate(x, y)
            smocu2 += p*self.SMOCU(pi_theta2, k, softtype)
        return smocu2
        
    def SoftMOCU_K(self, k, softtype):
#        smocu = self.SMOCU(self.pi_theta, k)
        utilitymat = np.zeros(len(self.xspace))
        for i, x in enumerate(self.xspace):
#            utilitymat[i] = smocu - self.D_SMOCU(i, k)
            utilitymat[i] = - self.D_SMOCU(i, k, softtype)
        return utilitymat
    
    def SoftMOCUWhole(self, k = 1, softtype = 1):
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
        self.pzmat_Theta = self.PzGivenData( self.pi_theta)
#        pymat = self.PzGivenData( self.pi_theta)
        if self.PYeqZ is None:
            pymat = self.pzmat_Theta
        else:
            pymat = self.PYeqZ(self.xspace)*self.pzmat_Theta+(1-self.pzmat_Theta)*(1-self.PYeqZ(self.xspace))
        posterior_entropy_mat = np.zeros(len(self.xspace))
#        posterior_entropy_mat2 = posterior_entropy_mat
        for i in range(len(self.thetalist)):
            theta = self.thetalist[i]
            pz_theta_mat = self.PzGivenXTheta(self.xspace, theta)
            if self.PYeqZ is None:
                py_theta_mat = pz_theta_mat
            else:
                py_theta_mat = pz_theta_mat*self.PYeqZ(self.xspace) +\
            (1-pz_theta_mat)*(1-self.PYeqZ(self.xspace))
#            posterior_entropy_mat += self.pi_theta[i]*bientropy(py_theta_mat)
            posterior_entropy_mat += self.pi_theta[i]*entropy([py_theta_mat, 1-py_theta_mat])
#        entropymat = bientropy(self.pzmat_Theta) - posterior_entropy_mat
        entropymat = entropy([pymat, 1-pymat]) - posterior_entropy_mat
        return entropymat
    
    
    def UncertaintyWhole(self):
#        pymat = self.PYeqZ(self.xspace)*self.pzmat_Theta+(1-self.pzmat_Theta)*(1-self.PYeqZ(self.xspace))
        objmat = -abs(self.pzmat_Theta - 0.5)
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
        for i, pi in enumerate(self.pi_theta):
            pz1_xtheta = self.PzGivenXTheta(xstar, self.thetalist[i])
            if self.PYeqZ is None:
                py1_xtheta = pz1_xtheta
            else:
                py1_xtheta = self.PYeqZ(xstar)*pz1_xtheta+(1-pz1_xtheta)*(1-self.PYeqZ(xstar))
            if ystar == 1:
                py_xtheta = py1_xtheta
            else:
                py_xtheta = (1 - py1_xtheta)
            self.pi_theta[i] *= py_xtheta
            
        self.pi_theta /= self.pi_theta.sum()
#        if self.yspace is not None:
#            self.xspace = np.delete(self.xspace, xidx, 0)
#            self.yspace = np.delete(self.yspace, xidx, 0)
        self.pzmat_Theta = self.PzGivenData(self.pi_theta)
        return


    def ObcEstimate(self, pzmat_Theta):
    #    py = PyGivenTheta(xspace, pi_theta)
        zhat = (pzmat_Theta>= 0.5)
        return zhat

    def ClassifierError(self, thetar, pi_theta):
    #    pymat is the prediction distribution of y given D
#        pzmat_Theta = self.PzGivenData(pi_theta)
        zhat = self.ObcEstimate(self.pzmat_Theta)
        zhat = zhat.astype(int)
        pzmat_r = self.PzGivenXTheta(self.xspace, thetar)
        error = np.mean(np.abs(zhat - pzmat_r))
    #    z = fc(xspace, thetar)
    #    error = np.mean(zhat^z)
        return error
        
    def BayesianError(self, thetar):
        pzmat = self.PzGivenXTheta(self.xspace, thetar)
        errormat = np.minimum(1 - pzmat, pzmat)
        error = np.mean(errormat)#assume x is uniform distributed
        return error
    
#
#    def PointWMOCU(self, xidx, pi_theta):
#        #weighted mocu on each point
#        x = self.xspace[xidx]
##        py_x = self.pzmat_Theta[xidx]
#        pzmat = self.PzGivenData(pi_theta)
#        
#        bayesian_error = 0
#        for i, theta in enumerate(self.thetalist):
#            bayesian_error += min(self.PzGivenXTheta(x, theta), 1-self.PzGivenXTheta(x, theta))*pi_theta[i]
#        
#        average_error = (min(pzmat[xidx], 1-pzmat[xidx])-bayesian_error)
#        weight = 1 - average_error
#        
##        weight = max(pzmat[xidx], 1-pzmat[xidx]) + bayesian_error
#        mocu = weight*average_error
#
#        return mocu
#
#    def DWeighted_MOCU(self, xidx):
#        #search acquisition function based on weighted mocu
#        x = self.xspace[xidx]
#        pz_x = self.pzmat_Theta[xidx]
#        if self.PYeqZ is None:
#            py_x = pz_x
#        else:
#            py_x = self.PYeqZ(x)*pz_x+(1 - self.PYeqZ(x))*(1 - pz_x)
#        
#        mocu = self.PointWMOCU(xidx, self.pi_theta)
#        mocu2 = 0
#        for i in range(2):
#            if i == 0:
#                p = py_x
#                y = 1
#            else:
#                p = 1-py_x
#                y = 0
#            pi_theta2 = self.ParameterUpdate(x, y)
#            mocu2 += p*self.PointWMOCU(xidx, pi_theta2)
#        return mocu - mocu2
#    
#    def Weighted_MOCUWhole(self):
#        utilitymat = np.zeros(len(self.xspace))
#        for i, x in enumerate(self.xspace):
##            py_x = self.pzmat_Theta[i]
#            utilitymat[i] = self.DWeighted_MOCU(i)
#        return utilitymat 