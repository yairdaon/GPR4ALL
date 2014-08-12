'''
Created on May 2, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import math
import numpy as np

import kriging as kg 
import truth 
import aux 
import config as cfg
import goal

import emcee as mc


def lnprob(s, CFG):
        '''
        return the interpolated f
        here this is interpreted as a log-likelihood \ log-probability
        input:
        s - the point in space fr which we estimate the log-likelihood
        X - a list of locations in space
        F - a list of corresponding precalculated log-likelihoods
        C -  an augmented covariance matrix
        M - an artificial bound on the distribution. we insist that
        if |x| > M (the sup norm) then the likelihood
        is zero (so the log-likelihood is -infinity)
        a hyper parameter, see the documentation for aux.cov(...) procedure
        '''
        
#         M = CFG.M
#         # we ensure M > |s| in sup norm
#         # we should also make sure that all observations 
#         # satisfy |X[j]| < M
#         if (np.linalg.norm(s, np.inf)  >  M):
#             return -np.inf
        
        # do kriging to estimate the the log likelihood in a new 
        # location, given previous observations
        mu, sig = kg.kriging(s, CFG)
    
        # return the interpolated value only - no use for the std dev
        return mu
    
class Sampler:
    '''
    a class that takes care of generating samples
    '''
    
    def __init__(self, CFG):
        '''
        create an instance, create walkers, let them walk
        '''
        
        # keep the configuration file till the rest of time
        self.CFG = CFG
    
         # the number of space dimensions
        self.ndim = len(CFG.X[0])
        
        # set number of walkers
        self.nwalkers =  10*self.ndim
        
        # set burn in time
        self.burn = 150*(self.ndim)**(1.5)
        
        # the initial set of positions are uniform  in the box [-M,M]^ndim
        self.pos = np.random.rand(self.ndim * self.nwalkers) #choose U[0,1]
        self.pos = ( 2*self.pos  - 1.0 )*CFG.M # shift and stretch
        self.pos = self.pos.reshape((self.nwalkers, self.ndim)) # reshape
        
        
        
        self.sam = mc.EnsembleSampler(self.nwalkers, self.ndim, lnprob, args=[ self.CFG ])
        
        self.state = np.random.get_state()
    
    
    def choosePointRegression( self ):
            ''' 
            current criterion for evaluating LL is choose position 
            of walker that maximizes  likelihood*variance
            this might be a good choice criterion when we want to 
            do INTERPOLATION.
            '''
            
            return self.pos[0,:]
        
            maxScore = 0
            ind = 0 
            for i in range(self.nwalkers):
                
                #ideally, the walker would carry this info 
                krig, sig = kg.kriging( self.pos[i,:] , self.CFG )
                
                # choose walker based on this made up score
                currScore = sig*krig 
                
                if  currScore > maxScore:
                    ind = i
                    maxScore = currScore
                
            return self.pos[ind,:]
    
    def choosePointOptimization( self ):
        '''
        if we seek to optimize, we choose a point according
        to the kriged dist and hope that it is closer to the 
        maximum
        '''
        return self.pos[0,:]
            
    def sample(self):
        '''
        this procedure samples a distribution. this distribution is defined by 
        kriging some previously collected data and interpolating it to give a 
        distribution over states space.
        input:    
        CFG is a container object that holds all required data. see the config.py module
        '''

        # run the MCMC 
        self.pos, self.prob, self.state = self.sam.run_mcmc(self.pos, self.burn, self.state ) 
        
        if self.CFG.addSamplesToDataSet == False:
            return self.pos[0,:]
    
        else:
    
            # choose the best point according to some criterion
            if self.CFG.goal == goal.REGRESSION:
                s = self.choosePointRegression()
            else: # i.e. if we choose optimization
                s = self.choosePointOptimization()
            
            # calculate the corresponding log likelihood
            f = np.array( [ self.CFG.LL(s) ] )
        
            # incorporate the new sample to our data set
            cfg.Config.addPair(self.CFG, s, f)
            
            # update the required matrices required for the kriging calculation
            cfg.Config.setMatrices(self.CFG)
            
            return s
            
            
        
        
        
        