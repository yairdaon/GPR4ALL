'''
Created on May 19, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import math  
import numpy as np

import aux
import type
import truth
import kriging as kg
import goal

class Config:   
    '''
    This will be referred to as a container class. It holds all
    the data needed to do kriging, resample etc. An instance of 
    this class is passed to the sampler, the kriging procedure
    etc.
    This module does not do heavy computations, that is left for the 
    kriging module. Here we store data and run specifications.
    '''
        
    def __init__(self):
        '''
        we set many default parameters here. you may change them if you 
        are sure you understand what they do. change them using their 
        setters method, though. not here
        '''
        
        
        self.X = [] # list observations... 
        self.F = []  # ...corresponding log-likelihoods
        self.Fmp = [] # F minus prior
        
        # the regularization used in the kriging procedure
        self.reg = 100*np.finfo(np.float).eps
        
        # are the matrices we use ready or do we need to calculate them
        self.matricesReady = False
        
        # some parameters, default values
        self.r = 1.0 # hyper paramenter
        self.M = 10.0
        self.algType = type.AUGMENTED_COVARIANCE # type of kriging algorithm
        
        # by default we incorporate new samples to our data set
        # we can set this to false. Then we can just sample from 
        # the posterior
        self.addSamplesToDataSet = True
        
        # point to the true log-likelihood that we'll use
        self.LL = truth.trueLL
        
        # are the limiting kriged value and variance ready? no!
        self.limitsReady = False
        
        # set prior to decay exponentially
        self.prior = lambda x: -np.inf if aux.infNorm(x)  > self.M else 0.0

        # the default task is regression
        self.goal = goal.REGRESSION
        
    def quickSetup(self, n):
        '''
        adds only two points so you can get things going quickly!!!
        n is the dimensionality of input of YOUR log-likelihood
        to be honest, we can start with only one point also, but
        starting with two is just as fine
        
        n - the dimension of input (e.g 2 if it's R^d)
        '''
        
        # the first points are close to the edges...
        x = np.ones(n)*2*self.M/3
        self.addPair(x, self.LL(x))
        self.addPair(-x, self.LL(-x) )
    
    def addPair(self,x,f):
        ''' 
        add a location, its log likelihood and the log-likelihood
        minus the prior to the lists
        '''
        self.X.append(x) # loactions
        self.F.append(f) # log-likelihood
        self.Fmp.append( f - self.prior(x) ) # Fmp is F minus prior
        
        # we need to recalculate the matrices, so the matrices aren't tready
        self.matricesReady = False
        
        # the limits need to be recalculated also
        self.limitsReady = False
        
    def setR(self,r):
        '''
        set the hyper parameter r and the regularization
        we used in the kriging procedure
        '''
        
        # the length scale of the covariance function
        self.r = r
        
        # the regularization we use in the tychonoff solver
        self.reg = 100*self.r*np.finfo(np.float).eps
        
        # parameters changed, so we need to recalculate the matrices
        self.matricesReady = False
        self.limitsReady = False
    
    def setLL(self, likelihood):
        '''
        we use this to tell the sampler what reality is.
        we say reality and mean a method to calculate TRUE
        log-likelihood.
        likelihood points to your favorite function that calculates log likelihood.
        your function's input should be a numpy array of length n (the value of n
        is up to you). 
        '''
        #print("settin LL")
        self.LL = likelihood   
    
    def setPrior(self ,prior):
        '''
        this is the log likelihood of the prior. we simply 
        subtract this from every observation. 
        f should be defined as prior = lambda x: someFunction(x)
        '''
        self.prior = prior   
                          
    def setMatrices(self):
        ''' 
        calculates the matrices needed to carry out the kriging calculation.
        we use this procedure when we add a sampled "ground truth" point 
        once the matrices are ready we set matricesReady = True
        '''
        
        if self.algType == type.AUGMENTED_COVARIANCE:
            self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
                                                           
        elif self.algType == type.COVARIANCE:
            self.cm = aux.covMat(self.X,self.r)    # cm  = covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
        
        elif self.algType == type.RASMUSSEN_WILLIAMS:
            self.cm = aux.covMat(self.X,self.r)    # cm  = covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
        else: 
            print("Your algorithm type is not valid. Algorithm type set to default.")
            self.algType = type.AUGMENTED_COVARIANCE
            self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        
        # tell everybody the matrices are ready
        self.matricesReady = True       
        
                
        
    def setAddSamplesToDataSet(self, addOrNot):
        '''
        addOrNot has to be boolean. The variable
        addSamplesToDataSet decides whether we simply sample the
        posterior (if set to False) or we sample the posterior, calcualte
        its corresponding log-likelihood and use that to get a better
        kriged interpolant
        '''
        
        self.addSamplesToDataSet = addOrNot
       
    def setType(self, algType):
        '''
        here we set the variable that decides which algorithm we use for kriging or not.
        if we use the augmented covariance matrix, we're using an UNBIASED predictor. This amounts to
        forcing the weights to sum to 1.
        '''        
        if self.algType != algType:
            self.algType = algType
            self.matricesReady = False
            self.limitsReady = False
    
    def setGoal(self, goal):
        '''
        decide on our objective - 
        optimize or regress
        '''
        self.goal = goal   
         
    def setM(self, M):
        ''' 
        M is the size, in the sup norm, of box weconsider. We set the probability outside the box
        to be zero. 
        Precisely: P(x) = 0 for all x such that |x|_inf > M. Equivalently: log P(x) = -inf for all
        x such that |x|_inf > M
        '''
        self.M = M


    def condition(self):
        '''
        use SVD to find condition number of matrix
        '''
        
        s = np.linalg.svd( self.acm )[1]
        self.acmCond = max(s)/min(s)
        
        s = np.linalg.svd( self.cm )[1]
        self.cmCond = max(s)/min(s)
        
    