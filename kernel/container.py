'''
Created on May 19, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import numpy as np
import copy


import aux
import kernel.rap as rap

class Container:   
    '''
    This will be referred to as a container class. It holds all
    the data needed to do kriging, resample etc. An instance of 
    this class is passed to the sampler, the kriging procedure
    etc.
    This module does not do heavy computations, that is left for the 
    kriging module. Here we store data and run specifications.
    We  use  algorithm 2.1 from the book Gaussian Processes for Machine Learning
    which can be found at <http://www.gaussianprocess.org/gpml/>.
    
    :param trueLL: the true log-likelihood that we will use. this 
        (supposedly) takes a long while to calculate, so we want to
        call it as less frequently as possible.
    
    :pararm prior: the prior log-likelihood. We use it so that probability
        goes to zero nicely and we may choose not to truncate the probability
        outside of the box of size M. By default, however, we do just that.
    
    :param args, kwargs: 
        these are the parameters of the true log-likelihood. it can be in any form,
        as long as it is one python object. The only place these parameters are used
        is when we call trueLL. It is defaulted as None.
    
    :param r: the characteristic lenght scale of the covariance function
    
    :Param M: a number. usually it is the size of the box outside of 
        which we assume the probability is zero. We also use it to put
        bounds on plots every once in a while
        
    :param X:
        a list of places in space for which we know the log-likelihood
    
    :param F: a list of log likelihoods corresponding to X
    
    :param Fmp: F minus prior. We subtract the prior log-likelihood
        from every entry of F. We do kriging using Fmp, then add 
        back what we subtracted
        
    :param reg: this is a regularizing term, used to solve linear equations
        stably using Tychonoff regularization. Some more information may be found
        at the documentation of the method tychonof_solver in module aux or online
        at <http://en.wikipedia.org/wiki/Tikhonov_regularization>.
        
    :param matricsReady: a boolean, which tells you if the SVD of the covariance
        matrix is up to date. This SVD is used in the kriging process.
        
    :param limitsReady: a boolean telling you if we already calculated the kriged
        value and kriged variance at infinity.
        
    :param normalization: the integral of e^{kriged log-likelihood} over all
    of space. It is set to -1 in case we think it is not up to date.
    
    :param args (optional):
        a list of parameters for the true log-likelihood function
        
    :param kwargs (optional):
        as above, a dictionary of key word argument for the true log-likelihood
    '''
        
    def __init__(self,trueLL,r=1.3, d=1.0 ,M=10.0, args =[] , kwargs = {} ):
        '''
        we set many default parameters here. you may change them if you 
        are sure you understand what they do. change them using their 
        setters method, though. not here
        '''
        
        # point to the true log-likelihood that we'll use
        self.trueLL = trueLL
        
        # set prior to uniform inside a box. Not using lambda 
        # because Guido said so.
        def default_prior(x):
            if np.amax(abs(x)) > self.M:
                return -np.Inf
            return 0.0
        self.prior = default_prior
        
        # parameters for the true log-likelihood
        self.args   = args
        self.kwargs = kwargs
        
        # some parameters
        self.r = r # hyper paramenter
        self.d = d
        self.M = M
        
        # the regularization used in the kriging procedure
        self.reg = 100*np.finfo(np.float).eps
        
        self.X = [] # list observations... 
        self.F = [] # ...corresponding log-likelihoods
        self.Fmp = [] # F minus prior
        
        # are the matrices we use ready or do we need to calculate them
        self.matricesReady = False       
    
    def add_pair(self,x,f):
        ''' 
        add a location, its log likelihood and the log-likelihood
        minus the prior to the lists. usually used when you let the
        program run and learn the probability distribution on its own
        '''
        f = float(f)
        self.X.append(np.ravel(x)) # loactions
        self.F.append(f) # log-likelihood
        self.Fmp.append( f - self.prior(x) ) # Fmp is F minus prior
        
        # we need to recalculate the matrices, so the matrices aren't ready
        self.matricesReady = False
        
        # the limits need to be recalculated also
        self.limitsReady = False
    
    def add_point(self , x):
        '''
        add the point x with its true LL. usually used when you have
        data from previous runs and you wwant to start your kriging 
        using that data
        '''
        
        # call the LL with the parameters
        f = float(rap.rapper( x, self.trueLL , self.args, self.kwargs))
        self.add_pair(x, f)
    
    def set_r(self,r):
        '''
        set the hyper parameter r and the regularization
        we used in the kriging procedure
        '''
        
        # the length scale of the covariance function
        self.r = r
        
        # parameters changed, so we need to recalculate stuff
        self.matricesReady = False
    
    def set_prior(self ,prior):
        '''
        this is the log likelihood of the prior. we simply 
        subtract this from every observation. 
        f should be defined as prior = lambda x: someFunction(x)
        '''
        self.prior = prior
        for i in range(len(self.X)):
            self.Fmp[i] = self.F[i] - self.prior(self.X[i])
                          
    def set_matrices(self):
        ''' 
        calculates the matrices needed to carry out the kriging calculation.
        we use this procedure when we add a sampled "ground truth" point 
        once the matrices are ready we set matricesReady = True
        '''
             
        cm = aux.cov_mat(self.X,self.r, self.d)    # cm  = covariance matrix
        self.U, self.S, self.V = np.linalg.svd(cm, full_matrices = True, compute_uv = True)
      
        # tell everybody the matrices are ready
        self.matricesReady = True

    def condition(self):
        '''
        find condition number of the matrix we use for kriging
        '''
        if not self.matricesReady:
            self.set_matrices()
            
        return max(self.S)/min(self.S)
        
        

    