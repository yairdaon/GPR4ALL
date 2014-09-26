'''
Created on May 19, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import math  
import numpy as np
import copy

import scipy.integrate
import mcint

import aux
import algorithm as alg
import kriging as kg
import kernel.rap as rap

class Container:   
    '''
    This will be referred to as a container class. It holds all
    the data needed to do kriging, resample etc. An instance of 
    this class is passed to the sampler, the kriging procedure
    etc.
    This module does not do heavy computations, that is left for the 
    kriging module. Here we store data and run specifications.
    
    :param trueLL: the true log-likelihood that we will use. this 
        (supposedly) takes a long while to calculate, so we want to
        call it as less frequently as possible.
    
    :pararm prior: the prior log-likelihood. We use it so that probability
        goes to zero nicely and we may choose not to truncate the probability
        outside of the box of size M. By default, however, we do just that.
    
    :param parameters: 
        these are the parameters of the true log-likelihood. it can be in any form,
        as long as it is one python object. The only place these parameters are used
        is when we call trueLL. It is defaulted as None.
    
    :param r: the characteristic lenght scale of the covariance function
    
    :Param M: a number. usually it is the size of the box outside of 
        which we assume the probability is zero. We also use it to put
        bounds on plots every once in a while
    
    :param algType: the type of algorithm used for kriging. We usually use
        algorithm 2.1 from the book Gaussian Processes for Machine Learning
        which can be found at <http://www.gaussianprocess.org/gpml/>. In other 
        occasions we use an unbiased version of the above. That can be found at
        Regression Models for Time Series Analysis, Kedem & Fokianos, 2002.   
    
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
        
    def __init__(self,trueLL,r=1.3,M=10.0,  
                 algType=alg.RASMUSSEN_WILLIAMS , args =[] , kwargs = {} ):
        '''
        we set many default parameters here. you may change them if you 
        are sure you understand what they do. change them using their 
        setters method, though. not here
        '''
        
        # point to the true log-likelihood that we'll use
        self.trueLL = trueLL
        
        # set prior to decay exponentially
        self.prior = lambda x: -np.inf if aux.inf_norm(x)  > self.M else 0.0
        
        # parameters for the true log-likelihood
        self.args   = args
        self.kwargs = kwargs
        
        # some parameters
        self.r = r # hyper paramenter
        self.M = M
        self.algType = algType # the kriging algorithm we use
        
        # the regularization used in the kriging procedure
        self.reg = 100*np.finfo(np.float).eps
        
        self.X = [] # list observations... 
        self.F = [] # ...corresponding log-likelihoods
        self.Fmp = [] # F minus prior

        # are the matrices we use ready or do we need to calculate them
        self.matricesReady = False
        
        # are the limiting kriged value and variance ready? no!
        self.limitsReady = False
        
        # we show we do not know the constant by making it negative
        self.normalization = -1
    
    def get_copy(self):
        '''
        we use this to create (almost) a deep copy of the currnet object.
        we override teh deepcopy method since it creates new instances of 
        class Algorithm. This is bad since these instances are used for 
        comparison tests.
        '''
        
        # create new instance
        c = Container( self.trueLL , M=self.M , r=self.r ,algType=self.algType)
        c.X = copy.deepcopy(self.X)
        c.F = copy.deepcopy(self.F)
        c.Fmp = copy.deepcopy(self.Fmp)
        c.set_prior( self.prior )

        return c
        
        
    def get_normalization( self , numSamples=1500 ,test=False ):
        '''
        calculate the normalization const
        for the data inside the container
        specs
        '''
        if self.normalization < 0:
                        
            # the (unnormalized) probability function 
            dim = self.X[0].size
            M = self.M
              
            if dim == 1 and not test: # do a 1D integration
                integrand = lambda x:  math.exp( kg.kriging(np.array([x]),self)[0])
                self.normalization , self.normError = scipy.integrate.quad(integrand, -M, M)
            
            elif dim == 2 and not test: # do a 2D integration
                integrand = lambda y,x:  math.exp( kg.kriging(np.array([y,x]),self)[0])
                self.normalization , self.normError = scipy.integrate.dblquad(integrand, -M, M, lambda x: -M , lambda x: M)             
            
            else: # use monte carlo
                
                
                integrand = lambda x:  math.exp( kg.kriging(x,self)[0])                 
    
                # Describe how Monte Carlo samples are taken
                def sampler(): 
                    while True:
                        yield (2*np.random.rand(dim) - 1 )*M
    
                # the size of the domain in question
                domainsize = (2*M)**dim
                self.normalization , self.normError  = mcint.integrate(integrand, 
                                                    sampler(), measure=domainsize, n=numSamples)    
            
        return self.normalization, self.normError
    
    def quick_setup(self, d):
        '''
        adds only two points so you can get things going quickly!!!
        n is the dimensionality of input of YOUR log-likelihood
        to be honest, we can start with only one point also, but
        starting with two is just as fine
        
        n - the dimension of input (e.g 2 if it's R^2)
        '''
        
        # the first points are close to the edges...
        x = np.ones(d)*2*self.M/3
        self.add_point( x )
        self.add_point(-x )
        self.normalization = -1
        
        # we need to recalculate the matrices, so the matrices aren't tready
        self.matricesReady = False
        
        # the limits need to be recalculated also
        self.limitsReady = False
    
    def add_pair(self,x,f):
        ''' 
        add a location, its log likelihood and the log-likelihood
        minus the prior to the lists. usually used when you let the
        program run and learn the probability distribution on its own
        '''
        self.X.append(x) # loactions
        self.F.append(f) # log-likelihood
        self.Fmp.append( f - self.prior(x) ) # Fmp is F minus prior
        
        # we need to recalculate the matrices, so the matrices aren't tready
        self.matricesReady = False
        
        # the limits need to be recalculated also
        self.limitsReady = False
        
        # let them know we do not know the normalization const
        self.normalization = -1
    
    def add_point(self , x):
        '''
        add the point x with its true LL. usually used when you have
        data from previous runs and you wwant to start your kriging 
        using that data
        '''
        
        # call the LL with the parameters
        f = rap.rapper( x, self.trueLL , self.args, self.kwargs)
        
        self.X.append(x)
        self.F.append(f)
        self.Fmp.append( f - self.prior(x) )
        
        # we need to recalculate the matrices, so the matrices aren't tready
        self.matricesReady = False
        
        # the limits need to be recalculated also
        self.limitsReady = False
        
        # let them know we do not know the normalization const
        self.normalization = -1
        
    def change_F(self, f):
        '''
        change the last entry of the likelihood vector. we use this when
        we calculate the information gain. see the module info.
        '''
        self.F[-1] = f
        self.Fmp[-1] = f - self.prior( self.X[-1])  
        
        # matrices didn't change but limits and normalization did
        self.limitsReady  = False
        self.normalization = -1
        
        # note that the matrices ARE ready, since we did not change
        # any location in space!!
    
    def set_r(self,r):
        '''
        set the hyper parameter r and the regularization
        we used in the kriging procedure
        '''
        
        # the length scale of the covariance function
        self.r = r
        
        # the regularization we use in the tychonoff solver
        self.reg = 100*self.r*np.finfo(np.float).eps
        
        # parameters changed, so we need to recalculate stuff
        self.matricesReady = False
        self.limitsReady = False
        self.normalization = -1
    
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
        
        if self.algType == alg.AUGMENTED_COVARIANCE:
            self.acm = aux.aug_cov_mat(self.X,self.r)    # acm  = augmented covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
                                                           
        elif self.algType == alg.RASMUSSEN_WILLIAMS:
            self.cm = aux.cov_mat(self.X,self.r)    # cm  = covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
        else: 
            print("Your algorithm type is not valid. Algorithm type set to default.")
            self.algType = alg.RASMUSSEN_WILLIAMS
            self.acm = aux.aug_cov_mat(self.X,self.r)    # acm  = augmented covariance matrix
            self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        
        # tell everybody the matrices are ready
        self.matricesReady = True       
        
                
    def set_type(self, algType):
        '''
        here we set the variable that decides which algorithm we use for kriging or not.
        if we use the augmented covariance matrix, we're using an UNBIASED predictor. This amounts to
        forcing the weights to sum to 1.
        '''        
        if self.algType != algType:
            self.algType = algType
            self.matricesReady = False
            self.limitsReady = False

    def set_M(self, M):
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
        
    