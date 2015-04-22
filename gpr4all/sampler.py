'''
Created on May 2, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import scipy.optimize
import numpy as np

import emcee as mc

import targets
import container as cot
import truth as truth
import kl as kl
import _g
i = np.array( [0.0] )


class Sampler(object):
    '''
    This class generates samples from the posterior. It can also 
    add points to its data via its learn() method. it calls the 
    package emcee hammer
    :param specs:
        an instance of container.Container. This object holds all
        the parameters and specifications of the current run.
    :param nwalkers:
        the number of Goodman & Weare walkers used. See documentation 
        of the package emcee at <http://dan.iel.fm/emcee/current/>
    :param burn:
        the number of burn in steps we let the emcee sampler take
    :param ndim:
        size of the space in which the walkers walk and probabilities 
        are calculated
    :param pos:
        current positions of the walkers, shape is (nwalkers, ndim)
    :param useInfoGain: boolean, whether we use the information gain 
		criterion for adding points. Set to False by default, until
		I can make it work faster and better.
    :param state:
        state of the random number generator
    :param walkerInd:
        the index of the nex walker we return if we take a single 
        sample. if walkerInd == nwalkers then we know we don't 
        have new walkers to return, so we run the emcee sampler
        to generate new samples
    :param decorTime:
        how many steps we take with the emcee sampler so we have new
        independent (really uncorrelated) samples 
    :param prob:
        a vector of current log-likelihoods of the walkers. has
        shape (nwalkers, ndim) 
    :param blobs:
        hold data of kriged variance - metadata of the calculation
        of log-likelihoods.
        
    '''
    
    def __init__(self, specs, nwalkers=100, burn=500 , noptimizers=20,
                                 maxiter = 7500, target = targets.atan_sig):
        '''
        create an instance, create walkers, let them walk
        '''
        
        # keep the configuration object till the rest of time
        self.specs = specs
    
        # the number of space dimensions
        self.ndim = len(specs.X[0])
            
        # set number of walkers
        self.nwalkers = nwalkers 
        
        # number of points to start optimization. 
        self.noptimizers = noptimizers  
        
        # set burn in time
        self.burn = burn 
        
        # the first decorrelation time. 
        self.decorTime = burn
        
        # the initial set of positions
        self.pos = np.random.rand(self.ndim * self.nwalkers) #choose U[0,1]
        self.pos = ( 2*self.pos  - 1.0 )*specs.r # shift and stretch
        self.pos = self.pos.reshape((self.nwalkers, self.ndim)) # reshape
        
        # set the initial state of the PRNG
        self.state = np.random.get_state()
        
        # create the emcee sampler, let it burn in and erase burn in
        self.sam = mc.EnsembleSampler(self.nwalkers, self.ndim, self.specs.kriging) 
        self.didBurnIn = False
        
        # tell samplers that they do not need to propagate the walkers
        self.walkerInd = 0
        
        # max num of optimization iterations
        self.maxiter = maxiter
        
        # the function which we optimize
        self.target = target
        
        self.variances = []
        
        self.choose_point = self.min_var
        


    def var_sample(self, nsteps):
        '''
        get the sample for the SSA approximation of the
        average variance. The only difference is that 
        we use twice the LL here
        '''

        #print("Starting SAA sample")
        # the initial set of positions
        pos = np.random.rand(self.ndim * self.nwalkers) #choose U[0,1]
        pos = ( 2*pos  - 1.0 )*self.specs.r # shift and stretch
        pos = pos.reshape((self.nwalkers, self.ndim)) # reshape
        
        # sample using the SSA LL which is twice the original LL
        sam = mc.EnsembleSampler(self.nwalkers, self.ndim, self.specs.ssaLL) 
        pos, _ , self.state = sam.run_mcmc( pos, self.burn,self.state )
        sam.reset()
        pos, _ , self.state = sam.run_mcmc( pos, nsteps,self.state )

        #print("Finished SAA sampler")
        return sam.flatchain
    


        
    def min_var(self):
        '''
        use minimum variance 
        criterion to choose next point
        '''           
        sample = self.var_sample(2000)
        target = _g.avg_var
        specs  = self.specs
        soArgs   = ( specs.U, specs.S, specs.V,
                   specs.Xarr, sample,
                   specs.r , specs.d, specs.reg )
            
        bestValue = np.inf

        for _ in range(self.noptimizers):
            
            # sample a starting point
            startPoint = self.sample_one()
            result = scipy.optimize.minimize(target, startPoint, args=soArgs, method='BFGS',
                                             jac = True, options= {'maxiter' : self.maxiter} )
            if result.fun < bestValue:
                bestValue = result.fun
                bestPoint = result.x
            
       	
        bestPoint = bestPoint.reshape(self.ndim,)
        print bestPoint        
        return bestPoint
                    

    def learn(self):
        '''
        if we want to choose another point to calculate the 
        log-likelihood at - this is the method we use
        '''
        
        # choose the next point 
        s = self.choose_point()
        
        # add the new sample to our data set
        self.specs.add_point( s )
        
        # let them know we need a new batch before we sample
        self.walkerInd = self.nwalkers
        
        # erase previous data so we do not accidentally use it
        self.sam.reset()
        
        # tell everybody we're not burned in
        self.didBurnIn = False

        # update our burnin time to something more reasonable
        #self.burn = 2*self.decorTime
    
           
    def run_mcmc(self, nsteps):
        '''
        run the emcee sampler for nsteps steps 
        :param nsteps:
            the number of steps we let the emcee sampler run
        '''   
        

        self.pos, self.prob, self.state = self.sam.run_mcmc(
                                                    self.pos, nsteps, self.state )
        
#         self.pos, self.prob, self.state, self.blobs = self.sam.run_mcmc(
#                                                     self.pos, nsteps, self.state ) 
        
#         self.variances = np.ravel(np.asarray( self.blobs ))
#         print("ran MCMC. shape of variances is " + str(self.variances.shape))
        
        # update the decorrelation time
        dec = 2*np.max( self.sam.acor )
        if nsteps > 10*dec:
            self.decorTime = max(dec , 20 )
        else:
            self.decorTime =self.burn
    
    
    def sample_batch(self):
        '''
        sample a bunch\ a batch
        return a new, unused batch of positions of the goodman
        & weare walkers
        * ``samples`` - numpy array of size (nwalkers , ndim)
        '''
        
        if not self.didBurnIn:
            self.burnIn()
            
        if self.walkerInd != 0:
            
            # run the MCMC to get new batch 
            self.run_mcmc(self.decorTime)
        
        # create a copy of the positions, so nothing unexpected happens if we parallelize
        samples = self.pos[:,:]
        
        # let them know we used this batch
        self.walkerInd = self.nwalkers
        
        return samples
     
        
    def sample_one(self):
        '''
        this method returns a single sample from the current posterior
        since we have a bunch of walkers, we return one of those 
        with every call to this method. if we have used them all 
        (i.e if walkerInd == nwalkers) then we are forced to run 
        emcee for some more time.
        '''
        if not self.didBurnIn:
            self.burnIn()
            
        # if we have no more unused walkers
        if self.walkerInd == self.nwalkers:
            
            # run the MCMC 
            self.run_mcmc(self.decorTime)
            
            # the walker we sample is the zeroth
            self.walkerInd = 0
        
        # we return the walker denoted by walkerInd     
        sample =  self.pos[self.walkerInd,:]
        
        # the next we return is the next unused walker in the list
        self.walkerInd = self.walkerInd + 1
        
        return sample
     
                    
    def flatchain(self):
        '''
        get the flattened chain
        '''
        
        # shape is (number of steps , dimension)
        return self.sam.flatchain     
     
     
    def burnIn(self):
        '''
        do the burn in phase and reset
        '''
        self.run_mcmc(self.burn)
        self.sam.reset()
        self.didBurnIn = True 
        self.walkerInd = 0
         
