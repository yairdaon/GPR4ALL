'''
Created on May 2, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np
from scipy import __version__ as ver
import scipy.optimize

import emcee as mc

import kriging as kg 
import gradient 
import targets


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
    
    def __init__(self, specs, nwalkers=20, burn=500):
        '''
        create an instance, create walkers, let them walk
        '''
        
        # keep the configuration object till the rest of time
        self.specs = specs
    
        # the number of space dimensions
        self.ndim = len(specs.X[0])
            
        # set number of walkers
        self.nwalkers = nwalkers #150*self.ndim
        
        # number of points to start optimization. 
        self.noptimizers = self.nwalkers
        
        # set burn in time
        self.burn = burn #500*(self.ndim)**(1.5)
        
        # the initial set of positions are uniform  in the box [-M,M]^ndim
        self.pos = np.random.rand(self.ndim * self.nwalkers) #choose U[0,1]
        self.pos = ( 2*self.pos  - 1.0 )*specs.M # shift and stretch
        self.pos = self.pos.reshape((self.nwalkers, self.ndim)) # reshape
        
        # set the initial state of the PRNG
        self.state = np.random.get_state()
        
        # create the emcee sampler and let it burn in
        self.sam = mc.EnsembleSampler(self.nwalkers, self.ndim, kg.kriging, args=[ self.specs ])
        self.run_mcmc(self.burn)
        
        # tell samplers that they do not need to propagate the walkers
        self.walkerInd = 0
        
        # default decorrelation time. 
        self.decorTime = burn/2

    def sample_one(self):
        '''
        this method returns a single sample from the current posterior
        since we have a bunch of walkers, we return one of those 
        with every call to this method. if we have used them all 
        (i.e if walerInd == nwalkers) then we are forced to run 
        the emcee for some more time.
        '''
        
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
            

    def sample_batch(self):
        '''
        sample a bunch\ a batch
        return a new, unused batch of positions of the goodman
        & weare walkers
        * ``samples`` - np array of size (nwalkers , ndim)
        '''
        
        if self.walkerInd != 0:
            
            # run the MCMC to get new batch 
            self.run_mcmc(self.decorTime)
        
        # create a copy of the positions, so nothing unexpected happens if we parallelize
        samples = self.pos[:,:]
        
        # let them know we used this batch
        self.walkerInd = self.nwalkers
        
        return samples
        
    def run_mcmc(self, nsteps):
        '''
        run the emcee sampler for nsteps steps 
        :param nsteps:
            the number of steps we let the emcee sampler run
        '''   
        self.pos, self.prob, self.state, self.blobs = self.sam.run_mcmc(
                                                    self.pos, nsteps, self.state ) 
         

    def learn(self):
        '''
        if we want to choose another point to calculate the 
        log-likelihood at - this is the method we use
        '''
        
        # choose the next point 
        s = self.choose_point_heuristic()
        
        # add the new sample to our data set
        self.specs.add_point( s )
        
        # let them know we need a new batch before we sample
        self.walkerInd = self.nwalkers
    
    def choose_point_heuristic(self):
        '''
        use some heuristic, made up, ad hoc 
        criterion to choose next point
        '''           
        
        target = targets.minus_exp_krig_times_sig_square  
        bestValue = np.inf
        
        for i in range(self.noptimizers):
            
            # sample a starting point
            startPoint = self.sample_one()
            result = scipy.optimize.minimize(target, startPoint, args=(self.specs,), 
                                                                method='Powell')
            
            if result.fun < bestValue:
                bestValue = result.fun
                bestPoint = result.x
            
       
        bestPoint = bestPoint.reshape(self.ndim,)        
        return bestPoint