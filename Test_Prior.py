'''
Created on Aug 2, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import matplotlib.pyplot as plt

import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import kernel.config as cfg
#import kernel.type as type


class Test(unittest.TestCase):


    def setUp(self):
        
        # for reproducibility purposes
        np.random.seed( 1243 )
        
        # create the container object
        CFG = cfg.Config()
        
        # set plot bounds
        CFG.M = 2
        
        # set prior loglikelihood to exponential
        prior = lambda x: -np.linalg.norm(x)**2
        CFG.setPrior(prior)
        
        # set true LL
        likelihood = truth.doubleWell1D
        CFG.setLL(likelihood)
        
        # use RW's algorithm
        CFG.setType(type.RASMUSSEN_WILLIAMS)
       
        # quick setup
        CFG.quickSetup(1)
        
        # create sampler...
        self.sampler = smp.Sampler ( CFG )
        k =  7 # ...decide how many initial points we take to resolve the log-likelihood
        for j in range(0,k): 
            print( "Initial samples " + str(j+1) + " of " + str(k))
            self.sampler.sample() # ... sample, incorporate into data set, repeat k times.
            
        self.CFG = CFG
     
    def testPrior(self):
        
        # allocating memory
        x = np.arange(-self.CFG.M, self.CFG.M, 0.05)
        n = len(x)
        f = np.zeros( n )
        true = np.zeros( n )
        prior = np.ones( n )

        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] , self.CFG) 
            f[j] =  (v[0]) # set the interpolant
            prior[j] = self.CFG.prior(x[j])   # set the limiting curve
            true[j]  = self.CFG.LL(   x[j])  
        
        #move to normal, non-exponential, scale
        fExp = np.exp(f)
        priorExp = np.exp(prior)
        trueExp = np.exp(true)
        samplesExp = np.exp(np.asarray( self.CFG.F ))
        X =  np.asarray( self.CFG.X )
        
        
        # first plot
        curve1  = plt.plot(x, f, label = "kriged LL")
        curve2  = plt.plot(x, true, label = "true LL")
        curve3  = plt.plot(x, prior, label = "prior log-likelihood")
        plt.plot(  self.CFG.X ,    self.CFG.F  , 'bo', label = "sampled points ")
        
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.setp( curve3, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Kriged log likelihood")
        plt.savefig("graphics/Test_Prior: Kriged LL")
        plt.close()
        
        # second plot
        curve4  = plt.plot(x, fExp, label = "exp(kriged LL)")
        curve5  = plt.plot(x, trueExp, label = "(unnormalized) likelihood")
        curve6  = plt.plot(x, priorExp, label = "exp(prior log-likelihood)")
        plt.plot(  X ,   samplesExp  , 'bo', label = "sampled points ")
        
        plt.setp( curve4, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve5, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )        
        plt.setp( curve6, 'linewidth', 3.0, 'color', 'b', 'alpha', .5 )
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Interpolated Likelihood")
        plt.savefig("graphics/Test_Prior: Interpolated Likelihood")
        plt.close()

        
        
        plt.close()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPrior']
    unittest.main()