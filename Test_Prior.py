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
import kernel.container as cot
import kernel.algorithm as alg


class Test(unittest.TestCase):

    def testPrior(self):
        
        # for reproducibility purposes
        np.random.seed( 1243 )
        
        # create the container object
        specs = cot.Container( truth.double_well_1D , M=3)
        
        # set prior loglikelihood to exponential
        specs.set_prior(   lambda x: -np.linalg.norm(x)**2  )
       
        # quick setup
        specs.quick_setup(1)
        
        # create sampler...
        sampler = smp.Sampler ( specs )
        k =  21 # ...decide how many initial points we take to resolve the log-likelihood
        for j in range(k): 
            print( "Sample " + str(j+1) + " of " + str(k))
            sampler.learn() # ... sample, incorporate into data set, repeat k times.
   
        # allocating memory
        x = np.arange(-specs.M, specs.M, 0.05)
        n = len(x)
        f = np.zeros( n )
        true = np.zeros( n )
        prior = np.ones( n )

        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] , specs) 
            f[j] =  (v[0]) # set the interpolant
            prior[j] = specs.prior(x[j])   # set the limiting curve
            true[j]  = specs.trueLL( x[j])  
        
        #move to normal, non-exponential, scale
        fExp = np.exp(f)
        priorExp = np.exp(prior)
        trueExp = np.exp(true)
        samplesExp = np.exp(np.asarray( specs.F ))
        X =  np.asarray( specs.X )
        
        
        # first plot
        curve1  = plt.plot(x, f, label = "kriged LL")
        curve2  = plt.plot(x, true, label = "true LL")
        curve3  = plt.plot(x, prior, label = "prior log-likelihood")
        plt.plot(  specs.X ,    specs.F  , 'bo', label = "sampled points ")
        
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPrior']
    unittest.main()