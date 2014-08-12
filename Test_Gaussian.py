'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import os
import numpy as np

import kernel.sampler as smp
import kernel.config as cfg
import kernel.type as type
import kernel.kriging as kg
import kernel.truth as truth

import pylab as P
import matplotlib.pyplot as plt




class Test(unittest.TestCase):
    '''
    we take the true log likelihood to be that of a gaussian.
    we sample from the gaussian some time, calculating true log 
    likelihood in each step and adding the new data to our data set. 
    Then, we sample from the posterior, with the hope that we've 
    resolved the gaussin structure. To see this, we plot a
    histogram
    '''


    def setUp(self):
        '''
        here we set the test up
        this means putting all required information 
        inside the container object
        '''
        # create target directory
        os.system("mkdir graphics")

        # set seed for reproducibility
        np.random.seed(89)
        
        # create an instance of the container
        self.CFG = cfg.Config()
        
        # set the true log-likelihood to be a gaussian
        self.CFG.setLL(truth.gaussian1D)
        f = self.CFG.LL # call it f for short

        # use one initial point        
        p = np.array( [0.0] )
        self.CFG.addPair(p, f(p))
        
        # parameters of the run:
                
        M = 10.0 # outside the box of size M the probability is zero
        self.CFG.setM(M)
        
        r = 1.3 # the typical length scale of the kriging. a hyper parameter
        self.CFG.setR(r)

        # take and incorporate to data an initial sample:        
        self.CFG.setAddSamplesToDataSet( True ) #... tell the container it does so...
        
        # create the sampler
        self.sampler = smp.Sampler ( self.CFG )
        
        k =  45 # ...decide how many initial points we take to resolve the log-likelihood
        for j in range(0,k): 
            print( "Initial samples " + str(j+1) + " of " + str(k))
            self.sampler.sample() # ... sample, incorporate into data set, repeat k times.
        
        
        # plot kriged LL
                
        # allocating memory
        x = np.arange(-10, 10, 0.05)
        n = len(x)
        f = np.zeros( n )
        
        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] ,self.CFG) 
            f[j] = v[0] # set the interpolant
            
        
        # do all the plotting here
        curve1  = plt.plot(x, f, label = "kriged value")
        curve2  = plt.plot( self.CFG.X, self.CFG.F, 'bo', label = "sampled points ")
        curve3  = plt.plot(x, truth.gaussian1D(x), label = "true log-likelihood")
        
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Kriging with bounds using " + self.CFG.algType.getDescription() )
        plt.savefig("graphics/Test_Gaussian: kriged LL")
        plt.close()
   
            
        
    
    def testGaussian(self):
        '''
        take samples from posterior (log) likelihood
        and plot in histogram
        '''
        
        # take 2000 samples. We DO NOT incorporate these into the data set
        n =  2000
        
        # allocate memory for the data
        samples = np.zeros(n)
        
        # do not incorporate newly sampled points in data set
        self.CFG.setAddSamplesToDataSet( False )
        
        # sample n points from the kriged posterior log likelihood
        print( "taking " + str(n) +  " samples from the posterior")
        for i in range(n):
            if (i % 50 == 0):
                print( "Gaussian test, posterior sample " + str(i+1) + " of " + str(n))
            samples[i] = self.sampler.sample() # ... sample, incorporate into data set, repeat k times.

        
        # do all the plotting business, copied from pylab's examples
        P.figure()
        # the histogram of the data with histtype='step'
        nn, bins, patches = P.hist(samples, 20, normed=1, histtype='stepfilled')
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        P.title(str(n) + " samples from the kriged (posterior) log-likelihood interpolating a Gaussian")
        P.savefig("graphics/Test_Gaussian: Posterior Histogram")
        P.close()
        
    def tearDown(self):
        '''
        plot the log likelihood the sampler used with the sampled points
        and the true log likelihood
        '''
        
        # allocating memory
        x = np.arange(-10, 10, 0.05)
        n = len(x)
        f = np.zeros( n )
        
        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] ,self.CFG) 
            f[j] = v[0] # set the interpolant
            
        
        # do all the plotting here
        curve1  = plt.plot(x, f, label = "kriged value")
        curve2  = plt.plot( self.CFG.X, self.CFG.F, 'bo', label = "sampled points ")
        curve3  = plt.plot(x, truth.gaussian1D(x), label = "true log-likelihood")
        
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Kriging with bounds using " + self.CFG.algType.getDescription() )
        plt.savefig("graphics/Test_Gaussian: kriged LL")
        plt.close()



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()