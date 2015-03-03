'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import pylab as P
import matplotlib.pyplot as plt

import gpr4all.sampler as smp
import gpr4all.container as cot
import gpr4all.truth as truth




class Test(unittest.TestCase):
    '''
    we take the true log likelihood to be that of a gaussian.
    we sample from the gaussian some time, calculating true log 
    likelihood in each step and adding the new data to our data set. 
    Then, we sample from the posterior, with the hope that we've 
    resolved the gaussin structure. To see this, we plot a
    histogram
    '''

    def testGaussian(self):
        '''
        take samples from posterior (log) likelihood
        and plot in histogram
        '''
        
        # set seed for reproducibility
        np.random.seed(89)
        
        # allocating memory
        x = np.arange(-10, 10, 0.05)
        n = len(x)
        f = np.zeros( n )
        
        # create an instance of the container
        specs = cot.Container(truth.gaussian_1D)
        specs.set_prior( lambda x: -np.linalg.norm(x)**2 , lambda x: -2*x)

        # use one initial point        
        specs.add_point( np.array([0.0]) )
       
        # create the sampler
        sampler = smp.Sampler ( specs )
        
        k =  11 # ...decide how many initial points we take to resolve the log-likelihood
        for j in range(0,k): 
            print( "Initial samples " + str(j+1) + " of " + str(k))
            sampler.learn() # ... sample, incorporate into data set, repeat k times.
        
        
        # plot kriged LL
        
        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging 
            f[j] =  specs.kriging(x[j], False, False)
            
        
        # do all the plotting here
        curve1  = plt.plot(x, f, label = "kriged value")
        plt.plot( specs.X, specs.F, 'bo', label = "sampled points ")
        plt.plot(x, specs.trueLL(x), label = "true log-likelihood")
        
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Kriged Log Likelihood")
        plt.savefig("graphics/Test_Gaussian: kriged LL")
        plt.close()
        
        # Done plotting kriged LL, now sample:
        
        # take some samples. We DO NOT incorporate these into the data set
        numSamples =  2000
        
        # allocate memory for the data
        samples   = np.zeros(numSamples)
        batchSize = sampler.nwalkers
        batch     = np.zeros(batchSize)
        
        # sample n points from the kriged posterior log likelihood
        print( "taking " + str(numSamples) +  " samples from the posterior:")
        for j in range(numSamples/batchSize):
            
            # get a batch of the current walkers
            batch = sampler.sample_batch()
            
            # iterate over this batch
            for i in range(batchSize):  
                
                # add every walker to the samples and print 
                samples[j*batchSize + i] = batch[i,:] 
            
            print( "Sample batch from psterior: " + str(j*batchSize) 
                                    + " to " + str((j+1)*batchSize) + " of "+ str(numSamples))

        # do all the plotting business, copied from pylab's examples
        P.figure()
        # the histogram of the data with histtype='step'
        _, _, patches = P.hist(samples, 30, normed=1, histtype='stepfilled')
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        P.title(str(numSamples) + " samples from the posterior likelihood interpolating a Gaussian")
        P.savefig("graphics/Test_Gaussian: Posterior Histogram")
        P.close()
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
