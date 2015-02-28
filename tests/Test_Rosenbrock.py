'''
Created on Nov 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np

import emcee as mc

import gpr4all.rosenbrock as rose

class Test(unittest.TestCase):

    def testNormalization(self):
        '''
        show the integral of the rosenbrock function we use
        is 1.
        '''
        M = 10
        delta = 0.005
        a  = np.arange(-M, M, delta)
        X, Y = np.meshgrid(a , a)         # create two meshgrid
        points    = np.asarray( [ np.ravel(X) , np.ravel(Y)])
        locRos    = rose.rosenbrock_2D
        rosen      = locRos(points, True)
        xpRosen    = np.exp(rosen)
        Zphi       = np.sum( xpRosen ) # no delta**2!! see below
        
        self.assertAlmostEqual(1.0, Zphi*delta*delta, 2)
 
 
    def testRosenbrockSampler(self):
        ''' 
        show we can calculate and sample from the 
        rosenbrock function using independent normals
        '''
          
        # we calculated this  analytically
        nwalkers=100
        burn=100
        nsteps = burn*10000
        indSamp = 1000000
           
        # the initial set of positions
        pos = np.random.rand(2 * nwalkers) #choose U[0,1]
        pos = ( 2*pos  - 1.0 ) # shift and stretch
        pos = pos.reshape((nwalkers, 2)) # reshape
           
           
        # create the emcee sampler and let it burn in
        sam = mc.EnsembleSampler(nwalkers, 2, rose.rosenbrock_2D)
           
        # burn in and then run the sampler
        pos , _ , _  = sam.run_mcmc(pos, burn)
        sam.reset()
        sam.run_mcmc(pos, nsteps)
  
        walkersMean = np.average(sam.flatchain, axis = 0)
              
        samples =  rose.sample_rosenbrock(indSamp)
        independentMean = np.mean(samples,0)
          
        diff = independentMean - walkersMean
        print( "ind samples mean = " + str(independentMean) + " MCMC mean = "
                + str(walkersMean) +  " difference norm = " + str(np.linalg.norm(diff))  )
          
        self.assertAlmostEqual(np.linalg.norm(diff), 0, 2)


        

    
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRose']
    unittest.main()
