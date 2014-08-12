'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import kernel.config as cfg
import kernel.truth as truth
import kernel.sampler as smp

class Test(unittest.TestCase):
    '''
    make sure we can reproduce results.
    try commenting out one of the seeds and see how this fails
    '''


    def setUp(self):
        
        np.random.seed(127)
        
        # parameters:
        r = 1.0 # length scale hyper parameter
        M = 10.0 # size of box with non-vanishing probability
        f = truth.trueLL # the real log-likelihood
        
        # start the algorithm using these points
        StartPoints = []
        StartPoints.append( np.array( [ -1.5 ] ) )
        StartPoints.append( np.array( [  1.5 ] ) )

        # creating the container object...
        CFG = cfg.Config()
        for point in StartPoints:
            CFG.addPair( point, f(point)) # ...populating it with (point,value) pairs...
        CFG.setR(r) # ...setting the hyper parametr r...
        CFG.setM(M) # ...setting the box size M...
        CFG.setMatrices() #... set the matrices we use for kriging...
        
        sampler = smp.Sampler(CFG)
        sampler.sample() # ... and sample two points using the given seed
        sampler.sample() # note: the sampler adds these points to the container a on its own
        
        # now we sample
        self.x = sampler.sample()
        
    def tearDown(self):
        pass


    def testReproducibility(self):
        '''
        tests that using the same seed, we can reproduce results.
        note: this is what we did in the setUp method above EXCEPT
        the last two lines.
        '''
        
        np.random.seed(127)
        r = 1.0
        M = 10.0
        f = truth.trueLL
        #    Start the algorithm using these points
        StartPoints = []
        StartPoints.append( np.array( [ -1.5 ] ) )
        StartPoints.append( np.array( [  1.5 ] ) )

        #     Initializations of the algorithm
        CFG = cfg.Config()
        for point in StartPoints:
            CFG.addPair(point, f(point))
        CFG.setR(r)
        CFG.setM(M)
        CFG.setMatrices()
        
        sampler = smp.Sampler(CFG)
        sampler.sample()
        sampler.sample()
        
        # now we put the sample in y
        self.y = sampler.sample()   
        
        # and compare
        self.assertEqual(self.x, self.y)  

    def testReproducibilityFails(self):
            '''
            tests that using different seeds, we cannot expect to
            reproduce results. 
            note: this is what we did in the testReproducibility method 
            above EXCEPT for the first line and the last.
            '''
            
            # Seed is commented, so we cannot expect reproducibility
            #np.random.seed(127)
            r = 1.0
            M = 10.0
            f = truth.trueLL
            #    Start the algorithm using these points
            StartPoints = []
            StartPoints.append( np.array( [ -1.5 ] ) )
            StartPoints.append( np.array( [  1.5 ] ) )
    
            #     Initializations of the algorithm
            CFG = cfg.Config()
            for point in StartPoints:
                CFG.addPair(point, f(point))
            CFG.setR(r)
            CFG.setM(M)
            CFG.setMatrices()
            
            sampler = smp.Sampler(CFG)
            sampler.sample()
            sampler.sample()
            
            # now we put the sample in y
            self.y = sampler.sample()   
            
            # and compare. they should be different!
            self.assertTrue( not np.array_equal(self.x, self.y) )        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()