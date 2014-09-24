'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import kernel.container as cot
import kernel.truth as truth
import kernel.sampler as smp

class Test(unittest.TestCase):
    '''
    make sure we can reproduce results.
    try commenting out one of the seeds and see how this fails
    '''


    def setUp(self):
        
        np.random.seed(127)

        # start the algorithm using these points
        StartPoints = []
        StartPoints.append( np.array([-1.5]) )
        StartPoints.append( np.array([ 1.5]) )

        # creating the container object...
        specs = cot.Container( truth.sin_1D , r=1.0)
        for x in StartPoints:
            specs.add_point( x ) # ...populating it with (point,value) pairs...
        
        sampler = smp.Sampler( specs )
        sampler.learn() # ... and sample two points using the given seed
        sampler.learn() # note: the sampler adds these points to the container a on its own
        
        # now we sample
        self.x = sampler.sample_one()
    
    def testReproducibility(self):
        '''
        tests that using the same seed, we can reproduce results.
        note: this is what we did in the setUp method above EXCEPT
        the last two lines.
        '''
        
        np.random.seed(127)
        
        #    Start the algorithm using these points
        StartPoints = []
        StartPoints.append( np.array( [ -1.5 ] ) )
        StartPoints.append( np.array( [  1.5 ] ) )

        #     Initializations of the algorithm
        specs = cot.Container( truth.sin_1D , r=1.0)
        for point in StartPoints:
            specs.add_point( point )
        
        sampler = smp.Sampler(specs)
        sampler.learn()
        sampler.learn()
        
        # now we put the sample in y
        self.y = sampler.sample_one()   
        
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
            
            
            #    Start the algorithm using these points
            StartPoints = []
            StartPoints.append( np.array( [ -1.5 ] ) )
            StartPoints.append( np.array( [  1.5 ] ) )
    
            #     Initializations of the algorithm
            specs = cot.Container(truth.sin_1D , r=1.0)
            for point in StartPoints:
                specs.add_point(point)
            
            sampler = smp.Sampler(specs)
            sampler.learn()
            sampler.learn()
            
            # now we put the sample in y
            self.y = sampler.sample_one()   
            
            # and compare. they should be different!
            self.assertTrue( not np.array_equal(self.x, self.y) )        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()