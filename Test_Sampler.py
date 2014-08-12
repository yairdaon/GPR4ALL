'''
Created on Jun 15, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import kernel.sampler as smp
import numpy as np
import kernel.config as cfg
import kernel.truth as truth


class Test(unittest.TestCase):
    '''
    we test the sampler.py module. 
    '''


    def setUp(self):
        '''
        set up the test. this means creating the container object 
        that holds all the data, settings, variables and flags 
        required for a successful run
        '''
        
        # for reproducibility purposes 
        np.random.seed(5012)
        
        # the size of the box outside of which we have zero probability
        M = 2.0
        
        # the length scale of the covariance function. this is a hyper parameter.
        r = 1.0
        
        # create and populate the container object:
        CFG = cfg.Config() # call the constructor...
        CFG.addPair(np.array ( [ 1.0, 1.0, 1.0 ]), np.array( [2.45])) #...add data...  
        CFG.setR(r) # ...set the length scale hyper parameter...
        CFG.setM(M) # ...set the box size...
        CFG.setMatrices() # ...calcualte the matrices neede for kriging...
        CFG.LL = truth.norm2D
        self.CFG = CFG # ... and keep the container in the scope of the test.

        self.sampler = smp.Sampler( CFG )
        
    def testSampler1(self):
        '''
        here we sample from the sampler. We choose to incorporate
        the sampled data into our data set. we set the appropirate variable to True
        (see below) although we do not need to - the container object chooses
        this option by default
        '''
                             
        # choose to add samples to data set
        self.CFG.setAddSamplesToDataSet( True )
                                                                                                                                      
        # take three samples
        self.sampler.sample()
        self.sampler.sample()
        self.sampler.sample()
                             
        self.assertEqual(len( self.CFG.X ) , 4 )
        self.assertEqual( len(self.CFG.X[0]) , len(self.CFG.X[1]) )             
        self.assertEqual( len(self.CFG.X[0]) , len(self.CFG.X[2]) )   
        self.assertEqual( len(self.CFG.X[0]) , len(self.CFG.X[3]) )             
          
                    
    def testSampler2(self):
        '''
        here we sample from the sampler. We choose NOT to incorporate
        the sampled data into our data set. we set the appropirate variable to False
        '''
        
        # choose to add samples to data set
        self.CFG.setAddSamplesToDataSet( False )
        
        # take four samples
        self.sampler.sample()
        self.sampler.sample()
        self.sampler.sample()
        self.sampler.sample()
        
        # take another one and incorporate it
        self.CFG.setAddSamplesToDataSet( True )
        self.sampler.sample()
        
        self.assertEqual(len( self.CFG.X ) , 2)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()