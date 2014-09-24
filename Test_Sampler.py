'''
Created on Jun 15, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import kernel.sampler as smp
import numpy as np
import kernel.container as cot
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
        
        # create and populate the container object:
        specs = cot.Container( truth.norm_2D , r=1.0, M =2.0) # call the constructor...
        specs.add_point( np.array([1.0,1.0,1.0]) ) #...add data...  
        
        self.specs = specs # ... and keep the container in the scope of the test.

        self.sampler = smp.Sampler( specs )
        
    def testSampler1(self):
        '''
        here we sample from the sampler. We choose to learn from
        the samples.
        '''
                                                                              
        # learn two new points
        self.sampler.learn()
        self.sampler.learn()
                              
        self.assertEqual( len( self.specs.X )  , 3 )
        self.assertEqual( len(self.specs.X[0]) , len(self.specs.X[1]) )             
        self.assertEqual( len(self.specs.X[0]) , len(self.specs.X[2]) )   
           
                     
    def testSampler2(self):
        '''
        here we sample from the sampler. We choose NOT to incorporate
        the sampled data into our data set. we set the appropirate variable to False
        '''
        
        # learn three new points
        self.sampler.learn()
        self.sampler.learn()
        self.sampler.learn()
        
        for i in range(self.sampler.nwalkers):
            self.sampler.sample_one()
         
        self.assertEqual(len( self.specs.X ) , 4)
         
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()