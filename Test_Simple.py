'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest

import kernel.kriging as kg
import kernel.aux as aux
import numpy as np
import kernel.config as cfg

class Test(unittest.TestCase):
    ''' 
    giving the kriging procedure symmetric values, 
    we expect it to predict zero for their center of mass
    '''
    
    def setUp(self):
        '''
        set the test up. define points and their
        log likelihood
        '''
        
        x1 =  np.matrix( [ 0 , 1 ] )
        x2 =  np.matrix( [ 1 , 0 ] )
        x3 =  np.matrix( [ 0 ,-1 ] )
        x4 =  np.matrix( [-1 , 0 ] )
        f1 = np.array( [0] )
        f2 = np.array( [2] )
        f3 = np.array( [0] )
        f4 = np.array( [-2] )

        # create the container object ...
        CFG = cfg.Config()
        
        # ...add the points to it ...
        CFG.addPair(x1, f1)
        CFG.addPair(x2, f2)
        CFG.addPair(x3, f3)
        CFG.addPair(x4, f4)
        
        # ...set the characteristic distance....
        CFG.setR(1.0)
        
        # ...and create the matrices used for kriging
        CFG.setMatrices()
        
        # keep the container in the right scope of the whole test
        self.CFG = CFG
        
    def testSimple(self):
        
        # the center of mass of the x1,...,x4 is (0,0)
        s = np.array( [0, 0] )
        
        # kriging for this center ...
        b , c = kg.kriging(s, self.CFG)
        
        # ... should be the average of the values (f1,...,f4) at the points
        self.assertTrue(np.allclose( np.array( [0] ) , b ))
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()