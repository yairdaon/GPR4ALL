'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest

import kernel.kriging as kg
import numpy as np
import kernel.container as cot
import kernel.truth as truth

class Test(unittest.TestCase):
    ''' 
    giving the kriging procedure symmetric values, 
    we expect it to predict zero for their center of mass
    '''
    
    def testSimple(self):
        
        x1 =  np.matrix( [ 0 , 1 ] )
        x2 =  np.matrix( [ 1 , 0 ] )
        x3 =  np.matrix( [ 0 ,-1 ] )
        x4 =  np.matrix( [-1 , 0 ] )
        f1 = np.array( [0] )
        f2 = np.array( [2] )
        f3 = np.array( [0] )
        f4 = np.array( [-2] )

        # create the container  (the true LL is zero since it is of no importance)
        specs = cot.Container( truth.zero , r=1.0)
        
        # ...add the points to it ...
        specs.add_pair(x1, f1)
        specs.add_pair(x2, f2)
        specs.add_pair(x3, f3)
        specs.add_pair(x4, f4)
    
        # the center of mass of the x1,...,x4 is (0,0)
        s = np.array( [0, 0] )
        
        # kriging for this center ...
        b , _ = kg.kriging(s, specs)
        
        # ... should be the average of the values (f1,...,f4) at the points
        self.assertTrue(np.allclose( np.array( [0] ) , b ))
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()