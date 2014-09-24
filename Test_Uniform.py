'''
Created on Jun 5, 2014

@author: daon
'''
import unittest 
import kernel.container as cot
import numpy as np
import kernel.kriging as kg
import kernel.truth as truth

class TestKriging(unittest.TestCase):
    ''' 
    test kriging by makin sure the procedure outputs
    a constant 0 when it is given constant 0
    input
    '''
    def testUniform(self):
        
        
        specs = cot.Container(truth.zero , r=1.0, M=25.0)
        
        # create locations where values of log 
        # likelihood are known
        X = []
        X.append( np.array( [ 0.5] ) )
        X.append( np.array( [ 1.0] ) )
        X.append( np.array( [ 1.5] ) )
        X.append( np.array( [ 1.25]) )
        
        # set all these known values to be the same
        for x in X:
            specs.add_point( x )        

        self.assertTrue(    np.allclose( np.array([0]) , kg.kriging(np.array([20.0]),specs )[0] )      )


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()