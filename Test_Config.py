'''
Created on Sep 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import math
import numpy as np
import scipy.integrate as integrate

import kernel.container as cot
import kernel.algorithm as alg
import kernel.truth as truth
import kernel.kriging as kg


class Test(unittest.TestCase):


    def setUp(self):
        '''
        create list of points, mainly
        '''
        # for reproducibility
        np.random.seed(162)    
        X = [] 
        X.append( np.array([ 0, 1.7]) )
        X.append( np.array([ 1.7, 0]) )
        X.append( np.array([ 0,-1.7]) )
        X.append( np.array([-1.7, 0]) )
        self.X = X
        
    def testGetNormalization1(self):
        '''
        we make sure the procedure we use for finding
        normalization constants works in the simplest
        of cases. Note that we must use the specified
        algorithm type, since the RW algorithm tends to
        zero even when given constant data, while the
        algorithm from 
        '''

        # create and populate the container object ...
        specs = cot.Container( truth.const , r=1.0 , M=2 , algType=alg.AUGMENTED_COVARIANCE )
        for x in self.X:
            specs.add_point(x)

        ownResult, _ = specs.get_normalization(test = True)
        trueResult = math.exp(2)*(2*specs.M)**specs.X[0].size
        self.assertTrue(np.allclose( ownResult , trueResult ))        
         
    def testGetNormalization2(self):
        '''
        we make sure the procedure we use for finding
        normalization constants works by comparing to scipy's
        integrate methods.
        '''
     
        # create the container object ...
        specs = cot.Container( truth.rosenbrock_2D , r = 1.0, M=5 ) # we don't care much bout the true LL
         
        # ...add the points to it ...
        for x in self.X:
            specs.add_point(x)
 
        # ...set the characteristic distance....
        M = specs.M 
 
        ownResult, ownErr = specs.get_normalization(test = True)
        f = lambda y,x:  math.exp(kg.kriging(np.array([y,x]),specs)[0])
        Qresult, Qerr = integrate.dblquad( f, -M, M, lambda x: -M , lambda x: M)     
        
        self.assertTrue( abs( ownResult - Qresult) < ownErr + Qerr )
         


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()