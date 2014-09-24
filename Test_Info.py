'''
Created on Sep 8, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import scipy.integrate as integrate
import math

import kernel.config as cfg
import kernel.info as nfo
import kernel.kriging as kg
import kernel.sampler as smp
import kernel.algorithm as alg
import kernel.truth as truth

class Test(unittest.TestCase):

 
    def setUp(self):
        '''
        create list of points, mainly
        '''
        # for reproducibility
        np.random.seed(1792)    
        X = [] 
        X.append( np.array([ 0, 1]) )
        X.append( np.array([ 1, 0]) )
        X.append( np.array([ 0,-1]) )
        X.append( np.array([-1, 0]) )
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

        # create the container object ...
        CFG = cfg.Config( truth.const )
         
        for x in self.X:
            CFG.addPoint(x)
 
        # ...set the characteristic distance....
        CFG.setR(1.0)
        CFG.M = 0.5
        CFG.setType( alg.AUGMENTED_COVARIANCE )
        result, _ = CFG.getNormalization()
        self.assertTrue(  np.allclose(result,math.exp(2))  )        
         
    def testGetNormalization2(self):
        '''
        we make sure the procedure we use for finding
        normalization constants works by comparing to scipy's
        integrate methods.
        '''
     
        # create the container object ...
        CFG = cfg.Config( truth.rosenbrock2D) # we don't care much bout the true LL
         
        # ...add the points to it ...
        for x in self.X:
            CFG.addPoint(x)
 
        # ...set the characteristic distance....
        CFG.setR(1.0)
        M = CFG.M = 5
        CFG.setType( alg.RASMUSSEN_WILLIAMS )
 
        MCresult, MCerr = CFG.getNormalization()
        f = lambda y,x:  math.exp(kg.kriging(np.array([y,x]),CFG)[0])
        Qresult, Qerr = integrate.dblquad( f, -M, M, lambda x: -M , lambda x: M)     
 
        # compare the two methods of integration
#         print("")     
#         print("Integral calculated using MC   = " + str(MCresult) + ", with error = " + str(MCerr)) 
#         print("Integral calculated using Quad = " + str(Qresult ) + ", with error = " + str(Qerr )) 
        
        self.assertTrue( abs( MCresult - Qresult) < MCerr + Qerr )
         
    def testI(self):
        '''
        test the function that calculates information gain
        '''
        
        # create the container object ...
        CFG = cfg.Config( truth.norm2D )
        
        # ...add the points to it ...
        for x in self.X:
            CFG.addPoint(x)

        # ...set the characteristic distance....
        CFG.setR(1.3)
        CFG.M = 2
        CFG.setType( alg.RASMUSSEN_WILLIAMS )
        #print(CFG.algType.getDescription())
        sampler = smp.Sampler( CFG )
        
        self.X.append( np.array([1.9,1.9]))
        for x in self.X: 
            print("Information gain for x = " + str(x) + " is " + str(nfo.I(x, sampler)))
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testgetNormalization']
    unittest.main()