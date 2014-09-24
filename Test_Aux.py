'''
Created on Sep 9, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np

import kernel.aux as aux

class Test(unittest.TestCase):


    def testCovVec(self):
        '''
        test the function that creates a covariance vector
        '''
        
        X = []
        X.append( np.array([1.0,2,3.3]) )
        X.append( np.array([2.0,4,5]) )
        X.append( np.array([-2,2,0]) )
        w = np.array( [ 2, 2 , 3])
        r = 1.0
        v = aux.cov_vec(X, w, r)
        #print(v)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCovVec']
    unittest.main()