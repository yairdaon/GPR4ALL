'''
Created on Jul 26, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest

import numpy as np
import kernel.point as pt

class Test(unittest.TestCase):
    '''
    tests whether the class PointWithError behaves like we want it to
    '''

    def testPointWithError(self):
        '''
        we want PointWithError to behave just like a numpy array
        '''
        
        # instantiate a point
        p = pt.PointWithError( [ 1. , 2. , 3 ] ) 
        
        # is it an instance?
        self.assertTrue( isinstance( p , pt.PointWithError ))
        
        # default error should be 0.0
        self.assertTrue( p.get_error() == 0.0 )
        
        # reference using array indexing
        self.assertTrue( p[0] == 1.0 )
        
        # length should be 3
        self.assertTrue( len(p) == 3 )
        
        # can we set the variable error?
        p.set_error(3)
        self.assertTrue( p.get_error() == 3.0 )
        
        # can we calculate its norm?
        q = np.array([1,2,3])
        self.assertTrue(    np.linalg.norm(p)  ==  np.linalg.norm(q)   )
        
        # does it equal a regular numpy array?
        self.assertTrue( np.all( p == q ))
        
        # can we add?
        self.assertTrue(  (p+q)[1] == 4 ) 
        
        # test matrix multiplication
        A = np.random.random((3, 3))
        self.assertTrue(  np.all( np.dot(A,p)==np.dot(A,q) )  )
        
        # can we call the constructor and set the error in one line?
        p1 = pt.PointWithError( [ 0. , 0. , 0 ] , error = 2.0)
        self.assertEqual(2.0, p1.get_error()) 

        p2 = pt.PointWithError( [ 1. , 2. , 3 ] , 2.5)
        self.assertEqual(2.5, p2.get_error()) 
                 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPoint']
    unittest.main()