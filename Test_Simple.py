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
import kernel.point as pt
import kernel.aux as aux

class Test(unittest.TestCase):
    '''
    assorted simple tests. These tests should never fail.
    If they do, something really bad has happened.
    '''
    
    def testSymmetric(self):
        ''' 
        giving the kriging procedure symmetric values, 
        we expect it to predict zero for their center of mass
        '''
        
        X = []
        X.append(np.array( [ 0 , 1 ] ))
        X.append(np.array( [ 1 , 0 ] ))
        X.append(np.array( [ 0 ,-1 ] ))
        X.append(np.array( [-1 , 0 ] ))
        
        F = []
        F.append(np.array( [ 2] ))
        F.append(np.array( [-1] ))
        F.append(np.array( [-2] ))
        F.append(np.array( [ 1] ))

        # create the container  (the true LL is zero since it is of no importance)
        specs = cot.Container( truth.zero , r=1.0 )
    
        for i in range(len(X)):
            specs.add_pair(X[i], F[i])
        
        # the center of mass of the x1,...,x4 is (0,0)
        s = np.array( [0, 0] )
        
        # kriging for this center ...
        b = kg.kriging(s, specs)[0]
        
        # ... should be zero, by symmetry
        self.assertAlmostEqual(b, 0, 14) # 15 is the number of decimal digits we consider
     
    def testUniform(self):
        ''' 
        test kriging by makin sure the procedure outputs
        a constant 0 when it is given constant 0
        input
        '''
        
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
   
        
    def testPointWithError(self):
        '''
        we want PointWithError to behave just like a numpy array
        '''
        
        # instantiate a point
        p = pt.PointWithError( [ 1. , 2. , 3 ] ) 
        
        # is it an instance?
        self.assertTrue( isinstance( p , pt.PointWithError ))
        
        # default error should be 0.0
        self.assertTrue( p.error == 0.0 )
        
        # reference using array indexing
        self.assertTrue( p[0] == 1.0 )
        
        # length should be 3
        self.assertTrue( len(p) == 3 )
        
        # can we set the variable error?
        p.error = 3
        self.assertTrue( p.error == 3.0 )
        
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
        self.assertEqual(2.0, p1.error ) 

        p2 = pt.PointWithError( [ 1. , 2. , 3 ] , 2.5)
        self.assertEqual(2.5, p2.error ) 
        
        
        
        

    def testCovVec(self):
        '''
        test the function that creates a covariance vector
        in the module aux
        '''
        
        X = []
        X.append( np.array([1.0,2,3.3]) )
        X.append( np.array([2.0,4,5]) )
        X.append( np.array([-2,2,0]) )
        w = np.array( [ 2, 2 , 3])
        r = 1.0
        d = 1.0
        v = aux.cov_vec(X, w, r, d)
        #print(v)
    
    def testTychonoffSimple(self):
        '''
        test the linear solver we use - This solver 
        uses tychonoff regularization. here we make sure the
        tychonoff regularization behaves reasonably for 
        a trivial example
        ''' 

        
        A = np.array( [[ 1, 2] , [ 3 ,4] ])
        U,  S,  V = np.linalg.svd( A, full_matrices= True, compute_uv=True)
        b = np.array([1 , 1])
        reg = 100*np.finfo(np.float).eps

        svdSol = aux.tychonoff_solver( U, S, V, b, reg )
        trueSol = np.array([ -1 , 1])
        self.assertTrue(np.allclose(svdSol, trueSol))

    def testTychonoffRandom(self):
        '''
        test the linear solver we use - This solver 
        uses tychonoff regularization. here we make sure the
        tychonoff regularization behaves reasonably for 
        well conditioned matrices
        ''' 

        
        # for reproducibility purposes
        np.random.seed(1792)
        
        # create some random matrix
        A = np.random.rand(50,50)
        
        # get SVD
        U,  S,  V = np.linalg.svd( A, full_matrices= True, compute_uv=True)
        
        # create three random target vectors
        b1 = np.array( np.random.rand(1,50) )
        b2 = np.array( np.random.rand(50,1) )
        b3 = np.array( np.random.rand(50) )
        
        # the regularization factor we ususlly use in the solver
        reg = 100*np.finfo(np.float).eps
        
        # solve using our solver
        x1 = aux.tychonoff_solver( U, S, V, b1, reg )
        x2 = aux.tychonoff_solver( U, S, V, b2, reg )
        x3 = aux.tychonoff_solver( U, S, V, b3, reg )
        
        # solve using standard package
        y1 = np.linalg.solve(A, np.ravel(b1))
        y2 = np.linalg.solve(A, np.ravel(b2))
        y3 = np.linalg.solve(A, np.ravel(b3))
        
        # compare solutions
        self.assertTrue(np.allclose(x1, y1))
        self.assertTrue(np.allclose(x2, y2))
        self.assertTrue(np.allclose(x3, y3))
        
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()