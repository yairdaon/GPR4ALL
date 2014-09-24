'''
Created on Jun 11, 2014

@author: daon
'''
import unittest
import numpy as np
import kernel.aux as aux

class Test(unittest.TestCase):
    '''
    we use a solver for linear equations. This solver 
    uses tychonoff regularization. here we make sure the
    tychonoff regularization behaves reasonably for 
    well conditioned matrices
    ''' 


    def testTychonoffRando(self):
        ''' 
        test the solver we use
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
        
    def testTychonoffSimple(self):
        '''
        check that the solver works for a trivial example
        '''
        
        A = np.array( [[ 1, 2] , [ 3 ,4] ])
        U,  S,  V = np.linalg.svd( A, full_matrices= True, compute_uv=True)
        b = np.array([1 , 1])
        reg = 100*np.finfo(np.float).eps

        svdSol = aux.tychonoff_solver( U, S, V, b, reg )
        trueSol = np.array([ -1 , 1])
        self.assertTrue(np.allclose(svdSol, trueSol))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testTyychonoff']
    unittest.main()
    
    
    