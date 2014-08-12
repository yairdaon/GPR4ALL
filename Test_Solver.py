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


    def setUp(self):
        ''' 
        set the entire ingredients needed for the test
        '''
        
        # for reproducibility purposes
        np.random.seed(1792)
        
        # create some random matrix
        self.A = np.random.rand(50,50)
        
        # get SVD
        self.U, self.S, self.V = np.linalg.svd(self.A, full_matrices= True, compute_uv=True)
        
        # create three random target vectors
        self.b1 = np.array( np.random.rand(1,50) )
        self.b2 = np.array( np.random.rand(50,1) )
        self.b3 = np.array( np.random.rand(50) )
        
        # the regularization factor we ususlly use in the solver
        self.reg = 100*np.finfo(np.float).eps
        

    def testTychonoff(self):
        
        # solve using our solver
        x1 = aux.tychonoffSvdSolver( self.U, self.S, self.V, self.b1, self.reg )
        x2 = aux.tychonoffSvdSolver( self.U, self.S, self.V, self.b2, self.reg )
        x3 = aux.tychonoffSvdSolver( self.U, self.S, self.V, self.b3, self.reg )
        
        # solve using standard package
        y1 = np.linalg.solve(self.A, np.ravel(self.b1))
        y2 = np.linalg.solve(self.A, np.ravel(self.b2))
        y3 = np.linalg.solve(self.A, np.ravel(self.b3))
        
        # compare solutions
        self.assertTrue(np.allclose(x1, y1))
        self.assertTrue(np.allclose(x2, y2))
        self.assertTrue(np.allclose(x3, y3))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testTyychonoff']
    unittest.main()
    
    
    