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
import kernel.aux as aux
import kernel.sampler as smp

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

        # create the container. we don'te use the true LL
        specs = cot.Container( truth.const )
        
        # make sure the prior doesn't bother us
        specs.set_prior( lambda x: 0.0 )

    
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
        test kriging by making sure the procedure outputs
        a constant 0 when it is given constant 0
        input
        '''
        
        specs = cot.Container( truth.zero )
        specs.set_prior( lambda x: 0.0)
        
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
        
        
        
            
        
    def testSampler(self):
        '''
        here we sample from the sampler. We choose to learn from
        the samples.
        '''

        # creating the container object...
        specs = cot.Container( truth.rosenbrock_2D )
        specs.set_prior( lambda x: -np.linalg.norm(x)**4)

        specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
        specs.add_point( np.array( [  1.5 ,-2.0  ] )  )
    
        sampler = smp.Sampler( specs )
        sampler.learn()
        sampler.learn()
        self.specs = specs
        self.sampler = sampler                                                 
                              
        self.assertEqual( len( self.specs.X )  , 4 )
        self.assertEqual( len(self.specs.X[0]) , len(self.specs.X[1]) )             
        self.assertEqual( len(self.specs.X[0]) , len(self.specs.X[2]) )   
        
        
        
         
    def testReproducibility(self):
        '''
        tests that using the same seed, we can reproduce results.
        note: this is what we did in the setUp method above EXCEPT
        the last two lines.
        '''

        def rep_func():
            np.random.seed(567)
        

            # creating the container object...
            specs = cot.Container( truth.rosenbrock_2D )
            specs.set_prior( lambda x: -np.linalg.norm(x)**4)

            specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
            specs.add_point( np.array( [  1.5 ,-2.0  ] )  )
        
            sampler = smp.Sampler( specs )
            sampler.learn()
            sampler.learn()
         
            # now we put the sample in y
            return  sampler.sample_one()   
       
        # and compare
        self.assertTrue( np.all(rep_func() == rep_func() )  ) 
        
        

    def testReproducibilityFails(self):
        '''
        tests that using different seeds, we cannot expect to
        reproduce results. 
        note: this is what we did in the testReproducibility method 
        above EXCEPT for the first line and the last.
        '''
        
        def non_rep_func():
            
            # the seed is now commented
            #np.random.seed(567)
        

            # creating the container object...
            specs = cot.Container( truth.rosenbrock_2D )
            specs.set_prior( lambda x: -np.linalg.norm(x)**4)

            specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
            specs.add_point( np.array( [  1.5 ,-2.0  ] )  )
        
            sampler = smp.Sampler( specs )
            sampler.learn()
            sampler.learn()
         
            # now we put the sample in y
            return  sampler.sample_one()   
       
        # compare
        self.assertFalse(  np.all( non_rep_func() == non_rep_func() )  )  
        
        
         
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()