'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import numpy as np

import gpr4all.container as cot
import gpr4all.sampler as smp
import gpr4all.truth as truth
import gpr4all._aux as _aux
import gpr4all.rosenbrock as rose

class Test(unittest.TestCase):
    '''
    assorted tests with varying levels of simplicity.
    These tests are relatively fast to run and produce
    no graphical output, unlike the other tests.
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
        specs = cot.Container( truth.zero )
         
        # make sure the prior doesn't bother us
        specs.set_prior( lambda x: 0.0, lambda x: 0.0  )
 
     
        for i in range(len(X)):
            specs.add_pair(X[i], F[i])
         
        # the center of mass of the x1,...,x4 is (0,0)
        s = np.array( [0, 0] )
         
        # kriging for this center ...
        b = specs.kriging(s, False, False)
         
        # ... should be zero, by symmetry
        self.assertAlmostEqual(b, 0, 8) # 8 is the number of decimal digits we consider
      
    def testUniform(self):
        ''' 
        test kriging by making sure the procedure outputs
        a constant 0 when it is given constant 0
        input
        '''
         
        specs = cot.Container( truth.zero )
        specs.set_prior( lambda x: 0.0, lambda x: 0.0  )
         
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
 
        self.assertTrue(    np.allclose( np.array([0]) , specs.kriging(np.array([20.0]) , False, False) )      )
    
         
    
 
    def testCovVec(self):
        '''
        test the function that creates a covariance vector
        in the module aux
        '''
         
        X = []
        X.append( np.array([1.0,2,3.3]) )
        X.append( np.array([2.0,4,5]) )
        X.append( np.array([-2,2,0]) )
        X = np.asarray(X)
        w = np.array( [ 2, 2 , 3])
        r = 1.0
        d = 1.0
        v = _aux.cov_vec(X, w, r, d)
             
    def testTychonoffSimple(self):
        '''
        test the linear solver we use - This solver 
        uses tychonoff regularization. here we make sure the
        tychonoff regularization behaves reasonably for 
        a trivial example
        ''' 
 
         
        A = np.array( [[ 1, 2] , [ 3 ,4] ])
         
        specs = cot.Container(truth.zero)
        specs.U,  specs.S,  specs.V = np.linalg.svd( A, full_matrices= True, compute_uv=True)
        specs.reg = 100*np.finfo(np.float).eps
        b = np.array([1 , 1])
 
        svdSol = _aux.solver( specs.U, specs.S, specs.V, b , specs.reg )
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
         
        # this is really a mocked container object.
        # it holds no data, only what we put in it
        specs = cot.Container(truth.zero)
         
        # get SVD
        specs.U,  specs.S,  specs.V = np.linalg.svd( A, full_matrices= True, compute_uv=True)
         
        # the regularization factor we ususlly use in the solver
        specs.reg = 100*np.finfo(np.float).eps
         
        # create three random target vectors
        b = np.array( np.random.rand(50) )
         
         
         
        # solve using our solver
        x = _aux.solver( specs.U, specs.S, specs.V, b , specs.reg )
 
         
        # solve using standard package
        y = np.linalg.solve(A, np.ravel(b))
         
        # compare solutions
        self.assertTrue(np.allclose(x, y))     
         
         
         
             
         
    def testSampler(self):
        '''
        here we sample from the sampler. We choose to learn from
        the samples.
        '''
 
        # creating the container object...
        specs = cot.Container( rose.rosenbrock_2D )
 
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
            specs = cot.Container( rose.rosenbrock_2D )
 
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
            specs = cot.Container( rose.rosenbrock_2D )
 
            specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
            specs.add_point( np.array( [  1.5 ,-2.0  ] )  )
         
            sampler = smp.Sampler( specs )
            sampler.learn()
            sampler.learn()
          
            # now we put the sample in y
            return  sampler.sample_one()   
        
        # compare
        self.assertFalse(  np.all( non_rep_func() == non_rep_func() )  )  

    def testGradientKrig(self):
        '''
        check that the gradient of the kriged function
        is calculated correctly (ok, approximately), 
        first in 1D, then in 2D
        '''

        r  = 1.72 
        d  = 2.28
        dx = 1e-8
        
        #1D:
        # container setup
        specs = cot.Container( truth.big_poly_1D , r=r, d=d)
        specs.add_point( np.array([ 1.0]) )
        specs.add_point( np.array([-1.0]) )
        specs.set_matrices()  
        
        # x is where derivative is calculated
        x         =    np.array( [ 0.47])
        
        # the derivative calculated using calculus differentiation
        analytic = specs.kriging( x, True, True)[2]
#    
        
        # derivative calculated using finite differences
        numeric   = ( specs.kriging(x+dx) - specs.kriging(x-dx) )/(2*dx)
        
        # should equal (ok, almost equal)
        self.assertAlmostEqual(numeric , analytic, 6)
        
        #2D:
        # container setup
        specs = cot.Container( rose.rosenbrock_2D , r=r, d=d)
        specs.add_point( np.array([ 1.0 , 2.11]) )
        specs.add_point( np.array([ 5.0 ,-4.3 ]) )
        specs.add_point( np.array([-2.0 , 0.22]) )
        specs.set_matrices() 
        
        # x is where we calculate the derivative 
        x = np.array([2.3 , 1.1])
        krig = specs.kriging(x)
        
        # tthe derivative using the rules of calculus
        analytic =  specs.kriging( x, True, True)[2]
        
        # derivative using finite differences
        numeric = np.zeros(2)
        numeric[0]  = (  specs.kriging(x + np.array([dx,0]) ) - krig  )/dx  
        numeric[1]  = (  specs.kriging(x + np.array([0,dx]) ) - krig  )/dx  
        
        # should equal (almost)
        self.assertTrue(np.allclose( analytic, numeric) )
 
    def testGradientSigSqr(self):
        '''
        check that the gradient of the kriged variance
        is calculated correctly (ok, approximately), 
        first in 1D, then in 2D. See the comments in 
        testGradientKrig, since these methods are 
        practically the same (different calculations
        carried out under the hood, though).
        '''
         
        r =1.3 
        d = 2.28
        dx = 1e-7
         
        #1D:
        # container setup
        specs = cot.Container( truth.big_poly_1D , r=r, d=d)
        specs.add_point( np.array([ 1.0]) )
        specs.add_point( np.array([-1.0]) )
        specs.set_matrices()  
         
        x         =    np.array( [ 0.47])
        analytic  =    specs.kriging(x, grads=True)[3]
        numeric   = ( specs.kriging(x+dx, var=True)[1] - specs.kriging(x-dx, var=True )[1] )/(2*dx)
        self.assertAlmostEqual(numeric , analytic, 8)
         
         
         
        #2D. Set the container
        specs = cot.Container( rose.rosenbrock_2D , r=r, d=d)
        specs.add_point( np.array([ 1.0 , 2.11]) )
        specs.add_point( np.array([ 5.0 ,-4.3 ]) )
        specs.add_point( np.array([-2.0 , 0.22]) )
        specs.set_matrices() 
          
        numeric = np.zeros(2)
        x = np.array([2.3 , 1.1])
        analytic  =    specs.kriging(x, grads=True)[3]
         
        # simple divided differences
        numeric[0]  = (  specs.kriging(x + np.array([dx,0]), var=True)[1] - specs.kriging(x + np.array([-dx,0]), var=True)[1]  )/(2*dx)  
        numeric[1]  = (  specs.kriging(x + np.array([0,dx]), var=True)[1] - specs.kriging(x + np.array([0,-dx]), var=True)[1]  )/(2*dx)  
        self.assertTrue(np.allclose( analytic, numeric) )         
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()