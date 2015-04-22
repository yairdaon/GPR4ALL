'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import numpy as np
import os
import math

import matplotlib.pyplot as plt
from matplotlib import gridspec

import sampler as smp
import _g as _g
import container as cot
import rosenbrock as rose
import targets as targets

class Test(unittest.TestCase):
    '''
    if this does not work, it is likely that don't have 
    ffmpeg. 
    this unit test creates a movie. run it and see for yourself!!
    '''

    def testSAA(self):
        '''
        create a 2D movie, based on the data we put in the container object 
        in the setUp method sthis method does all the graphics involved
        since this is a 2D running for lots of points might take a while
        '''
        
        # for reproducibility
        np.random.seed(1792) 
        
            
        # parameters to play with
        nSamples  = 5000      # number of samples we use for KL
        maxiter   = 30000    # max number of optimization steps
        nPoints   = 60       # The number of evaluations of the true likelihood
        M         = 7        # bound on the plot axes
        nopt      = 50
        nwalk     = 50
        burn      = 500
        delta     = 0.1


        
        # initialize container and sampler
        specs = cot.Container( rose.rosenbrock_2D )
        n = 1
        for i in range( -n , n+1 ):
            for j in range( -n, n+1 ):
                specs.add_point(np.array( [2*i , 2*j ] ))
        sampler = smp.Sampler( specs , target = targets.exp_krig_sigSqr, 
                               maxiter = maxiter , nwalkers = nwalk,
                               noptimizers = nopt,  burn = burn)
        sampler.run_mcmc(500)
        mc = sampler.flatchain()

        # memory allocations. constants etc
        a  = np.arange(-M, M, delta)
        X, Y = np.meshgrid(a , a)         # create two meshgrid
        grid = np.asarray( [ np.ravel(X) , np.ravel(Y)])
        xn = np.array([1.33, 2.45])


        avgC, gradAvgC = _g.avg_var(xn , specs.U, specs.S, specs.V, specs.Xarr, grid, specs.r , specs.d, specs.reg )
        avgPy,gradAvgPy= _g.avg_var(xn , specs.U, specs.S, specs.V, specs.Xarr, mc, specs.r , specs.d, specs.reg )
        
        print(avgC)
        print(avgPy)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
