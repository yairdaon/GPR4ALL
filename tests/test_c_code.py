'''
Created on Mar 4, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np
import math
import matplotlib.pyplot as plt	
import pytest

import gpr4all.container as cot
import gpr4all.sampler as smp
import gpr4all.truth as truth
import gpr4all._aux as _aux
import gpr4all._g as _g
import gpr4all.rosenbrock as rose

def test_g():
    
    def g_py(xn, s ,specs):
        '''
        calculate g, as naively as possible
        '''
        ks       = _aux.cov_vec(specs.Xarr, s, specs.r, specs.d)
        ksKinv   =  np.linalg.solve(specs.cm, ks)
        kxn      = _aux.cov_vec(specs.Xarr, xn, specs.r, specs.d)
        
        gValue   =  _aux.cov(s,xn, specs.r, specs.d) - np.einsum( 'i ,i' ,ksKinv , kxn)
        sigSqr   =  specs.kriging(xn, grads = False, var = True)[1]
        
        return       gValue*gValue/sigSqr
    
    dx = 1e-8

    # xn is where derivative is calculated
    xn         =    np.array( [ 0.0])
    s          =    np.array( [ 2.0])

    # container setup
    specs = cot.Container( truth.big_poly_1D , d = 1.0 , r = 1.0 )
    specs.set_prior( lambda x: 0.0 , lambda x: 0.0)
    specs.add_point( np.array([ -0.75]) )
    specs.add_point( np.array([  0.75]) )
    specs.set_matrices()
    
    # derivative calculated using C
    gC , gradC  = _g.g(specs.U,  specs.S, specs.V, specs.Xarr, s, xn, specs.r, specs.d, specs.reg)
    gPy         = g_py(xn,s,specs)
    
    gph      = g_py(xn+dx , s , specs )
    gmh      = g_py(xn-dx , s , specs )
    numeric  = ( gph - gmh )/(2.0*dx )

    # should equal (ok, almost equal)
    assert abs(gC - gPy) < 1e-14
    assert abs(gradC - numeric) < 1e-8 ## More slack for numeric diff
