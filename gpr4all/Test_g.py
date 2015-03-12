'''
Created on Mar 4, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import numpy as np

#import gpr4all.container as cot
#import gpr4all.sampler as smp
#import gpr4all.truth as truth
#import gpr4all._aux as _aux
#import gpr4all._g as _g
#import gpr4all.rosenbrock as rose


import container as cot
import sampler as smp
import truth as truth
import _aux as _aux
import _g as _g
import rosenbrock as rose
	




def gNaive(xn, s ,specs):
	'''
	calculate g, as naively as possible
	'''
		
	ks       = _aux.cov_vec(specs.Xarr, s, specs.r, specs.d)
	ksKinv   = _aux.solver(specs.U, specs.S, specs.V, ks, specs.reg)
	kxn      = _aux.cov_vec(specs.Xarr, s, specs.r, specs.d)

	gValue   = _aux.cov(s,xn, specs.r, specs.d) - np.einsum( 'i ,i' ,ksKinv , kxn)
	sigSqr   =  specs.kriging(xn, grads = False, var = True)[1]
	
	return       gValue*gValue/sigSqr
	
def gradNaive(xn, s, specs):
	grad = -_aux.cov(xn,s, specs.r , specs.d)/(r*r)*(xn -s)
	ks  = _aux.cov_vec(specs.Xarr, s, specs.r, specs.d)
	kxn = _aux.cov_vec(specs.Xarr, xn, specs.r, specs.d)
	ksKinv  = _aux.solver(specs.U, specs.S, specs.V, ks, specs.reg)

	for i in range(len(specs.X)):
		grad = grad + ksKinv[i] * kxn[i]*(xn - specs.X[i])/(specs.r*specs.r)

	print(grad)
	
	return grad

r  = 1.72 
d  = r
dx = 1e-8
	
#1D:
# container setup
specs = cot.Container( truth.big_poly_1D , r=r, d=d)
specs.add_point( np.array([ 1.0 ]) )
specs.add_point( np.array([-1.0 ]) )
specs.set_matrices()  

# x is where derivative is calculated
xn         =    np.array( [-4.55 ])
s          =    np.array( [ 3.47 ])

# the derivative calculated using calculus differentiation
gFromPy  = gNaive(xn, s, specs)
gradFromPy = gradNaive(xn,s,specs)

# derivative calculated using finite differences
gFromC , gradFromC  = _g.g(specs.U,  specs.S, specs.V, specs.Xarr, s, xn, specs.r, specs.d, specs.reg)

# should equal (ok, almost equal)
print( "g using c     = "  + str(gFromC) )
print( "g using py    = "  + str(gFromPy))
print( "grad using c  = "  + str(gradFromC) )
print( "grad using py = "  + str(gradFromPy))

	   


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
