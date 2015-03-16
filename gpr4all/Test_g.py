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
	




def g_naive(xn, s ,specs):
	'''
	calculate g, as naively as possible
	'''
		
	ks       = _aux.cov_vec(specs.Xarr, s, specs.r, specs.d)
	ksKinv   = _aux.solver(specs.U, specs.S, specs.V, ks, specs.reg)
	kxn      = _aux.cov_vec(specs.Xarr, xn, specs.r, specs.d)

	gValue   =  _aux.cov(s,xn, specs.r, specs.d) - np.einsum( 'i ,i' ,ksKinv , kxn)
	sigSqr   =  specs.kriging(xn, grads = False, var = True)[1]
	
	return       gValue*gValue/sigSqr
	
def grad_naive(xn, s, specs):

	r = specs.r
	d = specs.d


	f  = lambda x: _aux.cov(s, x, r, d)
	specs1 = cot.Container( f , r=r, d=d)
	specs1.set_prior(lambda x: 0.0 , lambda x: 0.0)
	for x in specs.X:
		specs1.add_point(x)



	covar = _aux.cov( xn ,s ,r, d)


	print(specs1.X)
	print
	print(specs1.y)
	print

	krig, sigSqr , gradKrig, gradSig = specs1.kriging( xn , True , True )
	
	first = -2.0*covar/sigSqr
	sec   = -(2.0/sigSqr) 
	third = -((covar  - krig)**2 / (sigSqr)**2 )* gradSig
	
	return first * (xn - s) + sec * gradKrig + third * gradSig

	

r  = 1.72 
d  = r
dx = 0.001
e1 = np.array([1.0 , 0.0])
e2 = np.array([0.0 , 1.0])
	

# container setup
specs = cot.Container( truth.norm, r=r, d=d)
specs.set_prior(lambda x: 0.0 , lambda x: 0.0)
specs.add_point( np.array([ 1.0 , 1.0]) )
specs.add_point( np.array([-1.0 ,-1.0]) )
specs.set_matrices()


# x is where derivative is calculated
xn         =    np.array( [ 2.99, 1.33 ])
s          =    np.array( [ 3.47, 1.52 ])



# the derivative calculated using calculus differentiation
gFromPy  = g_naive(xn, s, specs)
#gradFromPy = grad_naive(xn,s,specs)

# derivative calculated using finite differences
gFromC , gradFromC  = _g.g(specs.U,  specs.S, specs.V, specs.Xarr, s, xn, specs.r, specs.d, specs.reg)




print 
print
print






#gradKoverSig  = _g.gradKoverSig(specs.U,  specs.S, specs.V, specs.Xarr, s, xn, specs.r, specs.d, specs.reg)
#kOverSig      = _aux.cov(s, xn, r, d)/specs.kriging(xn, grads = False, var = True)[1]
 
#kOverSigPlusH     = _aux.cov(s, xn+h, r, d)/specs.kriging(xn+h, grads = False, var = True)[1]


#print "analytic grad of k(s,xn)/sigSqr(xn) using C   =" +str(gradKoverSig)
#print "numerical grad of  k(s,xn)/sigSqr(xn)  ="  +str(     (kOverSigPlusH - kOverSig)/h    )





gph1 , _ = _g.g(specs.U,  specs.S, specs.V, specs.Xarr, s, xn+e1*dx, specs.r, specs.d, specs.reg)
gph2 , _ = _g.g(specs.U,  specs.S, specs.V, specs.Xarr, s, xn+e2*dx, specs.r, specs.d, specs.reg)
numeric = np.array([ gph1- gFromC , gph2- gFromC ])/dx



# should equal (ok, almost equal)
print( "g using c     = "  + str(gFromC) )
print( "g using py    = "  + str(gFromPy))
print( "grad using c  = "  + str(gradFromC) )
#print( "grad using py = "  + str(gradFromPy) )
print( "numeric gradient  ="  +str(numeric)  )

