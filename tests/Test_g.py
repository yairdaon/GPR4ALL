'''
Created on Mar 4, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''


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

import math
import matplotlib.pyplot as plt	

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
	


def grad_k(xn , s, r ,d):
	return -_aux.cov(s,xn, r, d)/(r*r) *(xn - s)


def grad_py(xn, s, specs):


	r   = specs.r
	d   = specs.d
	reg = specs.reg
	g   = g_py(xn ,s , specs)




	f  = lambda x: _aux.cov(s, x, r, d)
	specs1 = cot.Container( f , r=r, d=d)
	specs1.set_prior(lambda x: 0.0 , lambda x: np.zeros( specs1.X[0].shape ) )
	for x in specs.X:
		specs1.add_point(x)
	specs1.set_matrices()



	covar = _aux.cov( xn ,s ,r, d)


	#print(specs1.X)
	print
	#print(specs1.y)
	print
	#print(specs1.U)

	krig, sigSqr , gradKrig, gradSig = specs1.kriging( xn , True , True )
	
	first = -2.0*covar/(sigSqr*r*r)*g
	sec   = -2.0/sigSqr*g
	third = -(covar  - krig)*(covar - krig) / (sigSqr*sigSqr) 	
	grad =  first * (xn - s)  +  sec * gradKrig  + third * gradSig  
	return grad 
	



def fake_num(xn, s, specs, dx):

	r   = specs.r
	d   = specs.d
	reg = specs.reg
	X   = specs.Xarr

	# k^n ( s  )
	ks  = _aux.cov_vec(X, s , r, d)

	# k^n ( xn )	
	kxn = _aux.cov_vec(X, xn, r, d)
	
	# k^n(xn + dx )
	kdx = _aux.cov_vec(X, xn+dx, r, d)

	# K^{-1} k^n ( s )
	ksKinv   = _aux.solver(specs.U ,specs.S, specs.V , ks , reg)
	#ksKinv   = np.linalg.solve(specs.cm, ks)

	# K^{-1} k^n ( s )
	kxnKinv  = _aux.solver(specs.U ,specs.S, specs.V , kxn , reg)
	#kxnKinv  = np.linalg.solve(specs.cm, kxn)
	
	# . . .
	kdxKinv  = _aux.solver(specs.U ,specs.S, specs.V , kdx , reg)
	#kdxKinv  = np.linalg.solve(specs.cm,  kdx)

	sigSqr   = d - np.sum(kxnKinv * kxn)
	sigSqrdx = d - np.sum(kdxKinv * kdx)

	#kxn^t * K^{-1} * ks
	kKk      = np.sum(kxnKinv * ks)

	first    =  2.0/sigSqr
	second   = -2.0/sigSqr
	t        = _aux.cov(s,xn,r,d) - kKk
	third    = -t*t/(sigSqr*sigSqr)
	
	grad1  =  ( _aux.cov(xn+dx,s,r,d) - _aux.cov(xn,s,r,d)    )/ dx
	grad2  =  ( np.sum(kdxKinv*ks - kxnKinv*ks) )/dx
	grad3  =  ( np.sum(kxn*kxnKinv - kdx*kdxKinv) )  / dx

	return first*grad1 + second*grad2 + third*grad3



dx = 1e-10



# xn is where derivative is calculated
xn         =    np.array( [ 0.0])
s          =    np.array( [ 2.0])

# container setup
specs = cot.Container( truth.big_poly_1D , d = 1.0 , r = 1.0 )
specs.set_prior( lambda x: 0.0 , lambda x: 0.0)
specs.add_point( np.array([ -0.75]) )
specs.add_point( np.array([  0.75]) )
specs.set_matrices()


print 
print
print

# the derivative calculated using calculus differentiation
gPy    = g_py(xn, s, specs)
gradPy = grad_py(xn ,s,specs)

# derivative calculated using finite differences
gC , gradC  = _g.g(specs.U,  specs.S, specs.V, specs.Xarr, s, xn, specs.r, specs.d, specs.reg)

gph      = g_py(xn+dx , s , specs )
gmh      = g_py(xn-dx , s , specs )
numeric  = ( gph - gC )/( dx )
fake     = fake_num(xn, s, specs, dx)



# should equal (ok, almost equal)
print( "g using c     = "  + str(gC) )
print( "g using py    = "  + str(gPy))
print( "grad using c  = "  + str(gradC) )
#print( "grad using py = "  + str(gradPy) )
#print( "half numeric  = "  + str(fake)  )
print( "numeric grad  ="   + str(numeric)  )


 # allocating memory
x = np.arange(-1,1, 0.05)
n  = len(x)
f = np.zeros( n )
yn = gC

for j in range(n):
	t = np.array([  x[j]  ])
	f[j] = g_py( t , s , specs )
	


plt.figure()
a = numeric
b = yn - a*xn
curve1  = plt.plot(x, f, label = "kriged value")
curve2  = plt.plot(x, a*x + b, label ="tangent")
plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
plt.setp( curve2, 'linewidth', 3.0, 'color', 'b', 'alpha', .5 )
plt.ylim([-0.2 , 1.0  ] )
plt.xlim([-1.0 , 1.0 ] )
#plt.show()
