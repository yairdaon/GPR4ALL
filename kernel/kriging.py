'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

from numpy import einsum as einsum
from numpy import array as array
from numpy import ravel as ravel
from numpy import asarray as asarray

from aux import tychonoff_solver as solver
from aux import cov as cov
from aux import cov_vec as cov_vec

def kriging(s, specs , gradients=False):
    '''
    use algortihm 2.1 from page 19 of the book "Gaussian
    Processes for Machine Learning" by Rasmussen and Williams.
    They solve using Cholesky. Here we just store the matrix
    [K + reg*Id]^{-1} where K is the covariance matrix and 
    reg is a regularization we add to the diagonal, for 
    stability.
    
    here this is interpreted as a log-likelihood \ log-probability
    
    according to our way of modeling the unknown log-likelihood, we 
    believe that the log-likelihood at x is a random variable distributed 
    normally N(kriged, std^2)
    
    :param s: location in space for which we want to calculate kriged log-likelihood
    
    :param specs: a container object holding all specifications required for the calculation
    
    returns:
    
    * ``kriged`` -the kriged value, base on the data in the container object.
    
    * ``std`` - the standard deviation at the point s.
    '''
    
    # make sure the matrices used in the kriging computation are ready    
    if not specs.matricesReady:
            specs.set_matrices()
   
    # unpack the variables
    X = specs.X
    y = array( specs.Fmp ) # F minus prior 
    y = ravel(y)

    # parameters
    r = specs.r
    d = specs.d
    
    # create the covariance vector:
    k = cov_vec(X, s, r, d)
    
    # k*K^{-1} in RW's notation. Inverse cov mat times cov vec. 
    kKinv = solver(specs ,  k)

    # the kriged value is...
    f = einsum( 'i ,i' ,  y , kKinv )
    
    # account for the prior we've subtracted
    krig = f + specs.prior(s)
    
    if gradients:
        
        # the variance is...
        sigSqr =  cov(0,0,r,d) - einsum( 'i , i  ', k, kKinv)
        sigSqr = max(sigSqr,0)   

        # calculate gradient of kriged value
        xMinusX = asarray(s - X) 
        FmpKinv = solver(specs, y)
        gradKrig = -einsum( 'i ,i , ij -> j '  , FmpKinv, k, xMinusX )/(r*r) + specs.gradPrior(s)
        
        # calculate  gradient of sigma squared
        gradSigSqr = 2*einsum( 'i ,i , ij -> j '  , kKinv, k, xMinusX )/(r*r)
        return krig , sigSqr, gradKrig, gradSigSqr
    
    return krig 