'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np
import math
import aux

def kriging(s, specs):
    '''
    use algortihm 2.1 from page 19 of the book "Gaussian
    Processes for Machine Learning" by Rasmussen and Williams.
    They solve using Cholesky. Here we just store the matrix
    [K + reg*Id]^{-1} where K is the covariance matrix and 
    reg is a regularization we add to the diagonal, for 
    stability.
    
    here this is interpreted as a log-likelihood \ log-probability
    
    according to our way of modelling the unknown log-likelihood, we 
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
    y = np.array( specs.Fmp ) # F minus prior 
    y = np.ravel(y)

    # parameters
    r = specs.r
    d = specs.d
    reg = specs.reg
    
    # create the covariance vector:
    k = aux.cov_vec(X, s, r, d)
    
    # K^{-1}*k in RW's notation. Inverse cov mat times cov vec. 
    lam = aux.tychonoff_solver(specs.U , specs.S , specs.V ,  k, reg)

    # the kriged value is...
    f = np.dot(  y , lam )
    
    # the variance is...
    sigmaSquare =  aux.cov(0,0,r,d) - np.dot(k,lam)
    if sigmaSquare  < 0: 
        print("")
        print("x = " + str(s))
        print("sigSquare = " + str(sigmaSquare) )
        print("condition number = " + str(specs.condition()))

    
    
    kriged = f + specs.prior(s)
    std = math.sqrt( max(sigmaSquare,0) )   
    
    return kriged , std  