'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np
import math
import aux
import algorithm as alg

def kriging(s, specs):
    '''
    this is where we decide actually which kriging subroutine we use.
    calling them always has the same syntax, though.
    return the interpolated f
    here this is interpreted as a log-likelihood \ log-probability
    
    according to our way of modelling the unknown log-likelihood, we 
    believe that the log-likelihood at x is a random variable distributed 
    normally N(kriged, std^2)
    
    :param s: location in space for which we want to calculate kriged log-likelihood
    
    :param specs: a container object holding all specifications required for the calculation
    
    returns:
    
    * ``kriged`` -the kriged value, base on the data in the container object.
    
    * ``std`` - the variance at the point s.
    '''
    
    # make sure we are using a valid algorithm
    assert (specs.algType == alg.AUGMENTED_COVARIANCE or 
                    specs.algType == alg.RASMUSSEN_WILLIAMS) , "Invalid algorithm type!" 

    # make sure the matrices used in the kriging computation are ready    
    if not specs.matricesReady:
        specs.set_matrices()
    
    # choose among the different algorithms
    if specs.algType == alg.AUGMENTED_COVARIANCE:
        f, sigSquare =  acm_kriging(s, specs)    
    
    elif specs.algType == alg.RASMUSSEN_WILLIAMS:
        f, sigSquare =  rw_kriging(s, specs)  
    
    kriged = f + specs.prior(s)
    std = math.sqrt( max(sigSquare,0) )   

    return kriged , std 
 
def acm_kriging(s, specs):
    '''
    do krigin using SVD and tychonoff regularization.

    we are looking to solve the following:
    [     |-1]   [   ]     [ ]
    [  C  |-1] * [lam]  =  [c]
    [     |-1]   [   ]     [ ]
    [1 1 1| 0]   [ m ]     [1]
    
    where:
    C is a covariance matrix between observations (have n of those)
    lambda are weights
    m is a lagrange multiplier
    c is the covariance between the given s and the n observations
    
    function parameters: 
    s - where we want to estimate our function \ process
    specs - an object that contains all the data we need for the computation
    
    returns - mean and variance for point s
    '''
    # unpack the variables
    X = specs.X
    F = specs.Fmp # F minus prior 
    r = specs.r
    reg = specs.reg

        
    # number of samples we have.
    n = len(X)
    
    # create the target c:
    c = np.zeros( n+1 )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    c[n] =  1.0
    
    lam = aux.tychonoff_solver(specs.U , specs.S , specs.V ,  c, reg)     # solve!!!
    
    m = lam[n]
    lam = lam[0:n]
    lam = lam/np.sum(lam) #make sure weights sum to 1
    
    # calculate the kriged estiamte
    f = 0
    for i in range(n):
        f = f + lam[i] * F[i]
        
    # calculate the variance    
    sigmaSquare = m + aux.cov(0,0,r) - np.sum(lam*c[0:n])  
    
    # we want to know if the variance turns out to be negative!
    if sigmaSquare  < 0 and -sigmaSquare > reg:
        print("Negative kriged variance. Probably because data points are too close.")
        
    return f, sigmaSquare   


def rw_kriging(s, specs):
    '''
    use algortihm 2.1 from page 19 of the book 
    "Gaussian Processes for Machine Learning" by
    Rasmussen and Wiliams. They solve using Cholesky. Here
    we use the SVD with tychonoff regularization instead.
    '''
    
    # unpack the variables
    X = specs.X
    y = np.array( specs.Fmp ) # F minus prior 
    y = np.ravel(y)

    # number of samples we have.
    n = len(X)
    
    # parameters
    r = specs.r
    reg = specs.reg
    
    # create the target c:
    k = np.zeros( n )
    for i in range(0,n):
        k[i] = aux.cov(s,X[i],r)
    
    
    
    # solve for lambda 
    alpha = aux.tychonoff_solver(specs.U, specs.S, specs.V, y, reg)
    
    f = 0 # F minus prior
    for i in range(n):
        f = f + alpha[i] * k[i]
        
    # solve using our tychonoff solver
    tmp = aux.tychonoff_solver(specs.U, specs.S, specs.V, k, reg)
    
    sigmaSquare =  aux.cov(0,0,r) - np.sum(k*tmp)
    if sigmaSquare  < 0 and -sigmaSquare > reg:
        print(" negative variance. s= " + str(s) )

    return f, sigmaSquare  

def set_get_limit(specs):
    '''
    Returns the kriged value "at infinity", along with
    the (prior) variance at infinity. Very similar to the 
    above kriging procedures.
    '''
    
    if not specs.limitsReady:
        
        # if we solve for the augmented covariance matrix
        if specs.algType == alg.AUGMENTED_COVARIANCE:
            
            # prepare for the following calculations
            if not specs.matricesReady:
                specs.setMatrices()
            
            # unpack
            F = specs.Fmp # F minus prior
            n = len(F)
            
            # set target
            c = np.zeros( n+1 )
            c[n] =  1.0
            
            # solve for the coefficients just like in kriging
            lam = aux.tychonoff_solver(specs.U, specs.S, specs.V, c, specs.reg)
            
            # calculate the function value according to these weights
            lim = 0
            for i in range(n):
                lim = lim + lam[i] * F[i] 
            
            specs.varAtInf = aux.cov(0,0,specs.r) + lam[n]
            specs.lim = lim
            
        
        # if we use algorithm 2.1 (page 19) from Rasmussen & Williams' book
        # title "Gaussian Processes for Machine Learning" we solve for the
        # not augmented covariance matrix   
        if specs.algType == alg.RASMUSSEN_WILLIAMS:
            
            # the variance at infinity
            specs.varAtInf = aux.cov(0,0,specs.r)
            specs.lim = 0.0
        
        # we've set the limits, so they're ready
        specs.limitsReady = True
        
    return specs.lim , specs.varAtInf