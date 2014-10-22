'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import numpy as np
import math

import point

def cov(x,y, r, d):
    '''
    calculate autocovariance as
    k(x,y) = exp(   -||x-y||^2 / r   )
    :param x:
        first point in space
    :param y: 
        second point in space
    :param r: the characteristic length scale of
        of the covariance
    :param d: the covariance for two points that
        are arbitrarily far apart
    * ``cov`` - the covariance between x,y with 
        characteristic length r
    '''
    
    if isinstance(x , point.PointWithError):
        if np.all(x==y):
            return x.error  + d
        
    temp = np.linalg.norm(x-y)
    return  d*math.exp(  -temp*temp/(2*r*r)  ) 

    # if we consider noisy observations
    
    
    return cov

def cov_mat(X,r,d): 
    '''
    create and return the covariance matrix for the observations
    :param X: 
        a list of locations in space, for which we calculate covariance
    :param r:
        characteristic length scale
    :param d: the covariance for two points that
        are arbitrarily far apart
    '''
    
    #decide on the size of
    n = len(X)
    
    # allocate memory
    C = np.zeros( (n,n) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(0,n):
        for j in range (0,i):
            C[i,j] = cov(X[i],X[j],r,d)
            C[j,i] = C[i,j]
        C[i,i] = cov(X[i], X[i], r, d)
    return C

def cov_vec(X,w,r,d):
    '''
    creates a vector of covariances, between 
    X(a list of numpy arrays) and w (a numpy
    array of same size).
    :param X: 
        a list of locations
    :param w:
        a specific location for which we calculate covariances
    :param r: 
        characteristic length scale
    :param d: the covariance for two points that
        are arbitrarily far apart
        
    returns a vector v s.t. v_i =cov(x_i,w)
    '''
    
    return np.array( [cov(x,w,r,d) for x in X] )

      
def tychonoff_solver( U, S, V, b, reg):
    '''
    solve Ax = b  using tychonoff regularization or, 
    equvalently, multiply A^{-1}v stably.
    we return the solution to the optimization problem
    x = argmin ||Ax -b||^2 + reg*||x||^2.
    U, S, V is the SVD of A ( A = USV and V is NOT transposed!! )
    reg is the regularization coefficient
    :param U,S,V: the SVD of the matrix A. It holds that
        U*S*V = K (yes, V and not V*, this is what python's
        numpy.linalg.svd(K) returns)
    :param b:
        the vector for which we solve
    * ``x`` - solution to the abovementioned optimization problem
    '''
    
    b = np.ravel(b)
    c = np.dot(np.transpose(U), b)  # c = U^t * b
    y = c*S/(S*S + reg )            # y_i  = s_i * c_i  /  (s_i^2 + reg
    x = np.dot( np.transpose(V) ,  np.transpose(y) )  # x = V^t * y
    
    return x
