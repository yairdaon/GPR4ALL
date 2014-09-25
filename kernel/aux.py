'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import numpy as np
import math

import point

def cov(x,y,r):
    '''
    calculate autocovariance
    :param x:
        first point in space
    :param y: 
        second point in space
    :param r: the characteristic length scale of
        of the covariance
    * ``cov`` - the covariance between x,y with 
        characteristic length r
    '''
    
    if np.all(x==y):
        if isinstance(x , point.PointWithError):
            return x.get_error() + 1.0
        else:
            return 1.0 
    else: 
        temp = np.linalg.norm(x-y)
        return  math.exp(  -temp*temp/(2*r*r)  ) 

    # if we consider noisy observations
    
    
    return cov

def aug_cov_mat(X,r):
    '''
    return the augmented covariance matrix for the observations,
    used for kriging. see 
    '''
    
    #find the size 
    n = len(X)
    
    # allocate memory
    C = np.zeros( (n+1,n+1) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(0,n):
        for j in range (0,i+1):
            C[i,j] = cov(X[i],X[j],r)
            C[j,i] = C[i,j]
    
    # set the values of the augmentation (see matrix above)        
    for i in range(0,n):
        C[i,n] = -1.0
        C[n,i] = 1.0
    C[n,n] = 0.0
    return C



def cov_mat(X,r): 
    '''
    create and return the covariance matrix for the observations
    :param X: 
        a list of locations in space, for which we calculate covariance
    :param r:
        characteristic length scale
    '''
    
    #decide on the size of
    n = len(X)
    
    # allocate memory
    C = np.zeros( (n,n) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(0,n):
        for j in range (0,i):
            C[i,j] = cov(X[i],X[j],r)
            C[j,i] = C[i,j]
        C[i,i] = cov(X[i], X[i], r)
    return C

def inv_cov(X,r):
    '''
    return the inverse of the covariance matrix defined above
    '''
    return np.linalg.inv(cov_mat(X,r))
    
def cov_vec(X,w,r):
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
        
    returns a vector v s.t. v_i =cov(x_i,w)
    '''
    
    return np.array( [cov(x,w,r) for x in X] )

# def multInvCovMat( a, specs, b):
#     '''
#     multiply by the inverse of the covariance 
#     '''
#     aU  = np.dot(a, specs.U)
#     Vtb = np.dot(specs.V,b)
#     aUSinv = np.dot(aU,np.diag(1/specs.S))
#     return np.dot(aUSinv, Vtb)   
    
      
def tychonoff_solver( U, S, V, b, reg):
    '''
    solve Ax = b  using tychonoff regularization.
    we return the solution to the optimization problem
    x = argmin ||Ax -b||^2 + reg*||x||^2.
    U, S, V is the SVD of A ( A = USV )
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

def inf_norm(s):
    if hasattr(s, "__len__"):
        return np.linalg.norm( np.asarray(s) , np.inf)
    else:
        return abs(s) 