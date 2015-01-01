'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
from numpy.linalg import norm
from numpy import zeros
from numpy import array
from numpy import einsum
from numpy import ravel

from math import exp

def cov(x,y, *args):
    '''
    calculate autocovariance as
    k(x,y) = exp(   -||x-y||^2 / r   )
    :param x:
        first point in space
    :param y: 
        second point in space
    :param r:
        the characteristic length scale of
        of the covariance
    :param d:
        the covariance for two points that
        are arbitrarily far apart
        
    * ``cov`` - the covariance between x,y with 
        characteristic length r
    '''
    
    # unpack
    r = args[0]
    d = args[1]
    
    if len(args) > 2:
        A = args[2]
        t = x - y
        dist = einsum(' i, ij, j ', t, A, t)/r

    else:
        dist = norm(x-y)/r  
#     t = dist*1.7320508075688772 # sqrt(3)
# 
#     return (1 + t)*math.exp(-t)

    return  d*exp(  -(dist*dist)/2  ) 

def cov_mat(X, *args): 
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
    C = zeros( (n,n) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(n):
        for j in range (i):
            C[i,j] = cov(X[i],X[j], *args)
            C[j,i] = C[i,j]
        C[i,i] = cov(X[i], X[i], *args)
    return C

def cov_vec(X,w, *args):
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
    
    return array( 
                    [cov(x,w, *args) for x in X]
#                     map(lambda x: cov(x,w, *args) , X )
                    )

      
def tychonoff_solver( specs, b ):
    '''
    solve Ax = b  using tychonoff regularization or, 
    equvalently, multiply A^{-1}b stably.
    we return the solution to the optimization problem
    x = argmin ||Ax -b||^2 + reg*||x||^2.
    U, S, V is the SVD of A ( A = USV and V is NOT transposed!! )
    reg is the regularization coefficient
    :param U,S,V: the SVD of the matrix A. It holds that
        U*S*V = A (V and not V*, this is what python's
        numpy.linalg.svd(A) returns)
    :param b:
        the vector for which we solve
    * ``x`` - solution to the abovementioned optimization problem
    '''
    S = specs.S
    b = ravel(b)
    c = einsum( ' ij ,  i -> j' , specs.U, b)  # c = U^t * b
    y = c*S/(S*S + specs.reg )            # y_i  = s_i * c_i  /  (s_i^2 + reg
    x = einsum( 'ij , i -> j', specs.V ,  y )  # x = V^t * y

    return x