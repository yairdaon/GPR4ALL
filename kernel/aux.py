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
    '''
    
    if np.all(x==y):
        if isinstance(x , point.PointWithError):
            return x.getError() + 1.0
        else:
            return 1.0
    else: 
        temp = np.linalg.norm(x-y)
        return  math.exp(  -temp*temp/(2*r*r)  ) 

    # if we consider noisy observations
    
    
    return cov

def augCovMat(X,r):
    '''
    return the augmented covariance matrix for the observations
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



def covMat(X,r): 
    '''
    create and return the covariance matrix for the observations
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

def tychonoffSvdSolver( U, S, V, b, reg):
    '''
    solve Ax = b  using tychonoff regularization.
    U, S, V is the SVD of A ( A = USV^t )
    reg is the regularization coefficient
    '''
    
    b = np.ravel(b)
    c = np.dot(np.transpose(U), b)  # c = U^t * b
    y = c*S/(S*S + reg )            # y_i  = s_i * c_i  /  (s_i^2 + reg
    x = np.dot( np.transpose(V) ,  np.transpose(y) )  # x = V^t * y
    
    return x

def infNorm(s):
    if hasattr(s, "__len__"):
        return np.linalg.norm( np.asarray(s) , np.inf)
    else:
        return abs(s) 