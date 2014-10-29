'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!

these methods return what we believe is the true log-likelihood.
you tell the container object about the function you want to use using the
LL variable of the container.
    
'''
import numpy as np
import math

# Some log likelihoods the tests use

def sin_1D(s):
    '''
    a wiggly sine log likelihood
    '''    
    return math.sin(3*s)
    
def gaussian_1D(s):
    '''
    the log likelihood of a 1D standard Gaussian
    '''
    return -s*s/2.0

def double_well_1D(s):
    '''
    this log likelihood is a double well potential (upside down)
    '''
    return -2*s**4 + 5*s**2

def big_poly_1D(s):
    '''
    the following polynomial
    '''
    return -(s**6  + 3.5*s**4  - 2.5*s**3 - 12.5*s**2 + 1.5*s )
    
def norm(s):
    '''
    some log likelihood function
    '''
    t = np.linalg.norm(s)
    return -t*t

def rosenbrock_2D(s):
    '''
    the (negative of the) 2D Rosenbrock function. google it if you've never
    heard of it.
    '''
    return -(  (1 - s[0])**2 + 100*(s[1] - s[0]**2)**2  )
    

def log_rosenbrock_2D(s):
    
    '''
    log of Rosenbrock's function
    '''
    return math.log(  (1-s[0])**2 + 100*(s[1]-s[0]**2)**2   )

def zero(s):
    '''
    identically zero, used when the LL 
    is of no importance
    '''
    return 0

def const(s):
    '''
    identically two, used when the LL 
    is of no importance
    '''
    return 2