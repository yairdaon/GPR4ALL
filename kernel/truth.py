'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!

these methods return what we believe is the true log-likelihood.
you tell the container object about the function you want to use using the
LL variable of the container.

ALL LOG LIKELIHOODS ARE ASSUMED TO BE NON POSITIVE!!!
    
'''
import numpy as np
import math

# Some log likelihoods the tests use

def sin_1D(s):
    '''
    a wiggly sine
    '''    
    return math.sin(3*s) -1
    
def gaussian_1D(s):
    '''
    the log likelihood of a 1D standard Gaussian
    '''
    return -s*s/2.0

def double_well_1D(s):
    '''
    this log likelihood is a double well potential (upside down)
    '''
    return -2*s**4 + 5*s**2 -3.25

def big_poly_1D(s):
    '''
    the following polynomial
    '''
    return -(s**6  + 3.5*s**4  - 2.5*s**3 - 12.5*s**2 + 1.5*s ) - 10.28
    
def norm(s):
    '''
    some log likelihood function
    '''
    t = np.linalg.norm(s)
    return -t*t

def zero(s):
    '''
    identically zero, used when the LL 
    is of no importance
    '''
    return 0

def dummy(s):
    '''
    this function will raise an error
    it is a dummy LL that should not be 
    called
    '''
    raise Exception("You have called the dummy log likelihood. Shame on you!!!")
