'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!

these methods return what we believe is the true log-likelihood.
you tell the container object about the function you want to use using the
LL variable of the container.
    
'''
import numpy as np


def trueLL(s):
    '''
    decide on the default log likelihood. you may also change 
    this if you want to use your own log-likelihood. in that case,
    don't tell the container - trueLL is the default log-likelihood
    '''
    return sin1D(s)


# Some log likelihoods the tests use
def sin1D(s):
    '''
    a wiggly sine log likelihood
    '''    
    return np.sin(3*s)
    
def gaussian1D(s):
    '''
    the log likelihood of a 1D standard Gaussian
    '''
    return -s*s/2.0

def doubleWell1D(s):
    '''
    this log likelihood is a double well potential (upside down)
    '''
    return -2*s**4 + 5*s**2

def bigPoly1D(s):
    '''
    the following polynomial
    '''
    return -(s**6  + 3.5*s**4  - 2.5*s**3 - 12.5*s**2 + 1.5*s )
    
def norm2D(s):
    '''
    some 2D log likelihood function
    '''
    t = np.linalg.norm(s)
    return -np.array( [t*t] )

def rosenbrock2D(s):
    '''
    the (negative of the) 2D Rosenbrock function. google it if you've never
    heard of it.
    '''
    return -np.array(     [ (1 - s[0])**2 + 100*(s[1] - s[0]**2)**2  ]  )
    
