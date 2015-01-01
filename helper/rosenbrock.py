'''
Created on Nov 21, 2014

A module to hold all rosenbrock related methods:
a likelihood function, a sampler and a Kullback-Liebler 
divergence estimating function.

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np
from  numpy.random import normal as rndn
import math


sig1 = math.sqrt(0.5) # sqrt(1/2)
sig2 = sig1/10        # sqrt(1/200)
logPiTen = math.log(math.pi/10)

def rosenbrock_2D(s, array = False):
    '''
    the (negative of the) 2D Rosenbrock function. 
    google it if you've never heard of it.
    this LL is actually normalized. So the integral of
    exp( the function below ) = 1
    '''
    
    if array:
        s0 = s[0,:]
        s1 = s[1,:]
        return -(  (1 - s0)**2 + 100*(s1 - s0**2)**2 ) - logPiTen
    else:
        return -(  (1 - s[0])**2 + 100*(s[1] - s[0]**2)**2 ) - logPiTen  



def sample_rosenbrock(n=1):
    '''
    generate a 2D sample from the exponent of 
    the Rosenbrock function
    '''
    
    x =  rndn(1.0  , sig1, n) 
    y =  rndn(x*x, sig2, n)
    return np.asarray([ x , y ] ).T
