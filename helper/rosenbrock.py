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
from functools import partial as partial

from kernel.kriging import kriging as kriging

def rosenbrock_2D(s):
    '''
    the (negative of the) 2D Rosenbrock function. 
    google it if you've never heard of it.
    THIS FUNCTION ALSO APPEARES IN THE TRUTH MODULE
    this LL is actually normalized. So the integral of
    exp( the function below ) = 1
    '''
    return -(  (1 - s[0])**2 + 100*(s[1] - s[0]**2)**2 + 0.31415926535897931 ) 


def sample_rosenbrock(n=1):
    '''
    generate a 2D sample from the exponent of 
    the Rosenbrock function
    '''
    
    sig1 = 0.70710678118654757 # sqrt(1/2)
    sig2 = 0.070710678118654752 # sqrt(1/200)
    x =  rndn(1  , sig1, n) 
    y =  rndn(x*x, sig2, n)
    return x , y

def rosenbrock_KL(specs, nSamp = 50000):
    
    # all the samples
    xSamples , ySamples = sample_rosenbrock(nSamp)
    
    # calculates the difference of log likelihoods
    def phi_minus_psi(specs, x, y):
        '''
        a.k.a phi minus psi
        '''
        s = (x,y)
        return rosenbrock_2D(s) -kriging(s, specs)      
    
    
    # differences between log likelihoods. IS THIS THE BOTTLENECK?
    phiMinPsi = np.asarray( map(partial(phi_minus_psi, specs), xSamples ,ySamples  ) )
    
    # means and std devs for error bars
    mu  = np.mean(phiMinPsi) # rosenbrock LL - kriged LL
    sig = math.sqrt(  np.std(phiMinPsi)  )

    # holds the negative exponent of the above
    expPsiMinPhi = np.exp(-phiMinPsi)
    
    # pull out the max to avoid overflow
    maximal   = max(expPsiMinPhi)
    modified  = expPsiMinPhi/maximal
    Z         = np.mean(modified)*maximal 
    tau       = np.std(modified)*maximal
    logZ      = math.log(Z)

    
    KL      =  mu    +  logZ
    lowBar  =  sig   +  logZ  -  math.log( max(1e-20, Z - tau)   ) 
    highBar =  sig   -  logZ  +  math.log(            Z + tau    )
    sumTerm =  mu
    lowSum  =  sig
    highSum =  sig
    lowLog  =  logZ           -  math.log( max(1e-20, Z - tau)   ) 
    highLog = -logZ           +  math.log(            Z + tau    )
#     assert lowLog > 0
#     assert highLog >0
#     assert sig > 0
#     print("         mu                sig                 logZ               lowLog            highLog")
#     print( mu, sig, logZ , lowLog, highLog)
#     print('- -- - - - -  - - -')

    return [KL , lowBar , highBar , sumTerm, lowSum , highSum , logZ , lowLog , highLog]

