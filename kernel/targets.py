'''
Created on Oct 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import math
from math import exp as xp
import numpy as np


pi2 = math.pi/2
xp100 = math.exp(100)

def goodmans_criterion(s , specs, variances):
    '''
    use goodman's criterion
    '''
    n = float(len(variances))
    v = specs.kriging(s, var=True)[1]
    
    numerator   = (n/n+1)*np.mean( np.exp( variances ) ) + v/(n+1)
    
    halfVars = variances/2.0
    denominator = (n/n+1)*np.mean( np.exp( halfVars ) ) + 0.5*v/(n+1)
    denominator = denominator*denominator
    
    result = numerator/denominator - 1.0
    if result < 0:
        print("shit it is too small!!! " + str(result))
    
    return result    

def exp_krig_sigSqr(s ,specs):
    '''
    this method returns the target
    -exp(krig(s))*sigSqr(s)
    and its gradient
    ''' 
    
    #preparatory calculations
    krig , sigSqr , gradKrig , gradSigSqr= specs.kriging(s, grads=True)
    
    if krig > 100:
        func = -xp100*sigSqr
        grad = -xp100*gradSigSqr
    else: 
        tmp = xp(krig)
        func     = -tmp*sigSqr
        grad     = -tmp*( gradKrig*sigSqr + gradSigSqr)
    
    return func  , grad 
exp_krig_sigSqr.desc = "-exp(krig)sig^2"
   
    


            
def krig_sig(s , specs):
    '''
    this makes sense if the function you're
    trying to interppolate has is non positive
    '''
    
    #preparatory calculations
    krig , sigSqr , gradKrig , gradSigSqr = specs.kriging(s, grads=True)
    
    func = -krig/sigSqr 
    grad =   krig*gradSigSqr/(sigSqr*sigSqr) - gradKrig/sigSqr
    
    return func , grad
krig_sig.desc = "krig*sig^2"


def atan_sig( s , specs):
    '''
    this method returns the target
    -[arctan(krig(s)) + pi/2]*sigmaSquare(s)
    and its gradient
    '''

    #preparatory calculations
    krig , sigSqr , gradKrig , gradSigSqr = specs.kriging(s, grads=True)
    atanKrig = math.atan(krig)
    
    func     = -(atanKrig + pi2)*sigSqr 
    grad     = -(sigSqr*gradKrig/(1 + krig*krig) 
                        + (atanKrig + pi2)*gradSigSqr )

    return func  , grad  
atan_sig.desc = "-(atan(krig) +pi/2)*sig^2"
         
    
def mod_atan_sig( s ,specs):
    '''
    this method returns the target
    -{[arctan(krig(s)) + pi/2]*krig(s) + 1}*sigmaSquare(s)
    and its gradient
    ''' 
    
    #preparatory calculations
    krig , sigSqr , gradKrig , gradSigSqr= specs.kriging(s, grads=True)
    atanKrig = math.atan(krig)
     
    func     = -((atanKrig + pi2)*krig + 1)*sigSqr 
    grad     = -gradSigSqr*func   -    sigSqr*    (sigSqr*gradKrig/(1 + krig*krig) 
                                                           + (atanKrig + pi2)*gradSigSqr )
    
    return func  , grad 
mod_atan_sig.desc = "-{[arctan(krig) + pi/2]*krig + 1}*sig^2"
    

def const (s ,specs):
        '''
        a BS const target
        '''
        return 0, 0 ,0 ,0
const.desc = "constant zero" 