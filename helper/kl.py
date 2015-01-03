'''
Created on Dec 15, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import numpy as np
import math


def expected_KL(s, specs, specsTmp, samplerTmp, n = 2000):
        '''
        our target is 
        E[ KL( p_n+1 || p_n )  ]
        '''
        
        pts = np.array( [ -1.5 , -1.0 , -0.5, 0.0, 0.5, 1.0, 1.5 ])
        
        tot = 0.0
        for p in pts:
            
            # remove previous point
            specsTmp.remove_last_point()
            
            # create and add a new value
            mu , sigSqr = specs.kriging(s, True)[0:2]
            f = mu + p * math.sqrt(sigSqr)
            specsTmp.add_pair(s, f)
            
            phi = specsTmp.kriging 
            psi = specs.kriging
            
            # generate samples for KL
            samplerTmp.run_mcmc(n)
            phiSamples = samplerTmp.flatchain()
            
            tot = tot + get_KL(phi, psi, phiSamples)[0]
        
        return -tot
        
def get_KL(phi ,psi, phiSamples):
    '''
    we calculate the kl divergence:
    
    p: = exp(phi)/Z(phi) 
    q: = exp(psi)/Z(psi)
    
    :param phiSamples:
        samples from exp(phi)/Z(phi). shape is 
        (number of samples , dimension)
    '''
    
#     phiMinPsi = np.asarray( map (lambda x: phi(x) - psi(x) , phiSamples)  ) 
#     print(phiSamples)
#     tmp = [x for x in phiSamples.T]
#     print(tmp)

    
    # computational bottleneck
    phiMinPsi = np.asarray( [phi(x) - psi(x) for x in phiSamples] )
    
    # we'll divide by the square root of the number of samples
    n = phiMinPsi.shape[0]
    s = 1.0/math.sqrt( n )
    
    #preparations...
    sumAvg = np.mean(phiMinPsi) # sample mean
    sumStd = np.std (phiMinPsi) # sample std
    Z , Zstd = Zpsi_over_Zphi(-phiMinPsi) # sample mean AND std!!!
    logZ = math.log(Z)

    # bars on the sum term
    sumLowBar    = sumStd*s
    sumHighBar   = sumStd*s
    

    # bars on log term (a bit sketchy...)
    logLowBar    = logZ - math.log( max(1e-20, Z - Zstd*s) )
    logHighBar   = math.log( Z + Zstd*s ) - logZ
    

    # KL divergence
    kl           = math.log(Z) + sumAvg
    klLowBar     = sumLowBar + logLowBar
    klHighBar    = sumHighBar + logHighBar
    

    return [kl, klLowBar , klHighBar,
                sumAvg, sumLowBar, sumHighBar,
                math.log(Z), logLowBar, logHighBar]
    
def Zpsi_over_Zphi(psiMinPhi):
    '''
    calculate the quotient of normalizatio constants.
    Z(phi) = int exp(phi(x))dx,
    Z(psi) = int exp(psi(x))dx.
    return the quotient and its estimated standard
    deviation (square root of sample variance)
    '''
    
    maxFactor    = np.max( psiMinPhi )
    
    if maxFactor > 200:
        
        # modified exponents
        modExp = np.exp( psiMinPhi - maxFactor )
        
        # the sample mean
        avg    =  np.mean(modExp ) * math.exp(200)
        
        # the sample variance
        std    =  np.std( modExp ) * math.exp(200)
        
        
    elif maxFactor < -200:
        
        avg = 1.0
        std = 0.01
    else:
        
        # modified exponents
        modExp = np.exp( psiMinPhi - maxFactor )
        
        # the sample mean
        avg    =  np.mean(modExp ) * math.exp(maxFactor)
        
        # the sample variance
        std    =  np.std( modExp ) * math.exp(maxFactor)
        
        
    #give back the love y'all!!!
    return avg, std

