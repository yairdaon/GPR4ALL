'''
Created on Dec 15, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import numpy as np
import math

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

    # bars on the sum term
    sumLowBar    = sumAvg - sumStd*s
    sumHighBar   = sumAvg + sumStd*s
    
    # bars on log term (a bit sketchy...)
    logLowBar    = math.log( max(1e-20, Z - Zstd*s) )
    logHighBar   = math.log( Z + Zstd*s )
    
    # KL divergence
    kl           = math.log(Z) + sumAvg
    klLowBar     = sumLowBar + logLowBar
    klHighBar    = sumHighBar + logHighBar
#     
#     if Zstd >= Zavg:
#         lowBar = sumStd
#         lowLog = 0
#     else:
#         lowBar = sumStd + math.log(1 - Zstd/Zavg)
#         lowLog = Zavg - Zstd
    
    
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
 
    if maxFactor < 200:
        maxFactor = 0
    else:
        print("please note that we factor out " + str(maxFactor) +" for numerical stability")
    
    #modified exponents
    modExp = np.exp( psiMinPhi - maxFactor )
    
    # the sample mean
    avg    =  np.mean(modExp ) * math.exp(maxFactor)
    
    # the sample variance
    std    =  np.std( modExp ) * math.exp(maxFactor)
    
    #give back the love y'all!!!
    return avg, std

