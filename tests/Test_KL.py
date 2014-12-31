'''
Created on Nov 5, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest

import numpy as np
import math
import helper.kl as kl
class Test(unittest.TestCase):

 
    def testKL1D(self):
        '''
        check we can calculate KL div for two gaussians
        '''
         
        np.random.seed(19)
        Zquo = 2
        
        # first gaussian
        mu    = 1.11 # mean
        sigma = 1.23 # std
        sig2  = sigma*sigma # variance
        phi   = lambda x: -math.log(sigma) -(x-mu)*(x-mu)/(2*sig2) # log-likelihood
 
        # second gaussian
        nu    = 1.51 # mean
        tau   = 0.24 #std 
        tau2  = tau*tau # var 
        psi   = lambda x: -math.log(tau) -(x-nu)*(x-nu)/(2*tau2) + Zquo # log likelihood
 
        # samles from first gaussian
        phiSamples = np.random.normal(loc = mu, scale = sigma, size=1000000)
         
        # our procedure for calculating KL divergence
        klDiv  = kl.get_KL(phi, psi, phiSamples)
 
        # analytic KL div, from wikipedia
        trueKL = ((mu - nu)**2)/(2*tau2) + (sig2/tau2 - 1.0 -math.log(sig2/tau2))/2.0
        self.assertAlmostEqual(klDiv[0], trueKL, 1)
        
        # compare normalization constants
        trueZpsiOverZphi = math.exp(Zquo)
        self.assertAlmostEqual(trueZpsiOverZphi, math.exp(klDiv[6]), 2)

         
        
    def testKL2D(self):
        '''
        check we can calculate KL div for two gaussians
        '''

        np.random.seed(19)
        Zquo = 0.54
                 
        # first gaussian
        mu     = np.array( [ 1.21 , 2.23] ) # mean
        sigma  = np.random.normal(0,1,4).reshape(2,2)
        sigma  = np.dot(sigma, sigma.T)
        detSig = np.linalg.det(sigma) 
        sigInv = np.linalg.inv(sigma)
        phi   = lambda x: -0.5*math.log(detSig) -0.5* np.einsum( 'i, ij ,j ' , x-mu, sigInv, x-mu )# log-likelihood
 
 
        # second gaussian
        nu     = np.array( [ 1.31 , 2.13] ) # mean
        tau    = np.random.normal(0,1,4).reshape(2,2)
        tau    = np.dot(tau,  tau.T)
        detTau = np.linalg.det(tau) 
        tauInv = np.linalg.inv(tau)
        psi    = lambda x: -0.5*math.log(detTau) -0.5* np.einsum( 'i, ij ,j ' , x-nu, tauInv, x-nu ) + Zquo # log-likelihood
 
        # samles from first gaussian
        phiSamples = np.random.multivariate_normal(mu, sigma, size=50000000)
 
        # our procedure for calculating KL divergence
        klDiv  = kl.get_KL(phi, psi, phiSamples)
 
        # analytic KL div, from wikipedia
        truth = ( np.trace( np.dot(tauInv , sigma)) + 
                  np.einsum( 'i, ij ,j ' , nu-mu, tauInv, nu-mu ) -
                  2  -math.log(detSig/detTau)
                  )/2.0
 
        self.assertAlmostEqual(klDiv[0], truth, 1)
        
        # compare normalization constants
        trueZpsiOverZphi = math.exp(Zquo)
        self.assertAlmostEqual(trueZpsiOverZphi, math.exp(klDiv[6]), 1)
                 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testTestKL']
    unittest.main()
               
    
    
    
