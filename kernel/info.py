'''
Created on Sep 8, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import math 
import numpy as np

import kernel.algorithm as alg
import kernel.aux as aux
import kernel.container as cot
import kernel.kriging as kg




def information_gain(x , sampler):
    '''
    calculate the expected information
    gained by evaluating the log likelihood
    at the point x. This information gain is
    the sum of two quantities, one being an
    average wrt the dist of values at x and the 
    other being a space integral.
    
    :param x:
        the position in which we want to calculate information gaing
    
    :param sampler:
        the sampler object representing the data we've seen so far
    
    returns
    
    * ``infoGain`` - the expected value (wrt the kriged distribution
     *at x*) of the KL divergence between the current posterior and 
     the posterior had we known the log likelihood at x.     
    '''
    
    
    
    # create a new copy of the container
    specs = sampler.specs.get_copy()
    
    # kriged log-likelihood and its variance
    px , vx = kg.kriging(x, specs) 

    # add a point so we may calculate Z(D,x,E[L])
    specs.add_pair( x, np.array([px])  )
    
    # get (effectively) 10 independent samples
    sampler.run_mcmc(sampler.decorTime*8)

    # get parameters and data of the run:
    nwalkers = sampler.nwalkers # number of walkers
    burn = sampler.burn # number of burn in steps   
    chain = sampler.sam.chain # chain of  the run: shape = (nwalkers, nsteps, dim)
    _ , nsteps , _ = chain.shape # number of steps in the chain
    nsteps = nsteps - burn # since we chop off the burn in time
    chain = chain[:,burn:,:] # get rid of the burn in steps
#     print(sampler.sam.blobs.len)
    
    # calculate the space mean
    walkersMean = 0
#     walkersMeanDiff = 0
    for i in range(nwalkers):
        for t in range(nsteps):
            w = chain[i,t,:]
#             value = sampler.blobs[i][t]
#             if (value != kg.kriging(w, sampler.specs )[0]):
#                 print("not good")
            walkersMean  = walkersMean + kg.kriging(w, sampler.specs )[0] - kg.kriging(w, specs )[0] 
#             walkersMeanDiff  = walkersMean + kg.kriging(w, sampler.specs )[0] - kg.kriging(w, specs )[0] 
    
    walkersMean = walkersMean/(nwalkers*nsteps)
    
    # create the samples from the dist at x
    numNormalSamples = 70
    normalSamples = np.random.normal(px, vx , numNormalSamples)
    
    # calculate the local mean
    normalMean = 0
    for s in normalSamples:
        specs.change_F(s) # incorporate the sample to data set
        normalMean = normalMean + math.log(specs.get_normalization()[0]
                                           /sampler.specs.get_normalization()[0])
    normalMean = normalMean/numNormalSamples
    
    
    infoGain = normalMean + walkersMean    
#     print("Information gain at " + str(x) + " is " + str(infoGain) )
    return  infoGain   

        
    
    