'''
Created on Sep 8, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import math 
import numpy as np

import kernel.algorithm as alg
import kernel.aux as aux
import kernel.config as cfg
import kernel.kriging as kg




def I(x , sampler):
    '''
    calculate the expected information 
    gained by evaluating the log likelihood
    at the point x
    '''
    
    # create a new copy of the container
    CFG = cfg.Config( sampler.CFG.trueLL )
    
    # we don't want to do learn, just do inference
    CFG.addSamplesToDataSet = False
    
    # or your favorite kriging algorithm
    CFG.setType(alg.RASMUSSEN_WILLIAMS)
    
    # copy the data from the old container file
    for i in range(len(sampler.CFG.X)):
        CFG.addPair( sampler.CFG.X[i] , sampler.CFG.F[i] )
    
    # create the samples from the dist at x
    #print("Generating normal samples with"),
    numNormalSamples = 70
    px , vx = kg.kriging(x, CFG) 
    print("Mean =  " + str(px) + ", variance = " + str(vx) + "..." )
    normalSamples = np.random.normal(px, vx , numNormalSamples)
    #print("done!")
    
    CFG.addPair( x, np.array([px])  )

    walkers = sampler.pos
    nwalkers = sampler.nwalkers

    walkersMean = 0
    for i in range(nwalkers):
        w = walkers[i,:]
        walkersMean  = walkersMean + kg.kriging(w, sampler.CFG)[0] - kg.kriging(w, CFG )[0]
    walkersMean = walkersMean/nwalkers
    
    normalMean = 0
    for s in normalSamples:
#         if (i+1) % 20 == 1:
#             print("Using " +str(i+1) + "th normal sample of " + 
#                   str(numNormalSamples))
        CFG.changeF(s) # incorporate the sample to data set
        normalMean = normalMean + math.log(CFG.getNormalization()[0]/sampler.CFG.getNormalization()[0])
    normalMean = normalMean/numNormalSamples
    
    return normalMean + walkersMean        

        
    
    