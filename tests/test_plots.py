'''Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!

a test suite that tests stuff works by creating simple 1D plots of
things.

'''

import matplotlib.pyplot as plt
import numpy as np
import math

import gpr4all.container as cot
import gpr4all.truth as truth
import gpr4all.sampler as smp

def testKriging(self):
    '''create a plot that shows kriging

    '''
        
    # locations where we know the log-likelihood value
    X = []
    X.append(np.array([ 1.1]))
    X.append(np.array([ 1.0]))
    X.append(np.array([-1.1]))
    X.append(np.array([-3.0]))
    
    # create the container object and populate it...
    specs = cot.Container( truth.sin_1D  )
    
    # set the prior so the pics look nice
    specs.set_prior( lambda x: -x*x/20.0 , lambda x: -x/10.0)
    
    for v in X: 
        specs.add_point(v)#... with (point, value) pair...
        
    # allocating memory
    x = np.arange(-4, 2, 0.05)
    n = len(x)
    f = np.zeros( n )
    upper = np.zeros( n )
    lower = np.zeros( n ) 
    fx     = np.zeros( n )
    upperx = np.zeros( n )
    lowerx = np.zeros( n )
        
    # the default prior
    prior = np.ravel( specs.prior(x) )
    
    # calculate the curves for the given input
    for j in range(0,n):    
            
        # do kriging, get avg value and std dev
        krig , sigSqr = specs.kriging(x[j], var=True) 
        f[j] = krig # set the interpolant
        sig = math.sqrt(sigSqr)
        upper[j] = krig + 1.96*sig # set the upper bound
        lower[j] = krig - 1.96*sig # set lower bound
        
    specs.add_pair(np.array([-3.5]), specs.kriging(np.array([-3.5]), grads=False, var=False))
    for j in range(0,n):    
            
        # do kriging, get avg value and std dev
        krig , sigSqr = specs.kriging(x[j], var=True) 
        fx[j] = krig # set the interpolant
        sig = math.sqrt(sigSqr)
        upperx[j] = krig + 1.96*sig # set the upper bound
        lowerx[j] = krig - 1.96*sig # set lower bound
            
            
    # do all the plotting here
    curve1  = plt.plot(x, f, label = "kriged value")
    curve2  = plt.plot(x, upper, label = "1.96 standard deviations")
    curve3  = plt.plot(x, lower)
    curve4  = plt.plot(x, prior, label = "prior")
    curve5  = plt.plot(x, fx, label = "with an extra point")
    curve6  = plt.plot(x, upperx, label = "1.96 std with point")
    curve7  = plt.plot(x, lowerx)
    
    plt.plot( specs.X, specs.F, 'bo', label = "sampled points ")
    
    plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
    plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
    plt.setp( curve3, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
    plt.setp( curve4, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
    plt.setp( curve5, 'linewidth', 1.5, 'color', 'g', 'alpha', .5 )
    plt.setp( curve6, 'linewidth', 1.5, 'color', 'y', 'alpha', .5 )
    plt.setp( curve7, 'linewidth', 1.5, 'color', 'y', 'alpha', .5 )
        
    plt.legend(loc=1,prop={'size':7})    
    plt.title("Kriging with bounds ")
    plt.savefig("graphics/testKriging: Kriged LL")
    plt.close()
    
def testLikelihood(self):
    '''test and plot kriging with and show the resulting (unnormalized)
        likelihood function. Here we let our sampler choose points on
        its own.

    '''
        
    # for reproducibility purposes
    np.random.seed( 1243 )
    
    # create the container object
    specs = cot.Container ( truth.double_well_1D )
    
    # note that this prior DOES NOT decay like the 
    # true LL. still, the plot of the likelihood looks good
    specs.set_prior( lambda x: -x*x , lambda x: -2*x)
       
    # quick setup
    pt = 2*np.ones(1)
    specs.add_point( pt)
    specs.add_point(-pt)

    # create sampler...
    sampler = smp.Sampler ( specs )
    k =  11 # ...decide how many initial points we take to resolve the log-likelihood
    for j in range(k): 
        print( "Sample " + str(j+1) + " of " + str(k))
        sampler.learn() # ... sample, incorporate into data set, repeat k times.
   
    # allocating memory
    M = 4
    x = np.arange(-M, M, 0.05)
    n = len(x)
    f = np.zeros( n )
    true = np.zeros( n )
    prior = np.ones( n )

    # calculate the curves for the given input
    for j in range(0,n):    
            
        # do kriging, get avg value and std dev
        krig = specs.kriging(x[j]) 
        f[j] =  (krig) # set the interpolant
        prior[j] = specs.prior(x[j])   # set the limiting curve
        true[j]  = specs.trueLL( x[j])  
        
    #move to normal, non-exponential, scale
    fExp = np.exp(f)
    priorExp = np.exp(prior)
    trueExp = np.exp(true)
    samplesExp = np.exp(np.asarray( specs.F ))
    X =  np.asarray( specs.X )
        
        
    # first plot
    curve1  = plt.plot(x, f, label = "kriged LL")
    curve2  = plt.plot(x, true, label = "true LL")
    curve3  = plt.plot(x, prior, label = "prior")
    plt.plot(  specs.X ,    specs.F  , 'bo', label = "sampled points ")
    
    plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
    plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
    plt.setp( curve3, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
    
    plt.legend(loc=1,prop={'size':7})    
    plt.title("Learned log likelihood")
    plt.savefig("graphics/testLikelihood: Learned log-likelihood")
    plt.close()
    
    # second plot
    curve4  = plt.plot(x, fExp, label = "exp(kriged LL)")
    curve5  = plt.plot(x, trueExp, label = "(unnormalized) likelihood")
    curve6  = plt.plot(x, priorExp, label = "exp(prior)")
    plt.plot(  X ,   samplesExp  , 'bo', label = "sampled points ")
    
    plt.setp( curve4, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
    plt.setp( curve5, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )        
    plt.setp( curve6, 'linewidth', 3.0, 'color', 'b', 'alpha', .5 )
    
    plt.legend(loc=1,prop={'size':7})    
    plt.title("Learned (unnormalized) Likelihood")
    plt.savefig("graphics/testLikelihood: Learned Likelihood")
    plt.close()
