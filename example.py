'''
Created on Sep 24, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np

import kernel.container as cot
import kernel.sampler as smp


# An  example of using this package. 
# Also, look at rap.py - very(!!) short.

#lets say our true LL is the following
def minus_norm_squared_LL(s, *args, **kwargs):
    
    print("We called the true log likelihood function.")
    print("Here we pretend we use the list:")
    for x in args:
            print(x)
    
    print("Here we pretend we use kwargs: ")
    print(kwargs)
    print("")
    result = -np.linalg.norm(s)**2
        
    return result
            

    
    
# Our data:
# x in R^d where we know LL ...
x1 = np.array( [2 ,3 ])
x2 = np.array( [11,-3])

#...and the corresponding log-likelihoods:
f1 = -13
f2 = -130 # we won't use it but it's still here

# Throw all parameters together.
l = [1 ,3 ,6]
d = {'hi!' : 2, 'my name is': 4, 'what?': 6}

# create a container to hold everything, explanations below.
specs = cot.Container( minus_norm_squared_LL , M=15, r=2.4, args=l, kwargs=d)

# 1st argument - your true log-likelihood.

# M is such that if ||x||_{inf} > M the log-likelihood is -inf.
# make sure your data does not violate this restriciton.
# if you don't want that restriction, ignore it and choose another prior by uncommenting:
# specs.set_prior( lambda x: -np.linalg.norm(x)**2)

# r is the characteristic length scale. set it to what you find right or
# use the magical default: 1.3

# parameters - everything you need to make your log-likelihood give results.
#If you don't want to use extra parameters, just erase the irrelevant part:
#specs = cot.Container( made_up_LL , M=15, r=2.4 ,X=X, F=F)







# three ways to add data:

# if we have the corresponding log-likelihood
specs.add_pair(x1, f1) 

# if we must add x2 but don't have its log-likelihood
print("First call to our true log-likelihood function!!!")
specs.add_point(x2)  

# we'll see the third way below...





# create a sampler with this data and specifications, comment below.
sampler = smp.Sampler(specs, nwalkers=6, burn=100, useInfoGain=False)
# simple, except for the useInfoGain=False. This curretnly runs too slow.
# we may use it in the future. False is the default value but change it
# if you feel like.

# third way to add data: if we want to calculate log-likelihood 
# and we trust the sampler to choose it wisely
print("Second call to our true log-likelihood function!!!")
sampler.learn() # now specs has three points!!!








# get ready to sample. You can sample a bunch (faster):
batch = sampler.sample_batch() 
# the shape of a batch is (nwalkers, ndim) = (6 , 2) here.

# and you can sample one (somewhat slower):
print("A single sample from the posterior: "+ str(sampler.sample_one()))
