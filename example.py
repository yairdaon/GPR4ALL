'''
Created on Sep 24, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np

import kernel.container as cot
import kernel.sampler as smp
import preparations
callsToLL = 0

# An  example of using this package. 
# Also, look at rap.py - very(!!) short.

#===============================================================================
# A. Our fake log likelihood
#===============================================================================
def minus_norm_squared_LL(s, *args, **kwargs):
    global callsToLL
    callsToLL +=1
    print("-------------------------------------------")
    print( str(callsToLL) +" calls to true log-likelihood function.")
    print("Here we pretend we use args:")
    print(args)
    print("Here we pretend we use kwargs: ")
    print(kwargs)
    print("-------------------------------------------")
    print("")
    
    return -4*np.linalg.norm(s)**2
        
        
            

#===============================================================================
# B. Creating the object that will hold the data
#===============================================================================

# Fake parameters that we pretend the log-likelihood needs.
args = [1 ,3 ,6]
kwargs = {'hi!' : 2, 'my name is': 4, 'what?': 6}

# create a container to hold everything, explanations below.
specs = cot.Container( minus_norm_squared_LL, r=2.4, args=args, kwargs=kwargs)
# 1st argument - your true log-likelihood.

# M determines the decay rate of your prior # if you don't want that,
# ignore it and choose another prior by (make sure it decays to minus infinity!!):
# specs.set_prior(  <put prior here>  )

# r is the characteristic length scale. set it to what you find right or
# use the default: 1.3

#If you don't want to use extra parameters, you may erase the irrelevant part:
#specs = cot.Container( made_up_LL   r=2.4)


# you should give the container a prior that decays with the same
# exponent as your function, otherwise, problems may (and will!) ensue.
specs.set_prior( lambda x: -np.linalg.norm(x)**2 , lambda x: -2*x)








#===============================================================================
# C. Three ways to add data
#===============================================================================

# (1) We know x1 and the log-likelihood at x1 , denoted f1:
# take x1 in R^d ... 
x1 = np.array( [2 ,3 ])
#... its log-likelihood ...
f1 = -13
# ...add both to the data set.
specs.add_pair(x1, f1) 



# (2) If we must add x2 but don't know the value of its log-likelihood
x2 = np.array( [11,-3])
specs.add_point( x2 )


# Before presenting the third way we have to
# create a sampler with the data and specifications.
sampler = smp.Sampler(specs, nwalkers=6, burn=100)

# (3) If we trust the sampler to choose the next point wisely
sampler.learn() # now specs has three points!!!




#===============================================================================
# D. Sample
#===============================================================================
# Sample a bunch:
batch = sampler.sample_batch() 
# the shape of a batch is (nwalkers, ndim) = (6 , 2) here.

# Sample one:
one = sampler.sample_one()

print("A single sample from the posterior: "+ str(one))
print("")
print("A batch sampled from the posterior. the shape is (nwalkers , ndim). ")
print(batch)
print("Don't forget to take a look at rap.py - it is super short.")


#===============================================================================
# Perform tests
#===============================================================================
import tests.testers_choice
