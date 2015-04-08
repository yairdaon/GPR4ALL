'''
Created on Apr 8, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''


import numpy as np
import math
import matplotlib.pyplot as plt	

#import gpr4all.container as cot
#import gpr4all.sampler as smp
#import gpr4all.truth as truth
#import gpr4all._aux as _aux
#import gpr4all._g as _g
#import gpr4all.rosenbrock as rose



import container as cot
import sampler as smp
import truth as truth
import _aux as _aux
import _g as _g
import rosenbrock as rose


# container setup
specs = cot.Container( truth.gaussian_1D , d = 1.0 , r = 1.0 )
specs.set_prior( lambda x: 0.0 , lambda x: 0.0)
specs.add_point( np.array([ -0.75]) )
specs.add_point( np.array([  0.75]) )
specs.set_matrices()

sampler = smp.Sampler( specs )
sampler.learn()
sampler.learn()
sampler.learn()
sampler.learn()
sampler.learn()
sampler.learn()
print(specs.X)
