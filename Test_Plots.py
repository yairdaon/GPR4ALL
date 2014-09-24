'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import matplotlib.pyplot as plt
import numpy as np

import kernel.kriging as kg
import kernel.container as cot
import kernel.truth as truth
import kernel.algorithm as alg

class Test(unittest.TestCase):
    
    def testPlots(self):
        '''
        create a plot that shows kriging
        '''
        
        # allocating memory
        x = np.arange(-10, 10, 0.05)
        n = len(x)
        f = np.zeros( n )
        upper = np.zeros( n )
        lower = np.zeros( n )
        limit = np.ones( n )

        # locations where we know the function value
        X = []
        X.append(np.array([ 1.1]))
        X.append(np.array([ 1.0]))
        X.append(np.array([-1.1]))
        X.append(np.array([-3.0]))

        # create the container object and populate it...
        specs = cot.Container( truth.sin_1D )
        for v in X: 
            specs.add_point(v)#... with (point, value) pair...
        
        # the value of the kriged function "at infinity"
        limAtInfty, _ = kg.set_get_limit(specs)

        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] ,specs) 
            f[j] = v[0] # set the interpolant
            upper[j] = v[0] + 1.96*v[1] # set the upper bound
            lower[j] = v[0] - 1.96*v[1] # set lower bound
            limit[j] = limAtInfty # set the limiting curve
        
        # do all the plotting here
        curve1  = plt.plot(x, f, label = "kriged value")
        curve2  = plt.plot(x, upper, label = "1.96 standard deviations")
        curve3  = plt.plot(x, lower)
        curve4  = plt.plot(x, limit, label = "kriged value at infinity")
        plt.plot( specs.X, specs.F, 'bo', label = "sampled points ")
        
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.setp( curve3, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.setp( curve4, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Kriging with bounds using " + specs.algType.get_description() )
        plt.savefig("graphics/Test_Plots: Kriged LL")
        plt.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPlots']
    unittest.main()