'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import matplotlib.pyplot as plt
import numpy as np

import kernel.kriging as kg
import kernel.config as cfg
import kernel.truth as truth
#import kernel.type as type

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
        x1 =  np.array( [ 1.1 ] )
        x2 =  np.array( [ 1.0 ] )
        x3 =  np.array( [ -1.1] )
        x4 =  np.array( [ -3.0] )
        X.append(x1)
        X.append(x2)
        X.append(x3)
        X.append(x4)

        # create the container object and populate it...
        CFG = cfg.Config()
        for v in X: 
            CFG.addPair(v, truth.trueLL(v)) #... with (point, value) pair...
        #a.setType(type.RASMUSSEN_WILLIAMS) #... with the algorithm we use...
        #a.setType(type.AUGMENTED_COVARIANCE)
        #a.setType(type.COVARIANCE)
        r = 1.3
        CFG.setR(r) # ...with the location scale hyper parameter r...
        CFG.setMatrices() # ... and with the matrices the kriging procedure uses
        
        # the value of the kriged function "at infinity"
        limAtInfty, _ = kg.setGetLimit(CFG)

        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] ,CFG) 
            f[j] = v[0] # set the interpolant
            upper[j] = v[0] + 1.96*v[1] # set the upper bound
            lower[j] = v[0] - 1.96*v[1] # set lower bound
            limit[j] = limAtInfty # set the limiting curve
        
        # do all the plotting here
        curve1  = plt.plot(x, f, label = "kriged value")
        curve2  = plt.plot(x, upper, label = "1.96 standard deviations")
        curve3  = plt.plot(x, lower)
        curve4  = plt.plot(x, limit, label = "kriged value at infinity")
        plt.plot( CFG.X, CFG.F, 'bo', label = "sampled points ")
        
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.setp( curve3, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.setp( curve4, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
        
        plt.legend(loc=1,prop={'size':7})    
        plt.title("Kriging with bounds using " + CFG.algType.getDescription() )
        plt.savefig("graphics/Test_Plots: Kriged LL")
        plt.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPlots']
    unittest.main()