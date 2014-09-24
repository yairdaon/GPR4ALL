'''
Created on Sep 8, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import math
import time


import kernel.container as cot
import kernel.info as nfo
import kernel.kriging as kg
import kernel.sampler as smp
import kernel.algorithm as alg
import kernel.truth as truth

class Test(unittest.TestCase):

 
    def setUp(self):
        '''
        create list of points, mainly
        '''
        # for reproducibility
        np.random.seed(162)    
        X = [] 
        X.append( np.array([ 0, 1.7]) )
        X.append( np.array([ 1.7, 0]) )
        X.append( np.array([ 0,-1.7]) )
        X.append( np.array([-1.7, 0]) )
        self.X = X

    def testInformationGain(self):
        '''
        test the function that calculates information gain
        '''
        
        # create the container object ...
        specs = cot.Container( truth.norm_2D , M=2 , r=1.4)
        
        # ...add the points to it ...
        for x in self.X:
            specs.add_point(x)

        # ...set the characteristic distance....
        sampler = smp.Sampler( specs , nwalkers=16 , burn=500 )
        
        start = time.time()
        self.X.append( np.array([0,0]) )
        for x in self.X: 
            infoGain = nfo.information_gain(x, sampler)
            print("Information gain for x = " + str(x) + " is " +
                                 str(infoGain))
        end = time.time()
        print("Calculating information gain takes " + str((end - start)/5) + " seconds on average")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testgetNormalization']
    unittest.main()