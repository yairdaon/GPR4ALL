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
import kernel.sampler as smp
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
        infoGainAvg = 0
        
        start = time.time()
        
        for x in self.X: 
            infoGainAvg += nfo.information_gain(x, sampler)  
        
        infoGainOrigin = nfo.information_gain(np.array( [0,0] ), sampler)          
        
        end = time.time()
        
        infoGainAvg = infoGainAvg/5
        
        print("Info gain calculation takes " + str((end - start)/5) + " secs on average.")
        print("Info gain at origin is " + str(infoGainOrigin/infoGainAvg) +
                         " larger than at known points.")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testgetNormalization']
    unittest.main()