'''
Created on Jan 2, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np

import helper.kl as kl

import container as cot
import truth as truth
import sampler as smp


class Test(unittest.TestCase):


    def testInbereed(self):
        '''
        
        '''
        #     Initializations of the container object
        specs = cot.Container(truth.big_poly_1D )
    
        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [ 0 ] )  )
        StartPoints.append( np.array( [0.5] )  )
        
        for point in StartPoints:
            specs.add_point(point)
        
        sampler = smp.Sampler( specs )
        
        s = np.array( [2.3] )
        
#         breeder = kl.InbreedKL(  sampler )
#         print( breeder(s, 100) ) 

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
