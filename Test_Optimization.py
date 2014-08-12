'''
Created on Aug 12, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''


import unittest
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import os.path
import pickle

import kernel.sampler as smp
import kernel.truth as truth
import kernel.config as cfg
import kernel.type as type


class Test(unittest.TestCase):


    def setUp(self):
        '''
        this is where prparatory work is done
        '''
        
        # for reproducibility
        np.random.seed(1792) 
               
        #     Initializations of the container object
        
        CFG = cfg.Config()
        
        # The length scale parameter in the Gaussian process covariance function.
        r =0.5
        CFG.setR(r) 
            
        # the size of the box outside of which the probability is zero
        M = 10
        CFG.setM(M)
    
        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [  7.54 , -2.45  ] ) )
        StartPoints.append( np.array( [ -8.88 ,  4.47  ] ) )
        #StartPoints.append( np.array( [  1.55 ,  1.55  ] ) )
    
        # set the true log likelihood
        CFG.setLL( truth.rosenbrock2D )
        f = CFG.LL # the true log-likelihood function 
        
        for point in StartPoints:
            CFG.addPair( point, f(point) )
        
        # we use algorithm 2.1 from Rasmussen & Williams book
        CFG.setType(  type.RASMUSSEN_WILLIAMS  )
        #CFG.setType( type.AUGMENTED_COVARIANCE)
        
        # keep the container in scope so we can use it later
        self.CFG = CFG
        
        # create the sampler
        self.sampler = smp.Sampler( self.CFG )
        
        
        self.fname = "Data/roseData.pyc"    
        if not os.path.isfile(self.fname):
            print "Creating data for the Rosenbrock plot..."
                 
            # define the grid over which the function should be plotted (xx and yy are matrices)
            xx, yy = pylab.meshgrid(
            pylab.linspace(-M,M, 1001),
            pylab.linspace(-M,M, 1001))
            
            # fill a matrix with the function values
            zz = pylab.zeros(xx.shape)
            for i in range(xx.shape[0]):
                for j in range(xx.shape[0]):
                    rosenbrock = self.CFG.LL( [ xx[i,j], yy[i,j] ] )
                    zz[i,j] = math.log( 0.5 - rosenbrock )
                    
            f = open(self.fname , 'w')
            pickle.dump( [xx,yy,zz] , f )  
            f.close()
                   
    def testOptimization(self):
        '''
        create a 2D movie, based on the data we put in the container object 
        in the setUp method this method does all the graphics involved
        since this is a 2D running for lots of points might take a while
        '''
        
        # number of samples
        ns = 99
        
        # number of initial
        ni = len(self.CFG.X)
        
        # create get samples
        for sample in range (ni, ns+ni): 
            if sample % 10 == 0:
                    print( "Taking sample " + str(sample) + " / " + str(ns + ni) )
            self.sampler.sample()
#             
#             last = self.CFG.X[sample-1]
#             oneBefore = self.CFG.X[sample - 2]
#             diff = np.linalg.norm( last - oneBefore )
#             if diff < 0.01:
#                 print("exp!!! in " + str(sample) + " sample.")
#                 self.CFG.setLL (  lambda x, f = self.CFG.LL: math.exp(f(x))  )
#                 for f in self.CFG.F:
#                     f = np.exp(f)
            #self.CFG.setR( math.sqrt(r) )    
                 
        print( "Done sampling." )
        
        xs = np.ravel( np.transpose( np.array( self.CFG.X ) )[0] )
        ys = np.ravel( np.transpose( np.array( self.CFG.X ) )[1] )
        zs = np.ravel( np.transpose( np.array( self.CFG.F ) )    )
        minInd = np.argmax(zs) #argmax, since we took the NEGATIVE of Rosenbtock function
        
        # load data of rosenbrock function
        print("Loading data of Rosenbrock function...")
        f = open(self.fname , 'r')
        xx, yy, zz = pickle.load(f)
        f.close()
        print("Finished loading the data.")
        
        # creating  the plot
        print("Creating and saving the plot...")
        plt.pcolor(xx,yy,zz) # ... the color plot ...
        plt.plot( xs, ys, 'k', lw=0.05) # ... the lines ...
        plt.scatter(xs,ys,s=0.5) # ... and the scatter plot.
        
        # making the plot understandable:
                
        #... creating the annotation
        annotation = "Minimal value: "  + str(-zs[minInd]) + ". True minimum is 0."
        plt.annotate( annotation , xy=(xs[minInd], ys[minInd]),  xycoords='data',
                xytext=(0.8, 0.95), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                ) 
        
        #creating title
        title = "Optimizing the Rosenbrock function. " + str(ns) + " samples, Algorithm: " + self.CFG.algType.getDescription()
        plt.title( title )
        
        # saving
        plt.savefig("graphics/Test_Optimization: Samples and Convergence" ) 
        
        # Done!!!
        print("Done!")   

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
