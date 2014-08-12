'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import kernel.config as cfg
import kernel.type as type


class Test(unittest.TestCase):
    '''
    if this does not work, it is most likely because you don't have 
    ffmpeg and\or vlc. go to the tearDown method to change these to
    whatever you have in your system.
    
    this unit test creates a movie. run it and see for yourself!!
    '''

    def setUp(self):
        '''
        this is where most of the setup is done for the movie
        '''
        
        # for reproducibility
        np.random.seed(1792) 
        
        # tell the OS to prepare for the movie and the frames
        os.system("rm -f Data/Movie2DSurfaceFrames/*.png") 
        os.system("rm -f Data/Movie2DContourFrames/*.png")     
    
        
        
        #     Initializations of the container object
        
        CFG = cfg.Config()
        
        # The length scale parameter in the Gaussian process covariance function.
        r = 2
        CFG.setR(r) 
            
        # the size of the box outside of which the probability is zero
        self.M = 3
        CFG.setM(self.M)

        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [ 0 , 0 ] ) )
        StartPoints.append( np.array( [0.5,1.0] ) )
        
        # the true log-likelihood function
        # CHANGE THIS IF YOU WANT YOUR OWN LOG-LIKELIHOOD!!!        
        CFG.setLL( truth.logRosenbrock2D )
        self.f = CFG.LL # the true log-likelihood function 
        
        for point in StartPoints:
            CFG.addPair( point, CFG.LL(point) )
        
        # we use algorithm 2.1 from Rasmussen & Williams book
        CFG.setType( type.RASMUSSEN_WILLIAMS  )
        #CFG.setType( type.AUGMENTED_COVARIANCE)
        
        # keep the container in scope so we can use it later
        self.sampler = smp.Sampler( CFG )
        self.CFG = CFG

        
    def tearDown(self):
        '''
        after the test was run - we create the movie.
        you need two programs installed on your machine to make this work:
        you need ffmpeg to create the movie from the frames python saves 
        and you need vlc to watch the movie
        feel free to change these two lines here according to whatever
        programs you have installed in your system
        '''
        
        # delete previous movie
        os.system("rm -f graphics/Movie2DSurface.mpg")   
        os.system("rm -f graphics/Movie2DContour.mpg")     
        
        # create new movie 
        os.system("ffmpeg -i Data/Movie2DSurfaceFrames/Frame%d.png graphics/Movie2DSurface.mpg") 
        os.system("ffmpeg -i Data/Movie2DContourFrames/Frame%d.png graphics/Movie2DContour.mpg") 
        
        #play new movie
        #os.system("vlc Movie2.mpg")     


    def testMovie2D(self):
        '''
        create a 2D movie, based on the data we put in the container object 
        in the setUp method this method does all the graphics involved
        since this is a 2D running for lots of points might take a while
        '''
        
        # The number of evaluations of the true likelihood
        # CHANGE THIS FOR A LONGER MOVIE!!!
        nf    =  6   
        
        # the bounds on the plot axes
        # CHANGE THIS IF STUFF HAPPEN OUTSIDE THE MOVIE FRAME
        xMin = -self.M
        xMax = self.M
        zMax = 50
        zMin = -500
        
        # create the two meshgrids the plotter needs
        a  = np.arange(xMin, xMax, 0.4)
        b  = np.arange(xMin, xMax, 0.4)
        X, Y = np.meshgrid(a, b)
        
        # we create each frame many times, so the movie is slower and easier to watch
        delay = 4
        
        # allocate memory for the arrays to be plotted
        kriged = np.zeros( X.shape )
        
        # allocate a two dimensional point, for which we calculate kriged value
        p = np.zeros(2)
        
        # create frames for the ffmpeg programs
        for frame in range (nf+1):

            # create the kriged curve 
            for j in range(len(a)):
                for i in range(len(b)):
                    p[0] = X[j,i]
                    p[1] = Y[j,i]    
                    kriged[j,i] = kg.kriging( p ,self.CFG )[0]
                                
            # create surface plot
            fig1 = plt.figure( frame*2 )
            ax1 = fig1.add_subplot(111 , projection='3d')
            
            ax1.plot_wireframe(X, Y, kriged, rstride=10, cstride=10)
            ax1.set_xlim(xMin, xMax)
            ax1.set_ylim(xMin, xMax)
            ax1.set_zlim(zMin, zMax)
            
            xs = np.ravel( np.transpose( np.array( self.CFG.X ) )[0] )
            ys = np.ravel( np.transpose( np.array( self.CFG.X ) )[1] )
            zs = np.ravel( np.transpose( np.array( self.CFG.F ) )    )
            ax1.scatter(xs, ys, zs)
    
            PlotTitle1 = 'Surface of interpolated Rosenbrock. ' + str(frame) + ' samples. r = ' + str(self.CFG.r) + " Algorithm: " + self.CFG.algType.getDescription()
            plt.title( PlotTitle1 )
            #textString = 'using  ' + str(frame ) + ' sampled points' 
            #plt.text( textString)
            plt.legend(loc=1,prop={'size':7}) 
            
            
            # create contour
            fig2 = plt.figure( frame*2 + 1 )
            ax2 = fig2.add_subplot(111)# , projection='2d')
            
            cs = ax2.contour(X, Y, kriged, levels = np.arange(zMin , zMax , 25)  ) 
            ax2.clabel(cs, fmt = '%.0f', inline = True) 
            ax2.scatter(xs, ys)
            PlotTitle2 = 'Contours of interpolated Rosenbrock. ' + str(frame) + ' samples. r = ' + str(self.CFG.r) + " Algorithm: " + self.CFG.algType.getDescription()
            plt.title( PlotTitle2 )
            # save the plot several times
            for k in range(delay):   
                FrameFileName1 = "Data/Movie2DSurfaceFrames/Frame" + str(frame*delay + k) + ".png"
                FrameFileName2 = "Data/Movie2DContourFrames/Frame" + str(frame*delay + k) + ".png"

                fig1.savefig(FrameFileName1)
                fig2.savefig(FrameFileName2)

                if (frame*delay + k) % 10 == 0:
                    print( "saved " + FrameFileName1 + " and " +FrameFileName2 + ".  " + str(frame*delay + k) +  " / " + str((nf+1)*delay) )
            
            plt.close( frame*2     )
            plt.close( frame*2 + 1 )

            # IMPORTANT - we sample from the kriged log-likelihood. this is crucial!!!!
            self.sampler.sample() 
            


   

plt.show()
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()