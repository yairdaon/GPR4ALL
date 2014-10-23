'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import numpy as np
import os
import math

import matplotlib.pyplot as plt

import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import kernel.container as cot


class Test(unittest.TestCase):
    '''
    if this does not work, it is likely that don't have 
    ffmpeg. 
    this unit test creates a movie. run it and see for yourself!!
    '''

    def testMovie2D(self):
        '''
        create a 2D movie, based on the data we put in the container object 
        in the setUp method sthis method does all the graphics involved
        since this is a 2D running for lots of points might take a while
        '''
        
        # for reproducibility
        np.random.seed(1792) 
        
        # tell the OS to prepare for the movie and the frames
        os.system("rm -f Data/Movie2DSurfaceFrames/*.png") 
        os.system("rm -f Data/Movie2DContourFrames/*.png")     
        
        #     Initializations of the container object
        
        specs = cot.Container( truth.rosenbrock_2D, M =5 )
        
        # hard prior makes the movie look much nicer
#         specs.set_prior( lambda x: 0 if np.linalg.norm(x, np.inf ) < 5 else -np.inf   )
        M = specs.M
        
        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [ 0 , 0 ] ) )
        StartPoints.append( np.array( [0.5,1.0] ) )
        
        for point in StartPoints:
            specs.add_point( point )

        # keep the container in scope so we can use it later
        sampler = smp.Sampler( specs )
 
        # The number of evaluations of the true likelihood
        # CHANGE THIS FOR A LONGER MOVIE!!!
        nf    =  55   
        
        # the bounds on the plot axes
        # CHANGE THIS IF STUFF HAPPEN OUTSIDE THE MOVIE FRAME
        xMin = -M
        xMax = M
        zMax = 1000
        zMin = -7000

        
        # create the two meshgrids the plotter needs
        a  = np.arange(xMin, xMax, 0.2)
        b  = np.arange(xMin, xMax, 0.2)
        X, Y = np.meshgrid(a, b)
        
        # the levels for which we plot contours
        ncontours = 400
        levels = np.arange(ncontours)
        levels = np.sqrt(np.sqrt(levels))
        levels = levels*(zMax - zMin)
        levels = levels/math.sqrt(math.sqrt(ncontours))
        levels = levels  + zMin
        levels = np.floor(levels)

        
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
                    kriged[j,i] = kg.kriging( p , specs )[0]
                                
  
            xs = np.ravel( np.transpose( np.array( specs.X ) )[0] )
            ys = np.ravel( np.transpose( np.array( specs.X ) )[1] )
            
            # cap everything so it fits our frame
            boolArr = (abs(xs) < M)*(abs(ys) < M)
            xs = xs[boolArr]
            ys = ys[boolArr]
            
            # create contour
            fig = plt.figure( frame )
            ax = fig.add_subplot(111) 
            

            cs = ax.contour(X, Y, kriged, levels = levels  ) 
            ax.clabel(cs, fmt = '%.0f', inline = True) 
            ax.scatter(xs, ys)
            plt.title('Contours of Learned Rosenbrock. ' + str(frame) + ' samples. r = ' + str(specs.r) )
            
            # save the plot several times
            for k in range(delay):   
                FrameFileName = "Data/Movie2DContourFrames/Frame" + str(frame*delay + k) + ".png"

                fig.savefig(FrameFileName)

                if (frame*delay + k) % 10 == 0:
                    print( "saved " +FrameFileName + ".  "
                            + str(frame*delay + k) +  " / " + str((nf+1)*delay) )
            
            plt.close( frame )

            # IMPORTANT - we sample from the kriged log-likelihood. this is crucial!!!!

            sampler.learn() 
            
        
        
#         after the test was run - we create the movie.
#         you need ffmpeg to create the movie from the frames python saves 
        
        # delete previous movie
        os.system("rm -f graphics/Movie2DContour.mpg")     
        
        # create new movie 
        os.system("ffmpeg -i Data/Movie2DContourFrames/Frame%d.png graphics/Movie2DContour.mpg") 
        

            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()