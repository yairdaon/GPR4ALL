'''
Created on Jun 15, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os

import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import kernel.container as cot
import kernel.algorithm as alg


class Test(unittest.TestCase):
    '''
    if this does not work, it is most likely because you don't have 
    ffmpeg and\or vlc. go to the tearDown method to change these to
    whatever you have in your system.
    
    this unit test creates a movie. run it and see for yourself!!
    '''

    def testMovie1D(self):
    
        # tell the OS to prepare for the movie and the frames
        
        os.system("rm -f Data/Movie1DFrames/*.png")    
        
        # for reproducibility
        np.random.seed(1792)    
        
        #     Initializations of the container object
        specs = cot.Container(truth.big_poly_1D, M =2.5)
        M = specs.M
    
        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [ 0 ] )  )
        StartPoints.append( np.array( [0.5] )  )
        
        for point in StartPoints:
            specs.add_point(point)
        
        sampler = smp.Sampler( specs )
        
        # the bounds on the plot axes
        # CHANGE THESE IF STUFF HAPPEN OUTSIDE THE FRAME
        xMin = -M
        xMax = M
        yMax = 100
        yMin = -300
        
        # all the x values for which we plot
        x = np.arange(xMin, xMax, 0.05)
        
        # we create each frame many times, so the movie is slower and easier to watch
        delay = 5
        
        # The number of evaluations of the true likelihood
        # change this if you want a longer\shorter movie
        nf    = 15      
        
        # allocate memory for the arrays to be plotted
        kriged = np.zeros( x.shape )
        limit = np.zeros( x.shape )
        true = np.zeros( x.shape )
        
        
        # create frames for the ffmpeg programs
        for frame in range (nf+1):
            
            # the current a value of the kriged interpolant "at infinity"
            
            limAtInfty = kg.set_get_limit(specs)[0]
            
            # create the kriged curve and the limit curve
            for j in range(0,len(x)):
                kriged[j] = kg.kriging(x[j] ,specs)[0]
                limit[j] = limAtInfty
                true[j] = specs.trueLL(x[j]) # the real log likelihood
            
            # each frame is saved delay times, so we can watch the movie at reasonable speed    
            #for k in range(delay):
            fig = plt.figure( frame*delay )
            
            # here we create the plot. nothing too fascinating here.
            curve1  = plt.plot(x, kriged , label = "kriged log-likelihood")
            curve2 =  plt.plot(x, true, label = "true log-likelihood")
            curve3 =  plt.plot( specs.X, specs.F, 'bo', label = "sampled points ")
            curve4  = plt.plot(x, limit, 'g', label = "kriged value at infinity")
    
            plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
            plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
    
            
            plt.axis([xMin, xMax, yMin, yMax])
            plt.title( 'Kriged Log-Likelihood Changes in Time. r = ' + str(specs.r)
                                         + " Algorithm: " + specs.algType.get_description())
            
            textString = 'using  ' + str(frame ) + ' sampled points' 
            plt.text(1.0, 1.0, textString)
            plt.legend(loc=1,prop={'size':7})  
            
            for k in range(delay):  
                FrameFileName = "Data/Movie1DFrames/Frame" + str(frame*delay + k) + ".png"
                plt.savefig(FrameFileName)
                if (frame*delay + k) % 10 == 0:
                    print( "saved file " + FrameFileName + ".  " + str(frame*delay + k) + 
                                                                 " / " + str(nf*delay) )
                    
            plt.close( frame*delay ) 
  
            # IMPORTANT - we sample from the kriged log-likelihood. this is crucial!!!!
            sampler.learn()
        
        
#         after the test was run, create and show the movie.
#         you need two programs installed on your machine so this works:
#         you need ffmpeg to create the movie from the frames python saves 
#         and you need vlc to watch the movie
#         feel free to change these two lines here according to whatever
#         programs you have installed in your system
       
        # delete previous movie
        os.system("rm -f graphics/Movie1D.mpg")    
        
        # create new movie 
        os.system("ffmpeg -i Data/Movie1DFrames/Frame%d.png graphics/Movie1D.mpg") 
        
        #os.system("vlc Movie1.mpg")     



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()