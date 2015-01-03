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
from matplotlib import gridspec

import kernel.sampler as smp
import kernel.container as cot
import helper.rosenbrock as rose
import helper.kl as kl
import kernel.targets as targets

class Test(unittest.TestCase):
    '''
    if this does not work, it is likely that don't have 
    ffmpeg. 
    this unit test creates a movie. run it and see for yourself!!
    '''

    def setUp(self):
        '''
        define helper methods and stuff
        '''
        # for reproducibility
        np.random.seed(1792) 
        
        # tell the OS to prepare for the movie and the frames
        os.system("rm -f Data/Movie2DContourFrames/*.png")
        
        def make_movie_frame( sample , frame, LLlevels , intLevels,  kriged, integrand, pts, X, Y, KL, desc, param ):
            '''
            well, create and save a movie frame. mostly 
            boring code.
            '''      
            
            # unpack 
            nSamp   = param[0]
            maxIter = param[1]
            nwalk   = param[2]
            nopt    = param[3]
            nPoints = param[4]
            M       = param[5]
            delay   = param[6] 
            delta   = param[7]
            
            
            
            xs = np.ravel( np.transpose( np.array( pts ) )[0] )
            ys = np.ravel( np.transpose( np.array( pts ) )[1] )
             
            # cap everything so it fits our frame
            boolArr = (abs(xs) < M)*(abs(ys) < M)
            xs = xs[boolArr]
            ys = ys[boolArr]
            
            KL      = np.asarray( KL ) 
            kl      =  KL[: , 0 ]
            lowBar  =  KL[: , 1 ]
            highBar =  KL[: , 2 ]
            sumTerm =  KL[: , 3 ]
            lowSum  =  KL[: , 4 ]
            highSum =  KL[: , 5 ]
            xp      =  KL[: , 6 ]
            lowExp  =  KL[: , 7 ]
            highExp =  KL[: , 8 ]
            uniKl   =  KL[: , 9 ]
            
            # create plot
            fig = plt.figure( figsize=(18, 9) )
            fig.suptitle( 'Total of ' + str(len(pts)) + ' LL evaluations. KL bars w\ ' + str(nSamp) +
                           ' samples. ' + str(nopt) + ' optimizers with ' + str(maxIter) +
                           ' optimization steps. Mesh size is ' + str(delta) +
                           '.\nOptimizing ' + desc , fontsize=12, verticalalignment = 'top') 
            
            gs = gridspec.GridSpec(6, 8)
            ax1 = plt.subplot(gs[0:4  , 0:4  ])
            ax5 = plt.subplot(gs[0:4  , 4:8  ])
            ax2 = plt.subplot(gs[4:6  , 0:3  ])
            ax3 = plt.subplot(gs[4:6  , 3:6  ])
            ax4 = plt.subplot(gs[4:6  , 6:8  ])
            
            
            # create big contour plot    
            cs1 = ax1.contour(X, Y, kriged, levels = LLlevels  ) 
            ax1.clabel(cs1, fmt = '%.0f', inline = False) 
            ax1.scatter(xs, ys)
            ax1.set_title('Learned Rosenbrock Contours and LL evaluations.' , fontsize=12)
            
            cs5 = ax5.contour(X, Y, integrand, levels = intLevels  ) 
            ax5.clabel(cs5, fmt = '%.0f', inline = False) 
            ax5.set_title('Contours of KL integrand' , fontsize=12)
            
                        
            x = np.asarray( [ range(len(kl))        , range(len(uniKl))       ])
            y = np.asarray( [ kl                    , uniKl                   ])
            t = np.asarray( [ np.zeros(len(kl))      , 100*np.ones(len(kl))     ])
            ax2.scatter(x, y, c=t)          
            ax2.errorbar(range(len(kl)), kl ,yerr=[lowBar, highBar], linestyle="None")
            ax2.set_ylim([-200,500])
            ax2.set_xlim([-1,nPoints+1])
            ax2.set_title('MC KL w\ bars (blue). uni KL (red)', fontsize=12)
            
            ax3.scatter(range(len(sumTerm)),sumTerm)
            ax3.errorbar(range(len(sumTerm)), sumTerm ,yerr=[lowSum , highSum], linestyle="None")
            ax3.set_ylim([-200,500])
            ax3.set_xlim([-1,nPoints+1])
            ax3.set_title('sum term w\ bars', fontsize=12)
            
            ax4.scatter(range(len(xp )),xp )
            ax4.errorbar(range(len(xp)), xp  ,yerr=[lowExp , highExp], linestyle="None")
            ax4.set_ylim([-200,500])
            ax4.set_xlim([-1,nPoints+1])
            ax4.set_title('log(sum exp) term w\ bars' , fontsize=12)
         
            # create the frame
            for _ in range(delay):
                
                if frame % 10 == 0:
                    print("Saved frame " + str(frame) + " of " +str(nPoints*delay))
                
                # save the plot              
                fig.savefig( "Data/Movie2DContourFrames/Frame" + str(frame) + ".png" )  
                frame = frame + 1                    
             
            plt.close()
            return frame
        
        def getLevels(ncontours, zMin, zMax):
            '''
            create levels for the contour plots
            '''
               
            # the levels for which we plot contours
            levels = np.arange(ncontours)
            levels = np.sqrt(np.sqrt(levels))
            levels = levels*(zMax - zMin)
            levels = levels/math.sqrt(math.sqrt(ncontours))
            levels = levels  + zMin
            levels = np.floor(levels)
            return levels             

        self.getLevels = getLevels
        self.make_movie_frame = make_movie_frame
           
    def testMovie2D(self):
        '''
        create a 2D movie, based on the data we put in the container object 
        in the setUp method sthis method does all the graphics involved
        since this is a 2D running for lots of points might take a while
        '''
        
            
        # parameters to play with
        nSamples  = 5000      # number of samples we use for KL
        maxiter   = 12500    # max number of optimization steps
        nPoints   = 80       # The number of evaluations of the true likelihood
        delay     = 3        # number of copies of each frame
        M         = 7        # bound on the plot axes
        nopt      = 50
        nwalk     = 50
        burn      = 500
        LLlevels  = self.getLevels(350 , -1e6 , 1e4) # levels of log likelihood contours
        intLevels = np.concatenate([np.arange(0,4,0.8),
                    np.arange(5,50,15),  np.arange(50,550,250)] )  # levels of integrand contours
        delta     = 0.05 # grid for the contour plots
        parameters = [nSamples, maxiter, nwalk, nopt , nPoints ,M ,delay, delta]
        
        # initialize container and sampler
        specs = cot.Container( rose.rosenbrock_2D )
        n = 1
        for i in range( -n , n+1 ):
            for j in range( -n, n+1 ):
                specs.add_point(np.array( [2*i , 2*j ] ))
        sampler = smp.Sampler( specs , target = targets.exp_krig_sigSqr, 
                               maxiter = maxiter , nwalkers = nwalk,
                               noptimizers = nopt,  burn = burn)
        
        
        # memory allocations. constants etc
        KL = [] # create list for KL div and its error bars
        a  = np.arange(-M, M, delta)
        X, Y = np.meshgrid(a , a)         # create two meshgrid
        form = X.shape
        points = np.asarray( [ np.ravel(X) , np.ravel(Y)])
        frame = 0   
        desc = sampler.target.desc
        locRos    = rose.rosenbrock_2D
        locKrig   = specs.kriging
       
        # some calculations we'll use again and again
        rosen      = np.reshape( locRos(points, True)  , form )
        xpRosen    = np.exp(rosen)
        xpRosenTimesRosen = xpRosen*rosen
        Zphi       = np.sum( xpRosen ) # no delta**2!! see below


        
        # create frames for the movie
        for sample in range (nPoints+1):
            
            # get the KL divergence estimate and error bars
            samples  = rose.sample_rosenbrock(nSamples)
            tmpKL = kl.get_KL(locRos ,locKrig, samples)
#             tmpKL = rose.rosenbrock_KL(specs, nSamples)
            
            # the kriged surface
            kriged     = np.reshape( np.asarray([ locKrig(x) for x in points.T]) , form )
            
            # the integrand (contour plot on left)
            integrand  = np.reshape( xpRosenTimesRosen - xpRosen*kriged   , form ) # Z(rosenbrock) = 1!!
       
            # estimate of log(Z) by using a riemann sum on the grid     
            Zpsi       = np.sum( np.exp(kriged)) #  no delta**2 ...
            logZpsiOverZphi  = math.log(Zpsi/Zphi) # ...it would've cancelled out!!
           
            # estimate of the KL divergence, from the grid we used for plotting
            uniKL = delta*delta*np.sum(integrand) + logZpsiOverZphi
            tmpKL.append( uniKL ) # add this to the other estimates
            
            # the value of KL integrand at every point on the grid
            integrand = integrand + logZpsiOverZphi

            
            tmpKL.append( uniKL )
            tmpKL.append( logZpsiOverZphi )
            KL.append( tmpKL  )   
            
#             print("Here's one problem - the log of the normalization constants don't agree.")         
#             print(tmpKL[6]) # 
#             print(logZpsiOverZphi)
            
            # make the frames
            frame =  self.make_movie_frame( sample, frame, LLlevels, intLevels ,
                                            kriged, integrand, specs.X, X, Y, KL, desc, parameters)

            # learn a new point and incorporate it and save
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
