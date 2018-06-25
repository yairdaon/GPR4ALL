import numpy as np

import gpr4all.container as cot
import gpr4all.sampler as smp
import gpr4all.truth as truth
import gpr4all._aux as _aux
import gpr4all.rosenbrock as rose


def testSampler():
    '''Here we sample from the sampler. We choose to learn from
    the samples.

    '''
                        
    # creating the container object...
    specs = cot.Container( rose.rosenbrock_2D )
    
    specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
    specs.add_point( np.array( [  1.5 ,-2.0  ] )  )

    sampler = smp.Sampler( specs, nwalkers = 20 )
    sampler.learn()
    sampler.learn()
    specs = specs
    sampler = sampler                                                 
    
    assert len( specs.X ) == 4 
    assert len(specs.X[0]) == len(specs.X[1])              
    assert len(specs.X[0]) == len(specs.X[2])    
                         
                        
def testReproducibility():
    '''Tests that using the same seed, we can reproduce results.

    '''
                        
    def rep_func():
        np.random.seed(567)
    
        # creating the container object...
        specs = cot.Container( rose.rosenbrock_2D )

        # Two arbitrary points
        specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
        specs.add_point( np.array( [  1.5 ,-2.0  ] )  )
                        
        sampler = smp.Sampler( specs )
        sampler.learn()
        sampler.learn()

        # now we put the sample in y
        return  sampler.sample_one()   
                    
    # and compare
    assert np.all(rep_func() == rep_func() ) 
                        
                        
def testReproducibilityFails():
    '''Tests that using different seeds, we cannot expect to reproduce
    results.  Note: this is what we did in the testReproducibility
    method above EXCEPT for the first line and the last.

    '''
                        
    def non_rep_func():
                        
        # the seed is now commented
        #np.random.seed(567)
        
                        
        # creating the container object...
        specs = cot.Container( rose.rosenbrock_2D )
        
        specs.add_point( np.array( [ -1.5 , 2.0  ] )  )
        specs.add_point( np.array( [  1.5 ,-2.0  ] )  )
                        
        sampler = smp.Sampler( specs )
        sampler.learn()
        sampler.learn()
                        
        # now we put the sample in y
        return  sampler.sample_one()   
                        
    # compare
    assert np.all( not non_rep_func() ==  non_rep_func() )  
                        
                        
                        
                        
