'''
Created on Jul 25, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import numpy as np
      
class PointWithError(np.ndarray):
    '''
    This class extends the numpy array by adding one variable.
    This is the calculation (of the LL routine) at the given
    INPUT. One can instantiate it just like one would do for
    a normal nump array.
    Copied from
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
    there it is found under "Slightly more realisic example"
    '''

    def __new__(cls, input_array, error=0.0):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.error = error
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        #self.info = getattr(obj, 'info', None)  # commented by myself   
        
    def getError(self):
        return self.error
    
    def setError(self , error):
        '''
        for some reason I feel it should always be a double
        '''
        self.error = error + 0.0
      
      
 