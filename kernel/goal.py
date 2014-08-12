'''
Created on Aug 12, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''


class Goal(object):
    '''
    a static class with the instances seen below.
    Do NOT create instances of this class on your own, this is NOT what this
    class is built for. The instances on the bottom are a substitute
    for a flag.
    '''


    def __init__(self, typeString):
        '''
        Constructor
        '''
        self._goal = typeString

    def getDescription(self):
        '''
        return the goal of our endeavors
        '''
        return self._goal

# here are the instances of the above class. 
REGRESSION   = Goal( "interpolate" ) # if our goal is to interpolate\ regress
OPTIMIZATION = Goal( "optimize" ) # If our goal is to optimize some function

  