'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!

This module creates static classes that are used as flags
The user (you!) SHOULD NOT create new instances of these classes
'''

class Algorithm(object):
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
        self._matrixType = typeString

    def getDescription(self):
        '''
        return the type of algorithm an instance defines
        '''
        return self._matrixType

# here are the instances of the above class. 
AUGMENTED_COVARIANCE = Algorithm( "Aug Cov" ) # Use augmented covariance matrix. Unbiased predictor.
COVARIANCE           = Algorithm( "Cov Mat" ) # Covariance matrix. Not an unbiased predictor.
RASMUSSEN_WILLIAMS   = Algorithm( "R&W") # algorithm 2.1 in R&W "gaussian Process for Machine Learning"      