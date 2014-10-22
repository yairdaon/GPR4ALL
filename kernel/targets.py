'''
Created on Oct 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import math

from kriging import kriging


def minus_exp_krig_times_sig_square(s, specs):
    '''
    return exp(kriged value) times sigma square
    '''
    
    krig , sig = kriging(s , specs)
     
    return -math.exp(krig)*sig*sig
    
    
