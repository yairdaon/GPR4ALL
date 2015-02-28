'''
Created on Sep 25, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

def rapper( s, trueLL, args, kwargs ):
    
    # do what you want here, I'll just ignore the additional
    # arguments for my purposes.  

    if args == None and kwargs == None:
        return float( trueLL(s))
    return float(  trueLL(s , *args , **kwargs )    )