'''
Created on Sep 25, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

def rapper( s, trueLL, args, kwargs ):
    
    # do what you want here, I'll just ignore the additional
    # arguments for my purposes.  

    if len(args) + len(kwargs) == 0:
        return trueLL(s)   
    
    return trueLL(s , *args , **kwargs )
    
         
        
