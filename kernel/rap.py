'''
Created on Sep 25, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

def rapper( s, trueLL, parameters ):
    
    if parameters==None:
        return trueLL(s)
    
    else:
        # Unpack parameters here, call trueLL with parameters
        # and return the value.
        # Here is unpacking that works with example.py
        num = parameters[0]
        someList = parameters[1]
        f = trueLL(s, num, someList)
        return f
        
        
