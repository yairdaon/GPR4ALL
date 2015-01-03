'''
Created on Jan 1, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest

print("performing tests:")

testmodules = [
    'Test_No_Graphics',
    'Test_Plots',
    'Test_KL',
    'Test_Rosenbrock',
    'Test_Movie1D',
    'Test_Movie2D',
    'Test_Gaussian',
    ]


for t in testmodules:
    print("running " + t + " :")
    print
    suite = unittest.TestSuite()

    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)

