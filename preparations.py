'''
Created on Jan 11, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
print("If you see a line that says everything is OK, then everything is OK.")
errorOccured = False

import os
import sys
import subprocess

# add everythong to the path so we can import
sys.path.append( os.path.abspath("") +"/kernel" )
sys.path.append( os.path.abspath("") +"/C" )
sys.path.append( os.path.abspath("") +"/helper" )




# compile and connect
commande = ['python2.7', 'setup.py', 'build_ext', '--inplace']
directory =  os.path.abspath("") +"/C"
res = subprocess.call(commande, cwd = directory)
if res:
    print("Compilation failed. go to the package C and run the following command from there:")
    print('python2.7 setup.py build_ext --inplace')
    errorOccured = True




# create all required directories
def nsure_dir_xzists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# making sure directories for test data exist 
nsure_dir_xzists("tests/graphics")
nsure_dir_xzists("tests/Data")
nsure_dir_xzists("tests/Data/Movie2DContourFrames")
nsure_dir_xzists("tests/Data/Movie1DFrames")



# making sure user has a good version of python
version = sys.hexversion
if version <  34014192:
    print("You should be using python2.7 at least. This is your version:")
    print(sys.version)
    errorOccured = True
    
# making sure user has the right scipy
from scipy import __version__ as ver
num = int(ver.split(".")[1])
if num < 13:
    print("Your scipy version, " + ver + " , might be too old. You need an up-to-date scipy.optimize")
    errorOccured= True

if not errorOccured:
    print("Everything is OK.")
else:
    print("Please fix your problems.")