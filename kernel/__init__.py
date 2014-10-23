import os
import sys

def nsure_dir_xzists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# making sure directories for test data exist 
nsure_dir_xzists("graphics")
nsure_dir_xzists("Data")
nsure_dir_xzists("Data/Movie2DContourFrames")
nsure_dir_xzists("Data/Movie1DFrames")

# making sure user has a good version of python
version = sys.hexversion
if version <  34014192:
    print("You should be using python2.7 at least. This is your version:")
    print(sys.version)
    
# making sure user has the right scipy
from scipy import __version__ as ver
num = int(ver.split(".")[1])
if num < 13:
    print("Your scipy version, " + ver + " , might be too old. You need an up-to-date scipy.optimize")

