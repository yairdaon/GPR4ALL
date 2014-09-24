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

version = sys.hexversion
if version <  33949424:
    print("You should be using python2.6 at least. This is your version:")
    print(sys.version)
