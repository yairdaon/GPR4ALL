# comile and connect
import subprocess
print( subprocess.call("python2.7 setup.py build_ext --inplace", shell=True) )
