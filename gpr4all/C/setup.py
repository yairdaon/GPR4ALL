'''
Created on Jan 8, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_krigger", ["gpr4all/C/_krigger.c", "gpr4all/C/krigger.c", "gpr4all/C/aux.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
 
setup(
    ext_modules=[Extension("_aux", ["gpr4all/C/_aux.c", "gpr4all/C/aux.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
