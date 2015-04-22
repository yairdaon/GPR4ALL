'''
Created on Jan 8, 2015

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_g", ["gpr4all/C/_g.c", "gpr4all/C/g.c" , "gpr4all/C/aux.c", "gpr4all/C/krigger.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
