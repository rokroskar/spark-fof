from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

import os

currdir = os.getcwd()

setup(
    ext_modules = cythonize([Extension("spark_fof_c", ["spark_fof_c.pyx"], include_dirs=[numpy.get_include(), './fof']),
    						 Extension("fof", 
    								   ["fof/fof.pyx"],
    								   libraries=['kd'],
    								   extra_compile_args=['-I', currdir+'/fof'],
    								   extra_link_args=['-L',currdir+'/fof'],
                                       include_dirs=[numpy.get_include()])])
)
