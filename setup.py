from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

import os

currdir = os.getcwd()

setup(name="spark_fof",
      author="Rok Roskar",
      author_email = "roskarr@ethz.ch",
      package_dir = {'spark_fof/': ''},
      packages = ['spark_fof', 'spark_fof/fof'],
      ext_modules = cythonize([Extension("spark_fof.spark_fof_c", 
                                         sources = ["spark_fof/spark_fof_c.pyx"], 
                                         include_dirs=[numpy.get_include(), 'spark_fof/fof']),
    	   			           Extension("spark_fof.fof.fof", 
    								     sources = ["spark_fof/fof/fof.pyx"],
    								     libraries = ['kd'],
    								     extra_compile_args = ['-I', os.path.join(currdir,'spark_fof/fof')],
    								     extra_link_args = ['-L',os.path.join(currdir,'spark_fof/fof')],
                                         include_dirs = [numpy.get_include()])])
)
