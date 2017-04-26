from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

import os

currdir = os.getcwd()

setup(name="spark_fof_test",
      author="Rok Roskar",
      author_email = "roskarr@ethz.ch",
      package_dir = {'spark_fof_test/': ''},
      packages = ['spark_fof_test', 'spark_fof_test/fof'],
      ext_modules = cythonize([Extension("spark_fof_test.spark_fof_c", 
                                         sources = ["spark_fof_test/spark_fof_c.pyx"], 
                                         include_dirs=[numpy.get_include(), 'spark_fof_test/fof']),
                               Extension("spark_fof_test.fof.fof", 
                                         sources = ["spark_fof_test/fof/fof.pyx"],
                                         libraries = ['kd'],
                                         extra_compile_args = ['-I', os.path.join(currdir,'spark_fof_test/fof')],
                                         extra_link_args = ['-L',os.path.join(currdir,'spark_fof_test/fof')],
                                         include_dirs = [numpy.get_include()])])
)
