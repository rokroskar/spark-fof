from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

setup(
    ext_modules = cythonize([Extension("spark_fof", ["spark_fof_c.pyx"], include_dirs=[numpy.get_include()]),
    						 Extension("fof", 
    								   ["fof/fof.pyx"],
    								   libraries=['kd'],
    								   extra_compile_args=['-I', '/Users/rokstar/projects/spark-fof/fof'],
    								   extra_link_args=['-L','/Users/rokstar/projects/spark-fof/fof'])])
)
