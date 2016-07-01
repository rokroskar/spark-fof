import numpy as np

def spark_cython(module, method):     
    def wrapped(*args, **kwargs):     
        global cython_function_      
        try:     
            return cython_function_(*args, **kwargs)     
        except: 
            import pyximport     
            pyximport.install(setup_args={"include_dirs":np.get_include()})     
            cython_function_ = getattr(__import__(module), method)     
            return cython_function_(*args, **kwargs) 
    return wrapped