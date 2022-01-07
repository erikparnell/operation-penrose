# run compile_pyxes.py

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

filename = 'fast_utils.pyx'

fast_utils_module = Extension(
   'fast_utils',
   sources=[filename],
   define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
   include_dirs=[numpy.get_include()],
)

setup(
    ext_modules=cythonize(fast_utils_module, annotate=True, compiler_directives={'language_level': "3"})
)
