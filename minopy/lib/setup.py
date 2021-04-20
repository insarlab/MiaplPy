#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Build import cythonize
#from Cython.Distutils import build_ext
from setuptools import Extension, setup
#from Cython.Build import cythonize
import numpy

#ext_modules=[
#    Extension("utils",    ["utils.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
#    Extension("invert",   ["invert.pyx"], include_dirs=[numpy.get_include()]),
#]

ext_modules=[
    Extension("utils",    ["utils.pyx"], include_dirs=[numpy.get_include()]),
    Extension("invert",   ["invert.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
  name = 'inversion_utils',
  script_args=["build_ext", "--inplace"],
  ext_modules = cythonize(ext_modules),
)



'''
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [Extension("inversion_utils", ["inversion_utils.pyx"], include_dirs=[numpy.get_include()])]

setup(name='inversion_utils', ext_modules=cythonize(extensions), script_args=["build_ext", "--inplace"])
'''
