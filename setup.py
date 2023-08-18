# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize

from setuptools import setup, find_packages, Extension

import numpy

ext_modules = [
    Extension(
        name="miaplpy.lib.utils",
        sources=["src/miaplpy/lib/utils.pyx"],
        # include_dirs        = ['src/miaplpy/lib'],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="miaplpy.lib.invert",
        sources=["src/miaplpy/lib/invert.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup_args = dict(
    packages=find_packages(where="src"),  # list
    # package_dir={"": "src"},  # mapping
    # ext_modules     = [ext],                            # list
    ext_modules=cythonize(ext_modules, language_level=3),
    # scripts=["examples/fbs_test.py"],  # list
)

setup(**setup_args)
