import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

ext_modules = [
    Extension(
        name="miaplpy.lib.utils",
        sources=["src/miaplpy/lib/utils.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="miaplpy.lib.invert",
        sources=["src/miaplpy/lib/invert.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup_args = dict(
    packages=find_packages(where="src"),
    ext_modules=cythonize(ext_modules, language_level=3),
)

setup(**setup_args)
