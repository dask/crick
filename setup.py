from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['crick/_wrapper.pyx']

extensions = [Extension("crick/_wrapper", sourcefiles)]

setup(ext_modules=cythonize(extensions))
