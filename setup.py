import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['crick/tdigest.pyx']

extensions = [Extension("crick/tdigest", sourcefiles)]

setup(ext_modules=cythonize(extensions),
      include_dirs=[np.get_include()])
