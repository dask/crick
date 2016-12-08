import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

sourcefiles = ['crick/tdigest.pyx']

extensions = [Extension("crick.tdigest", sourcefiles)]

setup(name='crick',
      version='0.0.1',
      packages=['crick'],
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include()])
