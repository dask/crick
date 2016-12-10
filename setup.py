import sys

import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


if '--debug' in sys.argv:
    kwargs = {'undef_macros': ["NDEBUG"],
              'extra_compile_args': ['-O0']}
else:
    kwargs = {}


extensions = [Extension("crick.tdigest",
                        ['crick/tdigest.pyx'],
                        **kwargs)
              ]

setup(name='crick',
      version='0.0.1',
      packages=['crick'],
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include()])
