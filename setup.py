import sys

import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


kwargs = {'extra_compile_args': ['-std=c99']}

if '--debug' in sys.argv:
    kwargs['undef_macros'] = ["NDEBUG"]
    kwargs['extra_compile_args'].append('-O0')


extensions = [Extension("crick.tdigest",
                        ['crick/tdigest.pyx'],
                        **kwargs)
              ]

setup(name='crick',
      version='0.0.1',
      packages=['crick'],
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include()])
