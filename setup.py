import sys
from os.path import exists

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
      description='High performance approximate and streaming algorithms',
      long_description=(open('README.md').read() if exists('README.rst')
                        else ''),
      keywords='streaming approximate algorithms',
      url='https://github.com/jcrist/crick',
      maintainer='Jim Crist',
      maintainer_email='crist042@umn.edu',
      license='BSD',
      packages=['crick', 'crick.tests'],
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include()],
      zip_safe=False)
