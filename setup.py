import sys
import os

import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython import Tempita as tempita

import versioneer


kwargs = {'extra_compile_args': ['-std=c99']}

if '--debug' in sys.argv:
    kwargs['undef_macros'] = ["NDEBUG"]
    kwargs['extra_compile_args'].append('-O0')


def generate_code(templates):
    """Generate code from template files"""
    for template in templates:
        # template extention must be .in
        assert template.endswith('.in')
        outfile = template[:-3]

        if (os.path.exists(outfile) and
                os.stat(template).st_mtime < os.stat(outfile).st_mtime):
            # if template is not updated, no need to generate
            continue

        with open(template, "r") as f:
            tmpl = f.read()
        output = tempita.sub(tmpl)

        with open(outfile, "w") as f:
            f.write(output)


templates = ['crick/stream_summary_stubs.c.in']

generate_code(templates)


extensions = [Extension("crick.tdigest",
                        ['crick/tdigest.pyx'],
                        **kwargs),
              Extension("crick.stream_summary",
                        ['crick/stream_summary.pyx'],
                        include_dirs=['crick/klib'],
                        **kwargs),
              Extension("crick.stats",
                        ['crick/stats.pyx'],
                        **kwargs)
              ]

setup(name='crick',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='High performance approximate and streaming algorithms',
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst') else ''),
      keywords='streaming approximate algorithms',
      url='https://github.com/jcrist/crick',
      maintainer='Jim Crist',
      maintainer_email='crist042@umn.edu',
      license='BSD',
      packages=['crick', 'crick.tests'],
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include()],
      zip_safe=False)
