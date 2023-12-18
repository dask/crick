import os
import sys

import numpy as np
from Cython import Tempita as tempita
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import versioneer

compile_args = dict(
    library_dirs=[
        os.path.abspath(os.path.join(np.get_include(), "..", "lib")),
    ],
    include_dirs=[np.get_include(), "crick/klib"],
    libraries=["npymath"],
)

if "--debug" in sys.argv:
    sys.argv.remove("--debug")
    compile_args["undef_macros"] = ["NDEBUG"]
    compile_args["extra_compile_args"] = ["-O0"]


def generate_code(templates):
    """Generate code from template files"""
    for template in templates:
        # template extention must be .in
        assert template.endswith(".in")
        outfile = template[:-3]

        if os.path.exists(outfile) and (
            not os.path.exists(template)
            or os.stat(template).st_mtime < os.stat(outfile).st_mtime
        ):
            # If output is present but template isn't, or if template is not
            # updated no need to generate
            continue

        with open(template, "r") as f:
            tmpl = f.read()
        output = tempita.sub(tmpl)

        with open(outfile, "w") as f:
            f.write(output)


templates = ["crick/space_saving_stubs.c.in"]

generate_code(templates)

extensions = [
    Extension("crick.tdigest", ["crick/tdigest.pyx"], **compile_args),
    Extension("crick.space_saving", ["crick/space_saving.pyx"], **compile_args),
    Extension("crick.stats", ["crick/stats.pyx"], **compile_args),
]

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=cythonize(extensions),
)
