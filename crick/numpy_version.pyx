cdef extern from "numpy_version_stubs.h":
    cdef char* NUMPY_VERSION

def numpy_version():
    """numpy_version()

    The NumPy version used to build and compile the Python and Cython code.
    Useful for troubleshooting runtime issues.

    References
    ----------

    .. https://github.com/dask/crick/issues/53
    """
    return NUMPY_VERSION
