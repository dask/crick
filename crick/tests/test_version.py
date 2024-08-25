from crick import __numpy_version__, __version__


def test_version():
    """Verify that crick's version is exported correctly."""
    assert __version__


def test_numpy_version():
    """Verify that the numpy version exists and is exports correctly.

    This value is created with Cython during the build-time, by the
    preprocessor that writes the value passed during build-time
    (by injecting `np.__version__`). It should match the version
    used to build, but not necessarily the version installed/available.

    i.e. you can have NumPy 2.0 installed, but the crick developers used
    2.1.0 to build and upload the wheel -- then it relies on NumPy's
    backward/forward compatibility during run-time.
    """
    assert __numpy_version__
