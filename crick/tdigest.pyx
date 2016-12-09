from libc.string cimport memcpy
from libc.math cimport NAN, isnan
cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "tdigest_stubs.c":
    ctypedef struct centroid_t:
        double mean
        np.npy_uint32 weight

    # Not the full struct, just the parameters we want
    ctypedef struct tdigest_t:
        double compression
        size_t size
        size_t buffer_size

        double min
        double max

        size_t last
        double total_weight
        double buffer_total_weight
        centroid_t *centroids

    tdigest_t *tdigest_new(double compression)
    void tdigest_free(tdigest_t *T)
    void tdigest_add(tdigest_t *T, double x, int w)
    void tdigest_flush(tdigest_t *T)
    void tdigest_merge(tdigest_t *T, tdigest_t *other)
    double tdigest_quantile(tdigest_t *T, double q)
    double tdigest_cdf(tdigest_t *T, double x)
    np.npy_intp tdigest_update_ndarray(tdigest_t *T, np.PyArrayObject *x, np.PyArrayObject *w)


CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.uint32)],
                          align=True)


cdef class TDigest:
    """TDigest(compression=100.0)

    An approximate histogram.

    Parameters
    ----------
    compression : float, optional
        The compression factor to use. Larger numbers provide more accurate
        estimates, but use more memory. For most uses, the default should
        suffice.
    """
    cdef tdigest_t *tdigest

    def __cinit__(self, compression=100.0):
        self.tdigest = tdigest_new(compression)

    def __dealloc__(self):
        if self.tdigest != NULL:
            tdigest_free(self.tdigest)

    def __repr__(self):
        return ("TDigest<compression={0}, "
                "count={1}>").format(self.compression, self.count())

    def add(self, double x, int w=1, skipna=True):
        """add(self, x, w, skipna=True)

        Add a sample to this digest.

        Parameters
        ----------
        x : float
            The value to add.
        w : int, optional
            The weight of the value to add. Default is 1.
        skipna : bool, optional
            If False, an error will be raised if ``x`` is ever ``NaN``.
            Otherwise, ``NaN`` values are ignored. Default is True.
        """
        if isnan(x) and not skipna:
            raise ValueError("NaN value encountered")
        tdigest_add(self.tdigest, x, w)

    @property
    def compression(self):
        """The compression factor for this digest"""
        return self.tdigest.compression

    def min(self):
        """min(self)

        The minimum value in the digest."""
        tdigest_flush(self.tdigest)
        if self.tdigest.total_weight > 0:
            return self.tdigest.min
        return NAN

    def max(self):
        """max(self)

        The maximum value in the digest."""
        tdigest_flush(self.tdigest)
        if self.tdigest.total_weight > 0:
            return self.tdigest.max
        return NAN

    def count(self):
        """count(self)

        The number of points in the digest."""
        return int(self.tdigest.total_weight + self.tdigest.buffer_total_weight)

    def cdf(self, double x):
        """cdf(self, x)

        Compute an estimate of the fraction of all points added to this digest
        which are <= x.

        Parameters
        ----------
        x : float
        """
        return tdigest_cdf(self.tdigest, x)

    def quantile(self, double q):
        """quantile(self, q)

        Compute an estimate of the qth percentile of the data added to
        this digest.

        Parameters
        ----------
        q : float
            A number between 0 and 1 inclusive.
        """
        return tdigest_quantile(self.tdigest, q)

    def centroids(self):
        """centroids(self)

        Returns a numpy array of all the centroids in the digest. Note that
        this array is a *copy* of the internal data.
        """
        cdef size_t n
        tdigest_flush(self.tdigest)
        if (self.tdigest.total_weight == 0):
            n = 0
        else:
            n = self.tdigest.last + 1

        cdef np.ndarray result = np.empty(n, dtype=CENTROID_DTYPE)
        if n > 0:
            memcpy(result.data, self.tdigest.centroids, n * sizeof(centroid_t))
        return result

    def __reduce__(self):
        return (TDigest, (self.compression,), self.__getstate__())

    def __getstate__(self):
        return (self.centroids(), self.tdigest.total_weight,
                self.tdigest.min, self.tdigest.max)

    def __setstate__(self, state):
        self.tdigest.total_weight = <double>state[1]
        self.tdigest.min = <double>state[2]
        self.tdigest.max = <double>state[3]

        cdef np.ndarray centroids = state[0]
        cdef int n = len(centroids)
        if n > 0:
            memcpy(self.tdigest.centroids, centroids.data,
                   n * sizeof(centroid_t))
            self.tdigest.last = n - 1

    def update(self, x, w=1, skipna=True):
        """update(self, x, w, skipna=True)

        Add many samples to this digest.

        Parameters
        ----------
        x : array_like
            The values to add.
        w : array_like, optional
            The weight (or weights) of the values to add. Default is 1.
        skipna : bool, optional
            If False, an error will be raised if ``x`` is ever ``NaN``.
            Otherwise, ``NaN`` values are ignored. Default is True.
        """
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.asarray(x)
        if np.isscalar(w):
            w = np.array([w])
        else:
            w = np.asarray(w)

        if not skipna:
            if np.isnan(x).any():
                raise ValueError("NaN value encountered")
        tdigest_update_ndarray(self.tdigest, <np.PyArrayObject*>x,
                               <np.PyArrayObject*>w)

    def merge(self, *args):
        """merge(self, *args)

        Update this digest inplace with data from other digests.

        Parameters
        ----------
        args : TDigests
            TDigests to merge into this one.
        """
        if not all(isinstance(i, TDigest) for i in args):
            raise TypeError("All arguments to merge must be TDigests")
        cdef TDigest td
        for td in args:
            tdigest_merge(self.tdigest, td.tdigest)
