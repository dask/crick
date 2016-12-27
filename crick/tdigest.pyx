from libc.string cimport memcpy
from numpy.math cimport NAN, isfinite

from copy import copy

cimport cython
cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "tdigest_stubs.c":
    ctypedef struct centroid_t:
        double mean
        double weight

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
    void tdigest_add(tdigest_t *T, double x, double w)
    void tdigest_flush(tdigest_t *T)
    void tdigest_merge(tdigest_t *T, tdigest_t *other)
    void tdigest_scale(tdigest_t *T, double factor)
    np.npy_intp tdigest_update_ndarray(tdigest_t *T, np.PyArrayObject *x, np.PyArrayObject *w) except -1
    np.PyArrayObject *tdigest_quantile_ndarray(tdigest_t *T, np.PyArrayObject *q) except NULL
    np.PyArrayObject *tdigest_cdf_ndarray(tdigest_t *T, np.PyArrayObject *x) except NULL


CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])


@cython.boundscheck(False)
cdef inline void _cdf_to_hist(double[:] cdf, double[:]  hist, double size, int hist_size):
    cdef size_t i
    for i in range(hist_size):
        hist[i] = (cdf[i + 1] - cdf[i]) * size


cdef class TDigest:
    """TDigest(compression=100.0)

    An approximate histogram.

    Parameters
    ----------
    compression : float, optional
        The compression factor to use. Larger numbers provide more accurate
        estimates, but use more memory. For most uses, the default should
        suffice.

    Notes
    -----
    This implements the "MergingDigest" variant of the T-Digest algorithm
    descibed in [1]_. The reference java implementation can be found at [2]_.

    References
    ----------
    .. [1] Dunning, Ted, and Otmar Ertl. "Computing Extremely Accurate
       Quantiles Using T-Digests." https://github.com/tdunning/t-digest/blob/
       master/docs/t-digest-paper/histo.pdf

    .. [2] https://github.com/tdunning/t-digest
    """
    cdef tdigest_t *tdigest

    def __cinit__(self, compression=100.0):
        if not isfinite(compression):
            raise ValueError("Compression must be finite")
        self.tdigest = tdigest_new(compression)
        if self.tdigest == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.tdigest != NULL:
            tdigest_free(self.tdigest)

    def __repr__(self):
        return ("TDigest<compression={0}, "
                "size={1}>").format(self.compression, self.size())

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

    def size(self):
        """size(self)

        The sum of the weights on all centroids."""
        return self.tdigest.total_weight + self.tdigest.buffer_total_weight

    def cdf(self, x):
        """cdf(self, x)

        Compute an estimate of the fraction of all points added to this digest
        which are <= x.

        Parameters
        ----------
        x : array_like or float
        """
        x = np.asarray(x)
        if not np.can_cast(x, 'f8', casting='safe'):
            raise TypeError("x must be numeric")
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        out = <np.ndarray>tdigest_cdf_ndarray(self.tdigest, <np.PyArrayObject*>x)
        if out.ndim == 0:
            return np.float64(out)
        return out

    def quantile(self, q):
        """quantile(self, q)

        Compute an estimate of the qth percentile of the data added to
        this digest.

        Parameters
        ----------
        q : array_like or float
            A number between 0 and 1 inclusive.
        """
        q = np.asarray(q)
        if not np.can_cast(q, 'f8', casting='safe'):
            raise TypeError("q must be numeric")
        if not np.isfinite(q).all():
            raise ValueError("q must be finite")
        out = <np.ndarray>tdigest_quantile_ndarray(self.tdigest, <np.PyArrayObject*>q)
        if out.ndim == 0:
            return np.float64(out)
        return out

    def histogram(self, bins=10, range=None):
        """histogram(self, bins=10, range=None)

        Compute a histogram from the digest.

        Parameters
        ----------
        bins : int or array_like, optional
            If ``bins`` is an int, it defines the number of equal width bins in
            the given range. If ``bins`` is an array_like, the values define
            the edges of the bins (rightmost edge inclusive), allowing for
            non-uniform bin widths. The default is 10.

        range : (float, float), optional
            The lower and upper bounds to use when generating bins. If not
            provided, the digest bounds ``(t.min(), t.max())`` are used. Note
            that this option is ignored if the bin edges are provided
            explicitly.

        Returns
        -------
        hist : array
            The values of the histogram.
        bin_edges : array
            The edges of the bins. ``len(bin_edges) == len(hist) + 1``.
        """
        cdef double left = 0
        cdef double right = 0
        cdef double size = self.size()
        if range is None:
            if size != 0:
                left = self.min()
                right = self.max()
        else:
            left = <double?>range[0]
            right = <double?>range[1]
            if not (isfinite(left) and isfinite(right)):
                raise ValueError("range parameters must be finite")
            elif right < left:
                raise ValueError("max must be larger than min for range "
                                 "parameter")
        if right == left:
            left -= 0.5
            right += 0.5

        if isinstance(bins, int):
            if bins < 1:
                raise ValueError("bins must be >= 1")
            bin_edges = np.linspace(left, right, bins + 1, endpoint=True)
        else:
            bin_edges = np.asarray(bins).astype('f8', copy=False)
            if bin_edges.ndim != 1:
                raise ValueError("bins must be a 1-dimensional array")
            elif not np.isfinite(bins).all():
                raise ValueError("bins must be finite")
            elif (np.diff(bin_edges) < 0).any():
                raise ValueError("bins must increase monotonically")

        cdef int hist_size = bin_edges.size - 1
        cdef np.ndarray[double, ndim=1] hist = np.zeros(hist_size,
                                                        dtype=np.float64)
        if size != 0:
            _cdf_to_hist(self.cdf(bin_edges), hist, size, hist_size)

        return hist, bin_edges

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

        cdef np.ndarray[centroid_t, ndim=1] result = np.empty(n, dtype=CENTROID_DTYPE)
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

    def add(self, double x, double w=1):
        """add(self, x, w=1)

        Add a sample to this digest.

        Parameters
        ----------
        x : float
            The value to add.
        w : float, optional
            The weight of the value to add. Default is 1.
        """
        # Don't check w in the common case where w is 1
        if w != 1 and (not isfinite(w) or w <= 0):
            raise ValueError("w must be > 0 and finite")
        tdigest_add(self.tdigest, x, w)

    def update(self, x, w=1):
        """update(self, x, w=1)

        Add many samples to this digest.

        Parameters
        ----------
        x : array_like
            The values to add.
        w : array_like, optional
            The weight (or weights) of the values to add. Default is 1.
        """
        x = np.asarray(x)
        check_w = not (np.isscalar(w) and w == 1)
        w = np.asarray(w)

        if not np.can_cast(x, 'f8', casting='safe'):
            raise TypeError("x must be numeric")

        if not np.can_cast(w, 'f8', casting='safe'):
            raise TypeError("w must be numeric")

        if check_w and (not np.isfinite(w).all() or (w <= 0).any()):
            raise ValueError("w must be > 0 and finite")

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

    def scale(self, double factor):
        """scale(self, factor)

        Return a new TDigest, with the weights all scaled by factor.

        Parameters
        ----------
        factor : number
            The factor to scale the weights by. Must be > 0.
        """
        if factor <= 0 or not isfinite(factor):
            raise ValueError("factor must be > 0 and finite")
        cdef TDigest out = copy(self)
        tdigest_scale(out.tdigest, factor)
        return out
