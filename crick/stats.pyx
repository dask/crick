from numpy.math cimport NAN
cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "stats_stubs.c":
    ctypedef struct stats_t:
        np.int64_t count
        np.float64_t sum
        np.float64_t min
        np.float64_t max
        np.float64_t m2
        np.float64_t m3
        np.float64_t m4

    stats_t *stats_new()
    void stats_free(stats_t *T)
    void stats_add(stats_t *T, np.float64_t x, np.int64_t count)
    void stats_merge(stats_t *T1, stats_t *T2)
    np.intp_t stats_update_ndarray(stats_t *T, np.PyArrayObject *x,
                                   np.PyArrayObject *w) except -1
    np.float64_t stats_mean(stats_t *T)
    np.float64_t stats_var(stats_t *T, np.int64_t ddof)
    np.float64_t stats_std(stats_t *T, np.int64_t ddof)
    np.float64_t stats_skew(stats_t *T, int bias)
    np.float64_t stats_kurt(stats_t *T, int fischer, int bias)


cdef class SummaryStats:
    """SummaryStats()

    Computes exact summary statistics on a data stream.

    Keeps track of enough information to compute:

    - count
    - sum
    - minimum
    - maximum
    - mean
    - variance
    - standard deviation
    - skewness
    - kurtosis

    Notes
    -----
    The update formulas for variance, standard deviation, skewness, and
    kurtosis were taken from [1]_.

    References
    ----------
    .. [1] Pebay, Philippe. "Formulas for robust, one-pass parallel computation
       of covariances and arbitrary-order statistical moments." Sandia Report
       SAND2008-6212, Sandia National Laboratories 94 (2008).
    """
    cdef stats_t *stats

    def __cinit__(self):
        self.stats = stats_new()

    def __dealloc__(self):
        if self.stats != NULL:
            stats_free(self.stats)

    def __repr__(self):
        return "SummaryStats<count=%d>" % self.count()

    def __reduce__(self):
        return (SummaryStats, (), self.__getstate__())

    def __getstate__(self):
        return (self.stats.count, self.stats.sum, self.stats.min,
                self.stats.max, self.stats.m2, self.stats.m3, self.stats.m4)

    def __setstate__(self, state):
        self.stats.count = state[0]
        self.stats.sum = state[1]
        self.stats.min = state[2]
        self.stats.max = state[3]
        self.stats.m2 = state[4]
        self.stats.m3 = state[5]
        self.stats.m4 = state[6]

    def add(self, double x, int count=1):
        """add(self, x)

        Add an element to the summary.

        Parameters
        ----------
        x : double
        count : int, optional
            The count of the item to add. Default is 1.
        """
        if count <= 0:
            raise ValueError("count must be > 0")
        stats_add(self.stats, x, count)

    def update(self, x, count=1):
        """update(self, x)

        Add many elements to the summary.

        Parameters
        ----------
        x : array_like
        count : array_like, optional
            The count (or counts) of the item to add. Default is 1.
        """
        x = np.asarray(x)
        check_count = not (np.isscalar(count) and count == 1)
        count = np.asarray(count, dtype='i8')
        if check_count and (count <= 0).any():
            raise ValueError("count must be > 0")

        stats_update_ndarray(self.stats, <np.PyArrayObject*>x,
                             <np.PyArrayObject*>count)

    def merge(self, *args):
        """merge(self, *args)

        Update this summary inplace with data from other summaries.

        Parameters
        ----------
        args : SummaryStats
            SummaryStats to merge into this one.
        """
        if not all(isinstance(i, SummaryStats) for i in args):
            raise TypeError("All arguments to merge must be SummaryStats")
        cdef SummaryStats s
        for s in args:
            stats_merge(self.stats, s.stats)

    def count(self):
        """count(self)

        The number of elements in the summary.
        """
        return self.stats.count

    def sum(self):
        """sum(self)

        The sum of all elements in the summary.
        """
        return self.stats.sum

    def min(self):
        """min(self)

        The minimum element added to the summary. Returns ``NaN`` if empty.
        """
        return self.stats.min if self.stats.count > 0 else NAN

    def max(self):
        """max(self)

        The maximum element added to the summary. Returns ``NaN`` if empty.
        """
        return self.stats.max if self.stats.count > 0 else NAN

    def mean(self):
        """mean(self)

        The mean of all elements in the summary. Returns ``NaN`` if empty.
        """
        return stats_mean(self.stats)

    def var(self, int ddof=0):
        """var(self, ddof=0)

        The variance of all elements in the summary. Returns ``NaN`` if empty.

        Parameters
        ----------
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By
            default ``ddof`` is zero.
        """
        return stats_var(self.stats, ddof)

    def std(self, int ddof=0):
        """std(self, ddof=0)

        The standard deviation  of all elements in the summary. Returns
        ``NaN`` if empty.

        Parameters
        ----------
        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By
            default ``ddof`` is zero.
        """
        return stats_std(self.stats, ddof)

    def skew(self, bint bias=True):
        """skew(self, bias=True)

        The skewness of all elements in the summary. Returns ``NaN`` if empty.

        Parameters
        ----------
        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.
            Default is True.
        """
        return stats_skew(self.stats, bias)

    def kurt(self, bint fisher=True, bint bias=True):
        """kurt(self, fisher=True, bias=True)

        The kurtosis (Fisher or Pearson) of all elements in the summary.
        Returns ``NaN`` if empty.

        Parameters
        ----------
        fisher : bool, optional
            If True [default], Fisher's definition is used (normal ==> 0.0). If
            False, Pearson's definition is used (normal ==> 3.0).
        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.
            Default is True.
        """
        return stats_kurt(self.stats, fisher, bias)
