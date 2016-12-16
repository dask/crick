cimport cython
from cpython.ref cimport PyObject

cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "space_saving_stubs.c":
    ctypedef struct counter_int64_t:
        size_t next
        size_t prev
        np.int64_t item
        long count
        long error

    ctypedef struct counter_object_t:
        size_t next
        size_t prev
        PyObject *item
        long count
        long error

    ctypedef struct summary_int64_t:
        size_t capacity
        size_t n_counters
        size_t head
        counter_int64_t *counters

    ctypedef struct summary_object_t:
        size_t capacity
        size_t n_counters
        size_t head
        counter_object_t *counters

    summary_int64_t *summary_int64_new(int capacity)
    void summary_int64_free(summary_int64_t *T)
    int summary_int64_add(summary_int64_t *T, np.int64_t item, int count) except -1

    summary_object_t *summary_object_new(int capacity)
    void summary_object_free(summary_object_t *T)
    int summary_object_add(summary_object_t *T, object item, int count) except -1
    np.int64_t asint64(np.float64_t key)
    np.float64_t asfloat64(np.int64_t key)


class TopKResult(tuple):
    """TopKResult(item, count, error)

    A result from ``StreamSummary.topk``.

    Attributes
    ----------
    item
        The item, matches the dtype from ``StreamSummary``.
    count : int
        The estimated frequency.
    error : int
        An upper bound on the error for ``count``. The number of occurrences of
        ``item`` is guaranteed to be in ``count <= actual <= count + error``.
    """

    __slots__ = ()

    def __new__(_cls, item, count, error):
        return tuple.__new__(_cls, (item, count, error))

    def __repr__(self):
        return 'TopKResult(item=%r, count=%r, error=%r)' % self

    def __getnewargs__(self):
        return tuple(self)

    def __dict__(self):
        return dict(zip(('item', 'count', 'error'), self))

    def __getstate__(self):
        pass

    @property
    def item(self):
        return self[0]

    @property
    def count(self):
        return self[1]

    @property
    def error(self):
        return self[2]


cdef class StreamSummary:
    """StreamSummary(capacity=20, dtype=float)

    Approximate TopK.

    Parameters
    ----------
    capacity : int, optional
        The number of items to track. A larger number gives higher accuracy,
        but uses more memory. Default is 20.
    dtype : dtype, optional
        The dtype of the objects to track. Currently only ``float``, ``int``,
        and ``object`` are supported. Default is ``float``

    Notes
    -----
    This implements the "Space-Saving" algorithm using the "Stream-Summary"
    datastructure described in [1]_.

    References
    ----------
    .. [1] Metwally, Ahmed, Divyakant Agrawal, and Amr El Abbadi. "Efficient
       computation of frequent and top-k elements in data streams."
       International Conference on Database Theory. Springer Berlin Heidelberg,
       2005.
    """
    cdef void *summary
    cdef readonly np.dtype dtype

    def __cinit__(self, int capacity=20, dtype=float):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        dtype = np.dtype(dtype)
        self.dtype = dtype

        if np.PyDataType_ISOBJECT(dtype):
            self.summary = <void *>summary_object_new(capacity)
        elif np.PyDataType_ISINTEGER(dtype):
            self.summary = <void *>summary_int64_new(capacity)
        elif np.PyDataType_ISFLOAT(dtype):
            self.summary = <void *>summary_int64_new(capacity)
        else:
            raise ValueError("dtype %s not supported" % dtype)

        if self.summary == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.summary != NULL:
            if np.PyDataType_ISOBJECT(self.dtype):
                summary_object_free(<summary_object_t *>self.summary)
            else:
                summary_int64_free(<summary_int64_t *>self.summary)

    def __repr__(self):
        return ("SpaceSaving<capacity={0}, dtype={1}, "
                "size={2}>").format(self.capacity, self.dtype, self.size())

    @property
    def capacity(self):
        return (<summary_int64_t *>self.summary).capacity

    cpdef int size(self):
        """size(self)

        The number of active counters."""
        return (<summary_int64_t *>self.summary).n_counters

    def add(self, item, int count=1):
        """add(self, item, count=1)

        Parameters
        ----------
        item
            The item to add. If it doesn't match the summary dtype, conversion
            will be attempted.
        count : int, optional
            The weight of the item to add. Default is 1.
        """
        if count <= 0:
            raise ValueError("count must be > 0")

        if np.PyDataType_ISOBJECT(self.dtype):
            summary_object_add(<summary_object_t *>self.summary, item, count)
            return

        cdef np.int64_t item2

        if np.PyDataType_ISFLOAT(self.dtype):
            item2 = asint64(<double>item)
        else:
            item2 = <np.int64_t>item
        summary_int64_add(<summary_int64_t *>self.summary, item2, count)

    def topk(self, int k):
        """Estimate the top k elements.

        Parameters
        ----------
        k : int
            The number of most frequent elements to estimate.

        Returns
        -------
        topk : list of TopKResult
            Returns a list of ``TopKResult`` tuples. The list will be of length
            ``min(self.size(), k)``
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        k = min(self.size(), k)

        if np.PyDataType_ISOBJECT(self.dtype):
            return object_topk(<summary_object_t*>self.summary, k)
        elif np.PyDataType_ISFLOAT(self.dtype):
            return float64_topk(<summary_int64_t*>self.summary, k)
        else:
            return int64_topk(<summary_int64_t*>self.summary, k)


cdef list object_topk(summary_object_t *summary, int k):
    cdef:
        int i = summary.head
        int count
        counter_object_t c
        list out = []

    for count in range(k):
        c = summary.counters[i]
        out.append(TopKResult(<object>c.item,
                              c.count,
                              c.error))
        i = c.next

    return out


cdef list int64_topk(summary_int64_t *summary, int k):
    cdef:
        int i = summary.head
        int count
        counter_int64_t c
        list out = []

    for count in range(k):
        c = summary.counters[i]
        out.append(TopKResult(c.item,
                              c.count,
                              c.error))
        i = c.next

    return out


cdef list float64_topk(summary_int64_t *summary, int k):
    cdef:
        int i = summary.head
        int count
        counter_int64_t c
        list out = []

    for count in range(k):
        c = summary.counters[i]
        out.append(TopKResult(asfloat64(c.item),
                              c.count,
                              c.error))
        i = c.next

    return out
