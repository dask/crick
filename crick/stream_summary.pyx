cimport cython
from cpython.ref cimport PyObject, Py_INCREF

cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "stream_summary_stubs.c":
    np.int64_t asint64(np.float64_t key)

    ctypedef struct summary_t:
        size_t capacity
        size_t size
        size_t head

    # int64
    ctypedef struct counter_int64_t:
        np.int64_t item
        np.int64_t count
        np.int64_t error

    ctypedef struct node_int64_t:
        size_t next
        size_t prev
        counter_int64_t counter

    ctypedef struct summary_int64_t:
        size_t size
        size_t head
        node_int64_t *list

    summary_int64_t *summary_int64_new(int capacity)
    void summary_int64_free(summary_int64_t *T)
    int summary_int64_add(summary_int64_t *T, np.int64_t item, np.int64_t count) except -1
    int summary_int64_merge(summary_int64_t *T1, summary_int64_t *T2) except -1
    int summary_int64_set_state(summary_int64_t *T, counter_int64_t *counters, size_t size) except -1
    int summary_int64_update_ndarray(summary_int64_t *T, np.PyArrayObject *item,
                                     np.PyArrayObject *count) except -1

    # object
    ctypedef struct counter_object_t:
        PyObject *item
        np.int64_t count
        np.int64_t error

    ctypedef struct node_object_t:
        size_t next
        size_t prev
        counter_object_t counter

    ctypedef struct summary_object_t:
        size_t size
        size_t head
        node_object_t *list

    summary_object_t *summary_object_new(int capacity)
    void summary_object_free(summary_object_t *T)
    int summary_object_add(summary_object_t *T, object item, np.int64_t count) except -1
    int summary_object_merge(summary_object_t *T1, summary_object_t *T2) except -1
    int summary_object_set_state(summary_object_t *T, counter_object_t *counters, size_t size) except -1
    int summary_object_update_ndarray(summary_object_t *T, np.PyArrayObject *item,
                                      np.PyArrayObject *count) except -1


# Repeat struct definition for numpy
_int64_offsets = [<size_t> &(<counter_int64_t*> NULL).item,
                  <size_t> &(<counter_int64_t*> NULL).count,
                  <size_t> &(<counter_int64_t*> NULL).error]

cdef np.dtype COUNTER_INT64_DTYPE = np.dtype(
        {'names': ['item', 'count', 'error'],
         'formats': [np.int64, np.int64, np.int64],
         'offsets': _int64_offsets})

cdef np.dtype COUNTER_FLOAT64_DTYPE = np.dtype(
        {'names': ['item', 'count', 'error'],
         'formats': [np.float64, np.int64, np.int64],
         'offsets': _int64_offsets})

cdef np.dtype COUNTER_OBJECT_DTYPE = np.dtype(
        {'names': ['item', 'count', 'error'],
         'formats': [np.object, np.int64, np.int64],
         'offsets': [<size_t> &(<counter_object_t*> NULL).item,
                     <size_t> &(<counter_object_t*> NULL).count,
                     <size_t> &(<counter_object_t*> NULL).error
                     ]
         })


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
    """StreamSummary(capacity=20, dtype='f8')

    Approximate TopK.

    Parameters
    ----------
    capacity : int, optional
        The number of items to track. A larger number gives higher accuracy,
        but uses more memory. Default is 20.
    dtype : dtype, optional
        The dtype of the objects to track. Currently only ``float64``,
        ``int64``, and ``object`` are supported. Other integer/float types will
        be upcast to match. Default is ``float64``.

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
    cdef summary_t *summary
    cdef readonly np.dtype dtype

    def __cinit__(self, int capacity=20, dtype='f8'):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        dtype = np.dtype(dtype)

        if np.PyDataType_ISOBJECT(dtype):
            self.summary = <summary_t *>summary_object_new(capacity)
        elif np.PyDataType_ISINTEGER(dtype):
            dtype = np.dtype(np.int64)
            self.summary = <summary_t *>summary_int64_new(capacity)
        elif np.PyDataType_ISFLOAT(dtype):
            dtype = np.dtype(np.float64)
            self.summary = <summary_t *>summary_int64_new(capacity)
        else:
            raise ValueError("dtype %s not supported" % dtype)
        self.dtype = dtype

        if self.summary == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.summary != NULL:
            if np.PyDataType_ISOBJECT(self.dtype):
                summary_object_free(<summary_object_t *>self.summary)
            else:
                summary_int64_free(<summary_int64_t *>self.summary)

    def __repr__(self):
        return ("StreamSummary<capacity={0}, dtype={1}, "
                "size={2}>").format(self.capacity, self.dtype, self.size())

    @property
    def capacity(self):
        return self.summary.capacity

    cpdef int size(self):
        """size(self)

        The number of active counters."""
        return self.summary.size

    def counters(self, astuples=False):
        """counters(self)

        Returns a numpy array of all the counters in the summary. Note that
        this array is a *copy* of the internal data.
        """
        return self.topk(self.size(), astuples=astuples)

    def __reduce__(self):
        return (StreamSummary, (self.capacity, self.dtype), self.__getstate__())

    def __getstate__(self):
        return self.counters()

    def __setstate__(self, state):
        cdef np.ndarray counters = state
        cdef size_t size = len(counters)
        if np.PyDataType_ISOBJECT(self.dtype):
            summary_object_set_state(<summary_object_t*>self.summary,
                                     <counter_object_t*>counters.data,
                                     size)
        else:
            if np.PyDataType_ISFLOAT(self.dtype):
                counters = counters.view(COUNTER_INT64_DTYPE)
            summary_int64_set_state(<summary_int64_t*>self.summary,
                                    <counter_int64_t*>counters.data,
                                    size)

    def add(self, item, int count=1):
        """add(self, item, count=1)

        Add an element to the summary.

        Parameters
        ----------
        item
            The item to add. If it doesn't match the summary dtype, conversion
            will be attempted.
        count : int, optional
            The count of the item to add. Default is 1.
        """
        if count <= 0:
            raise ValueError("count must be > 0")

        if np.PyDataType_ISOBJECT(self.dtype):
            summary_object_add(<summary_object_t *>self.summary, item, count)
        elif np.PyDataType_ISFLOAT(self.dtype):
            summary_int64_add(<summary_int64_t *>self.summary, asint64(item), count)
        else:
            summary_int64_add(<summary_int64_t *>self.summary, item, count)

    def update(self, item, count=1):
        """update(self, item, count=1)

        Add many elements to the summary.

        Parameters
        ----------
        item : array_like
            The item to add. If they don't match the summary dtype, conversion
            will be attempted.
        count : array_like, optional
            The count (or counts) of the item to add. Default is 1.
        """
        item = np.asarray(item)
        check_count = not (np.isscalar(count) and count == 1)
        count = np.asarray(count)

        item = item.astype(self.dtype, casting='safe', copy=False)
        count = count.astype(np.int64, casting='safe', copy=False)

        if check_count and (count <= 0).any():
            raise ValueError("count must be > 0")

        if np.PyDataType_ISOBJECT(self.dtype):
            summary_object_update_ndarray(<summary_object_t *>self.summary,
                                          <np.PyArrayObject*>item,
                                          <np.PyArrayObject*>count)
        else:
            if np.PyDataType_ISFLOAT(self.dtype):
                item = item.view('i8')
            summary_int64_update_ndarray(<summary_int64_t *>self.summary,
                                         <np.PyArrayObject*>item,
                                         <np.PyArrayObject*>count)

    cpdef topk(self, int k, astuples=False):
        """topk(self, k, astuples=True)

        Estimate the top k elements.

        Parameters
        ----------
        k : int
            The number of most frequent elements to estimate.
        astuples : bool, optional
            If False [default], the result is a numpy record array, otherwise
            it's a list of ``TopKResult`` tuples.

        Returns
        -------
        topk : array or list of TopKResult
            Returns a record array, or a list of ``TopKResult`` tuples. In both
            cases, the result has the following fields:

            - item: The item
            - count: The estimated frequency.
            - error: An upper bound on the error for ``count``. The number of
                     occurrences of ``item`` is guaranteed to be in
                     ``count <= actual <= count + error``.
        """
        cdef np.ndarray out
        if k < 0:
            raise ValueError("k must be >= 0")

        if np.PyDataType_ISOBJECT(self.dtype):
            out = object_counters(<summary_object_t*>self.summary, k)
        else:
            out = int64_counters(<summary_int64_t*>self.summary, k)
            if np.PyDataType_ISFLOAT(self.dtype):
                out = out.view(COUNTER_FLOAT64_DTYPE)
        if astuples:
            return [TopKResult(*i) for i in out]
        return out

    def merge(self, *args):
        """merge(self, *args)

        Update this summary inplace with data from other summaries.

        Parameters
        ----------
        args : StreamSummarys
            StreamSummarys to merge into this one.
        """
        if not all(isinstance(i, StreamSummary) for i in args):
            raise TypeError("All arguments to merge must be StreamSummarys")
        if not all(i.dtype == self.dtype for i in args):
            raise ValueError("All arguments to merge must have same dtype")
        cdef StreamSummary s
        if np.PyDataType_ISOBJECT(self.dtype):
            for s in args:
                summary_object_merge(<summary_object_t*>self.summary,
                                     <summary_object_t*>s.summary)
        else:
            for s in args:
                summary_int64_merge(<summary_int64_t*>self.summary,
                                    <summary_int64_t*>s.summary)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_counters(summary_int64_t *summary, int k):
    cdef:
        int i = summary.head
        int count
        node_int64_t c
        np.ndarray[counter_int64_t, ndim=1] out

    k = min(summary.size, k)
    out = np.empty(k, dtype=COUNTER_INT64_DTYPE)
    for count in range(k):
        c = summary.list[i]
        out[count] = c.counter
        i = c.next

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object_counters(summary_object_t *summary, int k):
    cdef:
        int i = summary.head
        int count
        node_object_t c
        np.ndarray out
        counter_object_t *data

    k = min(summary.size, k)
    out = np.empty(k, dtype=COUNTER_OBJECT_DTYPE)
    data = <counter_object_t *>out.data

    for count in range(k):
        c = summary.list[i]
        Py_INCREF(<object>c.counter.item)
        data[count] = c.counter
        i = c.next
    return out
