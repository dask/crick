cimport cython
from cpython.ref cimport PyObject, Py_INCREF

cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "stream_summary_stubs.c":
    ctypedef struct summary_t:
        size_t capacity
        size_t size
        size_t head

    # int64
    ctypedef struct counter_int64_t:
        np.int64_t item
        long count
        long error

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
    int summary_int64_add(summary_int64_t *T, np.int64_t item, int count) except -1
    int summary_int64_set_state(summary_int64_t *T, counter_int64_t *counters, size_t size) except -1
    np.npy_intp summary_int64_update_ndarray(summary_int64_t *T, np.PyArrayObject *item,
                                             np.PyArrayObject *count) except -1

    # float64
    ctypedef struct counter_float64_t:
        np.float64_t item
        long count
        long error

    ctypedef struct node_float64_t:
        size_t next
        size_t prev
        counter_float64_t counter

    ctypedef struct summary_float64_t:
        size_t size
        size_t head
        node_float64_t *list

    summary_float64_t *summary_float64_new(int capacity)
    void summary_float64_free(summary_float64_t *T)
    int summary_float64_add(summary_float64_t *T, np.float64_t item, int count) except -1
    int summary_float64_set_state(summary_float64_t *T, counter_float64_t *counters, size_t size) except -1

    # object
    ctypedef struct counter_object_t:
        PyObject *item
        long count
        long error

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
    int summary_object_add(summary_object_t *T, object item, int count) except -1
    int summary_object_set_state(summary_object_t *T, counter_object_t *counters, size_t size) except -1
    np.npy_intp summary_object_update_ndarray(summary_object_t *T, np.PyArrayObject *item,
                                              np.PyArrayObject *count) except -1


# Repeat struct definition for numpy
cdef np.dtype COUNTER_INT64_DTYPE = np.dtype(
        {'names': ['item', 'count', 'error'],
         'formats': [np.int64, np.int64, np.int64],
         'offsets': [<size_t> &(<counter_int64_t*> NULL).item,
                     <size_t> &(<counter_int64_t*> NULL).count,
                     <size_t> &(<counter_int64_t*> NULL).error
                     ]
         })

cdef np.dtype COUNTER_FLOAT64_DTYPE = np.dtype(
        {'names': ['item', 'count', 'error'],
         'formats': [np.float64, np.int64, np.int64],
         'offsets': [<size_t> &(<counter_float64_t*> NULL).item,
                     <size_t> &(<counter_float64_t*> NULL).count,
                     <size_t> &(<counter_float64_t*> NULL).error
                     ]
         })

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
    cdef summary_t *summary
    cdef readonly np.dtype dtype

    def __cinit__(self, int capacity=20, dtype=float):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        dtype = np.dtype(dtype)
        self.dtype = dtype

        if np.PyDataType_ISOBJECT(dtype):
            self.summary = <summary_t *>summary_object_new(capacity)
        elif np.PyDataType_ISINTEGER(dtype):
            self.summary = <summary_t *>summary_int64_new(capacity)
        elif np.PyDataType_ISFLOAT(dtype):
            self.summary = <summary_t *>summary_float64_new(capacity)
        else:
            raise ValueError("dtype %s not supported" % dtype)

        if self.summary == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.summary != NULL:
            if np.PyDataType_ISOBJECT(self.dtype):
                summary_object_free(<summary_object_t *>self.summary)
            elif np.PyDataType_ISINTEGER(self.dtype):
                summary_int64_free(<summary_int64_t *>self.summary)
            else:
                summary_float64_free(<summary_float64_t *>self.summary)

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

    def counters(self):
        """counters(self)

        Returns a numpy array of all the counters in the summary. Note that
        this array is a *copy* of the internal data.
        """
        return self.topk(self.size())

    def __reduce__(self):
        return (StreamSummary, (self.capacity, self.dtype), self.__getstate__())

    def __getstate__(self):
        return self.counters()

    def __setstate__(self, state):
        cdef np.ndarray counters = state
        cdef size = len(counters)
        if np.PyDataType_ISOBJECT(self.dtype):
            return summary_object_set_state(<summary_object_t*>self.summary,
                                            <counter_object_t*>counters.data,
                                            size)
        elif np.PyDataType_ISFLOAT(self.dtype):
            return summary_float64_set_state(<summary_float64_t*>self.summary,
                                             <counter_float64_t*>counters.data,
                                             size)
        else:
            return summary_int64_set_state(<summary_int64_t*>self.summary,
                                           <counter_int64_t*>counters.data,
                                           size)

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
        elif np.PyDataType_ISFLOAT(self.dtype):
            summary_float64_add(<summary_float64_t *>self.summary, item, count)
        else:
            summary_int64_add(<summary_int64_t *>self.summary, item, count)

    def update(self, item, count=1):
        """update(self, item, count=1)"""
        item = np.asarray(item)
        check_count = not (np.isscalar(count) and count == 1)
        count = np.asarray(count)

        item = item.astype(self.dtype, casting='safe', copy=False)
        count = count.astype(self.dtype, casting='safe', copy=False)

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

    def topk(self, int k, asarray=True):
        """topk(self, k, asarray=True)

        Estimate the top k elements.

        Parameters
        ----------
        k : int
            The number of most frequent elements to estimate.
        asarray : bool, optional
            If True [default], the result is a numpy record array, otherwise
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
        if k <= 0:
            raise ValueError("k must be > 0")

        if np.PyDataType_ISOBJECT(self.dtype):
            out = object_counters(<summary_object_t*>self.summary, k)
        elif np.PyDataType_ISFLOAT(self.dtype):
            out = float64_counters(<summary_float64_t*>self.summary, k)
        else:
            out = int64_counters(<summary_int64_t*>self.summary, k)
        if asarray:
            return out
        return [TopKResult(*i) for i in out]


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
cdef float64_counters(summary_float64_t *summary, int k):
    cdef:
        int i = summary.head
        int count
        node_float64_t c
        np.ndarray[counter_float64_t, ndim=1] out

    k = min(summary.size, k)
    out = np.empty(k, dtype=COUNTER_FLOAT64_DTYPE)
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
