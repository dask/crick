cimport cython

cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "space_saving_stubs.c":
    ctypedef struct counter_t:
        size_t next
        size_t prev
        long item
        long count
        long error

    ctypedef struct summary_t:
        size_t capacity
        size_t n_counters
        size_t head
        counter_t *counters

    summary_t *summary_new(int capacity)
    void summary_free(summary_t *T)
    int summary_add(summary_t *T, long item, int count)


cdef struct int64_result:
    long item
    long count
    long error


cdef class SpaceSaving:
    """SpaceSaving(capacity=50)"""
    cdef summary_t *summary

    def __cinit__(self, int capacity=50):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.summary = summary_new(capacity)
        if self.summary == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.summary != NULL:
            summary_free(self.summary)

    def __repr__(self):
        return ("SpaceSaving<capacity={0}, "
                "size={1}>").format(self.capacity, self.size())

    @property
    def capacity(self):
        return self.summary.capacity

    cpdef int size(self):
        """size(self)

        The number of active counters."""
        return self.summary.n_counters

    def add(self, int x, int w=1):
        """add(self, x, w)

        Parameters
        ----------
        x : int
            The value to add.
        w : int
            The weight of the value to add. Default is 1.
        """
        # Don't check w in the common case where w is 1
        if w <= 0:
            raise ValueError("w must be > 0")
        summary_add(self.summary, x, w)

    @cython.boundscheck(False)
    def topk(self, int k):
        """Get the topk elements"""
        cdef int i, count

        if k <= 0:
            raise ValueError("k must be > 0")
        k = min(self.size(), k)

        int64_result_dtype = np.dtype([('item', np.int64),
                                       ('count', np.int64),
                                       ('error', np.int64)])

        cdef np.ndarray[int64_result] out = np.empty(k, dtype=int64_result_dtype)
        cdef counter_t c

        i = self.summary.head
        for count in range(k):
            c = self.summary.counters[i]
            out[count].item = c.item
            out[count].count = c.count
            out[count].error = c.error
            i = c.next

        return out
