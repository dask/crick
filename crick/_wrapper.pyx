

cdef extern from "tdigest.c":

    ctypedef struct tdigest_t:
        pass

    tdigest_t *tdigest_new(double compression);

    void tdigest_free(tdigest_t *T);

    void tdigest_add(tdigest_t *T, double x, int w);

    void tdigest_flush(tdigest_t *T);

    double tdigest_quantile(tdigest_t *T, double q);

    double tdigest_min(tdigest_t *T);

    double tdigest_max(tdigest_t *T);

    int tdigest_count(tdigest_t *T);



cdef class TDigest:
    
    cdef tdigest_t *tdigest

    def __init__(self, compression=100.0):
        self.tdigest = tdigest_new(compression)

    def __dealloc__(self):
        if self.tdigest != NULL:
            tdigest_free(self.tdigest)
            self.tdigest == NULL

    def add(self, double x, int w=1):
        tdigest_add(self.tdigest, x, w)

    @property
    def min(self):
        return tdigest_min(self.tdigest)

    @property
    def max(self):
        return tdigest_max(self.tdigest)

    @property
    def count(self):
        return tdigest_count(self.tdigest)

    def quantile(self, double q):
        return tdigest_quantile(self.tdigest, q)
