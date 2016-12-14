/* This is a C port of the merging t-digest, as provided by the t-digest
 * library found https://github.com/tdunning/t-digest. The original license is
 * included under EXTERNAL_LICENSES */

#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#include <Python.h>
#include <numpy/arrayobject.h>


typedef struct centroid {
    double mean;
    double weight;
} centroid_t;


typedef struct tdigest {
    // tuning parameter
    double compression;

    // min/max values seen
    double min;
    double max;

    // long-term storage of centroids
    int size;
    int last;
    double total_weight;
    centroid_t *centroids;

    // merging buffer
    centroid_t *merge_centroids;

    // short-term storage of centroids
    int buffer_size;
    int buffer_last;
    double buffer_total_weight;
    centroid_t *buffer_centroids;

    // sorting buffer
    centroid_t *buffer_sort;
} tdigest_t;


static tdigest_t *tdigest_new(double compression) {
    tdigest_t *T = (tdigest_t *)malloc(sizeof(*T));
    if (T == NULL)
        return NULL;

    // Clip compression to bounds
    if (compression < 20)
        compression = 20;
    else if (compression > 1000)
        compression = 1000;
    T->compression = compression;

    // Select a good size and buffer_size
    int size = 2 * ceil(compression);
    int buffer_size = (7.5 + 0.37*compression - 2e-4*compression*compression);

    T->min = DBL_MAX;
    T->max = -DBL_MAX;

    T->size = size;
    T->total_weight = 0;
    T->last = 0;
    T->centroids = calloc(size, sizeof(centroid_t));
    if (T->centroids == NULL)
        goto fail;
    T->merge_centroids = calloc(size, sizeof(centroid_t));
    if (T->merge_centroids == NULL)
        goto fail;

    T->buffer_size = buffer_size;
    T->buffer_total_weight = 0;
    T->buffer_last = 0;
    T->buffer_centroids = calloc(buffer_size, sizeof(centroid_t));
    if (T->buffer_centroids == NULL)
        goto fail;
    T->buffer_sort = calloc(buffer_size, sizeof(centroid_t));
    if (T->buffer_sort == NULL)
        goto fail;

    return T;

fail:
    if (T->centroids != NULL) free(T->centroids);
    if (T->merge_centroids != NULL) free(T->merge_centroids);
    if (T->buffer_centroids != NULL) free(T->buffer_centroids);
    if (T->buffer_sort != NULL) free(T->buffer_sort);
    return NULL;
}


static void tdigest_free(tdigest_t *T) {
    free(T->centroids);
    free(T->merge_centroids);
    free(T->buffer_centroids);
    free(T->buffer_sort);
    free(T);
}


static inline int centroid_compare(centroid_t a, centroid_t b) {
    return a.mean < b.mean;
}


static void centroid_sort(size_t n, centroid_t array[], centroid_t buffer[])
{
    centroid_t *a2[2], *a, *b;
    int curr, shift;

    a2[0] = array;
    a2[1] = buffer;
    for (curr = 0, shift = 0; (1ul<<shift) < n; ++shift) {
        a = a2[curr];
        b = a2[1-curr];
        if (shift == 0) {
            centroid_t *p = b, *i, *eb = a + n;
            for (i = a; i < eb; i += 2) {
                if (i == eb - 1)
                    *p++ = *i;
                else {
                    if (centroid_compare(*(i+1), *i)) {
                        *p++ = *(i+1);
                        *p++ = *i;
                    } else {
                        *p++ = *i;
                        *p++ = *(i+1);
                    }
                }
            }
        } else {
            size_t i, step = 1ul<<shift;
            for (i = 0; i < n; i += step<<1) {
                centroid_t *p, *j, *k, *ea, *eb;
                if (n < i + step) {
                    ea = a + n;
                    eb = a;
                } else {
                    ea = a + i + step;
                    eb = a + (n < i + (step<<1)? n : i + (step<<1));
                }
                j = a + i;
                k = a + i + step;
                p = b + i;
                while (j < ea && k < eb) {
                    if (centroid_compare(*k, *j))
                        *p++ = *k++;
                    else
                        *p++ = *j++;
                }
                while (j < ea)
                    *p++ = *j++;
                while (k < eb)
                    *p++ = *k++;
            }
        }
        curr = 1 - curr;
    }
    if (curr == 1) {
        centroid_t *p = a2[0], *i = a2[1], *eb = array + n;
        for (; p < eb; ++i)
            *p++ = *i;
    }
}


static inline double integrate(double c, double q) {
    // TODO: Rarely (but sometimes) 1 < q < epsilon, due to some roundoff
    // issues. There's probably a way to rearrange computations so this doesn't
    // ever happen. For now, we threshold here.
    q = fmin(q, 1);

    double out = (c * (asin(2 * q - 1) + M_PI_2) / M_PI);
    assert(out <= c);
    assert(0 <= out);
    return out;
}


static double centroid_merge(tdigest_t *T, double weight_so_far, double k1,
                             double u, double w) {
    double k2 = integrate(T->compression, (weight_so_far + w) / T->total_weight);
    int n = T->last;

    if (weight_so_far == 0) {
        assert(n < T->size);
        T->merge_centroids[n].weight = w;
        T->merge_centroids[n].mean = u;
    }
    else if ((k2 - k1) <= 1) {
        assert(n < T->size);
        T->merge_centroids[n].weight += w;
        T->merge_centroids[n].mean += ((u - T->merge_centroids[n].mean) *
                                       w / T->merge_centroids[n].weight);
    } else {
        T->last = ++n;
        assert(n < T->size);
        T->merge_centroids[n].weight = w;
        T->merge_centroids[n].mean = u;
        k1 = integrate(T->compression, weight_so_far / T->total_weight);
    }
    return k1;
}


static void tdigest_flush(tdigest_t *T) {
    if (T->buffer_last == 0)
        return;

    centroid_sort(T->buffer_last, T->buffer_centroids, T->buffer_sort);

    T->min = fmin(T->min, T->buffer_centroids[0].mean);
    T->max = fmax(T->max, T->buffer_centroids[T->buffer_last - 1].mean);

    int n = (T->total_weight > 0) ? T->last + 1 : 0;

    T->last = 0;
    T->total_weight += T->buffer_total_weight;
    T->buffer_total_weight = 0;

    int i = 0, j = 0;
    double k1 = 0, weight_so_far = 0;
    centroid_t c;

    while (i < T->buffer_last && j < n) {
        if (centroid_compare(T->buffer_centroids[i], T->centroids[j])) {
            assert(i < T->buffer_size);
            c = T->buffer_centroids[i];
            i++;
        } else {
            assert(j < T->size);
            c = T->centroids[j];
            j++;
        }
        k1 = centroid_merge(T, weight_so_far, k1, c.mean, c.weight);
        weight_so_far += c.weight;
    }

    for (; i < T->buffer_last; i++) {
        assert(i < T->buffer_size);
        c = T->buffer_centroids[i];
        k1 = centroid_merge(T, weight_so_far, k1, c.mean, c.weight);
        weight_so_far += c.weight;
    }

    for (; j < n; j++) {
        assert(j < T->size);
        c = T->centroids[j];
        k1 = centroid_merge(T, weight_so_far, k1, c.mean, c.weight);
        weight_so_far += c.weight;
    }

    T->buffer_last = 0;

    centroid_t *swap = T->centroids;
    T->centroids = T->merge_centroids;
    T->merge_centroids = swap;
}


static inline void tdigest_add(tdigest_t *T, double x, double w) {
    // w must be > 0 and finite, for speed we assume the caller has checked this
    assert(w > 0);
    assert(isfinite(w));

    // Ignore x = NAN, INF, and -INF
    // Ignore w <= eps
    if (!isfinite(x) || w <= DBL_EPSILON)
        return;

    if (T->buffer_last >= T->buffer_size) {
        tdigest_flush(T);
    }

    int n = T->buffer_last++;
    assert(n < T->buffer_size);

    T->buffer_centroids[n].mean = x;
    T->buffer_centroids[n].weight = w;
    T->buffer_total_weight += w;
}


static void tdigest_query_prep(tdigest_t *T) {
    // Prep for computing quantiles or cdf
    tdigest_flush(T);

    centroid_t a, b = T->centroids[0];
    T->merge_centroids[0].mean = 0;

    for (int i = 1; i < T->last + 1; i++) {
        a = b;
        b = T->centroids[i];

        T->merge_centroids[i].mean = (T->merge_centroids[i - 1].mean +
                                      T->centroids[i].weight);
        T->merge_centroids[i - 1].weight = (a.mean + (b.mean - a.mean) *
                                            a.weight / (a.weight + b.weight));
    }
}


static inline int bisect(centroid_t *arr, double index, int lo, int hi) {
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid].mean < index)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}


static inline double interpolate(double x, double x0, double x1) {
    return (x - x0) / (x1 - x0);
}


static double tdigest_cdf(tdigest_t *T, double x) {
    if (T->total_weight == 0)
        return NAN;
    if (x < T->min)
        return 0;
    if (x > T->max)
        return 1;

    if (T->last == 0) {
        if (T->max - T->min < DBL_EPSILON)
            return 0.5;
        return interpolate(x, T->min, T->max);
    }

    int i = bisect(T->centroids, x, 0, T->last + 1);
    double l = (i > 0) ? T->merge_centroids[i - 1].weight : T->min;
    double r = (i < T->last) ? T->merge_centroids[i].weight : T->max;
    if (x < l) {
        r = l;
        l = --i ? T->merge_centroids[i - 1].weight : T->min;
    }
    double weight = (i > 0) ? T->merge_centroids[i - 1].mean : 0;
    return ((weight + T->centroids[i].weight * interpolate(x, l, r)) /
            T->total_weight);
}


static PyArrayObject *tdigest_cdf_ndarray(tdigest_t *T, PyArrayObject *x) {
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext = NULL;
    PyArrayObject *ret = NULL;
    PyArrayObject *op[2] = {NULL};
    npy_uint32 flags;
    npy_uint32 op_flags[2];
    PyArray_Descr *dtypes[2] = {NULL};

    npy_intp *innersizeptr, *strideptr;
    char **dataptr;

    op[0] = x;
    op[1] = NULL;
    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_ZEROSIZE_OK;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    dtypes[0] = PyArray_DescrFromType(NPY_FLOAT64);
    if (dtypes[0] == NULL) {
        goto finish;
    }
    dtypes[1] = dtypes[0];
    Py_INCREF(dtypes[1]);

    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        goto finish;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // Preprocess the centroids
    tdigest_query_prep(T);
    do {
        char *data_x = dataptr[0];
        char *data_out = dataptr[1];
        npy_intp stride_x = strideptr[0];
        npy_intp stride_out = strideptr[1];
        npy_intp count = *innersizeptr;

        while (count--) {
            *(npy_float64 *)data_out = tdigest_cdf(T, *(npy_float64 *)data_x);

            data_x += stride_x;
            data_out += stride_out;
        }
    } while (iternext(iter));

    ret = NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);

finish:
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_XDECREF(ret);
            ret = NULL;
        }
    }
    return ret;
}


static double tdigest_quantile(tdigest_t *T, double q) {
    if (T->total_weight == 0)
        return NAN;
    if (q <= 0)
        return T->min;
    if (q >= 1)
        return T->max;
    if (T->last == 0)
        return T->centroids[0].mean;

    double index = q * T->total_weight;
    int i = bisect(T->merge_centroids, index, 0, T->last + 1);
    double l = (i > 0) ? T->merge_centroids[i - 1].weight : T->min;
    double r = (i < T->last) ? T->merge_centroids[i].weight : T->max;
    double weight = (i > 0) ? T->merge_centroids[i - 1].mean : 0;
    return l + (r - l) * (index - weight) / T->centroids[i].weight;
}


static PyArrayObject *tdigest_quantile_ndarray(tdigest_t *T, PyArrayObject *q) {
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext = NULL;
    PyArrayObject *ret = NULL;
    PyArrayObject *op[2] = {NULL};
    npy_uint32 flags;
    npy_uint32 op_flags[2];
    PyArray_Descr *dtypes[2] = {NULL};

    npy_intp *innersizeptr, *strideptr;
    char **dataptr;

    op[0] = q;
    op[1] = NULL;
    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_ZEROSIZE_OK;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    dtypes[0] = PyArray_DescrFromType(NPY_FLOAT64);
    if (dtypes[0] == NULL) {
        goto finish;
    }
    dtypes[1] = dtypes[0];
    Py_INCREF(dtypes[1]);

    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        goto finish;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // Preprocess the centroids
    tdigest_query_prep(T);
    do {
        char *data_q = dataptr[0];
        char *data_out = dataptr[1];
        npy_intp stride_q = strideptr[0];
        npy_intp stride_out = strideptr[1];
        npy_intp count = *innersizeptr;

        while (count--) {
            *(npy_float64 *)data_out = tdigest_quantile(T, *(npy_float64 *)data_q);

            data_q += stride_q;
            data_out += stride_out;
        }
    } while (iternext(iter));

    ret = NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);

finish:
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_XDECREF(ret);
            ret = NULL;
        }
    }
    return ret;
}


static void tdigest_merge(tdigest_t *T, tdigest_t *other) {
    tdigest_flush(other);
    if (other->total_weight) {
        centroid_t *centroids = other->centroids;
        for (int i=0; i < other->last + 1; i++) {
            tdigest_add(T, centroids[i].mean, centroids[i].weight);
        }
        T->min = fmin(T->min, other->min);
        T->max = fmax(T->max, other->max);
    }
}


static void tdigest_scale(tdigest_t *T, double factor) {
    tdigest_flush(T);
    double total_weight = 0;
    if (T->total_weight) {
        centroid_t *centroids = T->centroids;
        double w;
        int j = 0;
        for (int i=0; i < T->last + 1; i++) {
            w = centroids[i].weight * factor;
            // If the scaled weight is approximately 0, skip the centroid
            if (w > DBL_EPSILON) {
                centroids[j].weight = w;
                total_weight += w;
                j++;
            }
        }
        T->total_weight = total_weight;
        T->last = j == 0 ? j : j - 1;
    }
}


static npy_intp tdigest_update_ndarray(tdigest_t *T, PyArrayObject *x,
                                       PyArrayObject *w) {
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    PyArrayObject *op[2];
    npy_uint32 flags;
    npy_uint32 op_flags[2];
    PyArray_Descr *dtypes[2] = {NULL};

    npy_intp *innersizeptr, *strideptr;
    char **dataptr;

    npy_intp ret = -1;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(x) == 0 || PyArray_SIZE(w) == 0) {
        return 0;
    }

    op[0] = x;
    op[1] = w;
    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;

    dtypes[0] = PyArray_DescrFromType(NPY_FLOAT64);
    if (dtypes[0] == NULL) {
        goto finish;
    }
    dtypes[1] = dtypes[0];
    Py_INCREF(dtypes[1]);

    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        goto finish;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));

    do {
        char *data_x = dataptr[0];
        char *data_w = dataptr[1];
        npy_intp stride_x = strideptr[0];
        npy_intp stride_w = strideptr[1];
        npy_intp count = *innersizeptr;

        while (count--) {
            tdigest_add(T, *(npy_float64 *)data_x,
                           *(npy_float64 *)data_w);

            data_x += stride_x;
            data_w += stride_w;
        }
    } while (iternext(iter));

    NPY_END_THREADS;

    ret = 0;

finish:
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            return -1;
        }
    }
    return ret;
}
