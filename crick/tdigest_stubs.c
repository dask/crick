/* This is a C port of the merging t-digest, as provided by the t-digest
 * library found https://github.com/tdunning/t-digest. The original license is
 * included under EXTERNAL_LICENSES */

#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "common.h"


typedef struct centroid {
    double mean;
    double weight;
} centroid_t;


typedef struct tdigest {
    /* tuning parameter */
    double compression;

    /* min/max values seen */
    double min;
    double max;

    /* long-term storage of centroids */
    int size;
    int ncentroids;
    double total_weight;
    centroid_t *centroids;

    /* merging buffer */
    centroid_t *merge_centroids;

    /* short-term storage of centroids */
    int buffer_size;
    int buffer_ncentroids;
    double buffer_total_weight;
    centroid_t *buffer_centroids;

    /* sorting buffer */
    centroid_t *buffer_sort;
} tdigest_t;


CRICK_INLINE tdigest_t *tdigest_new(double compression) {
    int size, buffer_size;

    tdigest_t *T = (tdigest_t *)malloc(sizeof(*T));
    if (T == NULL)
        return NULL;

    /* Clip compression to bounds */
    if (compression < 20)
        compression = 20;
    else if (compression > 1000)
        compression = 1000;
    T->compression = compression;

    /* Select a good size and buffer_size */
    size = 2 * npy_ceil(compression);
    buffer_size = (7.5 + 0.37*compression - 2e-4*compression*compression);

    T->min = DBL_MAX;
    T->max = -DBL_MAX;

    T->size = size;
    T->total_weight = 0;
    T->ncentroids = 0;
    T->centroids = calloc(size, sizeof(centroid_t));
    if (T->centroids == NULL)
        goto fail;
    T->merge_centroids = calloc(size, sizeof(centroid_t));
    if (T->merge_centroids == NULL)
        goto fail;

    T->buffer_size = buffer_size;
    T->buffer_total_weight = 0;
    T->buffer_ncentroids = 0;
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


CRICK_INLINE void tdigest_free(tdigest_t *T) {
    free(T->centroids);
    free(T->merge_centroids);
    free(T->buffer_centroids);
    free(T->buffer_sort);
    free(T);
}


CRICK_INLINE int centroid_compare(centroid_t a, centroid_t b) {
    return a.mean < b.mean;
}


CRICK_INLINE void centroid_sort(size_t n, centroid_t array[],
                                centroid_t buffer[])
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


CRICK_INLINE double integrate(double c, double q) {
    /* TODO: Rarely (but sometimes) 1 < q < epsilon, due to some roundoff
     * issues. There's probably a way to rearrange computations so this doesn't
     * ever happen. For now, we threshold here. */
    double out;
    q = (q > 1) ? 1 : q;

    out = (c * (npy_asin(2 * q - 1) + NPY_PI_2) / NPY_PI);
    assert(out <= c);
    assert(0 <= out);
    return out;
}


CRICK_INLINE double centroid_merge(tdigest_t *T, double weight_so_far,
                                   double k1, double u, double w) {
    double k2 = integrate(T->compression, (weight_so_far + w) / T->total_weight);
    int n = T->ncentroids;

    if (n == 0) {
        /* First centroid */
        T->ncentroids = ++n;
        T->merge_centroids[n - 1].weight = w;
        T->merge_centroids[n - 1].mean = u;
    }
    else if ((k2 - k1) <= 1) {
        assert(n < T->size);
        T->merge_centroids[n - 1].weight += w;
        T->merge_centroids[n - 1].mean += ((u - T->merge_centroids[n - 1].mean) *
                                           w / T->merge_centroids[n - 1].weight);
    } else {
        T->ncentroids = ++n;
        assert(n < T->size);
        T->merge_centroids[n - 1].weight = w;
        T->merge_centroids[n - 1].mean = u;
        k1 = integrate(T->compression, weight_so_far / T->total_weight);
    }
    return k1;
}


CRICK_INLINE void tdigest_flush(tdigest_t *T) {
    int n, i = 0, j = 0;
    double k1 = 0, weight_so_far = 0;
    centroid_t c, *swap;

    if (T->buffer_ncentroids == 0)
        return;

    centroid_sort(T->buffer_ncentroids, T->buffer_centroids, T->buffer_sort);

    if (T->min > T->buffer_centroids[0].mean)
        T->min = T->buffer_centroids[0].mean;
    if (T->max < T->buffer_centroids[T->buffer_ncentroids - 1].mean)
        T->max = T->buffer_centroids[T->buffer_ncentroids - 1].mean;

    n = T->ncentroids;

    T->ncentroids = 0;
    T->total_weight += T->buffer_total_weight;
    T->buffer_total_weight = 0;

    while (i < T->buffer_ncentroids && j < n) {
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

    for (; i < T->buffer_ncentroids; i++) {
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

    T->buffer_ncentroids = 0;

    swap = T->centroids;
    T->centroids = T->merge_centroids;
    T->merge_centroids = swap;
}


CRICK_INLINE void tdigest_add(tdigest_t *T, double x, double w) {
    int n;

    /* w must be > 0 and finite, for speed we assume the caller has checked this */
    assert(w > 0);
    assert(npy_isfinite(w));

    /* Ignore x = NAN, INF, and -INF
     * Ignore w <= eps */
    if (!npy_isfinite(x) || w <= DBL_EPSILON)
        return;

    if (T->buffer_ncentroids == T->buffer_size) {
        tdigest_flush(T);
    }

    n = T->buffer_ncentroids++;
    assert(n < T->buffer_size);

    T->buffer_centroids[n].mean = x;
    T->buffer_centroids[n].weight = w;
    T->buffer_total_weight += w;
}


CRICK_INLINE void tdigest_query_prep(tdigest_t *T) {
    int i;
    centroid_t current;
    double cumulative_weight;

    /* Prep for computing quantiles or cdf */
    tdigest_flush(T);

    cumulative_weight = 0.0;
    for (i = 0; i < T->ncentroids; i++) {
        current = T->centroids[i];
        T->merge_centroids[i].mean = current.mean;
        T->merge_centroids[i].weight = cumulative_weight + current.weight / 2.0;
        cumulative_weight += current.weight;
    }
}

CRICK_INLINE int bisect_weight(centroid_t *arr, double index, int lo, int hi) {
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid].weight < index)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}


CRICK_INLINE int bisect_left_mean(centroid_t *arr, double index, int lo, int hi) {
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid].mean < index)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

CRICK_INLINE int bisect_right_mean(centroid_t *arr, double index, int lo, int hi) {
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (index < arr[mid].mean)
            hi = mid;
        else
            lo = mid + 1;
    }
    if (lo && arr[lo - 1].mean == index)
        lo--;
    return lo;
}


CRICK_INLINE double tdigest_cdf(tdigest_t *T, double x) {
    int i_l, i_r;
    double x0, x1, dw;

    /* No data */
    if (T->ncentroids == 0)
        return NPY_NAN;

    /* Single centroid */
    if (T->ncentroids == 1) {
        if (x < T->min)
            return 0;
        if (x > T->max)
            return 1;
        if (T->max - T->min < DBL_EPSILON)
            return 0.5;
        return (x - T->min) / (T->max - T->min);
    }

    /* Equality checks only apply if > 1 centroid */
    if (x >= T->max)
        return 1;
    if (x <= T->min)
        return 0;

    i_l = bisect_left_mean(T->merge_centroids, x, 0, T->ncentroids);

    if (x < T->centroids[0].mean) {
        /* min < x < first centroid */
        x0 = T->min;
        x1 = T->merge_centroids[0].mean;
        dw = T->merge_centroids[0].weight / 2;
        return dw * (x - x0) / (x1 - x0) / T->total_weight;
    } else if (i_l == T->ncentroids) {
        /* last centroid < x < max */
        x0 = T->centroids[i_l - 1].mean;
        x1 = T->max;
        dw = T->centroids[i_l - 1].weight / 2;
        return 1 - dw * (x1 - x) / (x1 - x0) / T->total_weight;
    } else if (T->centroids[i_l].mean == x) {
        /* x is equal to one or more centroids */
        i_r = bisect_right_mean(T->merge_centroids, x, i_l, T->ncentroids);
        return T->merge_centroids[i_r].weight / T->total_weight;
    } else {
        /* x is between centroids i_l - 1 and i_l */
        assert(T->centroids[i_l].mean > x);
        x0 = T->centroids[i_l - 1].mean;
        x1 = T->centroids[i_l].mean;
        dw = (T->centroids[i_l - 1].weight + T->centroids[i_l].weight) / 2;
        return (T->merge_centroids[i_l - 1].weight
                + dw * (x - x0) / (x1 - x0)) / T->total_weight;
    }
}


CRICK_INLINE PyArrayObject *tdigest_cdf_ndarray(tdigest_t *T,
                                                PyArrayObject *x) {
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

    /* Preprocess the centroids */
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


CRICK_INLINE double tdigest_quantile(tdigest_t *T, double q) {
    double index, x0, y0, x1, y1;
    int i;

    if (T->total_weight == 0)
        return NPY_NAN;
    if (q <= 0)
        return T->min;
    if (q >= 1)
        return T->max;
    if (T->ncentroids == 1)
        return T->centroids[0].mean;

    index = q * T->total_weight;
    i = bisect_weight(T->merge_centroids, index, 0, T->ncentroids);

    if (i == 0) {
        x0 = 0;
        y0 = T->min;
    } else {
        x0 = T->merge_centroids[i - 1].weight;
        y0 = T->merge_centroids[i - 1].mean;
    }

    if (i == T->ncentroids) {
        x1 = T->total_weight;
        y1 = T->max;
    } else {
        x1 = T->merge_centroids[i].weight;
        y1 = T->merge_centroids[i].mean;
    }

    return y0 + (index - x0) * (y1 - y0) / (x1 - x0);
}


CRICK_INLINE PyArrayObject *tdigest_quantile_ndarray(tdigest_t *T,
                                                     PyArrayObject *q) {
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

    /* Preprocess the centroids */
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


CRICK_INLINE void tdigest_merge(tdigest_t *T, tdigest_t *other) {
    int i;

    tdigest_flush(other);
    if (other->total_weight) {
        centroid_t *centroids = other->centroids;
        for (i=0; i < other->ncentroids; i++) {
            tdigest_add(T, centroids[i].mean, centroids[i].weight);
        }
        if (T->min > other->min)
            T->min = other->min;
        if (T->max < other->max)
            T->max = other->max;
    }
}


CRICK_INLINE void tdigest_scale(tdigest_t *T, double factor) {
    double total_weight = 0;
    tdigest_flush(T);

    if (T->total_weight) {
        centroid_t *centroids = T->centroids;
        double w;
        int i, j = 0;
        for (i=0; i < T->ncentroids; i++) {
            w = centroids[i].weight * factor;
            /* If the scaled weight is approximately 0, skip the centroid */
            if (w > DBL_EPSILON) {
                centroids[j].weight = w;
                total_weight += w;
                j++;
            }
        }
        T->total_weight = total_weight;
        T->ncentroids = j;
    }
}


CRICK_INLINE npy_intp tdigest_update_ndarray(tdigest_t *T, PyArrayObject *x,
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
    NPY_BEGIN_THREADS_DEF;

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
