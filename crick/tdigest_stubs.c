/* This is a C port of the merging t-digest, as provided by the t-digest
 * library found https://github.com/tdunning/t-digest. The original license is
 * included under EXTERNAL_LICENSES */

#include <stdlib.h>
#include <float.h>
#include <stdint.h>
#include <math.h>

#include <Python.h>
#include <numpy/arrayobject.h>


typedef struct centroid {
    double mean;
    uint32_t weight;
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


tdigest_t *tdigest_new(double compression) {
    tdigest_t *T = (tdigest_t *)malloc(sizeof(*T));

    // Clip compression to bounds
    if (compression < 20)
        compression = 20;
    else if (compression > 1000)
        compression = 1000;
    T->compression = compression;

    // Select a good size and buffer_size
    int size = M_PI_2 * compression + 0.5;
    int buffer_size = (7.5 + 0.37*compression - 2e-4*compression*compression);

    T->min = DBL_MAX;
    T->max = -DBL_MAX;

    T->size = size;
    T->total_weight = 0;
    T->last = 0;
    T->centroids = calloc(size, sizeof(centroid_t));
    T->merge_centroids = calloc(size, sizeof(centroid_t));

    T->buffer_size = buffer_size;
    T->buffer_total_weight = 0;
    T->buffer_last = 0;
    T->buffer_centroids = calloc(buffer_size, sizeof(centroid_t));
    T->buffer_sort = calloc(buffer_size, sizeof(centroid_t));

    return T;
}


void tdigest_free(tdigest_t *T) {
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
    return (c * (asin(2 * q - 1) + M_PI_2) / M_PI);
}


static double centroid_merge(tdigest_t *T, double weight_so_far, double k1,
                             double u, int w) {
    double k2 = integrate(T->compression, (weight_so_far + w) / T->total_weight);
    int n = T->last;

    if (weight_so_far == 0) {
        T->merge_centroids[n].weight = w;
        T->merge_centroids[n].mean = u;
    }
    else if ((k2 - k1) <= 1) {
        T->merge_centroids[n].weight += w;
        T->merge_centroids[n].mean += ((u - T->merge_centroids[n].mean) *
                                       w / T->merge_centroids[n].weight);
    } else {
        T->last = ++n;
        T->merge_centroids[n].weight = w;
        T->merge_centroids[n].mean = u;
        k1 = integrate(T->compression, weight_so_far / T->total_weight);
    }
    return k1;
}


void tdigest_flush(tdigest_t *T) {
    if (T->buffer_total_weight == 0)
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
            c = T->buffer_centroids[i];
            i++;
        } else {
            c = T->centroids[j];
            j++;
        }
        k1 = centroid_merge(T, weight_so_far, k1, c.mean, c.weight);
        weight_so_far += c.weight;
    }

    for (; i < T->buffer_last; i++) {
        c = T->buffer_centroids[i];
        k1 = centroid_merge(T, weight_so_far, k1, c.mean, c.weight);
        weight_so_far += c.weight;
    }

    for (; j < n; j++) {
        c = T->centroids[j];
        k1 = centroid_merge(T, weight_so_far, k1, c.mean, c.weight);
        weight_so_far += c.weight;
    }

    T->buffer_last = 0;

    centroid_t *swap = T->centroids;
    T->centroids = T->merge_centroids;
    T->merge_centroids = swap;
}


void tdigest_add(tdigest_t *T, double x, int w) {
    if isnan(x)
        return;

    if (T->buffer_last >= T->buffer_size) {
        tdigest_flush(T);
    }

    int n = T->buffer_last++;
    T->buffer_centroids[n].mean = x;
    T->buffer_centroids[n].weight = w;
    T->buffer_total_weight += w;
}


static inline double interpolate(double x, double x0, double x1) {
    return (x - x0) / (x1 - x0);
}


double tdigest_cdf(tdigest_t *T, double x) {
    tdigest_flush(T);

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

    double r = 0;
    double left = 0, right=0;
    centroid_t a, b = {T->min, 0};

    for (int i = 0; i < T->last + 1; i++) {
        a = b;
        b = T->centroids[i];

        right = (b.mean - a.mean) * a.weight / (a.weight + b.weight);

        if (x < a.mean + right) {
            double value = (r + a.weight * interpolate(x, a.mean - left, a.mean + right)) / T->total_weight;
            return value > 0.0 ? value : 0.0;
        }

        r += a.weight;
        left = b.mean - (a.mean + right);
    }

    a = b;
    right = T->max - a.mean;
    if (x < a.mean + right)
        return (r + a.weight * interpolate(x, a.mean - left, a.mean + right)) / T->total_weight;
    return 1;
}


double tdigest_quantile(tdigest_t *T, double q) {
    tdigest_flush(T);

    if (T->total_weight == 0)
        return NAN;
    if (q <= 0)
        return T->min;
    if (q >= 1)
        return T->max;
    if (T->last == 0)
        return T->centroids[0].mean;

    double index = q * T->total_weight;
    double weight_so_far = 0;
    double left, right = T->min;
    centroid_t a, b = T->centroids[0];

    for (int i = 1; i < T->last + 1; i++) {
        a = b;
        b = T->centroids[i];

        left = right;
        right = (b.weight * a.mean + a.weight * b.mean) / (a.weight + b.weight);

        if (index < weight_so_far + a.weight) {
            double p = (index - weight_so_far) / a.weight;
            return left * (1 - p) + right * p;
        }
        weight_so_far += a.weight;
    }

    left = right;
    right = T->max;
    a = b;

    if (index < weight_so_far + a.weight) {
        double p = (index - weight_so_far) / a.weight;
        return left * (1 - p) + right * p;
    } else {
        return T->max;
    }
}


void tdigest_merge(tdigest_t *T, tdigest_t *other) {
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


npy_intp tdigest_update_ndarray(tdigest_t *T, PyArrayObject *x, PyArrayObject *w) {
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    PyArrayObject *op[2];
    npy_uint32 flags;
    npy_uint32 op_flags[2];
    PyArray_Descr *dtypes[2] = {NULL};

    npy_intp *innersizeptr, *strideptr;
    char **dataptr;

    npy_intp ret = -1;

    op[0] = x;
    op[1] = w;
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;

    dtypes[0] = PyArray_DescrFromType(NPY_FLOAT64);
    if (dtypes[0] == NULL) {
        goto finish;
    }
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[1] == NULL) {
        goto finish;
    }

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
    if (!NpyIter_IterationNeedsAPI(iter)) {
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
    }

    do {
        char *data_x = dataptr[0];
        char *data_w = dataptr[1];
        npy_intp stride_x = strideptr[0];
        npy_intp stride_w = strideptr[1];
        npy_intp count = *innersizeptr;

        while (count--) {
            tdigest_add(T, *(npy_float64 *)data_x,
                           *(npy_int64 *)data_w);

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
