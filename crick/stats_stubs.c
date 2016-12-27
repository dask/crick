#include <stdlib.h>
#include <float.h>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "common.h"


typedef struct {
    npy_int64 count;
    npy_float64 sum;
    npy_float64 min;
    npy_float64 max;
    npy_float64 m2;
    npy_float64 m3;
    npy_float64 m4;
} stats_t;


CRICK_INLINE stats_t *stats_new() {
    stats_t *T = (stats_t *)malloc(sizeof(*T));
    if (T == NULL) return NULL;

    T->count = 0;
    T->sum = 0;
    T->min = DBL_MAX;
    T->max = -DBL_MIN;
    T->m2 = 0;
    T->m3 = 0;
    T->m4 = 0;

    return T;
}


CRICK_INLINE void stats_free(stats_t *T) {
    free(T);
}


CRICK_INLINE void stats_do_update(stats_t *T, npy_int64 n2, npy_float64 sum2,
                                   npy_float64 min_2, npy_float64 max_2,
                                   npy_float64 m4_2, npy_float64 m3_2,
                                   npy_float64 m2_2) {
    npy_float64 u2 = sum2/n2;
    npy_float64 delta = T->count ? u2 - (T->sum / T->count) : u2;
    npy_int64 n1 = T->count;
    npy_int64 n = n1 + n2;
    npy_int64 n1n2 = n1 * n2;
    npy_int64 n1_2 = n1 * n1;
    npy_int64 n2_2 = n2 * n2;
    npy_float64 delta_div_n = delta / n;
    npy_float64 delta_div_n_2 = (delta_div_n * delta_div_n);
    npy_float64 delta_div_n_3 = delta_div_n_2 * delta_div_n;
    if (min_2 < T->min)
        T->min = min_2;
    if (max_2 > T->max)
        T->max = max_2;
    T->m4 += (m4_2 +
              n1n2 * (n1_2 - n1n2 + n2_2) * delta * delta_div_n_3 +
              6 * (n1_2 * m2_2 + n2_2 * T->m2) * delta_div_n_2 +
              4 * (n1 * m3_2 - n2 * T->m3) * delta_div_n);
    T->m3 += (m3_2 +
              n1n2 * (n1 - n2) * delta * delta_div_n_2 +
              3 * (n1 * m2_2 - n2 * T->m2) * delta_div_n);
    T->m2 += m2_2 + n1n2 * delta * delta_div_n;
    T->count += n2;
    T->sum += sum2;
}


CRICK_INLINE void stats_merge(stats_t *T1, stats_t *T2) {
    if (T2->count == 0) return;
    stats_do_update(T1, T2->count, T2->sum, T2->min, T2->max,
                    T2->m4, T2->m3, T2->m2);
}

CRICK_INLINE void stats_add(stats_t *T, npy_float64 x, npy_int64 count) {
    if (npy_isnan(x)) return;
    stats_do_update(T, count, x, x, x, 0, 0, 0);
}


CRICK_INLINE double stats_mean(stats_t *T) {
    return T->count ? T->sum / T->count : NPY_NAN;
}


CRICK_INLINE double stats_var(stats_t *T, long ddof) {
    return T->count ? T->m2 / (T->count - ddof) : NPY_NAN;
}


CRICK_INLINE double stats_std(stats_t *T, long ddof) {
    return npy_sqrt(stats_var(T, ddof));
}


CRICK_INLINE double stats_skew(stats_t *T, int bias) {
    double n, m2, m3, skew;
    if (!T->count) return NPY_NAN;
    n = T->count;
    m2 = T->m2 / T->count;
    m3 = T->m3 / T->count;
    skew = m2 ? m3 / (npy_sqrt(m2) * m2) : 0;
    if (!bias && n > 2 && m2 > 0)
        return npy_sqrt((n - 1) * n) / (n - 2) * skew;
    return skew;
}


CRICK_INLINE double stats_kurt(stats_t *T, int fisher, int bias) {
    double n, m2, m4, kurt;
    if (!T->count) return NPY_NAN;
    n = T->count;
    m2 = T->m2 / T->count;
    m4 = T->m4 / T->count;
    kurt = m2 ? kurt = m4 / (m2 * m2) : 0;
    if (!bias && n > 3 && m2 > 0)
        kurt = ((n*n - 1)*kurt - 9*n + 15)/((n - 2)*(n - 3));
    return fisher ? kurt - 3 : kurt;
}


CRICK_INLINE npy_intp stats_update_ndarray(stats_t *T, PyArrayObject *x,
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
    Py_INCREF(dtypes[0]);
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[1] == NULL) {
        goto finish;
    }
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
            stats_add(T, *(npy_float64 *)data_x,
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
