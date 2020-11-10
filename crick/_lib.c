#include <stddef.h>
#include <stdlib.h>
#include <float.h>

#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "common.h"

typedef struct {
    PyObject_HEAD
    npy_int64 count;
    npy_float64 sum;
    npy_float64 min;
    npy_float64 max;
    npy_float64 m2;
    npy_float64 m3;
    npy_float64 m4;
    PyObject* weakreflist;
} statsobject;


static PyTypeObject stats_type;


static PyObject *
stats_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    statsobject *self = (statsobject *)type->tp_alloc(type, 0);

    if (self == NULL)
        return NULL;

    self->weakreflist = NULL;
    self->count = 0;
    self->count = 0;
    self->sum = 0;
    self->min = DBL_MAX;
    self->max = -DBL_MIN;
    self->m2 = 0;
    self->m3 = 0;
    self->m4 = 0;

    return (PyObject *)self;
}


static void
stats_dealloc(statsobject *self)
{
    if (self->weakreflist != NULL)
        PyObject_ClearWeakRefs((PyObject *) self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject*
stats_setstate(statsobject *self, PyObject *state)
{
    if (!PyArg_ParseTuple(state, "Ldddddd", &self->count, &self->sum,
                          &self->min, &self->max, &self->m2, &self->m3,
                          &self->m4)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
stats_reduce(statsobject *self)
{
    return Py_BuildValue("(O()(Ldddddd))", Py_TYPE(self), self->count,
                         self->sum, self->min, self->max, self->m2,
                         self->m3, self->m4);
}


static PyObject *
stats_repr(statsobject *self)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("SummaryStats<count=%lld>", self->count);
#else
    return PyString_FromFormat("SummaryStats<count=%lld>", (PY_LONG_LONG)self->count);
#endif
}


static void
stats_do_update(statsobject *self, npy_int64 n2, npy_float64 sum2,
                npy_float64 min_2, npy_float64 max_2,
                npy_float64 m4_2, npy_float64 m3_2,
                npy_float64 m2_2)
{
    npy_float64 u2 = sum2/n2;
    npy_float64 delta = self->count ? u2 - (self->sum / self->count) : u2;
    npy_int64 n1 = self->count;
    npy_int64 n = n1 + n2;
    npy_int64 n1n2 = n1 * n2;
    npy_int64 n1_2 = n1 * n1;
    npy_int64 n2_2 = n2 * n2;
    npy_float64 delta_div_n = delta / n;
    npy_float64 delta_div_n_2 = delta_div_n * delta_div_n;
    npy_float64 delta_div_n_3 = delta_div_n_2 * delta_div_n;

    if (min_2 < self->min)
        self->min = min_2;

    if (max_2 > self->max)
        self->max = max_2;

    self->m4 += (m4_2 +
              n1n2 * (n1_2 - n1n2 + n2_2) * delta * delta_div_n_3 +
              6 * (n1_2 * m2_2 + n2_2 * self->m2) * delta_div_n_2 +
              4 * (n1 * m3_2 - n2 * self->m3) * delta_div_n);

    self->m3 += (m3_2 +
              n1n2 * (n1 - n2) * delta * delta_div_n_2 +
              3 * (n1 * m2_2 - n2 * self->m2) * delta_div_n);

    self->m2 += m2_2 + n1n2 * delta * delta_div_n;

    self->count += n2;

    self->sum += sum2;
}


static PyObject*
stats_add(statsobject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"x", "count", 0};
    double x;
    long count = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d|l", kwlist,
                                     &x, &count)) {
        return NULL;
    }
    if (count <= 0) {
        (void)PyErr_SetString(PyExc_ValueError, "count must be >= 1");
        return NULL;
    }

    if (!npy_isnan(x)) {
        stats_do_update(self, count, x, x, x, 0, 0, 0);
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(stats_add_doc,
"add(self, x, count=1)\n\
\n\
Add an element to the summary.\n\
\n\
Parameters\n\
----------\n\
x : double\n\
count : int, optional\n\
    The count of the item to add. Default is 1.");


static npy_intp
crick_is_nonnegative(PyArrayObject* x)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp* strideptr,* innersizeptr;
    npy_intp ret = 1;

    if (PyArray_SIZE(x) == 0) {
        return ret;
    }

    iter = NpyIter_New(x, NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP,
                       NPY_KEEPORDER, NPY_SAFE_CASTING, NULL);
    if (iter == NULL) {
        return -1;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    do {
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            if (*(npy_int64 *)data <= 0) {
                ret = 0;
                break;
            }
            data += stride;
        }
    } while(ret && iternext(iter));

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        return -1;
    }

    return ret;
}


static npy_intp
stats_update_ndarray(statsobject *self, PyArrayObject *x, PyArrayObject *w)
{
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
            npy_float64 xval = *(npy_float64 *)data_x;
            npy_int64 wval = *(npy_int64 *)data_w;
            if (!npy_isnan(xval)) {
                stats_do_update(self, wval, xval, xval, xval, 0, 0, 0);
            }

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


static PyObject*
stats_update(statsobject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"x", "count", 0};
    PyObject *py_x = NULL;
    PyObject *py_count = NULL;
    PyArrayObject *ar_count = NULL;
    PyArrayObject *ar_x = NULL;
    int decref_count = 0, failed=0, nonnegative=0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &py_x, &py_count)) {
        return NULL;
    }

    if (py_count == NULL) {
        decref_count = 1;
        py_count = PyLong_FromLong(1);
    }

    if (py_count != NULL) {
        ar_count = (PyArrayObject *)PyArray_FROM_OTF(py_count, NPY_INT64, 0);
        if (ar_count == NULL) {
            failed = 1;
            goto finish;
        }
    }

    nonnegative = crick_is_nonnegative(ar_count);
    if (nonnegative == -1) {
        failed = 1;
        goto finish;
    } else if (nonnegative == 0) {
        PyErr_SetString(PyExc_ValueError, "count must not be <= 0");
        failed = 1;
        goto finish;
    }

    ar_x = (PyArrayObject *)PyArray_FROM_OTF(py_x, NPY_FLOAT64, 0);
    if (ar_x == NULL) {
        failed = 1;
        goto finish;
    }

    failed = stats_update_ndarray(self, ar_x, ar_count);

finish:
    if (decref_count) {
        Py_DECREF(py_count);
    }
    Py_XDECREF(ar_count);
    Py_XDECREF(ar_x);

    if (failed)
        return NULL;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(stats_update_doc,
"update(self, x, count=1)\n\
\n\
Add many elements to the summary.\n\
\n\
Parameters\n\
----------\n\
x : array_like\n\
count : array_like, optional\n\
    The count (or counts) of the item to add. Default is 1.");


static PyObject*
stats_merge(statsobject *self, PyObject *args)
{
    Py_ssize_t len, i;

    len = (args != NULL) ? PyTuple_GET_SIZE(args) : 0;
    for (i = 0; i < len; i++) {
        PyObject *other = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(other, &stats_type)) {
            PyErr_SetString(PyExc_TypeError,
                            "All arguments to merge must be SummaryStats");
            return NULL;
        }
    }

    for (i = 0; i < len; i++) {
        statsobject *other = (statsobject *)PyTuple_GET_ITEM(args, i);
        if (other->count > 0) {
            stats_do_update(self, other->count, other->sum, other->min,
                            other->max, other->m4, other->m3, other->m2);
        }
    }

    Py_RETURN_NONE;
}

PyDoc_STRVAR(stats_merge_doc,
"merge(self, *args)\n\
\n\
Update this summary inplace with data from other summaries.\n\
\n\
Parameters\n\
----------\n\
args : SummaryStats\n\
    SummaryStats to merge into this one.");


static PyObject*
stats_count(statsobject *self)
{
    return PyLong_FromLongLong(self->count);
}

PyDoc_STRVAR(stats_count_doc,
"count(self)\n\
\n\
The number of elements in the summary.");


static PyObject*
stats_sum(statsobject *self)
{
    return PyFloat_FromDouble(self->sum);
}

PyDoc_STRVAR(stats_sum_doc,
"sum(self)\n\
\n\
The sum of all elements in the summary.");


static PyObject*
stats_min(statsobject *self)
{
    return PyFloat_FromDouble(self->count ? self->min : NPY_NAN);
}

PyDoc_STRVAR(stats_min_doc,
"min(self)\n\
\n\
The minimum element added to the summary. Returns ``NaN`` if empty.");


static PyObject*
stats_max(statsobject *self)
{
    return PyFloat_FromDouble(self->count ? self->max : NPY_NAN);
}

PyDoc_STRVAR(stats_max_doc,
"max(self)\n\
\n\
The maximum element added to the summary. Returns ``NaN`` if empty.");


static PyObject*
stats_mean(statsobject *self)
{
    return PyFloat_FromDouble(self->count ? self->sum / self->count : NPY_NAN);
}

PyDoc_STRVAR(stats_mean_doc,
"mean(self)\n\
\n\
The mean of all elements in the summary. Returns ``NaN`` if empty.");


static PyObject*
stats_var(statsobject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"ddof", 0};
    long long ddof = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|L", kwlist, &ddof)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->count
                              ? self->m2 / (self->count - ddof)
                              : NPY_NAN);
}

PyDoc_STRVAR(stats_var_doc,
"var(self, ddof=0)\n\
\n\
The variance of all elements in the summary. Returns ``NaN`` if empty.\n\
\n\
Parameters\n\
----------\n\
ddof : int, optional\n\
    'Delta Degrees of Freedom': the divisor used in the calculation is\n\
    ``N - ddof``, where ``N`` represents the number of elements. By\n\
    default ``ddof`` is zero.");


static PyObject*
stats_std(statsobject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"ddof", 0};
    long long ddof = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|L", kwlist, &ddof)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->count
                              ? npy_sqrt(self->m2 / (self->count - ddof))
                              : NPY_NAN);
}

PyDoc_STRVAR(stats_std_doc,
"std(self, ddof=0)\n\
\n\
The standard deviation of all elements in the summary. Returns ``NaN``\n\
if empty.\n\
\n\
Parameters\n\
----------\n\
ddof : int, optional\n\
    'Delta Degrees of Freedom': the divisor used in the calculation is\n\
    ``N - ddof``, where ``N`` represents the number of elements. By\n\
    default ``ddof`` is zero.");


static PyObject*
stats_skew(statsobject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"bias", 0};
    int bias = 1;
    double skew = NPY_NAN;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &bias)) {
        return NULL;
    }

    if (self->count) {
        double n, m2, m3;
        n = self->count;
        m2 = self->m2 / self->count;
        m3 = self->m3 / self->count;
        skew = m2 ? m3 / (npy_sqrt(m2) * m2) : 0;
        if (!bias && n > 2 && m2 > 0)
            skew *= npy_sqrt((n - 1) * n) / (n - 2);
    }

    return PyFloat_FromDouble(skew);
}

PyDoc_STRVAR(stats_skew_doc,
"skew(self, bias=True)\n\
\n\
The skewness of all elements in the summary. Returns ``NaN`` if empty.\n\
\n\
Parameters\n\
----------\n\
bias : bool, optional\n\
    If False, then the calculations are corrected for statistical bias.\n\
    Default is True.");


static PyObject*
stats_kurt(statsobject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"fisher", "bias", 0};
    int bias = 1;
    int fisher = 1;
    double kurt = NPY_NAN;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii", kwlist,
                                     &fisher, &bias)) {
        return NULL;
    }

    if (self->count) {
        double n, m2, m4;
        n = self->count;
        m2 = self->m2 / self->count;
        m4 = self->m4 / self->count;
        kurt = m2 ? m4 / (m2 * m2) : 0;
        if (!bias && n > 3 && m2 > 0)
            kurt = ((n*n - 1)*kurt - 9*n + 15)/((n - 2)*(n - 3));
        if (fisher)
            kurt -= 3;
    }

    return PyFloat_FromDouble(kurt);
}

PyDoc_STRVAR(stats_kurt_doc,
"kurt(self, fisher=True, bias=True)\n\
\n\
The kurtosis (Fisher or Pearson) of all elements in the summary.\n\
Returns ``NaN`` if empty.\n\
\n\
Parameters\n\
----------\n\
fisher : bool, optional\n\
    If True [default], Fisher's definition is used (normal ==> 0.0). If\n\
    False, Pearson's definition is used (normal ==> 3.0).\n\
bias : bool, optional\n\
    If False, then the calculations are corrected for statistical bias.\n\
    Default is True.");


static PyMethodDef stats_methods[] = {
    {"__reduce__", (PyCFunction)stats_reduce, METH_NOARGS , ""},
    {"__setstate__", (PyCFunction)stats_setstate, METH_O , ""},
    {"add", (PyCFunction)stats_add, METH_VARARGS | METH_KEYWORDS, stats_add_doc},
    {"update", (PyCFunction)stats_update, METH_VARARGS | METH_KEYWORDS, stats_update_doc},
    {"merge", (PyCFunction)stats_merge, METH_VARARGS, stats_merge_doc},
    {"count", (PyCFunction)stats_count, METH_NOARGS, stats_count_doc},
    {"sum", (PyCFunction)stats_sum, METH_NOARGS, stats_sum_doc},
    {"min", (PyCFunction)stats_min, METH_NOARGS, stats_min_doc},
    {"max", (PyCFunction)stats_max, METH_NOARGS, stats_max_doc},
    {"mean", (PyCFunction)stats_mean, METH_NOARGS, stats_mean_doc},
    {"var", (PyCFunction)stats_var, METH_VARARGS | METH_KEYWORDS, stats_var_doc},
    {"std", (PyCFunction)stats_std, METH_VARARGS | METH_KEYWORDS, stats_std_doc},
    {"skew", (PyCFunction)stats_skew, METH_VARARGS | METH_KEYWORDS, stats_skew_doc},
    {"kurt", (PyCFunction)stats_kurt, METH_VARARGS | METH_KEYWORDS, stats_kurt_doc},
    {NULL,              NULL}   /* sentinel */
};


PyDoc_STRVAR(stats_doc,
"SummaryStats()\n\
\n\
Computes exact summary statistics on a data stream.\n\
\n\
Keeps track of enough information to compute:\n\
\n\
- count\n\
- sum\n\
- minimum\n\
- maximum\n\
- mean\n\
- variance\n\
- standard deviation\n\
- skewness\n\
- kurtosis\n\
\n\
Notes\n\
-----\n\
The update formulas for variance, standard deviation, skewness, and\n\
kurtosis were taken from [1]_.\n\
\n\
References\n\
----------\n\
.. [1] Pebay, Philippe. 'Formulas for robust, one-pass parallel computation\n\
    of covariances and arbitrary-order statistical moments.' Sandia Report\n\
    SAND2008-6212, Sandia National Laboratories 94 (2008).");


static PyTypeObject stats_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "crick._lib.SummaryStats",                  /* tp_name */
    sizeof(statsobject),                        /* tp_basicsize */
    0,                         		            /* tp_itemsize */
    (destructor)stats_dealloc,                  /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    (reprfunc)stats_repr,                       /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash  */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    stats_doc,                                  /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    offsetof(statsobject, weakreflist),         /* tp_weaklistoffset*/
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    stats_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    stats_new,                                  /* tp_new */
};


#if PY_MAJOR_VERSION >= 3
static PyModuleDef lib_module = {
    PyModuleDef_HEAD_INIT,
    "crick._lib",
    NULL,
    -1,
    NULL, NULL, NULL, NULL, NULL
};
#define MODINIT_NAME PyInit__lib
#define MODINIT_RETURN(mod) return mod;
#else
#define MODINIT_NAME init_lib
#define MODINIT_RETURN(mod) return;
#endif


PyMODINIT_FUNC
MODINIT_NAME(void)
{
    PyObject* m;

    if (PyType_Ready(&stats_type) < 0)
        MODINIT_RETURN(NULL);

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&lib_module);
#else
    m = Py_InitModule3("_lib", NULL, "");
#endif
    if (m == NULL)
        MODINIT_RETURN(NULL);

    Py_INCREF(&stats_type);
    PyModule_AddObject(m, "SummaryStats", (PyObject *)&stats_type);

    import_array();

    MODINIT_RETURN(m);
}
