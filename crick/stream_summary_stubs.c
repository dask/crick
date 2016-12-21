#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "khash.h"


// The value indicating missingess
#define NIL -1


/* Define the hashtable structs
 * These map the items to counters for each dtype */

// int64 dtypes
KHASH_MAP_INIT_INT64(int64, size_t)

// object dtype
static inline int pyobject_cmp(PyObject* a, PyObject* b) {
	int result = PyObject_RichCompareBool(a, b, Py_EQ);
	if (result < 0) {
		PyErr_Clear();
		return 0;
	}
	return result;
}

KHASH_INIT(object, PyObject*, size_t, 1, PyObject_Hash, pyobject_cmp)


/* Generic Summary struct, used for casting to access the consistent fields */
#define SUMMARY_HEAD   \
    size_t capacity;   \
    size_t size;       \
    size_t head;

typedef struct {
    SUMMARY_HEAD
} summary_t;

/* Macros for creating summary implementations for different dtypes
 * Note that these expect the KHASH_INIT for the equivalent dtype to
 * already be called. */

#define INIT_SUMMARY_TYPES(name, item_t)    \
typedef struct {                            \
    item_t item;                            \
    npy_int64 count;                        \
    npy_int64 error;                        \
} counter_##name##_t;                       \
                                            \
typedef struct {                            \
    size_t next;                            \
    size_t prev;                            \
    counter_##name##_t counter;             \
} node_##name##_t;                          \
                                            \
typedef struct summary_##name##_s {         \
    SUMMARY_HEAD                            \
    node_##name##_t *list;                  \
    khash_t(name) *hashmap;                 \
} summary_##name##_t;


#define INIT_SUMMARY_METHODS(name, item_t, refcount)                            \
static inline summary_##name##_t *summary_##name##_new(int capacity) {          \
    summary_##name##_t *T = (summary_##name##_t *)malloc(sizeof(*T));           \
    if (T == NULL)                                                              \
        goto fail;                                                              \
                                                                                \
    T->list = (node_##name##_t *)malloc(capacity * sizeof(node_##name##_t));    \
    if (T->list == NULL)                                                        \
        goto fail;                                                              \
                                                                                \
    T->capacity = capacity;                                                     \
    T->size = 0;                                                                \
    T->head = NIL;                                                              \
    T->hashmap = kh_init(name);                                                 \
                                                                                \
    return T;                                                                   \
                                                                                \
fail:                                                                           \
    if (T->list != NULL) free(T->list);                                         \
    if (T != NULL) free(T);                                                     \
    return NULL;                                                                \
}                                                                               \
                                                                                \
                                                                                \
static inline void summary_##name##_free(summary_##name##_t *T) {               \
    if (refcount) {                                                             \
        khiter_t iter;                                                          \
        for (iter = kh_begin(T->hashmap); iter != kh_end(T->hashmap); ++iter) { \
            if (kh_exist(T->hashmap, iter))                                     \
                Py_DECREF(kh_key(T->hashmap, iter));                            \
        }                                                                       \
    }                                                                           \
    kh_destroy(name, T->hashmap);                                               \
    free(T->list);                                                              \
    free(T);                                                                    \
}                                                                               \
                                                                                \
                                                                                \
static inline int counter_##name##_ge(counter_##name##_t c1,                    \
                                      counter_##name##_t c2,                    \
                                      npy_int64 offset) {                       \
    npy_int64 count = c2.count + offset;                                        \
    npy_int64 error = c2.error + offset;                                        \
    return (c1.count > count || (c1.count == count && c1.error <= error));      \
}                                                                               \
                                                                                \
                                                                                \
static inline void summary_##name##_counter_insert(summary_##name##_t *T,       \
                                                   size_t c, size_t prev) {     \
    size_t tail = T->list[T->head].prev;                                        \
    while(1) {                                                                  \
        if (counter_##name##_ge(T->list[prev].counter,                          \
                                T->list[c].counter, 0))                         \
            break;                                                              \
        prev = T->list[prev].prev;                                              \
        if (prev == tail) {                                                     \
            T->head = c;                                                        \
            break;                                                              \
        }                                                                       \
    }                                                                           \
    T->list[c].next = T->list[prev].next;                                       \
    T->list[c].prev = prev;                                                     \
    T->list[T->list[prev].next].prev = c;                                       \
    T->list[prev].next = c;                                                     \
}                                                                               \
                                                                                \
                                                                                \
static inline size_t summary_##name##_counter_new(summary_##name##_t *T,        \
                                                  item_t item, npy_int64 count, \
                                                  npy_int64 error) {            \
    if (refcount)                                                               \
        Py_INCREF(item);                                                        \
    size_t c = T->size;                                                         \
    T->size++;                                                                  \
                                                                                \
    T->list[c].counter.item = item;                                             \
    T->list[c].counter.count = count;                                           \
    T->list[c].counter.error = error;                                           \
                                                                                \
    if (T->head == NIL) {                                                       \
        T->head = c;                                                            \
        T->list[c].prev = c;                                                    \
        T->list[c].next = c;                                                    \
    }                                                                           \
    else {                                                                      \
        size_t tail = T->list[T->head].prev;                                    \
        summary_##name##_counter_insert(T, c, tail);                            \
    }                                                                           \
    return c;                                                                   \
}                                                                               \
                                                                                \
                                                                                \
static inline void summary_##name##_rebalance(summary_##name##_t *T,            \
                                              size_t index) {                   \
    if (T->head == index) {                                                     \
        /* Counts can only increase */                                          \
        return;                                                                 \
    }                                                                           \
    size_t prev = T->list[index].prev;                                          \
                                                                                \
    if (counter_##name##_ge(T->list[prev].counter,                              \
                            T->list[index].counter, 0))                         \
        return;                                                                 \
                                                                                \
    /* Counter needs to be moved. Remove then insert. */                        \
    T->list[T->list[index].next].prev = prev;                                   \
    T->list[prev].next = T->list[index].next;                                   \
    summary_##name##_counter_insert(T, index, prev);                            \
}                                                                               \
                                                                                \
                                                                                \
static inline int summary_##name##_swap(summary_##name##_t *T,                  \
                                        size_t index, item_t item,              \
                                        npy_int64 count, npy_int64 error) {     \
    /* Remove the old item from the hashmap */                                  \
    item_t old_item = T->list[index].counter.item;                              \
    khiter_t iter = kh_get(name, T->hashmap, old_item);                         \
    if (iter == kh_end(T->hashmap)) return -1;                                  \
    if (refcount && PyErr_Occurred()) return -1;                                \
    kh_del(name, T->hashmap, iter);                                             \
    if (refcount) {                                                             \
        Py_DECREF(old_item);                                                    \
        Py_INCREF(item);                                                        \
    }                                                                           \
                                                                                \
    T->list[index].counter.item = item;                                         \
    T->list[index].counter.error = error;                                       \
    T->list[index].counter.count = count;                                       \
    summary_##name##_rebalance(T, index);                                       \
    return 0;                                                                   \
}                                                                               \
                                                                                \
                                                                                \
static inline int summary_##name##_add(summary_##name##_t *T,                   \
                                       item_t item, npy_int64 count) {          \
    int absent;                                                                 \
    size_t index;                                                               \
                                                                                \
    /* Get the pointer to the bucket */                                         \
    khiter_t iter = kh_put(name, T->hashmap, item, &absent);                    \
    /* If the key is an object, we need to check for hash failure */            \
    if (refcount && PyErr_Occurred())                                           \
        return -1;                                                              \
    if (absent > 0) {                                                           \
        /* New item */                                                          \
        if (T->size == T->capacity) {                                           \
            /* we're full, replace the min counter */                           \
            index = T->list[T->head].prev;                                      \
            int err = summary_##name##_swap(T, index, item,                     \
                                            T->list[index].counter.count + 1,   \
                                            T->list[index].counter.count);      \
            if (err < 0) return -1;                                             \
        } else {                                                                \
            /* Not full, allocate a new counter */                              \
            index = summary_##name##_counter_new(T, item, count, 0);            \
        }                                                                       \
        kh_val(T->hashmap, iter) = index;                                       \
    }                                                                           \
    else if (absent == 0) {                                                     \
        /* The counter exists, just update it */                                \
        index = kh_val(T->hashmap, iter);                                       \
        T->list[index].counter.count += count;                                  \
        summary_##name##_rebalance(T, index);                                   \
    }                                                                           \
    else {                                                                      \
        PyErr_NoMemory();                                                       \
        return -1;                                                              \
    }                                                                           \
    return 1;                                                                   \
}                                                                               \
                                                                                \
static int summary_##name##_set_state(summary_##name##_t *T,                    \
                                      counter_##name##_t *counters,             \
                                      size_t size) {                            \
    int absent;                                                                 \
    size_t index;                                                               \
    if (size > T->capacity) {                                                   \
        PyErr_SetString(PyExc_ValueError,                                       \
                        "deserialization failed, size > capacity");             \
        return -1;                                                              \
    }                                                                           \
    for (int i=0; i < size; i++) {                                              \
        counter_##name##_t c = counters[i];                                     \
        /* Get the pointer to the bucket */                                     \
        khiter_t iter = kh_put(name, T->hashmap, c.item, &absent);              \
        /* If the key is an object, we need to check for hash failure */        \
        if (refcount && PyErr_Occurred())                                       \
            return -1;                                                          \
        if (absent > 0) {                                                       \
            index = summary_##name##_counter_new(T, c.item, c.count, c.error);  \
            kh_val(T->hashmap, iter) = index;                                   \
        }                                                                       \
        else if (absent == 0) {                                                 \
            PyErr_SetString(PyExc_ValueError,                                   \
                            "deserialization failed, duplicate items found");   \
            return -1;                                                          \
        }                                                                       \
        else {                                                                  \
            PyErr_NoMemory();                                                   \
            return -1;                                                          \
        }                                                                       \
    }                                                                           \
    return 1;                                                                   \
}                                                                               \
                                                                                \
                                                                                \
static int summary_##name##_merge(summary_##name##_t *T1,                       \
                                  summary_##name##_t *T2) {                     \
    /* Nothing to do */                                                         \
    if (T2->size == 0) return 0;                                                \
                                                                                \
    /* Get the minimum counts from each */                                      \
    npy_int64 m1, m2;                                                           \
    if (T1->size < T1->capacity)                                                \
        m1 = 0;                                                                 \
    else                                                                        \
        m1 = T1->list[T1->list[T1->head].prev].counter.count;                   \
                                                                                \
    if (T2->size < T2->capacity)                                                \
        m2 = 0;                                                                 \
    else                                                                        \
        m2 = T2->list[T2->list[T2->head].prev].counter.count;                   \
                                                                                \
    /* Iterate through T1, updating it inplace */                               \
    size_t i1, i2;                                                              \
    for (i1 = 0; i1 < T1->size; i1++) {                                         \
        khiter_t iter = kh_get(name, T2->hashmap, T1->list[i1].counter.item);   \
                                                                                \
        if (refcount && PyErr_Occurred()) return -1;                            \
                                                                                \
        if (iter != kh_end(T2->hashmap)) {                                      \
            /* item is in both T1 and T2 */                                     \
            i2 = kh_val(T2->hashmap, iter);                                     \
            T1->list[i1].counter.count += T2->list[i2].counter.count;           \
            T1->list[i1].counter.error += T2->list[i2].counter.error;           \
        }                                                                       \
        else {                                                                  \
            /* item is in only T1 */                                            \
            T1->list[i1].counter.count += m2;                                   \
            T1->list[i1].counter.error += m2;                                   \
        }                                                                       \
        summary_##name##_rebalance(T1, i1);                                     \
    }                                                                           \
                                                                                \
    /* Iterate through T2, adding in any missing items */                       \
    i2 = T2->head;                                                              \
    for (int i = 0; i < T2->size; i++) {                                        \
        int absent;                                                             \
        counter_##name##_t c2 = T2->list[i2].counter;                           \
        khiter_t iter = kh_put(name, T1->hashmap, c2.item, &absent);            \
        if (refcount && PyErr_Occurred()) return -1;                            \
                                                                                \
        if (absent > 0) {                                                       \
            /* Item isn't in T1, maybe add it */                                \
            if (T1->size == T1->capacity) {                                     \
                i1 = T1->list[T1->head].prev;                                   \
                                                                                \
                /* If all counts in T1 are >= T2 + m1 then we're done here */   \
                if (counter_##name##_ge(T1->list[i1].counter, c2, m1)) break;   \
                                                                                \
                int err = summary_##name##_swap(T1, i1, c2.item, c2.count + m1, \
                                                c2.error + m1);                 \
                if (err < 0) return -1;                                         \
            }                                                                   \
            else {                                                              \
                i1 = summary_##name##_counter_new(T1, c2.item, c2.count + m1,   \
                                                  c2.error + m1);               \
            }                                                                   \
            kh_val(T1->hashmap, iter) = i1;                                     \
        }                                                                       \
        else if (absent < 0) {                                                  \
            PyErr_NoMemory();                                                   \
            return -1;                                                          \
        }                                                                       \
        i2 = T2->list[i2].next;                                                 \
    }                                                                           \
    return 0;                                                                   \
}


#define INIT_SUMMARY_NUMPY(name, item_t, DTYPE)                                 \
static int summary_##name##_update_ndarray(summary_##name##_t *T,               \
                                                PyArrayObject *x,               \
                                                PyArrayObject *w) {             \
    NpyIter *iter = NULL;                                                       \
    NpyIter_IterNextFunc *iternext;                                             \
    PyArrayObject *op[2];                                                       \
    npy_uint32 flags;                                                           \
    npy_uint32 op_flags[2];                                                     \
    PyArray_Descr *dtypes[2] = {NULL};                                          \
                                                                                \
    npy_intp *innersizeptr, *strideptr;                                         \
    char **dataptr;                                                             \
                                                                                \
    npy_intp ret = -1;                                                          \
                                                                                \
    /* Handle zero-sized arrays specially */                                    \
    if (PyArray_SIZE(x) == 0 || PyArray_SIZE(w) == 0) {                         \
        return 0;                                                               \
    }                                                                           \
                                                                                \
    op[0] = x;                                                                  \
    op[1] = w;                                                                  \
    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_REFS_OK;      \
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;                         \
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;                         \
                                                                                \
    dtypes[0] = PyArray_DescrFromType(DTYPE);                                   \
    if (dtypes[0] == NULL) {                                                    \
        goto finish;                                                            \
    }                                                                           \
    Py_INCREF(dtypes[0]);                                                       \
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);                               \
    if (dtypes[1] == NULL) {                                                    \
        goto finish;                                                            \
    }                                                                           \
    Py_INCREF(dtypes[1]);                                                       \
                                                                                \
    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,      \
                            op_flags, dtypes);                                  \
    if (iter == NULL) {                                                         \
        goto finish;                                                            \
    }                                                                           \
                                                                                \
    iternext = NpyIter_GetIterNext(iter, NULL);                                 \
    if (iternext == NULL) {                                                     \
        goto finish;                                                            \
    }                                                                           \
    dataptr = NpyIter_GetDataPtrArray(iter);                                    \
    strideptr = NpyIter_GetInnerStrideArray(iter);                              \
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);                           \
                                                                                \
    NPY_BEGIN_THREADS_DEF;                                                      \
    if (!NpyIter_IterationNeedsAPI(iter))                                       \
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));               \
                                                                                \
    do {                                                                        \
        char *data_x = dataptr[0];                                              \
        char *data_w = dataptr[1];                                              \
        npy_intp stride_x = strideptr[0];                                       \
        npy_intp stride_w = strideptr[1];                                       \
        npy_intp count = *innersizeptr;                                         \
                                                                                \
        while (count--) {                                                       \
            summary_##name##_add(T, *(item_t *)data_x,                          \
                                 *(npy_int64 *)data_w);                         \
                                                                                \
            data_x += stride_x;                                                 \
            data_w += stride_w;                                                 \
        }                                                                       \
    } while (iternext(iter));                                                   \
                                                                                \
    NPY_END_THREADS;                                                            \
                                                                                \
    ret = 0;                                                                    \
                                                                                \
finish:                                                                         \
    Py_XDECREF(dtypes[0]);                                                      \
    Py_XDECREF(dtypes[1]);                                                      \
    if (iter != NULL) {                                                         \
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {                          \
            return -1;                                                          \
        }                                                                       \
    }                                                                           \
    return ret;                                                                 \
}                                                                               \


#define INIT_SUMMARY(name, item_t, refcount, DTYPE) \
    INIT_SUMMARY_TYPES(name, item_t)                \
    INIT_SUMMARY_METHODS(name, item_t, refcount)    \
    INIT_SUMMARY_NUMPY(name, item_t, DTYPE)



INIT_SUMMARY(int64, npy_int64, 0, NPY_INT64)
INIT_SUMMARY(object, PyObject*, 1, NPY_OBJECT)

/* float64 definitions are just a thin wrapper around int64, viewing the bytes
 * as int64. Define a small helper to view float64 as int64: */

static inline npy_int64 asint64(npy_float64 key) {
  return *(npy_int64 *)(&key);
}
