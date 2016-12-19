#include <stdlib.h>
#include <Python.h>
#include <signal.h>

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
#define kh_python_hash_func(key) (PyObject_Hash(key))
#define kh_python_hash_equal(a, b) (pyobject_cmp(a, b))

typedef PyObject* kh_pyobject_t;


KHASH_INIT(object, kh_pyobject_t, size_t, 1, kh_python_hash_func,
           kh_python_hash_equal)


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

#define INIT_SUMMARY_TYPES(name, hash, item_t)\
typedef struct {                            \
    item_t item;                            \
    long count;                             \
    long error;                             \
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
    khash_t(hash) *hashmap;                 \
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
static inline void summary_##name##_counter_insert(summary_##name##_t *T,       \
                                                   size_t c, size_t prev) {     \
    long count = T->list[c].counter.count;                                      \
    size_t tail = T->list[T->head].prev;                                        \
    while(1) {                                                                  \
        if (T->list[prev].counter.count >= count)                               \
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
                                                  item_t item, long count,      \
                                                  long error) {                 \
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
static inline size_t summary_##name##_replace_min(summary_##name##_t *T,        \
                                                  item_t item, long count) {    \
    size_t tail = T->list[T->head].prev;                                        \
                                                                                \
    /* Remove the min item from the hashmap */                                  \
    item_t min_item = T->list[tail].counter.item;                               \
    if (refcount)                                                               \
        Py_DECREF(min_item);                                                    \
    khiter_t iter = kh_get(name, T->hashmap, min_item);                         \
    kh_del(name, T->hashmap, iter);                                             \
                                                                                \
    T->list[tail].counter.item = item;                                          \
    T->list[tail].counter.error = T->list[tail].counter.count;                  \
    T->list[tail].counter.count++;                                              \
    return tail;                                                                \
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
    if (T->list[prev].counter.count >= T->list[index].counter.count)            \
        return;                                                                 \
                                                                                \
    /* Counter needs to be moved. Remove then insert. */                        \
    T->list[T->list[index].next].prev = prev;                                   \
    T->list[prev].next = T->list[index].next;                                   \
    summary_##name##_counter_insert(T, index, prev);                            \
}                                                                               \
                                                                                \
                                                                                \
static inline int summary_##name##_add(summary_##name##_t *T,                   \
                                       item_t item, int count) {                \
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
            index = summary_##name##_replace_min(T, item, count);               \
            summary_##name##_rebalance(T, index);                               \
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
}


#define INIT_SUMMARY(name, item_t, refcount)        \
    INIT_SUMMARY_TYPES(name, name, item_t)          \
    INIT_SUMMARY_METHODS(name, item_t, refcount)


INIT_SUMMARY(int64, khint64_t, 0)
INIT_SUMMARY(object, PyObject*, 1)
INIT_SUMMARY_TYPES(float64, int64, double)

/* float64 definitions are just a thin wrapper around int64, viewing the bytes
 * as int64 */

static inline khint64_t asint64(double key) {
  return *(khint64_t *)(&key);
}

static inline summary_float64_t *summary_float64_new(int capacity) {
    return (summary_float64_t *)summary_int64_new(capacity);
}

static inline void summary_float64_free(summary_float64_t *T) {
    summary_int64_free((summary_int64_t *)T);
}

static inline int summary_float64_add(summary_float64_t *T, double item,
                                      long count) {
    return summary_int64_add((summary_int64_t *)T, asint64(item), count);
}

static inline int summary_float64_set_state(summary_float64_t *T,
                                            counter_float64_t *counters,
                                            size_t size) {
    return summary_int64_set_state((summary_int64_t *)T,
                                   (counter_int64_t *)counters, size);
}
