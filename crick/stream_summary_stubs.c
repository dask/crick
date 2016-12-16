#include <stdlib.h>
#include <python.h>

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



// Helpers for viewing floats as ints, and ints as floats
// Used for the float64 dtype
static inline khint64_t asint64(double key) {
  return *(khint64_t *)(&key);
}

static inline double asfloat64(khint64_t key) {
  return *(double *)(&key);
}


/* Macros for creating summary implementations for different dtypes
 * Note that these expect the KHASH_INIT for the equivalent dtype to
 * already be called. */

#define INIT_SUMMARY_TYPES(name, item_t)    \
typedef struct counter_##name##_s {         \
    size_t next;                            \
    size_t prev;                            \
    item_t item;                            \
    long count;                             \
    long error;                             \
} counter_##name##_t;                       \
                                            \
typedef struct summary_##name##_s {         \
    size_t capacity;                        \
    size_t n_counters;                      \
    size_t head;                            \
    counter_##name##_t *counters;           \
    khash_t(name) *hashmap;                 \
} summary_##name##_t;


#define INIT_SUMMARY_METHODS(name, item_t, refcount)                                    \
static inline summary_##name##_t *summary_##name##_new(int capacity) {                  \
    summary_##name##_t *T = (summary_##name##_t *)malloc(sizeof(*T));                   \
    if (T == NULL)                                                                      \
        goto fail;                                                                      \
                                                                                        \
    T->counters = (counter_##name##_t *)malloc(capacity * sizeof(counter_##name##_t));  \
    if (T->counters == NULL)                                                            \
        goto fail;                                                                      \
                                                                                        \
    T->capacity = capacity;                                                             \
    T->n_counters = 0;                                                                  \
    T->head = NIL;                                                                      \
    T->hashmap = kh_init(name);                                                         \
                                                                                        \
    return T;                                                                           \
                                                                                        \
fail:                                                                                   \
    if (T->counters != NULL) free(T->counters);                                         \
    if (T != NULL) free(T);                                                             \
    return NULL;                                                                        \
}                                                                                       \
                                                                                        \
                                                                                        \
static inline void summary_##name##_free(summary_##name##_t *T) {                       \
    if (refcount) {                                                                     \
        khiter_t iter;                                                                  \
        for (iter = kh_begin(T->hashmap); iter != kh_end(T->hashmap); ++iter) {         \
            if (kh_exist(T->hashmap, iter))                                             \
                Py_DECREF(kh_key(T->hashmap, iter));                                    \
        }                                                                               \
    }                                                                                   \
    kh_destroy(name, T->hashmap);                                                       \
    free(T->counters);                                                                  \
    free(T);                                                                            \
}                                                                                       \
                                                                                        \
                                                                                        \
static inline void summary_##name##_counter_insert(summary_##name##_t *T,               \
                                                   size_t c, size_t prev) {             \
    long count = T->counters[c].count;                                                  \
    size_t tail = T->counters[T->head].prev;                                            \
    while(1) {                                                                          \
        if (T->counters[prev].count >= count)                                           \
            break;                                                                      \
        prev = T->counters[prev].prev;                                                  \
        if (prev == tail) {                                                             \
            T->head = c;                                                                \
            break;                                                                      \
        }                                                                               \
    }                                                                                   \
    T->counters[c].next = T->counters[prev].next;                                       \
    T->counters[c].prev = prev;                                                         \
    T->counters[T->counters[prev].next].prev = c;                                       \
    T->counters[prev].next = c;                                                         \
}                                                                                       \
                                                                                        \
                                                                                        \
static inline size_t summary_##name##_counter_new(summary_##name##_t *T,                \
                                                  item_t item, int count) {             \
    if (refcount)                                                                       \
        Py_INCREF(item);                                                                \
    size_t c = T->n_counters;                                                           \
    T->n_counters++;                                                                    \
                                                                                        \
    T->counters[c].item = item;                                                         \
    T->counters[c].count = count;                                                       \
    T->counters[c].error = 0;                                                           \
                                                                                        \
    if (T->head == NIL) {                                                               \
        T->head = c;                                                                    \
        T->counters[c].prev = c;                                                        \
        T->counters[c].next = c;                                                        \
    }                                                                                   \
    else {                                                                              \
        size_t tail = T->counters[T->head].prev;                                        \
        summary_##name##_counter_insert(T, c, tail);                                    \
    }                                                                                   \
    return c;                                                                           \
}                                                                                       \
                                                                                        \
                                                                                        \
static inline size_t summary_##name##_replace_min(summary_##name##_t *T,                \
                                                  item_t item, long count) {            \
    size_t tail = T->counters[T->head].prev;                                            \
                                                                                        \
    /* Remove the min item from the hashmap */                                          \
    item_t min_item = T->counters[tail].item;                                           \
    if (refcount)                                                                       \
        Py_DECREF(min_item);                                                            \
    khiter_t iter = kh_get(name, T->hashmap, min_item);                                 \
    kh_del(name, T->hashmap, iter);                                                     \
                                                                                        \
    T->counters[tail].item = item;                                                      \
    T->counters[tail].error = T->counters[tail].count;                                  \
    T->counters[tail].count++;                                                          \
    return tail;                                                                        \
}                                                                                       \
                                                                                        \
                                                                                        \
static inline void summary_##name##_rebalance(summary_##name##_t *T,                    \
                                              size_t counter) {                         \
    if (T->head == counter) {                                                           \
        /* Counts can only increase */                                                  \
        return;                                                                         \
    }                                                                                   \
    size_t prev = T->counters[counter].prev;                                            \
                                                                                        \
    if (T->counters[prev].count >= T->counters[counter].count)                          \
        return;                                                                         \
                                                                                        \
    /* Counter needs to be moved. Remove then insert. */                                \
    T->counters[T->counters[counter].next].prev = prev;                                 \
    T->counters[prev].next = T->counters[counter].next;                                 \
    summary_##name##_counter_insert(T, counter, prev);                                  \
}                                                                                       \
                                                                                        \
                                                                                        \
static inline int summary_##name##_add(summary_##name##_t *T,                           \
                                       item_t item, int count) {                        \
    int absent;                                                                         \
    size_t counter;                                                                     \
                                                                                        \
    /* Get the pointer to the bucket */                                                 \
    khiter_t iter = kh_put(name, T->hashmap, item, &absent);                            \
    /* If the key is an object, we need to check for hash failure */                    \
    if (refcount && PyErr_Occurred())                                                   \
        return -1;                                                                      \
    if (absent > 0) {                                                                   \
        /* New item */                                                                  \
        if (T->n_counters == T->capacity) {                                             \
            /* we're full, replace the min counter */                                   \
            counter = summary_##name##_replace_min(T, item, count);                     \
            summary_##name##_rebalance(T, counter);                                     \
        } else {                                                                        \
            /* Not full, allocate a new counter */                                      \
            counter = summary_##name##_counter_new(T, item, count);                     \
        }                                                                               \
        kh_val(T->hashmap, iter) = counter;                                             \
    }                                                                                   \
    else if (absent == 0) {                                                             \
        /* The counter exists, just update it */                                        \
        counter = kh_val(T->hashmap, iter);                                             \
        T->counters[counter].count += count;                                            \
        summary_##name##_rebalance(T, counter);                                         \
    }                                                                                   \
    else {                                                                              \
        PyErr_NoMemory();                                                               \
        return -1;                                                                      \
    }                                                                                   \
    return 1;                                                                           \
}                                                                                       \


#define INIT_SUMMARY(name, item_t, refcount)        \
    INIT_SUMMARY_TYPES(name, item_t)                \
    INIT_SUMMARY_METHODS(name, item_t, refcount)


/* Define summary for int64 dtype */
INIT_SUMMARY(int64, khint64_t, 0)

/* Define summary for object dtype */
INIT_SUMMARY(object, PyObject*, 1)
