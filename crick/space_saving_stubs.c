#include <stdlib.h>
#include "khash.h"
#include <signal.h>


// The value indicating missingess
#define NIL -1


typedef struct counter {
    // Organizational
    size_t next;
    size_t prev;
    // Data
    long item;
    long count;
    long error;
} counter_t;


// Define the hashtable
KHASH_MAP_INIT_INT64(int64, size_t)


typedef struct summary {
    // max number of counters allowed
    size_t capacity;
    // Current number of counters
    size_t n_counters;
    // The head of the linked list
    size_t head;
    // The array of counters
    counter_t *counters;
    // Hashmap from item -> counter*
    khash_t(int64) *hashmap;
} summary_t;


summary_t *summary_new(int capacity) {
    // Allocate the summary structure
    summary_t *T = (summary_t *)malloc(sizeof(*T));
    if (T == NULL)
        return NULL;

    T->counters = (counter_t *)malloc(capacity * sizeof(counter_t));
    if (T->counters == NULL) {
        free(T);
        return NULL;
    }

    T->capacity = capacity;
    T->n_counters = 0;
    T->head = NIL;
    T->hashmap = kh_init(int64);

    return T;
}


void summary_free(summary_t *T) {
    kh_destroy(int64, T->hashmap);
    free(T->counters);
    free(T);
}


void summary_counter_insert(summary_t *T, size_t c, size_t prev) {
    long count = T->counters[c].count;
    size_t tail = T->counters[T->head].prev;
    while(1) {
        if (T->counters[prev].count >= count)
            break;
        prev = T->counters[prev].prev;
        if (prev == tail) {
            if (T->head != tail)
                T->head = c;
            break;
        }
    }
    T->counters[c].next = T->counters[prev].next;
    T->counters[c].prev = prev;
    T->counters[T->counters[prev].next].prev = c;
    T->counters[prev].next = c;
}


size_t summary_counter_new(summary_t *T, long item, int count) {
    size_t c = T->n_counters;
    T->n_counters++;

    T->counters[c].item = item;
    T->counters[c].count = count;
    T->counters[c].error = 0;

    if (T->head == NIL) {
        T->head = c;
        T->counters[c].prev = c;
        T->counters[c].next = c;
    }
    else {
        size_t tail = T->counters[T->head].prev;
        summary_counter_insert(T, c, tail);
    }
    return c;
}


size_t summary_replace_min(summary_t *T, long item, int count) {
    size_t tail = T->counters[T->head].prev;

    // Remove the min item from the hashmap
    long min_item = T->counters[tail].item;
    khiter_t iter = kh_get(int64, T->hashmap, min_item);
    kh_del(int64, T->hashmap, iter);

    T->counters[tail].item = item;
    T->counters[tail].error = T->counters[tail].count;
    T->counters[tail].count++;
    return tail;
}


void summary_rebalance(summary_t *T, size_t counter) {
    if (T->head == counter) {
        // Counts can only increase
        return;
    }
    size_t prev = T->counters[counter].prev;

    if (T->counters[prev].count >= T->counters[counter].count)
        return;

    // Counter needs to be moved. Remove then insert.
    T->counters[T->counters[counter].next].prev = prev;
    T->counters[prev].next = T->counters[counter].next;
    summary_counter_insert(T, counter, prev);
}


int summary_add(summary_t *T, long item, int count) {
    int absent;
    size_t counter;

    // Get the pointer to the bucket
    khiter_t iter = kh_put(int64, T->hashmap, item, &absent);
    if (absent > 0) {
        // New item
        if (T->n_counters == T->capacity) {
            // we're full, replace the min counter
            counter = summary_replace_min(T, item, count);
            summary_rebalance(T, counter);
        } else {
            // Not full, allocate a new counter
            counter = summary_counter_new(T, item, count);
        }
        kh_val(T->hashmap, iter) = counter;
    }
    else if (absent == 0) {
        // The counter exists, just update it
        counter = kh_val(T->hashmap, iter);
        T->counters[counter].count += count;
        summary_rebalance(T, counter);
    }
    else {
        return -1;
    }
    return 1;
}
