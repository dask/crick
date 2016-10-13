#ifndef T_DIGEST_H
#define T_DIGEST_H

typedef struct tdigest tdigest_t;

tdigest_t *tdigest_new(double compression);

void tdigest_free(tdigest_t *T);

void tdigest_add(tdigest_t *T, double x, int w);

void tdigest_flush(tdigest_t *T);

double tdigest_quantile(tdigest_t *T, double q);

double tdigest_min(tdigest_t *T);

double tdigest_max(tdigest_t *T);

int tdigest_count(tdigest_t *T);

#endif
