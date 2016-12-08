from __future__ import print_function, division, absolute_import

import pickle

import numpy as np
import pytest

from crick import TDigest


# -- Distributions for testing --
N = 100000

# A highly skewed gamma distribution
gamma = np.random.gamma(0.1, 0.1, N)

# Uniform from [0, 1]
uniform = np.random.uniform(0, 1, N)

# A very narrow normal distribution
normal = np.random.normal(0, 1e-5, N)

# Sequential, non-repeated, sorted
sequential = np.arange(N) * 1e-5

# Reversed of above
reverse_sequential = np.arange(N, 0, -1) * 1e-5

# A mix of a narrow normal and a uniform distribution
mixed = np.concatenate([np.random.normal(0, 1e-5, N//2),
                        np.random.uniform(-1, 1, N//2)])
np.random.shuffle(mixed)

# A step function, each value repeated 100 times
step = np.concatenate([np.arange(N/100)] * 100)
np.random.shuffle(step)

# Sorted of above
sorted_step = np.sort(step)


distributions = [gamma, uniform, normal, sequential, reverse_sequential,
                 mixed, step, sorted_step, ]


def quantiles_to_q(data, quant):
    """Convert quantiles of data to the q they represent"""
    N = len(data)
    quant2 = quant[:, None]
    return (2 * (data < quant2).sum(axis=1) +
            (data == quant2).sum(axis=1)) / (2 * N)


def q_to_x(data, q):
    """Convert q to x such that cdf(x) == q"""
    N = len(data)
    sorted_data = np.sort(data)
    x = np.empty_like(q)
    for i in range(len(q)):
        ix = N * q[i] - 0.5
        index = int(ix)
        p = ix - index
        x[i] = sorted_data[index] * (1 - p) + sorted_data[index + 1] * p
    return x


@pytest.mark.parametrize('data', distributions)
def test_distributions(data):
    t = TDigest()
    t.update(data)

    assert t.count() == len(data)
    assert t.min() == data.min()
    assert t.max() == data.max()

    # *Quantile
    q = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    est = np.array([t.quantile(i) for i in q])
    q_est = quantiles_to_q(data, est)
    np.testing.assert_allclose(q, q_est, atol=0.012, rtol=0)

    # *CDF
    x = q_to_x(data, q)
    q_est = np.array(map(t.cdf, x))
    np.testing.assert_allclose(q, q_est, atol=0.005)


def test_empty():
    t = TDigest()
    assert t.count() == 0
    assert np.isnan(t.min())
    assert np.isnan(t.max())
    assert np.isnan(t.quantile(0.5))
    assert np.isnan(t.cdf(0.5))


def test_single():
    t = TDigest()
    t.add(10)
    assert t.min() == 10
    assert t.max() == 10
    assert t.count() == 1

    assert t.quantile(0) == 10
    assert t.quantile(0.5) == 10
    assert t.quantile(1) == 10

    assert t.cdf(9) == 0
    assert t.cdf(10) == 0.5
    assert t.cdf(11) == 1


def test_nan():
    t = TDigest()
    data = gamma.copy()
    data[::10] = np.nan
    t.update(data)
    non_nan = data[~np.isnan(data)]
    assert t.count() == len(non_nan)
    assert t.min() == non_nan.min()
    assert t.max() == non_nan.max()

    t = TDigest()
    t.add(np.nan)
    assert t.count() == 0

    with pytest.raises(ValueError):
        t = TDigest()
        t.update(data, skipna=False)

    with pytest.raises(ValueError):
        t = TDigest()
        t.add(np.nan, skipna=False)


def test_weights():
    t = TDigest()
    t.add(1, 10)
    assert t.count() == 10

    x = np.arange(5)
    w = np.array([1, 2, 1, 2, 1])

    t = TDigest()
    t.update(x, 10)
    assert t.count() == len(x) * 10

    t = TDigest()
    t.update(x, w)
    assert t.count() == w.sum()


def test_serialize():
    not_empty = TDigest()
    not_empty.update(gamma)
    empty = TDigest()
    for t in [not_empty, empty]:
        t2 = pickle.loads(pickle.dumps(t))
        assert t.compression == t2.compression
        assert (t.centroids() == t2.centroids()).all()
        np.testing.assert_equal(t.min(), t2.min())
        np.testing.assert_equal(t.max(), t2.max())
        np.testing.assert_equal(t.count(), t2.count())


def test_merge():
    t = TDigest()
    t2 = TDigest()
    t3 = TDigest()
    a = np.random.uniform(0, 1, N)
    b = np.random.uniform(2, 3, N)
    data = np.concatenate([a, b])
    t2.update(a)
    t3.update(b)

    t2_centroids = t2.centroids()

    t.merge(t2, t3)
    assert t.min() == min(t2.min(), t3.min())
    assert t.max() == max(t2.max(), t3.max())
    assert t.count() == t2.count() + t3.count()
    # Check no mutation of args
    assert (t2.centroids() == t2_centroids).all()

    # *Quantile
    q = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    est = np.array([t.quantile(i) for i in q])
    q_est = quantiles_to_q(data, est)
    np.testing.assert_allclose(q, q_est, atol=0.012, rtol=0)

    # *CDF
    x = q_to_x(data, q)
    q_est = np.array(map(t.cdf, x))
    np.testing.assert_allclose(q, q_est, atol=0.005)

    with pytest.raises(TypeError):
        t.merge(t2, 'not a tdigest')
