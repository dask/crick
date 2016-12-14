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

    assert t.size() == len(data)
    assert t.min() == data.min()
    assert t.max() == data.max()

    # *Quantile
    q = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    est = t.quantile(q)
    q_est = quantiles_to_q(data, est)
    np.testing.assert_allclose(q, q_est, atol=0.012, rtol=0)

    # *CDF
    x = q_to_x(data, q)
    q_est = t.cdf(x)
    np.testing.assert_allclose(q, q_est, atol=0.005)


def test_init():
    t = TDigest(500)
    assert t.compression == 500

    with pytest.raises(TypeError):
        TDigest('foo')

    for c in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            TDigest(c)


def test_repr():
    t = TDigest(500)
    assert str(t) == "TDigest<compression=500.0, size=0.0>"
    t.update(np.arange(100))
    assert str(t) == "TDigest<compression=500.0, size=100.0>"


def test_empty():
    t = TDigest()
    assert t.size() == 0
    assert len(t.centroids()) == 0
    assert np.isnan(t.min())
    assert np.isnan(t.max())
    assert np.isnan(t.quantile(0.5))
    assert np.isnan(t.cdf(0.5))


def test_single():
    t = TDigest()
    t.add(10)
    assert t.min() == 10
    assert t.max() == 10
    assert t.size() == 1

    assert t.quantile(0) == 10
    assert t.quantile(0.5) == 10
    assert t.quantile(1) == 10

    assert t.cdf(9) == 0
    assert t.cdf(10) == 0.5
    assert t.cdf(11) == 1


def test_nonfinite():
    t = TDigest()
    data = gamma.copy()
    data[::10] = np.nan
    data[::7] = np.inf
    t.update(data)
    finite = data[np.isfinite(data)]
    assert t.size() == len(finite)
    assert t.min() == finite.min()
    assert t.max() == finite.max()

    t = TDigest()
    t.add(np.nan)
    t.add(np.inf)
    t.add(-np.inf)
    assert t.size() == 0

    for w in [np.inf, -np.inf, np.nan]:
        t = TDigest()
        with pytest.raises(ValueError):
            t.add(1, w)

        w = np.array([1, 2, w, 3, 4])
        t = TDigest()
        with pytest.raises(ValueError):
            t.update(np.ones(5), w)


def test_small_w():
    eps = np.finfo('f8').eps
    t = TDigest()
    t.update(gamma, eps)
    assert t.size() == 0
    assert len(t.centroids()) == 0

    t = TDigest()
    t.add(1, eps)
    assert t.size() == 0
    assert len(t.centroids()) == 0


def test_update_non_numeric_errors():
    data = np.array(['foo', 'bar', 'baz'])
    t = TDigest()

    with pytest.raises(TypeError):
        t.update(data)

    with pytest.raises(TypeError):
        t.update(1, data)

    with pytest.raises(TypeError):
        t.add('foo')

    with pytest.raises(TypeError):
        t.add(1, 'foo')


def test_quantile_and_cdf_non_numeric():
    t = TDigest()
    t.update(np.arange(5))

    with pytest.raises(TypeError):
        t.quantile('foo')

    with pytest.raises(TypeError):
        t.update(['foo'])

    with pytest.raises(TypeError):
        t.cdf('foo')

    with pytest.raises(TypeError):
        t.cdf(['foo'])


def test_quantile_and_cdf_shape():
    t = TDigest()
    t.update(np.arange(5))

    assert isinstance(t.quantile(0.5), np.float64)
    assert isinstance(t.cdf(2), np.float64)

    res = t.quantile(())
    assert res.shape == (0,)
    res = t.cdf(())
    assert res.shape == (0,)

    q = np.array([0.5, 0.9])
    res = t.quantile(q)
    assert res.shape == (2,)
    res = t.cdf(q)
    assert res.shape == (2,)

    q = np.array([[0.5, 0.9], [0, 1]])
    res = t.quantile(q)
    assert res.shape == (2, 2)
    res = t.cdf(q)
    assert res.shape == (2, 2)


def test_weights():
    t = TDigest()
    t.add(1, 10)
    assert t.size() == 10

    x = np.arange(5)
    w = np.array([1, 2, 1, 2, 1])

    t = TDigest()
    t.update(x, 10)
    assert t.size() == len(x) * 10

    t = TDigest()
    t.update(x, w)
    assert t.size() == w.sum()


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
        np.testing.assert_equal(t.size(), t2.size())


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
    assert t.size() == t2.size() + t3.size()
    # Check no mutation of args
    assert (t2.centroids() == t2_centroids).all()

    # *Quantile
    q = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    est = t.quantile(q)
    q_est = quantiles_to_q(data, est)
    np.testing.assert_allclose(q, q_est, atol=0.012, rtol=0)

    # *CDF
    x = q_to_x(data, q)
    q_est = t.cdf(x)
    np.testing.assert_allclose(q, q_est, atol=0.005)

    with pytest.raises(TypeError):
        t.merge(t2, 'not a tdigest')


def test_scale():
    t = TDigest()
    t.update(uniform)

    for factor in [0.5, 2]:
        t2 = t.scale(factor)
        assert t is not t2
        assert t.size() * factor == t2.size()
        assert t.min() == t2.min()
        assert t.max() == t2.max()
        a = t.centroids()
        b = t2.centroids()
        np.testing.assert_array_equal(a['mean'], b['mean'])
        np.testing.assert_allclose(a['weight'] * factor, b['weight'])

    for val in [-0.5, 0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            t.scale(val)

    with pytest.raises(TypeError):
        t.scale('foobar')

    # Test scale compacts
    eps = np.finfo('f8').eps
    t = TDigest()
    t.update([1, 2, 3, 4, 5],
             [1, 1000, 1, 10000, 1])
    t2 = t.scale(eps)
    assert len(t2.centroids()) == 2

    # Compacts to 0
    t = TDigest()
    t.update([1, 2, 3, 4, 5])
    t2 = t.scale(eps)
    assert len(t2.centroids()) == 0
