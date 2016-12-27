from __future__ import print_function, division, absolute_import

import pickle
from copy import copy

import numpy as np
import pytest

from crick import SummaryStats

normal = np.random.normal(50, scale=100, size=10000)
missing = np.random.randint(0, 10000, size=1000)
normal[missing] = np.nan

empty = np.array([])
one = np.array([1])
duplicate = np.array([1, 1])
different = np.array([1, 2])

datasets = [normal, empty, one, duplicate, different]

RTOL = 0
ATOL = 1e-8


@pytest.mark.parametrize('x', datasets)
def test_basic_stats(x):
    s = SummaryStats()
    s.update(x)

    assert s.count() == np.count_nonzero(~np.isnan(x))
    np.testing.assert_allclose(s.sum(), np.nansum(x), rtol=RTOL, atol=ATOL)
    np.testing.assert_equal(s.min(), np.nanmin(x) if len(x) else np.nan)
    np.testing.assert_equal(s.max(), np.nanmax(x) if len(x) else np.nan)
    np.testing.assert_allclose(s.mean(), np.nanmean(x) if len(x) else np.nan,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.var(), np.nanvar(x) if len(x) else np.nan,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.std(), np.nanstd(x) if len(x) else np.nan,
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('x', datasets)
@pytest.mark.parametrize('bias', [True, False])
def test_skew(x, bias):
    stats = pytest.importorskip('scipy.stats')
    s = SummaryStats()
    s.update(x)
    res = s.skew(bias=bias)
    sol = stats.skew(x[~np.isnan(x)], bias=bias) if len(x) else np.nan
    np.testing.assert_allclose(res, sol, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('x', datasets)
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('fisher', [True, False])
def test_kurt(x, bias, fisher):
    stats = pytest.importorskip('scipy.stats')
    s = SummaryStats()
    s.update(x)

    res = s.kurt(bias=bias, fisher=fisher)
    if len(x):
        sol = stats.kurtosis(x[~np.isnan(x)], bias=bias, fisher=fisher)
    else:
        sol = np.nan
    np.testing.assert_allclose(res, sol, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('x', [empty, normal])
def test_pickle(x):
    s = SummaryStats()
    s.update(x)
    s2 = pickle.loads(pickle.dumps(s, protocol=2))
    np.testing.assert_equal(s.count(), s2.count())
    np.testing.assert_equal(s.sum(), s2.sum())
    np.testing.assert_equal(s.min(), s2.min())
    np.testing.assert_equal(s.max(), s2.max())
    np.testing.assert_equal(s.var(), s2.var())
    np.testing.assert_equal(s.skew(), s2.skew())
    np.testing.assert_equal(s.kurt(), s2.kurt())


def test_repr():
    s = SummaryStats()
    assert str(s) == 'SummaryStats<count=0>'
    s.add(10)
    assert str(s) == 'SummaryStats<count=1>'


def test_weights():
    s = SummaryStats()
    s.add(10, 2)
    assert s.count() == 2
    assert s.sum() == 10

    x = np.array([1, 2, 3, 4, 5])
    s.update(x, 2)
    assert s.count() == 12
    assert s.sum() == x.sum() + 10

    s = SummaryStats()
    x = np.array([1, 2, 3, 4, 5])
    s.update(x, x)
    assert s.count() == x.sum()


def test_add_update_errors():
    s = SummaryStats()

    x = np.array([1, 2, 3])
    for c in [-1, 0, np.array([1, 1, -1])]:
        with pytest.raises(ValueError):
            s.update(x, c)

    for c in [-1, 0]:
        with pytest.raises(ValueError):
            s.update(1, c)

    with pytest.raises(ValueError):
        s.update(x, np.nan)

    with pytest.raises(ValueError):
        s.add(1, np.nan)


def test_merge():
    s = SummaryStats()
    half = int(len(normal)/2)
    s.update(normal[:half])
    s2 = SummaryStats()
    s2.update(normal[half:])
    sol = SummaryStats()
    sol.update(normal)

    s.merge(s2)
    np.testing.assert_allclose(s.count(), sol.count(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.sum(), sol.sum(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.min(), sol.min(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.max(), sol.max(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.var(), sol.var(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.skew(), sol.skew(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.kurt(), sol.kurt(), rtol=RTOL, atol=ATOL)

    # Test merge both ways is idempotent
    empty_with_full = SummaryStats()
    empty_with_full.merge(sol)
    full_with_empty = copy(sol)
    full_with_empty.merge(SummaryStats())

    for s in [empty_with_full, full_with_empty]:
        np.testing.assert_equal(s.count(), sol.count())
        np.testing.assert_equal(s.sum(), sol.sum())
        np.testing.assert_equal(s.min(), sol.min())
        np.testing.assert_equal(s.max(), sol.max())
        np.testing.assert_equal(s.var(), sol.var())
        np.testing.assert_equal(s.skew(), sol.skew())
        np.testing.assert_equal(s.kurt(), sol.kurt())
