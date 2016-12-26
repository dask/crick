from __future__ import print_function, division, absolute_import

import sys
import pickle
from copy import copy
from collections import namedtuple

import numpy as np
import pytest

from crick import SpaceSaving


data_f8 = np.random.RandomState(42).gamma(0.1, 0.1, size=10000).round(2) * 100
data_i8 = data_f8.astype('i8')
data_object = data_i8.astype(object)
data_string = np.array([str(i) for i in data_object], dtype=object)


def topk(data, k):
    unique, counts = np.unique(data, return_counts=True)
    inds = counts.argsort()[::-1]
    return unique[inds[:k]], counts[inds[:k]]


@pytest.mark.parametrize('dtype,data',
                         [(float, data_f8),
                          (int, data_i8),
                          (object, data_object),
                          (object, data_string),
                          (object, data_i8),
                          (float, data_i8)])
def test_space_saving(dtype, data):
    s = SpaceSaving(capacity=20, dtype=dtype)
    s.update(data)
    assert s.size() == s.capacity
    top = s.topk(10)
    vals, counts = topk(data, 10)
    assert (top['item'] == vals).all()
    assert (top['count'] == counts).all()


def test_add_matches_update():
    s = SpaceSaving(capacity=20, dtype=int)
    s2 = SpaceSaving(capacity=20, dtype=int)
    s.update(data_i8)
    for i in data_i8:
        s2.add(i)
    np.testing.assert_equal(s.counters(), s2.counters())


def test_init():
    s = SpaceSaving(capacity=10, dtype='f8')
    assert s.dtype == np.dtype('f8')
    assert s.capacity == 10
    assert SpaceSaving(dtype='i4').dtype == np.dtype('i8')

    for c in [-10, 0, np.nan]:
        with pytest.raises(ValueError):
            SpaceSaving(capacity=c)

    with pytest.raises(ValueError):
        SpaceSaving(dtype='S1')


def test_algorithm():
    s = SpaceSaving(capacity=5, dtype='i8')
    assert s.size() == 0
    assert len(s.counters()) == 0

    # Add 5 independent items
    for i in range(5):
        s.add(i)
    c = s.counters()
    assert len(c) == 5
    np.testing.assert_equal(c['item'], np.arange(5, dtype='i8'))
    np.testing.assert_equal(c['count'], 1)
    np.testing.assert_equal(c['error'], 0)

    # Add the tail one twice, should move to front
    s.add(4, 2)
    c = s.counters()
    np.testing.assert_equal(c['item'], np.array([4, 0, 1, 2, 3]))
    np.testing.assert_equal(c['count'], np.array([3, 1, 1, 1, 1]))
    np.testing.assert_equal(c['error'], 0)

    # Add a middle one, should move to 2nd
    s.add(2)
    c = s.counters()
    np.testing.assert_equal(c['item'], np.array([4, 2, 0, 1, 3]))
    np.testing.assert_equal(c['count'], np.array([3, 2, 1, 1, 1]))
    np.testing.assert_equal(c['error'], 0)

    # Add new element, tail should drop off
    s.add(5)
    c = s.counters()
    np.testing.assert_equal(c['item'], np.array([4, 2, 5, 0, 1]))
    np.testing.assert_equal(c['count'], np.array([3, 2, 2, 1, 1]))
    np.testing.assert_equal(c['error'], np.array([0, 0, 1, 0, 0]))

    # Update some more elements
    s.update([6, 7] * 5)
    c = s.counters()
    np.testing.assert_equal(c['item'], np.array([6, 7, 4, 2, 5]))
    np.testing.assert_equal(c['count'], np.array([6, 6, 3, 2, 2]))
    np.testing.assert_equal(c['error'], np.array([1, 1, 0, 0, 1]))


def test_object_reference_counting():
    s = SpaceSaving(capacity=5, dtype=object)

    data = ('this', 'should', 'have', 'no', 'other', 'refs')
    data2 = ('neither', 'should', 'this')
    data3 = ('added', 'later', 'to', 'check', 'swapping', 'counts')
    array_data2 = np.array([None], dtype=object)
    array_data2[0] = data2
    array_data3 = np.array([1, 2, 3, data3] * 2, dtype=object)

    # Adding increases the reference count
    orig = sys.getrefcount(data)
    s.add(data)
    after = sys.getrefcount(data)
    assert after == orig + 1

    # Updating increases the reference count
    orig2 = sys.getrefcount(data2)
    s.update(array_data2)
    after2 = sys.getrefcount(data2)
    assert after2 == orig2 + 1

    # Adding again doesn't change things
    s.add(data)
    assert sys.getrefcount(data) == after

    # Dropping decreases the reference count, swapped items
    # have reference counts increased
    orig3 = sys.getrefcount(data3)
    s.update(array_data3)
    after3 = sys.getrefcount(data3)
    assert after3 == orig3 + 1
    assert sys.getrefcount(data2) == orig2

    # check that deleting decreases the reference count
    del s
    assert sys.getrefcount(data) == orig
    assert sys.getrefcount(data2) == orig2
    assert sys.getrefcount(data3) == orig3


def test_object_hash_errors():
    s = SpaceSaving(capacity=5, dtype=object)
    data = ['lists', 'are', 'unhashable']
    orig = sys.getrefcount(data)

    with pytest.raises(TypeError):
        s.add(data)

    # nothing was added and no change in refcount
    assert s.size() == 0
    assert sys.getrefcount(data) == orig


def test_weights():
    s = SpaceSaving(capacity=5, dtype='i8')
    s.add(1, 10)
    c = s.counters()
    np.testing.assert_equal(c['item'], 1)
    np.testing.assert_equal(c['count'], 10)

    s.update([1, 2, 3], 5)
    c = s.counters()
    np.testing.assert_equal(c['item'], [1, 2, 3])
    np.testing.assert_equal(c['count'], [15, 5, 5])

    s.update(3, [5, 10, 5])
    c = s.counters()
    np.testing.assert_equal(c['item'], [3, 1, 2])
    np.testing.assert_equal(c['count'], [25, 15, 5])

    s.update([1, 2, 3], [1, 2, 3])
    c = s.counters()
    np.testing.assert_equal(c['item'], [3, 1, 2])
    np.testing.assert_equal(c['count'], [28, 16, 7])


def test_add_raises():
    s = SpaceSaving(capacity=5, dtype='i8')

    for x, c in [(1, 0), (1, -10), (1, np.nan), (np.nan, 1)]:
        with pytest.raises(ValueError):
            s.add(x, c)

    for x, c in [('foo', 1), (1, 'foo')]:
        with pytest.raises(TypeError):
            s.add(x, c)

    for x, c in [(1, np.inf), (1, -np.inf), (np.inf, 1), (-np.inf, 1)]:
        with pytest.raises(OverflowError):
            s.add(x, c)


def test_update_raises():
    s = SpaceSaving(capacity=5, dtype='i8')

    for x, c in [(1, 0), (1, -10)]:
        for x2, c2 in [([x], [c]), ([x], c), (x, [c])]:
            with pytest.raises(ValueError):
                s.update(x2, c2)

    for x, c in [(1, np.nan), (1, np.inf), (1, -np.inf), (1, 'foo'),
                 (np.nan, 1), (np.inf, 1), (-np.inf, 1), ('foo', 1)]:
        with pytest.raises(TypeError):
            for x2, c2 in [([x], [c]), ([x], c), (x, [c])]:
                s.update(x2, c2)


def test_topk_invariants():
    s = SpaceSaving(capacity=5, dtype='f8')
    s.update(data_f8)
    for k in [0, 5]:
        top = s.topk(k)
        assert isinstance(top, np.ndarray)
        dtype = np.dtype([('item', 'f8'), ('count', 'i8'), ('error', 'i8')])
        assert top.dtype == dtype
        assert len(top) == k
        assert (np.diff(top['count']) <= 0).all()

        top2 = s.topk(k, astuples=True)
        assert len(top2) == k
        np.testing.assert_equal(top['item'], [i.item for i in top2])
        np.testing.assert_equal(top['count'], [i.count for i in top2])
        np.testing.assert_equal(top['error'], [i.error for i in top2])

    with pytest.raises(ValueError):
        s.topk(-1)


@pytest.mark.parametrize('dtype', [float, int, object])
def test_serialize(dtype):
    empty = SpaceSaving(capacity=5, dtype=dtype)
    nonempty = SpaceSaving(capacity=5, dtype=dtype)
    # Results in nonuniform count and error, and non-sorted counters
    nonempty.update([1, 2, 3, 4, 5, 3, 3, 3, 6, 7])

    for s in [empty, nonempty]:
        s2 = pickle.loads(pickle.dumps(s))
        assert s2.capacity == s.capacity
        assert s2.size() == s.size()
        np.testing.assert_equal(s.counters(), s2.counters())
        # Add an element already in the table, to ensure hashing still works
        s.add(1)
        s2.add(1)
        np.testing.assert_equal(s.counters(), s2.counters())


Counter = namedtuple('Counter', ['item', 'count', 'error'])


def counter_sort_key(c):
    return (c.count, -c.error)


def merge(s1, s2):
    """A reimplementation of the merging algorithm, for testing purposes."""
    capacity = s1.capacity
    c1 = s1.counters()
    c2 = s2.counters()
    m1 = c1['count'][-1] if s1.size() == s1.capacity else 0
    m2 = c2['count'][-1] if s2.size() == s2.capacity else 0

    s1 = {i['item']: Counter(*i) for i in c1}
    s2 = {i['item']: Counter(*i) for i in c2}

    out = {}
    for item in set(s1).intersection(s2):
        out[item] = Counter(item,
                            s1[item].count + s2[item].count,
                            s1[item].error + s2[item].error)

    for item in set(s1).difference(s2):
        out[item] = Counter(item,
                            s1[item].count + m2,
                            s1[item].error + m2)

    for item in set(s2).difference(s1):
        out[item] = Counter(item,
                            s2[item].count + m1,
                            s2[item].error + m1)

    res = sorted(out.values(), key=counter_sort_key, reverse=True)
    return res[:capacity]


d1 = [1, 2, 3, 4, 5, 5, 5, 6]
d2 = [1, 2, 3, 5, 5, 6, 6, 7]

# s1 and s2 are designed to hit each case of the merge algorithm
s1 = SpaceSaving(capacity=5, dtype='i8')
s1.update(d1)

s2 = SpaceSaving(capacity=5, dtype='i8')
s2.update(d2)

empty = SpaceSaving(capacity=5, dtype='i8')

two = SpaceSaving(capacity=7, dtype='i8')
two.update([1, 2])

half = SpaceSaving(capacity=10, dtype='i8')
half.update(d1)

big_1 = SpaceSaving(capacity=20, dtype='i8')
big_1.update(data_i8)

data_i8b = np.random.RandomState(7).gamma(0.1, 0.1, size=10000).round(2) * 100
big_2 = SpaceSaving(capacity=20, dtype='i8')
big_2.update(data_i8b.astype('i8'))

object_1 = SpaceSaving(capacity=20, dtype=object)
object_1.update(data_i8)
object_2 = SpaceSaving(capacity=20, dtype=object)
object_2.update(data_i8b)

summaries = [(s1, s2),
             (empty, empty),
             (s1, s1),
             (s1, empty),
             (empty, s1),
             (empty, two),
             (empty, big_1),
             (two, big_1),
             (big_1, two),
             (half, s1),
             (big_1, big_2),
             (object_1, object_2)]


@pytest.mark.parametrize('s1, s2', summaries)
def test_merge_algorithm(s1, s2):
    s3 = SpaceSaving(capacity=s1.capacity, dtype=s1.dtype)
    s3.merge(s1, s2)
    s4 = copy(s1)
    s4.merge(s2)
    # Equivalent operations
    np.testing.assert_equal(s3.counters(), s4.counters())

    res = [Counter(*i) for i in s3.counters()]
    # Check that the output is sorted
    res2 = sorted(res, key=counter_sort_key, reverse=True)
    assert res == res2
    # Due to equivalent counters maybe sorting differently we just check that
    # the C code returns only counters that *may* have sorted into the topk
    sol = merge(s1, s2)
    assert len(sol) == len(res)
    for c in res:
        assert counter_sort_key(c) >= counter_sort_key(sol[-1])


def test_merge_errors():
    s = SpaceSaving(capacity=10, dtype=object)
    s1 = SpaceSaving(capacity=10, dtype=object)
    s1.update(data_object)

    with pytest.raises(TypeError):
        s.merge(s1, 'not correct type')

    with pytest.raises(ValueError):
        s.merge(s1, big_1)

    # Nothing added before error checking
    assert s.size() == 0
