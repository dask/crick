from __future__ import print_function, division, absolute_import

import sys
import pickle

import numpy as np
import pytest

from crick import StreamSummary


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
def test_stream_summary(dtype, data):
    s = StreamSummary(capacity=20, dtype=dtype)
    s.update(data)
    assert s.size() == s.capacity
    top = s.topk(10)
    vals, counts = topk(data, 10)
    assert (top['item'] == vals).all()
    assert (top['count'] == counts).all()


def test_add_matches_update():
    s = StreamSummary(capacity=20, dtype=int)
    s2 = StreamSummary(capacity=20, dtype=int)
    s.update(data_i8)
    for i in data_i8:
        s2.add(i)
    np.testing.assert_equal(s.counters(), s2.counters())


def test_init():
    s = StreamSummary(capacity=10, dtype='f8')
    assert s.dtype == np.dtype('f8')
    assert s.capacity == 10
    assert StreamSummary(dtype='i4').dtype == np.dtype('i8')

    for c in [-10, 0, np.nan]:
        with pytest.raises(ValueError):
            StreamSummary(capacity=c)

    with pytest.raises(ValueError):
        StreamSummary(dtype='S1')


def test_algorithm():
    s = StreamSummary(capacity=5, dtype='i8')
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
    s = StreamSummary(capacity=5, dtype=object)

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
    s = StreamSummary(capacity=5, dtype=object)
    data = ['lists', 'are', 'unhashable']
    orig = sys.getrefcount(data)

    with pytest.raises(TypeError):
        s.add(data)

    # nothing was added and no change in refcount
    assert s.size() == 0
    assert sys.getrefcount(data) == orig


def test_weights():
    s = StreamSummary(capacity=5, dtype='i8')
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
    s = StreamSummary(capacity=5, dtype='i8')

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
    s = StreamSummary(capacity=5, dtype='i8')

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
    s = StreamSummary(capacity=5, dtype='f8')
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
    empty = StreamSummary(capacity=5, dtype=dtype)
    nonempty = StreamSummary(capacity=5, dtype=dtype)
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
