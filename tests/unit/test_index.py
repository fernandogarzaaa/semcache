import numpy as np
import pytest

from semcache.index import NumpyIndex, _as_vec, make_index

try:
    import faiss  # noqa: F401

    HAVE_FAISS = True
except ImportError:  # pragma: no cover
    HAVE_FAISS = False

KINDS = ["numpy"] + (["faiss"] if HAVE_FAISS else [])


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("kind", KINDS)
def test_add_search_ranked(kind):
    idx = make_index(kind, 3)
    assert idx.name == kind
    idx.add("ns", 1, unit(1, 0, 0))
    idx.add("ns", 2, unit(1, 1, 0))
    idx.add("ns", 3, unit(0, 0, 1))
    res = idx.search("ns", unit(1, 0.1, 0), k=3)
    assert [i for i, _ in res] == [1, 2, 3]
    assert res[0][1] > res[1][1] > res[2][1]
    assert res[0][1] == pytest.approx(float(unit(1, 0, 0) @ unit(1, 0.1, 0)), abs=1e-5)
    assert len(idx) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_k_larger_than_size_and_k1(kind):
    idx = make_index(kind, 2)
    idx.add("ns", 7, unit(1, 0))
    assert idx.search("ns", unit(1, 0), k=10) == [(7, pytest.approx(1.0, abs=1e-5))]
    idx.add("ns", 8, unit(0, 1))
    assert [i for i, _ in idx.search("ns", unit(0, 1), k=1)] == [8]


@pytest.mark.parametrize("kind", KINDS)
def test_namespaces_are_isolated(kind):
    idx = make_index(kind, 2)
    idx.add("a", 1, unit(1, 0))
    idx.add("b", 2, unit(1, 0))
    assert [i for i, _ in idx.search("a", unit(1, 0), k=5)] == [1]
    assert idx.search("missing", unit(1, 0)) == []


@pytest.mark.parametrize("kind", KINDS)
def test_remove_and_readd(kind):
    idx = make_index(kind, 2)
    idx.add("ns", 1, unit(1, 0))
    idx.add("ns", 2, unit(0, 1))
    idx.add("ns", 3, unit(1, 1))
    assert idx.remove("ns", 1) is True
    assert idx.remove("ns", 1) is False
    assert idx.remove("nope", 1) is False
    assert sorted(i for i, _ in idx.search("ns", unit(1, 0), k=5)) == [2, 3]
    assert len(idx) == 2
    idx.remove("ns", 2)
    idx.remove("ns", 3)
    assert len(idx) == 0
    assert idx.search("ns", unit(1, 0)) == []
    idx.add("ns", 4, unit(1, 0))
    assert [i for i, _ in idx.search("ns", unit(1, 0))] == [4]


@pytest.mark.parametrize("kind", KINDS)
def test_readd_same_id_replaces_vector(kind):
    idx = make_index(kind, 2)
    idx.add("ns", 1, unit(1, 0))
    idx.add("ns", 1, unit(0, 1))
    assert len(idx) == 1
    ((i, s),) = idx.search("ns", unit(0, 1))
    assert i == 1 and s == pytest.approx(1.0, abs=1e-5)


def test_numpy_partition_grows_beyond_initial_capacity():
    idx = NumpyIndex(4)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 4)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for i, v in enumerate(vecs):
        idx.add("ns", i, v)
    assert len(idx) == 50
    for probe in (0, 9, 31, 49):
        assert idx.search("ns", vecs[probe], k=1)[0][0] == probe


def test_numpy_remove_middle_keeps_mapping_consistent():
    idx = NumpyIndex(2)
    vs = {1: unit(1, 0), 2: unit(0, 1), 3: unit(-1, 0)}
    for i, v in vs.items():
        idx.add("ns", i, v)
    idx.remove("ns", 1)  # last element (3) moves into slot 0
    assert idx.search("ns", unit(-1, 0), k=1)[0][0] == 3
    assert idx.search("ns", unit(0, 1), k=1)[0][0] == 2
    idx.remove("ns", 3)
    assert idx.search("ns", unit(-1, 0), k=5) == [(2, pytest.approx(0.0, abs=1e-6))]


def test_numpy_remove_last_element():
    idx = NumpyIndex(2)
    idx.add("ns", 1, unit(1, 0))
    idx.add("ns", 2, unit(0, 1))
    assert idx.remove("ns", 2)
    assert idx.search("ns", unit(0, 1), k=5) == [(1, pytest.approx(0.0, abs=1e-6))]


def test_as_vec_checks_dimension_and_flattens():
    assert _as_vec(np.ones((1, 3)), 3).shape == (3,)
    assert _as_vec(np.ones(3, dtype=np.float64), 3).dtype == np.float32
    with pytest.raises(ValueError, match="3-dimensional"):
        _as_vec(np.ones(2), 3)


def test_make_index_kinds():
    assert make_index("NUMPY", 2).name == "numpy"
    assert make_index("auto", 2).name in ("numpy", "faiss")
    with pytest.raises(ValueError):
        make_index("annoy", 2)
