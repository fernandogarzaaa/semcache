import numpy as np
import pytest

from semcache.embedders import HashingEmbedder, _stem, l2_normalize, make_embedder
from semcache.normalize import normalize_query


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("What is Python?", "what is python"),
        ("  Hey, can you please tell me what is Python?? ", "what is python"),
        ("Could you explain recursion, thanks!", "explain recursion"),
        ("ok so quick question: how do I sort a dict", "how do i sort a dict"),
        ("Explain   decorators\n please", "explain decorators"),
        ("Hello", "hello"),
        ("please", "please"),
        ("I would like to know the capital of France.", "the capital of france"),
        ("thanks for the help with sql joins", "thanks for the help with sql joins"),
        ("hiking trails near denver", "hiking trails near denver"),
    ],
)
def test_normalize_query(raw, expected):
    assert normalize_query(raw) == expected


def test_normalize_never_empty():
    assert normalize_query("?") == "?"
    assert normalize_query("") == ""


def test_stem():
    assert _stem("running") == "runn"
    assert _stem("lists") == "list"
    assert _stem("boxes") == "box"
    assert _stem("sorted") == "sort"
    assert _stem("is") == "is"
    assert _stem("ages") == "age"
    assert _stem("bus") == "bus"
    assert _stem("ring") == "ring"
    assert _stem("seed") == "seed"


def test_l2_normalize_rows_and_zero_rows():
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(m)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_hashing_embedder_shape_unit_norm_deterministic():
    e = HashingEmbedder(dim=64)
    assert e.dim == 64 and e.name == "hashing-64"
    v = e.embed(["what is python", "what is python", "x"])
    assert v.shape == (3, 64)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v[0], v[1])
    # Pinned values: the embedding must be identical across runs, processes and machines.
    assert float(np.round(v[0] @ e.embed(["what is java"])[0], 4)) == pytest.approx(0.5721, abs=1e-4)
    nz = np.nonzero(v[0])[0][:5]
    assert list(nz) == [6, 8, 16, 18, 19]
    assert np.allclose(v[0][nz], [0.2132, -0.2132, 0.4264, -0.2132, 0.4264], atol=1e-4)


def test_hashing_embedder_similarity_ordering():
    e = HashingEmbedder()
    a, b, c = e.embed(["how do i reverse a list in python", "how to reverse a python list", "best pizza in naples"])
    assert a @ b > 0.6
    assert a @ c < 0.2
    assert a @ b > a @ c


def test_hashing_embedder_features():
    e = HashingEmbedder()
    feats = e.features("Lists ok")
    assert "w:list" in feats and "w:ok" in feats
    assert "c:#li" in feats and "c:st#" in feats and "c:#ok" in feats and "c:ok#" in feats
    assert sum(f.startswith("w:") for f in feats) == 2
    assert len([f for f in e.features("ab") if f.startswith("c:")]) == 2


def test_hashing_embedder_weights_words_double():
    e = HashingEmbedder(dim=4096)
    raw = np.zeros(4096)
    from semcache.embedders import _bucket

    for f in e.features("zz"):
        i, s = _bucket(f, 4096)
        raw[i] += s * (2.0 if f.startswith("w:") else 1.0)
    expected = raw / np.linalg.norm(raw)
    assert np.allclose(e.embed(["zz"])[0], expected, atol=1e-6)


def test_bucket_range_and_sign():
    from semcache.embedders import _bucket

    seen_signs = set()
    for i in range(200):
        idx, sign = _bucket(f"f{i}", 7)
        assert 0 <= idx < 7
        seen_signs.add(sign)
    assert seen_signs == {1.0, -1.0}


def test_empty_text_embeds_to_zero_vector():
    v = HashingEmbedder(dim=8).embed([""])
    assert v.shape == (1, 8)
    assert not v.any()


def test_hashing_embedder_rejects_bad_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_make_embedder():
    assert make_embedder("hashing", dim=32).dim == 32
    assert make_embedder("HASHING").dim == 256
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("nope")


# ---- order guard -------------------------------------------------------------------------------
from semcache.normalize import _directional, _stem_token, content_tokens, order_swapped  # noqa: E402


def test_stem_token_rules():
    assert [_stem_token(w) for w in "miles class watches dishes buzzes boxes bus lists ages".split()] == [
        "mile",
        "class",
        "watch",
        "dish",
        "buzz",
        "box",
        "bus",
        "list",
        "age",
    ]
    assert _stem_token("sees") == "see"
    assert _stem_token("goes") == "goe"
    assert _stem_token("ties") == "tie"
    assert _stem_token("cats") == "cat"
    assert _stem_token("bed") == "bed"
    assert _stem_token("seeded") == "seed"
    assert _stem_token("sing") == "sing"
    assert _stem_token("boss") == "boss"
    assert _stem_token("axes") == "axe"  # len 4 is not > 4, falls through to the plain "s" rule


def test_content_tokens_drops_stopwords_and_stems():
    assert content_tokens("How do I convert the Miles to KM?") == ["convert", "mile", "to", "km"]
    assert content_tokens("") == []


@pytest.mark.parametrize(
    "a,b,swapped",
    [
        ("convert km to miles", "convert miles to km", True),
        ("convert degrees fahrenheit into celsius", "how do i convert celsius to fahrenheit", True),
        ("how many kilometers are in a mile", "how many miles are in a kilometer", True),
        ("is a cat faster than a dog", "is a dog faster than a cat", True),
        ("python reverse list", "how to reverse a list in python", False),
        ("celsius to fahrenheit formula", "how to change celsius into fahrenheit", False),
        ("left join vs inner join", "explain inner join versus left join in sql", False),
        ("what is x", "what is x", False),
        ("python", "python", False),
        ("convert km to miles", "convert km to miles please", False),
        ("go from a to b", "go from a to b", False),
        ("translate english to french", "how to translate", False),
        ("map to list", "list map", False),
    ],
)
def test_order_swapped(a, b, swapped):
    assert order_swapped(a, b) is swapped
    assert order_swapped(b, a) is swapped


def test_single_shared_word_is_not_a_swap():
    assert not order_swapped("send money to bob", "money from alice")


def test_directional():
    assert _directional(["convert", "a", "to", "b"])
    assert _directional(["into"])
    assert _directional(["to", "x"])  # leading "to" has no verb before it
    assert not _directional(["how", "to", "cook"])
    assert not _directional(["want", "to", "go"])
    assert _directional(["a", "than", "b"]) and _directional(["from", "x"])
    assert not _directional(["a", "vs", "b"])
