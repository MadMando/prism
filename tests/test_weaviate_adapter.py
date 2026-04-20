"""
Tests for prism.adapters.weaviate.WeaviateAdapter.

Mocks the Weaviate v4 collection so no running Weaviate is required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("weaviate")

from prism.adapters.weaviate import WeaviateAdapter


def _obj(uuid, props, vector=None, distance=None):
    return SimpleNamespace(
        uuid       = uuid,
        properties = props,
        vector     = vector,
        metadata   = SimpleNamespace(distance=distance),
    )


def _make_adapter():
    adapter = WeaviateAdapter(collection_name="Test", id_property="chunk_id")
    adapter._embedder = MagicMock()
    adapter._embedder.embed.return_value = [0.1] * 4
    adapter._embedder.model = "mock-embed"
    adapter._client     = MagicMock()
    adapter._collection = MagicMock()
    return adapter


# ── seed_scores ───────────────────────────────────────────────────────────────

def test_seed_scores_maps_distance_to_similarity():
    adapter = _make_adapter()
    adapter._collection.query.near_vector.return_value = SimpleNamespace(objects=[
        _obj("u1", {"chunk_id": "a", "source": "s"}, distance=0.1),
        _obj("u2", {"chunk_id": "b", "source": "s"}, distance=0.5),
    ])
    scores = adapter.seed_scores("q", top_k=2)
    assert scores["a"] > scores["b"]
    assert scores["a"] == 0.9


def test_seed_scores_source_filter_client_side():
    adapter = _make_adapter()
    adapter._collection.query.near_vector.return_value = SimpleNamespace(objects=[
        _obj("u1", {"chunk_id": "a", "source": "dmbok"}, distance=0.1),
        _obj("u2", {"chunk_id": "b", "source": "nist"},  distance=0.1),
    ])
    scores = adapter.seed_scores("q", top_k=5, source_filter="dmbok")
    assert set(scores.keys()) == {"a"}


# ── get_chunks ────────────────────────────────────────────────────────────────

def test_get_chunks_empty_input():
    adapter = _make_adapter()
    assert adapter.get_chunks([]) == {}


def test_get_chunks_returns_shape():
    adapter = _make_adapter()
    adapter._collection.query.fetch_objects.return_value = SimpleNamespace(
        objects=[_obj("u1", {
            "chunk_id": "a", "source": "s", "page": 2, "section": "§", "text": "T"
        })]
    )
    chunks = adapter.get_chunks(["a"])
    assert chunks["a"]["text"] == "T"
    assert chunks["a"]["page"] == 2


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_no_dead_info_variable():
    """Stats must not call client.collections.get — regression for dead-code fix."""
    adapter = _make_adapter()
    # Return no objects to end the pagination loop immediately
    adapter._collection.query.fetch_objects.return_value = SimpleNamespace(objects=[])
    s = adapter.stats()
    assert s["total_chunks"] == 0
    # Verify the removed dead code really is gone.
    adapter._client.collections.get.assert_not_called()


def test_stats_counts_sources():
    adapter = _make_adapter()
    objs = [
        _obj("u1", {"source": "x"}),
        _obj("u2", {"source": "x"}),
        _obj("u3", {"source": "y"}),
    ]
    adapter._collection.query.fetch_objects.side_effect = [
        SimpleNamespace(objects=objs),
        SimpleNamespace(objects=[]),
    ]
    s = adapter.stats()
    assert s["total_chunks"] == 3
    assert s["sources"]      == {"x": 2, "y": 1}


# ── candidate_pairs_for N+1 fix ───────────────────────────────────────────────

def test_candidate_pairs_for_single_scan_no_n_plus_one():
    """
    Regression test for the N+1 bug: for each chunk_id, the fix uses the vector
    already loaded in the scan loop — it must NOT call fetch_objects again with
    a per-ID filter.
    """
    adapter = _make_adapter()

    scan_objs = [
        _obj("u1", {"chunk_id": "a", "source": "s1"}, vector=[0.1] * 4),
        _obj("u2", {"chunk_id": "b", "source": "s2"}, vector=[0.2] * 4),
    ]

    fetch_calls = {"count": 0}
    def fetch_objects_side_effect(*args, **kwargs):
        fetch_calls["count"] += 1
        # First call returns the scan objects; second call ends pagination.
        if fetch_calls["count"] == 1:
            assert "include_vector" in kwargs and kwargs["include_vector"] is True
            return SimpleNamespace(objects=scan_objs)
        return SimpleNamespace(objects=[])

    adapter._collection.query.fetch_objects.side_effect = fetch_objects_side_effect
    adapter._collection.query.near_vector.return_value = SimpleNamespace(objects=[
        _obj("u2", {"chunk_id": "b", "source": "s2"}),
    ])

    pairs = adapter.candidate_pairs_for(["a"], k_neighbors=2)

    # Only two fetch_objects calls — the scan pages. No per-ID fetch.
    assert fetch_calls["count"] == 2
    assert len(pairs) == 1
    assert pairs[0][0]["id"] == "a"
    assert pairs[0][1]["id"] == "b"
