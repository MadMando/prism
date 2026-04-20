"""
Tests for prism.adapters.chroma.ChromaAdapter.

Mocks the chromadb client so no running Chroma is required. Skipped entirely
when the chromadb package isn't installed.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("chromadb")

from prism.adapters.chroma import ChromaAdapter


def _make_adapter(query_result=None, get_result=None):
    adapter = ChromaAdapter(collection_name="test")
    adapter._embedder = MagicMock()
    adapter._embedder.embed.return_value = [0.1] * 4
    adapter._embedder.model = "mock-embed"

    collection = MagicMock()
    if query_result is not None:
        collection.query.return_value = query_result
    if get_result is not None:
        collection.get.return_value = get_result
    collection.count.return_value = len((get_result or {"ids": []}).get("ids", []))

    adapter._collection = collection
    adapter._client     = MagicMock()
    return adapter, collection


# ── seed_scores ───────────────────────────────────────────────────────────────

def test_seed_scores_basic_shape():
    adapter, _ = _make_adapter(query_result={
        "ids":       [["a", "b", "c"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [[{"source": "x"}, {"source": "y"}, {"source": "z"}]],
    })
    scores = adapter.seed_scores("q", top_k=3)
    assert set(scores.keys()) == {"a", "b", "c"}
    assert scores["a"] > scores["b"] > scores["c"]


def test_seed_scores_source_filter_client_side():
    """The $contains bugfix: filter must apply client-side, no where clause."""
    adapter, collection = _make_adapter(query_result={
        "ids":       [["a", "b", "c"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [[{"source": "dmbok"}, {"source": "nist"}, {"source": "dmbok"}]],
    })
    scores = adapter.seed_scores("q", top_k=5, source_filter="dmbok")
    assert set(scores.keys()) == {"a", "c"}
    # Must not pass a `where` — that was the bug.
    kwargs = collection.query.call_args.kwargs
    assert "where" not in kwargs or kwargs["where"] is None


def test_seed_scores_empty_result():
    adapter, _ = _make_adapter(query_result={
        "ids": [[]], "distances": [[]], "metadatas": [[]],
    })
    assert adapter.seed_scores("q", top_k=5) == {}


def test_seed_scores_top_k_enforced():
    adapter, _ = _make_adapter(query_result={
        "ids":       [["a", "b", "c", "d", "e"]],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        "metadatas": [[{}, {}, {}, {}, {}]],
    })
    scores = adapter.seed_scores("q", top_k=2)
    assert len(scores) == 2


# ── get_chunks ────────────────────────────────────────────────────────────────

def test_get_chunks_returns_shape():
    adapter, _ = _make_adapter(get_result={
        "ids":       ["a", "b"],
        "documents": ["text A", "text B"],
        "metadatas": [{"source": "s1", "page": 1, "section": "1.0"},
                      {"source": "s2", "page": 2, "section": "2.0"}],
    })
    chunks = adapter.get_chunks(["a", "b"])
    assert chunks["a"]["text"]    == "text A"
    assert chunks["a"]["source"]  == "s1"
    assert chunks["a"]["page"]    == 1
    assert chunks["b"]["section"] == "2.0"


def test_get_chunks_empty_input():
    adapter, _ = _make_adapter()
    assert adapter.get_chunks([]) == {}


def test_get_chunks_handles_none_metadata():
    adapter, _ = _make_adapter(get_result={
        "ids":       ["a"],
        "documents": ["text"],
        "metadatas": [None],
    })
    chunks = adapter.get_chunks(["a"])
    assert chunks["a"]["source"] == ""
    assert chunks["a"]["page"]   == 0


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_counts_sources():
    adapter, _ = _make_adapter(get_result={
        "ids":       ["a", "b", "c"],
        "metadatas": [{"source": "x"}, {"source": "x"}, {"source": "y"}],
    })
    adapter._collection.count.return_value = 3
    s = adapter.stats()
    assert s["total_chunks"] == 3
    assert s["sources"]      == {"x": 2, "y": 1}
