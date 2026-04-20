"""
Tests for prism.adapters.qdrant.QdrantAdapter.

Mocks the QdrantClient so no running Qdrant is required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("qdrant_client")

from prism.adapters.qdrant import QdrantAdapter


def _point(id_val, score=None, payload=None, vector=None):
    return SimpleNamespace(
        id      = id_val,
        score   = score if score is not None else 0.0,
        payload = payload or {},
        vector  = vector,
    )


def _make_adapter():
    adapter = QdrantAdapter(collection_name="test")
    adapter._embedder = MagicMock()
    adapter._embedder.embed.return_value = [0.1] * 4
    adapter._embedder.model = "mock-embed"
    adapter._client = MagicMock()
    return adapter


# ── seed_scores ───────────────────────────────────────────────────────────────

def test_seed_scores_maps_cosine_to_0_1():
    adapter = _make_adapter()
    adapter._client.search.return_value = [
        _point(1, score= 1.0, payload={"id": "a", "source": "s1"}),
        _point(2, score= 0.0, payload={"id": "b", "source": "s1"}),
        _point(3, score=-1.0, payload={"id": "c", "source": "s1"}),
    ]
    scores = adapter.seed_scores("q", top_k=3)
    assert scores["a"] == 1.0
    assert scores["b"] == 0.5
    assert scores["c"] == 0.0


def test_seed_scores_source_filter_client_side():
    adapter = _make_adapter()
    adapter._client.search.return_value = [
        _point(1, score=0.9, payload={"id": "a", "source": "dmbok"}),
        _point(2, score=0.8, payload={"id": "b", "source": "nist"}),
    ]
    scores = adapter.seed_scores("q", top_k=5, source_filter="dmbok")
    assert set(scores.keys()) == {"a"}


def test_seed_scores_top_k_respected():
    adapter = _make_adapter()
    adapter._client.search.return_value = [
        _point(i, score=1.0 - i * 0.1, payload={"id": f"c{i}", "source": "s"})
        for i in range(5)
    ]
    assert len(adapter.seed_scores("q", top_k=2)) == 2


# ── get_chunks ────────────────────────────────────────────────────────────────

def test_get_chunks_empty_input():
    adapter = _make_adapter()
    assert adapter.get_chunks([]) == {}


def test_get_chunks_scroll_stops_when_offset_none():
    adapter = _make_adapter()
    adapter._client.scroll.return_value = (
        [_point(1, payload={"id": "a", "source": "s", "page": 1, "section": "§1", "text": "T"})],
        None,
    )
    chunks = adapter.get_chunks(["a"])
    assert chunks["a"]["text"]   == "T"
    assert chunks["a"]["source"] == "s"


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_reports_source_counts():
    adapter = _make_adapter()
    adapter._client.get_collection.return_value = SimpleNamespace(points_count=2)
    adapter._client.scroll.return_value = (
        [
            _point(1, payload={"source": "x"}),
            _point(2, payload={"source": "y"}),
        ],
        None,
    )
    s = adapter.stats()
    assert s["total_chunks"] == 2
    assert s["sources"]      == {"x": 1, "y": 1}
