"""
Tests for prism.retriever — fallback behaviour, bucketing, top_k enforcement.

All tests mock the LanceDB adapter so no vector store is required.
"""

import pytest
from unittest.mock import MagicMock, patch

from prism.retriever import PRISMRetriever
from prism.graph import EpistemicGraph
from prism.edges import EpistemicEdgeType


# ── helpers ───────────────────────────────────────────────────────────────────

def make_adapter(seed_scores=None, chunks=None):
    """Return a mock LanceDBAdapter with configurable responses."""
    adapter = MagicMock()
    adapter.seed_scores.return_value = seed_scores if seed_scores is not None else {
        "chunk_a": 0.92,
        "chunk_b": 0.85,
        "chunk_c": 0.78,
    }
    adapter.get_chunks.return_value = chunks or {
        "chunk_a": {"id": "chunk_a", "source": "doc-1", "page": 1,  "section": "1.0", "text": "Chunk A text"},
        "chunk_b": {"id": "chunk_b", "source": "doc-2", "page": 5,  "section": "2.0", "text": "Chunk B text"},
        "chunk_c": {"id": "chunk_c", "source": "doc-1", "page": 10, "section": "3.0", "text": "Chunk C text"},
    }
    return adapter


def make_graph_with_edges():
    """Small graph: chunk_a → chunk_b (supports), chunk_b → chunk_c (refutes)."""
    g = EpistemicGraph()
    g.add_node("chunk_a", source="doc-1", page=1, section="1.0")
    g.add_node("chunk_b", source="doc-2", page=5, section="2.0")
    g.add_node("chunk_c", source="doc-1", page=10, section="3.0")
    g.add_edge("chunk_a", "chunk_b", EpistemicEdgeType.SUPPORTS, confidence=0.9)
    g.add_edge("chunk_b", "chunk_c", EpistemicEdgeType.REFUTES,  confidence=0.8)
    return g


# ── fallback (no graph) ───────────────────────────────────────────────────────

def test_fallback_returns_epistemic_result():
    from prism.result import EpistemicResult
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("test query", top_k=3)
    assert isinstance(result, EpistemicResult)


def test_fallback_query_preserved():
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("my specific query", top_k=3)
    assert result.query == "my specific query"


def test_fallback_graph_was_not_used():
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("test query", top_k=3)
    assert not result.graph_was_used


def test_fallback_seeds_land_in_primary():
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("test query", top_k=5)
    assert len(result.primary) > 0
    assert all(c.is_seed for c in result.primary)


def test_fallback_empty_seeds_returns_empty_result():
    retriever = PRISMRetriever(adapter=make_adapter(seed_scores={}), graph=None)
    result = retriever.retrieve("test query", top_k=5)
    assert result.n_seeds == 0
    assert len(result.primary) == 0


def test_fallback_top_k_respected():
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("test query", top_k=1)
    assert len(result.primary) <= 1


# ── fallback with empty graph ─────────────────────────────────────────────────

def test_empty_graph_falls_back_to_vector():
    empty_graph = EpistemicGraph()  # no nodes
    retriever = PRISMRetriever(adapter=make_adapter(), graph=empty_graph)
    result = retriever.retrieve("test query", top_k=3)
    assert not result.graph_was_used


# ── with graph ────────────────────────────────────────────────────────────────

def test_graph_used_when_loaded():
    g = make_graph_with_edges()
    retriever = PRISMRetriever(adapter=make_adapter(), graph=g)
    result = retriever.retrieve("test query", top_k=5)
    assert result.graph_was_used


def test_n_seeds_reported():
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("test query", top_k=5)
    assert result.n_seeds == 3  # 3 seed scores in mock


def test_persona_passed_through():
    retriever = PRISMRetriever(adapter=make_adapter(), graph=None)
    result = retriever.retrieve("test query", top_k=3, persona="data engineer")
    assert result.persona == "data engineer"


def test_adapter_called_with_correct_query():
    adapter = make_adapter()
    retriever = PRISMRetriever(adapter=adapter, graph=None)
    retriever.retrieve("my query text", top_k=3)
    adapter.seed_scores.assert_called_once()
    call_args = adapter.seed_scores.call_args
    assert "my query text" in call_args[0] or call_args[1].get("query") == "my query text"


def test_source_filter_passed_to_adapter():
    adapter = make_adapter()
    retriever = PRISMRetriever(adapter=adapter, graph=None)
    retriever.retrieve("query", top_k=3, source_filter="doc-1")
    call_kwargs = adapter.seed_scores.call_args[1]
    assert call_kwargs.get("source_filter") == "doc-1"


# ── missing chunk data handled gracefully ─────────────────────────────────────

def test_missing_chunk_data_skipped():
    """If get_chunks doesn't return data for a node ID, it should be silently skipped."""
    adapter = make_adapter(chunks={
        "chunk_a": {"id": "chunk_a", "source": "doc-1", "page": 1, "section": "1.0", "text": "Text A"},
        # chunk_b and chunk_c intentionally missing
    })
    retriever = PRISMRetriever(adapter=adapter, graph=None)
    result = retriever.retrieve("test query", top_k=5)
    # Should not crash; only chunk_a returned
    ids = [c.id for c in result.all_chunks]
    assert "chunk_a" in ids
    assert "chunk_b" not in ids
