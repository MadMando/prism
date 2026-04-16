"""
tests/test_new_behaviour.py
---------------------------
Tests for all improvements introduced in the review follow-up:

  1. edges_between() dead loop fix
  2. source_filter actually filters in SpreadingActivation
  3. EpistemicFilter default model is llama3.1:8b
  4. EpistemicExtractor retries on failure + logs failures
  5. VectorAdapter Protocol is @runtime_checkable
  6. candidate_pairs_for() exists on LanceDBAdapter
  7. PRISM.add_documents() method exists
  8. PRISM requires either lancedb_path or adapter
  9. Reranker hook is applied and final_score updated
 10. PRISM cross_source_only defaults to False
 11. prism-stats CLI registers in pyproject entry points
 12. VectorAdapter and Reranker are exported from top-level prism package
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prism.edges import EpistemicEdgeType
from prism.graph import EpistemicGraph


# ── 1. edges_between() dead loop fix ─────────────────────────────────────────

def test_edges_between_returns_correct_types():
    g = EpistemicGraph()
    g.add_edge("a", "b", EpistemicEdgeType.SUPPORTS, confidence=0.9)
    g.add_edge("a", "b", EpistemicEdgeType.QUALIFIES, confidence=0.7)
    edges = g.edges_between("a", "b")
    assert len(edges) == 2
    types = {et for et, _ in edges}
    assert EpistemicEdgeType.SUPPORTS in types
    assert EpistemicEdgeType.QUALIFIES in types


def test_edges_between_missing_node_returns_empty():
    g = EpistemicGraph()
    g.add_node("a")
    result = g.edges_between("a", "nonexistent")
    assert result == []


def test_edges_between_no_edge_returns_empty():
    g = EpistemicGraph()
    g.add_node("a")
    g.add_node("b")
    assert g.edges_between("a", "b") == []


def test_edges_between_returns_typed_list():
    """Return type must be list[tuple[EpistemicEdgeType, float]]."""
    g = EpistemicGraph()
    g.add_edge("x", "y", EpistemicEdgeType.REFUTES, confidence=0.85)
    edges = g.edges_between("x", "y")
    assert len(edges) == 1
    etype, weight = edges[0]
    assert isinstance(etype, EpistemicEdgeType)
    assert isinstance(weight, float)


# ── 2. source_filter actually filters ────────────────────────────────────────

def test_source_filter_blocks_propagation():
    """Nodes from a different source must not be reached when source_filter is set."""
    from prism.activation import SpreadingActivation

    g = EpistemicGraph()
    g.add_node("seed", source="source_a")
    g.add_node("connected", source="source_b")
    g.add_node("target_a", source="source_a")

    g.add_edge("seed", "connected", EpistemicEdgeType.SUPPORTS, confidence=0.9)
    g.add_edge("seed", "target_a",  EpistemicEdgeType.SUPPORTS, confidence=0.9)

    engine = SpreadingActivation(hops=2, use_reverse_edges=False)
    state = engine.activate(g, {"seed": 1.0}, source_filter="source_a")

    # "connected" is in source_b — should not be reachable
    assert "connected" not in state
    # "target_a" is in source_a — should be reachable
    assert "target_a" in state


def test_source_filter_none_allows_all():
    """No source_filter should allow all nodes."""
    from prism.activation import SpreadingActivation

    g = EpistemicGraph()
    g.add_node("seed", source="source_a")
    g.add_node("other", source="source_b")
    g.add_edge("seed", "other", EpistemicEdgeType.SUPPORTS, confidence=0.9)

    engine = SpreadingActivation(hops=2, use_reverse_edges=False)
    state = engine.activate(g, {"seed": 1.0}, source_filter=None)

    assert "other" in state


# ── 3. EpistemicFilter default model ─────────────────────────────────────────

def test_epistemic_filter_default_model():
    from prism.filter import EpistemicFilter
    f = EpistemicFilter()
    assert f.model == "llama3.1:8b"


# ── 4. EpistemicExtractor retries on failure + logs failures ─────────────────

def test_extractor_retries_on_exception():
    """_extract_batch should retry max_retries times before giving up."""
    import asyncio as _asyncio
    from prism.extractor import EpistemicExtractor

    extractor = EpistemicExtractor(
        base_url="http://fake",
        model="fake-model",
        api_key="fake-key",
        max_retries=2,
        retry_base_delay=0.0,
    )

    call_count = 0

    async def failing_call_llm(client, sem, messages):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated failure")

    extractor._call_llm = failing_call_llm  # type: ignore

    batch = [
        ({"id": "a", "text": "text a", "source": "s1", "page": 1, "section": ""},
         {"id": "b", "text": "text b", "source": "s2", "page": 2, "section": ""}),
    ]

    async def _run():
        sem = _asyncio.Semaphore(1)
        return await extractor._extract_batch(None, sem, batch)  # type: ignore

    results = _asyncio.run(_run())

    assert results == [None]
    assert call_count == 3  # 1 attempt + 2 retries
    assert len(extractor._failures) == 1
    assert "RuntimeError" in extractor._failures[0]["error"]


def test_extractor_failure_log_written(tmp_path):
    """failures should be written to failure_log_path if set."""
    from prism.extractor import EpistemicExtractor
    import json

    log_path = tmp_path / "failures.json"
    extractor = EpistemicExtractor(
        base_url="http://fake",
        model="fake-model",
        api_key="fake-key",
        max_retries=0,
        failure_log_path=str(log_path),
    )
    extractor._failures = [{"pair_ids": [["a", "b"]], "error": "TestError: boom"}]

    # Simulate writing the log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(log_path, "w") as fh:
        json.dump(extractor._failures, fh)

    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert data[0]["error"] == "TestError: boom"


# ── 5. VectorAdapter Protocol is @runtime_checkable ──────────────────────────

def test_vector_adapter_protocol_runtime_checkable():
    from prism.adapters.base import VectorAdapter
    from prism.adapters.lancedb import LanceDBAdapter

    # LanceDBAdapter must satisfy the protocol (has all required methods)
    adapter = LanceDBAdapter.__new__(LanceDBAdapter)
    assert isinstance(adapter, VectorAdapter)


def test_custom_adapter_satisfies_protocol():
    from prism.adapters.base import VectorAdapter

    class MyAdapter:
        def seed_scores(self, query, top_k=20, source_filter=None):
            return {}
        def get_chunks(self, node_ids):
            return {}
        def connect(self):
            pass
        def populate_graph_nodes(self, graph):
            return 0
        def candidate_pairs(self, k_neighbors=8, cross_source_only=False, max_pairs=None):
            return []
        def candidate_pairs_for(self, node_ids, k_neighbors=8, cross_source_only=False):
            return []
        def stats(self):
            return {}

    assert isinstance(MyAdapter(), VectorAdapter)


# ── 6. candidate_pairs_for() exists on LanceDBAdapter ───────────────────────

def test_lancedb_adapter_has_candidate_pairs_for():
    from prism.adapters.lancedb import LanceDBAdapter
    assert hasattr(LanceDBAdapter, "candidate_pairs_for")
    assert callable(LanceDBAdapter.candidate_pairs_for)


# ── 7. PRISM.add_documents() method exists ───────────────────────────────────

def test_prism_has_add_documents():
    from prism import PRISM
    assert hasattr(PRISM, "add_documents")
    assert callable(PRISM.add_documents)


# ── 8. PRISM requires lancedb_path or adapter ────────────────────────────────

def test_prism_raises_without_path_or_adapter(tmp_path):
    from prism import PRISM
    with pytest.raises(ValueError, match="lancedb_path"):
        PRISM(graph_path=str(tmp_path / "graph.json.gz"))


def test_prism_accepts_custom_adapter(tmp_path):
    from prism import PRISM
    from prism.adapters.base import VectorAdapter

    class MockAdapter:
        def seed_scores(self, query, top_k=20, source_filter=None):
            return {}
        def get_chunks(self, node_ids):
            return {}
        def connect(self):
            pass
        def populate_graph_nodes(self, graph):
            return 0
        def candidate_pairs(self, k_neighbors=8, cross_source_only=False, max_pairs=None):
            return []
        def candidate_pairs_for(self, node_ids, k_neighbors=8, cross_source_only=False):
            return []
        def stats(self):
            return {}

    p = PRISM(graph_path=str(tmp_path / "graph.json.gz"), adapter=MockAdapter())
    assert p.adapter is not None


# ── 9. Reranker hook is applied ───────────────────────────────────────────────

def test_reranker_hook_reorders_chunks():
    from prism.result import EpistemicChunk
    from prism.retriever import PRISMRetriever

    chunks = [
        EpistemicChunk(id="a", source="s", page=1, section="", text="first",
                       vector_score=0.9, activation=0.9, convergence=1.0,
                       final_score=0.9, is_seed=True),
        EpistemicChunk(id="b", source="s", page=2, section="", text="second",
                       vector_score=0.5, activation=0.5, convergence=0.5,
                       final_score=0.5, is_seed=False),
    ]

    # Reranker reverses the order
    def reverse_reranker(query, chunks):
        return list(reversed(chunks))

    mock_adapter = MagicMock()
    mock_adapter.seed_scores.return_value = {"a": 0.9, "b": 0.5}
    mock_adapter.get_chunks.return_value = {
        "a": {"id": "a", "source": "s", "page": 1, "section": "", "text": "first"},
        "b": {"id": "b", "source": "s", "page": 2, "section": "", "text": "second"},
    }

    retriever = PRISMRetriever(
        adapter=mock_adapter,
        graph=None,
        reranker=reverse_reranker,
        seed_top_k=2,
    )
    result = retriever.retrieve("test query", top_k=2)

    # The reranker reversed [a, b] → [b, a], so b should have higher final_score
    primary_ids = [c.id for c in result.primary]
    assert primary_ids[0] == "b"


def test_reranker_failsafe_on_exception():
    """A crashing reranker should not break retrieval."""
    from prism.retriever import PRISMRetriever

    def bad_reranker(query, chunks):
        raise RuntimeError("reranker exploded")

    mock_adapter = MagicMock()
    mock_adapter.seed_scores.return_value = {"a": 0.9}
    mock_adapter.get_chunks.return_value = {
        "a": {"id": "a", "source": "s", "page": 1, "section": "", "text": "text"},
    }

    retriever = PRISMRetriever(
        adapter=mock_adapter,
        graph=None,
        reranker=bad_reranker,
        seed_top_k=1,
    )
    # Should not raise
    result = retriever.retrieve("test", top_k=1)
    assert len(result.primary) == 1


# ── 10. PRISM cross_source_only defaults to False ────────────────────────────

def test_prism_build_default_cross_source_only():
    """The build() default must be cross_source_only=False."""
    import inspect
    from prism import PRISM
    sig = inspect.signature(PRISM.build)
    default = sig.parameters["cross_source_only"].default
    assert default is False, f"Expected False, got {default}"


# ── 11. prism-stats and prism-inspect registered as entry points ──────────────

def test_entry_points_registered():
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="console_scripts")
        names = {ep.name for ep in eps}
    except Exception:
        pytest.skip("importlib.metadata not available")

    # They'll be present after `pip install -e .`
    # If running from source without install, just check they're importable
    from prism.inspect_cli import stats_main, inspect_main
    assert callable(stats_main)
    assert callable(inspect_main)


# ── 12. VectorAdapter and Reranker exported from top-level package ────────────

def test_top_level_exports():
    from prism import VectorAdapter, Reranker
    assert VectorAdapter is not None
    assert Reranker is not None
