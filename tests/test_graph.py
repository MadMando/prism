"""
Tests for prism.graph — EpistemicGraph construction, traversal, and serialisation.
"""

import os
import tempfile

import pytest
from prism.graph import EpistemicGraph
from prism.edges import EpistemicEdgeType, PROPAGATION_WEIGHTS


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_triangle_graph():
    """A → B (supports), B → C (refutes). Three nodes, two edges."""
    g = EpistemicGraph()
    g.add_node("a", source="doc-one",   page=1,  section="1.1", text_preview="Alpha text")
    g.add_node("b", source="doc-two",   page=5,  section="2.0", text_preview="Beta text")
    g.add_node("c", source="doc-three", page=12, section="3.0", text_preview="Gamma text")
    g.add_edge("a", "b", EpistemicEdgeType.SUPPORTS,  confidence=0.9, rationale="A supports B")
    g.add_edge("b", "c", EpistemicEdgeType.REFUTES,   confidence=0.75)
    return g


# ── node management ───────────────────────────────────────────────────────────

def test_add_node_basic():
    g = EpistemicGraph()
    g.add_node("x")
    assert g.has_node("x")
    assert g.node_count() == 1


def test_add_node_with_metadata():
    g = EpistemicGraph()
    g.add_node("x", source="my-doc", page=42, section="4.2", text_preview="Some preview text")
    assert g.has_node("x")


def test_add_multiple_nodes():
    g = EpistemicGraph()
    for i in range(10):
        g.add_node(f"node_{i}")
    assert g.node_count() == 10


def test_has_node_false_for_missing():
    g = EpistemicGraph()
    assert not g.has_node("nonexistent")


def test_text_preview_truncated_to_200():
    g = EpistemicGraph()
    long_text = "x" * 500
    g.add_node("x", text_preview=long_text)
    stored = g._g.nodes["x"].get("text_preview", "")
    assert len(stored) <= 200


# ── edge management ───────────────────────────────────────────────────────────

def test_add_edge_creates_missing_nodes():
    g = EpistemicGraph()
    g.add_edge("x", "y", EpistemicEdgeType.SUPPORTS)
    assert g.has_node("x")
    assert g.has_node("y")
    assert g.edge_count() == 1


def test_edge_weight_scaled_by_confidence():
    g = EpistemicGraph()
    g.add_edge("a", "b", EpistemicEdgeType.SUPPORTS, confidence=0.5)
    neighbors = list(g.neighbors("a"))
    assert len(neighbors) == 1
    _, _, weight, _ = neighbors[0]
    expected = PROPAGATION_WEIGHTS[EpistemicEdgeType.SUPPORTS] * 0.5
    assert abs(weight - expected) < 1e-6


def test_multiple_edges_between_same_nodes():
    """MultiDiGraph allows multiple edge types between the same pair."""
    g = EpistemicGraph()
    g.add_edge("a", "b", EpistemicEdgeType.SUPPORTS)
    g.add_edge("a", "b", EpistemicEdgeType.EXEMPLIFIES)
    assert g.edge_count() == 2


def test_has_edge():
    g = make_triangle_graph()
    assert g.has_edge("a", "b")
    assert not g.has_edge("a", "c")


# ── traversal ─────────────────────────────────────────────────────────────────

def test_neighbors_returns_correct_nodes():
    g = make_triangle_graph()
    neighbors = list(g.neighbors("a"))
    assert len(neighbors) == 1
    nbr_id, etype, weight, rationale = neighbors[0]
    assert nbr_id == "b"
    assert etype == EpistemicEdgeType.SUPPORTS
    assert rationale == "A supports B"


def test_neighbors_of_isolated_node_is_empty():
    g = EpistemicGraph()
    g.add_node("lone")
    assert list(g.neighbors("lone")) == []


def test_incoming_edges():
    g = make_triangle_graph()
    incoming = list(g.incoming("b"))
    assert len(incoming) == 1
    src_id, etype, _, _ = incoming[0]
    assert src_id == "a"
    assert etype == EpistemicEdgeType.SUPPORTS


def test_incoming_empty_for_root_node():
    g = make_triangle_graph()
    assert list(g.incoming("a")) == []


def test_neighbors_missing_node_is_empty():
    g = EpistemicGraph()
    assert list(g.neighbors("ghost")) == []


# ── serialisation round-trips ─────────────────────────────────────────────────

def test_save_load_roundtrip_gzip():
    g = make_triangle_graph()
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
        path = f.name
    try:
        g.save(path)
        assert os.path.exists(path)
        g2 = EpistemicGraph.load(path)
        assert g2.node_count() == g.node_count()
        assert g2.edge_count() == g.edge_count()
    finally:
        os.unlink(path)


def test_save_load_roundtrip_plain_json():
    g = make_triangle_graph()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        g.save(path, compress=False)
        g2 = EpistemicGraph.load(path)
        assert g2.node_count() == g.node_count()
        assert g2.edge_count() == g.edge_count()
    finally:
        os.unlink(path)


def test_load_preserves_edge_types():
    g = make_triangle_graph()
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
        path = f.name
    try:
        g.save(path)
        g2 = EpistemicGraph.load(path)
        nbrs = list(g2.neighbors("a"))
        assert len(nbrs) == 1
        assert nbrs[0][1] == EpistemicEdgeType.SUPPORTS
    finally:
        os.unlink(path)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        EpistemicGraph.load("/tmp/definitely_does_not_exist_prism.json.gz")


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_counts():
    g = make_triangle_graph()
    stats = g.stats()
    assert stats["n_nodes"] == 3
    assert stats["n_edges"] == 2


def test_stats_edge_types():
    g = make_triangle_graph()
    stats = g.stats()
    assert "supports" in stats["edge_types"]
    assert "refutes" in stats["edge_types"]
    assert stats["edge_types"]["supports"] == 1
    assert stats["edge_types"]["refutes"] == 1


def test_repr():
    g = make_triangle_graph()
    r = repr(g)
    assert "3" in r
    assert "2" in r
