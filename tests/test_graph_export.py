"""Tests for EpistemicGraph export methods: to_networkx, to_cypher, to_neo4j."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from prism.graph import EpistemicGraph
from prism.edges import EpistemicEdgeType


@pytest.fixture()
def small_graph():
    g = EpistemicGraph()
    g.add_node("a", source="doc1.pdf", page=1, section="Intro", text_preview="Alpha text")
    g.add_node("b", source="doc1.pdf", page=2, section="Body",  text_preview="Beta text")
    g.add_node("c", source="doc2.pdf", page=5, section="Conc",  text_preview="Gamma text")
    g.add_edge("a", "b", EpistemicEdgeType.SUPPORTS,   confidence=0.9)
    g.add_edge("b", "c", EpistemicEdgeType.REFUTES,    confidence=0.7)
    g.add_edge("a", "c", EpistemicEdgeType.SUPERSEDES, confidence=0.8)
    return g


# ── to_networkx ───────────────────────────────────────────────────────────────

def test_to_networkx_returns_copy(small_graph):
    import networkx as nx
    G = small_graph.to_networkx()
    assert isinstance(G, nx.MultiDiGraph)


def test_to_networkx_node_count(small_graph):
    G = small_graph.to_networkx()
    assert G.number_of_nodes() == 3


def test_to_networkx_edge_count(small_graph):
    G = small_graph.to_networkx()
    assert G.number_of_edges() == 3


def test_to_networkx_is_independent_copy(small_graph):
    G = small_graph.to_networkx()
    G.remove_node("a")
    # Original graph must be unchanged
    assert small_graph.node_count() == 3


def test_to_networkx_preserves_node_attrs(small_graph):
    G = small_graph.to_networkx()
    assert G.nodes["a"]["source"] == "doc1.pdf"
    assert G.nodes["a"]["page"] == 1


def test_to_networkx_preserves_edge_attrs(small_graph):
    G = small_graph.to_networkx()
    edges = list(G.edges("a", data=True))
    types = {d["type"] for _, _, d in edges}
    assert "supports" in types


def test_to_networkx_usable_with_pagerank(small_graph):
    import networkx as nx
    G = small_graph.to_networkx()
    pr = nx.pagerank(G, weight="weight")
    assert set(pr.keys()) == {"a", "b", "c"}


# ── to_cypher ─────────────────────────────────────────────────────────────────

def test_to_cypher_creates_file(small_graph, tmp_path):
    out = tmp_path / "graph.cypher"
    result = small_graph.to_cypher(out)
    assert result == out
    assert out.exists()


def test_to_cypher_contains_create_nodes(small_graph, tmp_path):
    out = tmp_path / "graph.cypher"
    small_graph.to_cypher(out)
    text = out.read_text()
    assert "CREATE (:Chunk" in text
    assert "'a'" in text
    assert "'doc1.pdf'" in text


def test_to_cypher_contains_relationships(small_graph, tmp_path):
    out = tmp_path / "graph.cypher"
    small_graph.to_cypher(out)
    text = out.read_text()
    assert ":SUPPORTS" in text
    assert ":REFUTES" in text
    assert ":SUPERSEDES" in text


def test_to_cypher_contains_index(small_graph, tmp_path):
    out = tmp_path / "graph.cypher"
    small_graph.to_cypher(out)
    text = out.read_text()
    assert "CREATE INDEX" in text
    assert "chunk_id" in text


def test_to_cypher_transactions_batched(small_graph, tmp_path):
    out = tmp_path / "graph.cypher"
    small_graph.to_cypher(out, batch_size=2)
    text = out.read_text()
    # 3 nodes with batch_size=2 → at least 2 :begin/:commit blocks for nodes
    assert text.count(":begin") >= 2


def test_to_cypher_escapes_single_quotes(tmp_path):
    g = EpistemicGraph()
    g.add_node("x", source="it's a doc.pdf", page=1, section="", text_preview="")
    out = tmp_path / "graph.cypher"
    g.to_cypher(out)
    text = out.read_text()
    assert "it\\'s a doc.pdf" in text


def test_to_cypher_returns_path_object(small_graph, tmp_path):
    out = tmp_path / "graph.cypher"
    result = small_graph.to_cypher(str(out))   # accepts str too
    assert isinstance(result, Path)


# ── to_neo4j ──────────────────────────────────────────────────────────────────

def _make_mock_driver():
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


def test_to_neo4j_raises_without_driver(small_graph):
    with patch.dict("sys.modules", {"neo4j": None}):
        with pytest.raises(ImportError, match="neo4j"):
            small_graph.to_neo4j("bolt://localhost:7687", "neo4j", "secret")


def test_to_neo4j_calls_driver_with_uri(small_graph):
    mock_driver, _ = _make_mock_driver()
    neo4j_mod = MagicMock()
    neo4j_mod.GraphDatabase.driver.return_value = mock_driver
    with patch.dict("sys.modules", {"neo4j": neo4j_mod}):
        small_graph.to_neo4j("bolt://localhost:7687", "neo4j", "secret")
    neo4j_mod.GraphDatabase.driver.assert_called_once_with(
        "bolt://localhost:7687", auth=("neo4j", "secret")
    )


def test_to_neo4j_returns_counts(small_graph):
    mock_driver, _ = _make_mock_driver()
    neo4j_mod = MagicMock()
    neo4j_mod.GraphDatabase.driver.return_value = mock_driver
    with patch.dict("sys.modules", {"neo4j": neo4j_mod}):
        result = small_graph.to_neo4j("bolt://localhost:7687", "neo4j", "secret")
    assert result["nodes_created"] == 3
    assert result["edges_created"] == 3


def test_to_neo4j_clear_existing_runs_delete(small_graph):
    mock_driver, session = _make_mock_driver()
    neo4j_mod = MagicMock()
    neo4j_mod.GraphDatabase.driver.return_value = mock_driver
    with patch.dict("sys.modules", {"neo4j": neo4j_mod}):
        small_graph.to_neo4j(
            "bolt://localhost:7687", "neo4j", "secret", clear_existing=True
        )
    calls = [str(c) for c in session.run.call_args_list]
    assert any("DETACH DELETE" in c for c in calls)


def test_to_neo4j_closes_driver(small_graph):
    mock_driver, _ = _make_mock_driver()
    neo4j_mod = MagicMock()
    neo4j_mod.GraphDatabase.driver.return_value = mock_driver
    with patch.dict("sys.modules", {"neo4j": neo4j_mod}):
        small_graph.to_neo4j("bolt://localhost:7687", "neo4j", "secret")
    mock_driver.close.assert_called_once()
