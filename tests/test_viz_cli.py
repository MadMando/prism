"""
Smoke tests for prism.viz_cli — exercise the GEXF and D3 exporters against a
small in-memory graph.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from prism.graph import EpistemicGraph
from prism.edges import EpistemicEdgeType
from prism.viz_cli import _build_subgraph, _export_d3, _export_gexf, _sample_by_degree


def _tiny_graph():
    g = EpistemicGraph()
    g.add_node("a", source="doc-1", page=1, section="1.0")
    g.add_node("b", source="doc-2", page=2, section="2.0")
    g.add_node("c", source="doc-1", page=3, section="3.0")
    g.add_edge("a", "b", EpistemicEdgeType.SUPPORTS, confidence=0.9)
    g.add_edge("b", "c", EpistemicEdgeType.REFUTES,  confidence=0.7)
    return g


def test_build_subgraph_no_filter_preserves_all():
    g   = _tiny_graph()
    sub = _build_subgraph(g)
    assert sub.number_of_nodes() == 3
    assert sub.number_of_edges() == 2


def test_build_subgraph_edge_type_filter():
    g   = _tiny_graph()
    sub = _build_subgraph(g, edge_types={"supports"})
    assert sub.number_of_edges() == 1
    for _, _, data in sub.edges(data=True):
        assert data["type"] == "supports"


def test_build_subgraph_min_confidence_filter():
    g   = _tiny_graph()
    sub = _build_subgraph(g, min_confidence=0.8)
    assert sub.number_of_edges() == 1


def test_build_subgraph_source_filter():
    g   = _tiny_graph()
    sub = _build_subgraph(g, source_filter="doc-1")
    assert set(sub.nodes()) == {"a", "c"}


def test_sample_by_degree():
    g   = _tiny_graph()
    sub = _build_subgraph(g)
    sampled = _sample_by_degree(sub, max_nodes=2)
    assert sampled.number_of_nodes() == 2
    # node "b" has highest degree (2), so it must be kept
    assert "b" in sampled.nodes()


def test_export_d3_produces_valid_json(tmp_path):
    g   = _tiny_graph()
    sub = _build_subgraph(g)
    out = tmp_path / "graph.json"
    _export_d3(sub, out)

    data = json.loads(out.read_text())
    assert len(data["nodes"]) == 3
    assert len(data["links"]) == 2
    assert {n["id"] for n in data["nodes"]} == {"a", "b", "c"}
    link_types = {l["type"] for l in data["links"]}
    assert link_types == {"supports", "refutes"}


def test_export_gexf_produces_valid_xml(tmp_path):
    g   = _tiny_graph()
    sub = _build_subgraph(g)
    out = tmp_path / "graph.gexf"
    _export_gexf(sub, out)

    tree = ET.parse(out)
    root = tree.getroot()
    # GEXF root element — namespace-qualified, just assert we can parse it.
    assert root.tag.endswith("gexf")
