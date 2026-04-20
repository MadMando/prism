"""
prism.viz_cli
-------------
prism-viz — export a PRISM epistemic graph to Gephi (GEXF) or D3 (JSON)
for visualisation and exploration.

Usage
-----
    # Export to Gephi GEXF
    prism-viz graph.json.gz --format gexf --output graph.gexf

    # Export to D3 JSON (force-directed graph)
    prism-viz graph.json.gz --format d3 --output graph.json

    # Filter: only include specific edge types
    prism-viz graph.json.gz --format d3 --edge-types supports,refutes

    # Filter: only high-confidence edges
    prism-viz graph.json.gz --format gexf --min-confidence 0.8

    # Filter: only nodes from a specific source
    prism-viz graph.json.gz --format d3 --source-filter "dmbok"

    # Cap size for large graphs (sample by degree centrality)
    prism-viz graph.json.gz --format d3 --max-nodes 500

D3 JSON format
--------------
    {
      "nodes": [{"id": "...", "source": "...", "page": 1, "section": "...",
                 "group": "source-name", "degree": 4}],
      "links": [{"source": "...", "target": "...", "type": "supports",
                 "weight": 0.85, "confidence": 0.9}]
    }

    Load in D3 with d3.forceSimulation — colour nodes by "group", scale
    link opacity by "weight".

GEXF format
-----------
    Standard Gephi GEXF — open directly in Gephi for layout and analysis.
    Edge type, weight, confidence, and rationale are exported as attributes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_subgraph(graph, edge_types=None, min_confidence=0.0, source_filter=None):
    """Return a filtered copy of the underlying networkx graph."""
    import networkx as nx

    g = graph._g
    sub = nx.MultiDiGraph()

    for node_id, attrs in g.nodes(data=True):
        src = attrs.get("source", "")
        if source_filter and source_filter.lower() not in src.lower():
            continue
        sub.add_node(node_id, **attrs)

    for u, v, data in g.edges(data=True):
        if u not in sub.nodes or v not in sub.nodes:
            continue
        etype = data.get("type", "")
        conf  = float(data.get("confidence", 1.0))
        if edge_types and etype not in edge_types:
            continue
        if conf < min_confidence:
            continue
        sub.add_edge(u, v, **data)

    return sub


def _sample_by_degree(sub, max_nodes: int):
    """Keep the top-N nodes by total degree."""
    import networkx as nx
    degree = dict(sub.degree())
    top_ids = set(sorted(degree, key=lambda n: -degree[n])[:max_nodes])
    return sub.subgraph(top_ids).copy()


def _export_gexf(sub, output_path: Path) -> None:
    import networkx as nx

    # GEXF doesn't allow "type" as an attribute name — rename to edge_type
    renamed = nx.MultiDiGraph()
    for node_id, attrs in sub.nodes(data=True):
        renamed.add_node(node_id, **attrs)
    for u, v, data in sub.edges(data=True):
        edge_attrs = {
            "edge_type":  data.get("type", ""),
            "weight":     float(data.get("weight", 0.5)),
            "confidence": float(data.get("confidence", 1.0)),
            "rationale":  str(data.get("rationale", "")),
        }
        renamed.add_edge(u, v, **edge_attrs)

    nx.write_gexf(renamed, str(output_path))
    print(f"[prism-viz] GEXF written → {output_path}")
    print(f"[prism-viz] {renamed.number_of_nodes():,} nodes, {renamed.number_of_edges():,} edges")
    print(f"[prism-viz] Open in Gephi: File → Open → {output_path}")


def _export_d3(sub, output_path: Path) -> None:
    degree = dict(sub.degree())

    nodes = []
    for node_id, attrs in sub.nodes(data=True):
        nodes.append({
            "id":      node_id,
            "source":  attrs.get("source", ""),
            "page":    attrs.get("page", 0),
            "section": attrs.get("section", ""),
            "preview": attrs.get("text_preview", "")[:120],
            "group":   attrs.get("source", "unknown"),
            "degree":  degree.get(node_id, 0),
        })

    links = []
    for u, v, data in sub.edges(data=True):
        links.append({
            "source":     u,
            "target":     v,
            "type":       data.get("type", ""),
            "weight":     round(float(data.get("weight", 0.5)), 4),
            "confidence": round(float(data.get("confidence", 1.0)), 4),
        })

    payload = {"nodes": nodes, "links": links}

    if str(output_path) == "-":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[prism-viz] D3 JSON written → {output_path}")
        print(f"[prism-viz] {len(nodes):,} nodes, {len(links):,} links")
        print(f"[prism-viz] Load with: d3.json('{output_path}').then(data => ...)")


def viz_main() -> None:
    """Entry point for `prism-viz`."""
    parser = argparse.ArgumentParser(
        prog="prism-viz",
        description="Export a PRISM epistemic graph for visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph_path",
                        help="Path to prism_graph.json.gz")
    parser.add_argument("--format", choices=["gexf", "d3"], default="d3",
                        help="Output format: 'gexf' (Gephi) or 'd3' (D3.js JSON)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path. Defaults to graph.<format>. Use '-' for stdout (d3 only).")
    parser.add_argument("--edge-types", default=None,
                        help="Comma-separated edge types to include, e.g. supports,refutes,supersedes")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum edge confidence to include (0.0–1.0)")
    parser.add_argument("--source-filter", default=None,
                        help="Only include nodes whose source contains this substring")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Cap node count by keeping top-N highest-degree nodes")

    args = parser.parse_args()

    from .graph import EpistemicGraph

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"[prism-viz] ERROR: graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    g = EpistemicGraph.load(graph_path)

    edge_types = None
    if args.edge_types:
        edge_types = {e.strip() for e in args.edge_types.split(",")}

    sub = _build_subgraph(
        g,
        edge_types     = edge_types,
        min_confidence = args.min_confidence,
        source_filter  = args.source_filter,
    )

    if args.max_nodes and sub.number_of_nodes() > args.max_nodes:
        print(f"[prism-viz] sampling top {args.max_nodes} nodes by degree ...")
        sub = _sample_by_degree(sub, args.max_nodes)

    print(f"[prism-viz] exporting {sub.number_of_nodes():,} nodes, {sub.number_of_edges():,} edges ...")

    # Determine output path
    if args.output == "-":
        output_path = Path("-")
    elif args.output:
        output_path = Path(args.output)
    else:
        stem = graph_path.name.replace(".json.gz", "").replace(".json", "")
        ext  = "gexf" if args.format == "gexf" else "json"
        output_path = graph_path.parent / f"{stem}.{ext}"

    if args.format == "gexf":
        if str(output_path) == "-":
            print("[prism-viz] ERROR: GEXF cannot be written to stdout", file=sys.stderr)
            sys.exit(1)
        _export_gexf(sub, output_path)
    else:
        _export_d3(sub, output_path)


if __name__ == "__main__":
    viz_main()
