"""
prism.inspect_cli
-----------------
Diagnostic CLI tools for inspecting PRISM graphs and LanceDB stores.

Commands
--------
prism-stats  <graph-path> [--lancedb-path PATH]
    Print a summary of the graph and optionally the vector store.

prism-inspect <graph-path> --node <node-id>
    Inspect a specific node: its edges, neighbours, and metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── prism-stats ───────────────────────────────────────────────────────────────

def stats_main() -> None:
    """Entry point for `prism-stats`."""
    parser = argparse.ArgumentParser(
        prog="prism-stats",
        description="Print summary statistics for a PRISM epistemic graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph_path", help="Path to prism_graph.json.gz")
    parser.add_argument("--lancedb-path", default=None,
                        help="Optional: also print LanceDB store stats")
    parser.add_argument("--table-name", default="knowledge",
                        help="LanceDB table name (used with --lancedb-path)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable text")
    args = parser.parse_args()

    from .graph import EpistemicGraph

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"[prism-stats] ERROR: graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    g = EpistemicGraph.load(graph_path)
    stats = g.stats()

    output: dict = {"graph": stats}

    if args.lancedb_path:
        try:
            from .adapters.lancedb import LanceDBAdapter
            adapter = LanceDBAdapter(
                db_path    = args.lancedb_path,
                table_name = args.table_name,
            )
            adapter.connect()
            output["lancedb"] = adapter.stats()
        except Exception as exc:
            output["lancedb_error"] = str(exc)

    if args.json:
        print(json.dumps(output, indent=2, default=str))
        return

    # Human-readable output
    g_stats = output["graph"]
    print()
    print("━" * 50)
    print("  PRISM Graph Statistics")
    print("━" * 50)
    print(f"  File     : {graph_path}")
    print(f"  Nodes    : {g_stats['n_nodes']:,}")
    print(f"  Edges    : {g_stats['n_edges']:,}")
    print(f"  Avg deg  : {g_stats['avg_out_degree']:.2f} edges/node")
    print()
    print("  Edge type distribution:")
    edge_types = sorted(g_stats["edge_types"].items(), key=lambda x: -x[1])
    total_edges = g_stats["n_edges"] or 1
    for etype, count in edge_types:
        bar_len = int(count / total_edges * 30)
        bar = "█" * bar_len
        print(f"    {etype:<20} {count:>6,}  {count/total_edges*100:5.1f}%  {bar}")
    print()
    meta = g_stats.get("meta", {})
    if meta:
        print("  Build metadata:")
        for k, v in meta.items():
            if k in ("saved_at", "n_nodes", "n_edges"):
                continue
            print(f"    {k:<24} {v}")
    print()

    if "lancedb" in output:
        lb = output["lancedb"]
        print("  LanceDB store:")
        print(f"    Total chunks : {lb.get('total_chunks', '?'):,}")
        print(f"    Embed model  : {lb.get('embed_model', '?')}")
        print(f"    Table        : {lb.get('table', '?')}")
        sources = lb.get("sources", {})
        if sources:
            print("    Sources:")
            for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                print(f"      {src:<40} {count:>6,}")
        print()

    if "lancedb_error" in output:
        print(f"  LanceDB error: {output['lancedb_error']}", file=sys.stderr)

    print("━" * 50)
    print()


# ── prism-inspect ─────────────────────────────────────────────────────────────

def inspect_main() -> None:
    """Entry point for `prism-inspect`."""
    parser = argparse.ArgumentParser(
        prog="prism-inspect",
        description="Inspect a node in a PRISM epistemic graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph_path", help="Path to prism_graph.json.gz")
    parser.add_argument("--node", required=True,
                        help="Node ID to inspect")
    parser.add_argument("--max-edges", type=int, default=20,
                        help="Maximum edges to show per direction")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    from .graph import EpistemicGraph

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"[prism-inspect] ERROR: graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    g = EpistemicGraph.load(graph_path)

    if not g.has_node(args.node):
        print(f"[prism-inspect] ERROR: node '{args.node}' not found in graph", file=sys.stderr)
        # Suggest partial matches
        all_nodes = list(g._g.nodes())
        matches = [n for n in all_nodes if args.node.lower() in n.lower()][:5]
        if matches:
            print(f"[prism-inspect] Similar node IDs: {matches}", file=sys.stderr)
        sys.exit(1)

    node_attrs = dict(g._g.nodes[args.node])

    outgoing = list(g.neighbors(args.node))[:args.max_edges]
    incoming = list(g.incoming(args.node))[:args.max_edges]

    if args.json:
        print(json.dumps({
            "node_id": args.node,
            "attrs": node_attrs,
            "outgoing": [
                {"to": nbr, "type": et.value, "weight": w, "rationale": r}
                for nbr, et, w, r in outgoing
            ],
            "incoming": [
                {"from": src, "type": et.value, "weight": w, "rationale": r}
                for src, et, w, r in incoming
            ],
        }, indent=2))
        return

    # Human-readable
    print()
    print("━" * 60)
    print(f"  Node: {args.node}")
    print("━" * 60)
    for k, v in node_attrs.items():
        print(f"  {k:<16}: {v}")
    print()

    if outgoing:
        print(f"  Outgoing edges ({len(outgoing)}):")
        for nbr, etype, weight, rationale in outgoing:
            rat = f" — {rationale[:60]}" if rationale else ""
            print(f"    → {etype.value:<20} {nbr:<40} w={weight:.2f}{rat}")
    else:
        print("  Outgoing edges: none")

    print()

    if incoming:
        print(f"  Incoming edges ({len(incoming)}):")
        for src, etype, weight, rationale in incoming:
            rat = f" — {rationale[:60]}" if rationale else ""
            print(f"    ← {etype.value:<20} {src:<40} w={weight:.2f}{rat}")
    else:
        print("  Incoming edges: none")

    print()
    print("━" * 60)
    print()


if __name__ == "__main__":
    stats_main()
