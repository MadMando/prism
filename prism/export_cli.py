"""
prism.export_cli
----------------
prism-export — export a PRISM epistemic graph to Neo4j or Cypher.

Usage
-----
    # Write a .cypher file (run with cypher-shell or Neo4j Browser)
    prism-export graph.json.gz --format cypher --output graph.cypher

    # Push directly into a running Neo4j instance
    prism-export graph.json.gz --format neo4j \\
        --uri bolt://localhost:7687 --user neo4j --password secret

    # Clear existing Chunk nodes before import
    prism-export graph.json.gz --format neo4j --clear \\
        --uri bolt://localhost:7687 --user neo4j --password secret

NetworkX
--------
    The graph is directly accessible as a NetworkX MultiDiGraph via Python:

        from prism.graph import EpistemicGraph
        g = EpistemicGraph.load("graph.json.gz")
        G = g.to_networkx()          # nx.MultiDiGraph copy
        import networkx as nx
        pr = nx.pagerank(G, weight="weight")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_main() -> None:
    parser = argparse.ArgumentParser(
        prog="prism-export",
        description="Export a PRISM epistemic graph to Neo4j or Cypher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph_path", help="Path to prism_graph.json.gz")
    parser.add_argument(
        "--format", choices=["cypher", "neo4j"], default="cypher",
        help="'cypher' writes a .cypher script; 'neo4j' pushes via Bolt driver",
    )
    parser.add_argument("--output", "-o", default=None,
                        help="Output .cypher file path (cypher format only)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Nodes/edges per transaction batch")

    neo4j_grp = parser.add_argument_group("Neo4j connection (--format neo4j)")
    neo4j_grp.add_argument("--uri",      default="bolt://localhost:7687")
    neo4j_grp.add_argument("--user",     default="neo4j")
    neo4j_grp.add_argument("--password", default=None)
    neo4j_grp.add_argument("--database", default="neo4j")
    neo4j_grp.add_argument("--clear",    action="store_true",
                           help="Delete all :Chunk nodes before importing")

    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"[prism-export] ERROR: graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    from .graph import EpistemicGraph
    print(f"[prism-export] Loading graph from {graph_path} …")
    g = EpistemicGraph.load(graph_path)
    print(f"[prism-export] {g.node_count():,} nodes, {g.edge_count():,} edges")

    if args.format == "cypher":
        out = Path(args.output) if args.output else graph_path.with_suffix("").with_suffix(".cypher")
        g.to_cypher(out, batch_size=args.batch_size)
        print(f"[prism-export] Cypher script written → {out}")
        print(f"[prism-export] Run with: cypher-shell -u {args.user} -p <password> < {out}")

    else:  # neo4j
        if not args.password:
            print("[prism-export] ERROR: --password is required for --format neo4j", file=sys.stderr)
            sys.exit(1)
        print(f"[prism-export] Connecting to {args.uri} (database: {args.database}) …")
        result = g.to_neo4j(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            batch_size=args.batch_size,
            clear_existing=args.clear,
        )
        print(f"[prism-export] Done — {result['nodes_created']:,} nodes, "
              f"{result['edges_created']:,} relationships created")


if __name__ == "__main__":
    export_main()
