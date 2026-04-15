"""
prism.cli
---------
Entry point for the `prism-build` CLI command.
Delegates to scripts/build_graph.py logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """
    Build (or rebuild) a PRISM epistemic graph from an existing LanceDB.

    Examples
    --------
    prism-build \\
        --lancedb-path  /path/to/lancedb \\
        --graph-path    /path/to/prism_graph.json.gz \\
        --llm-api-key   sk-...

    # Against the OpenClaw governance corpus:
    prism-build \\
        --lancedb-path  /home/mando/.openclaw/knowledge/lancedb \\
        --graph-path    /home/mando/.openclaw/knowledge/prism_graph.json.gz \\
        --llm-api-key   $DEEPSEEK_API_KEY \\
        --max-pairs     50000
    """
    parser = argparse.ArgumentParser(
        prog="prism-build",
        description="Build a PRISM epistemic graph from a LanceDB vector store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument("--lancedb-path", required=True,
                        help="Path to the LanceDB directory")
    parser.add_argument("--graph-path", required=True,
                        help="Output path for the graph (e.g. prism_graph.json.gz)")

    # ── LanceDB / embedding ───────────────────────────────────────────────────
    parser.add_argument("--table-name",  default="knowledge",
                        help="LanceDB table name")
    parser.add_argument("--ollama-url",  default="http://localhost:11434",
                        help="Ollama API base URL (used for embeddings)")
    parser.add_argument("--embed-model", default="qwen3-embedding:4b",
                        help="Embedding model (must match the model used at ingest time)")

    # ── LLM extraction ────────────────────────────────────────────────────────
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com",
                        help="OpenAI-compatible API base URL for epistemic extraction")
    parser.add_argument("--llm-model",    default="deepseek-chat",
                        help="LLM model for epistemic extraction")
    parser.add_argument("--llm-api-key",  default="",
                        help="API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--min-confidence", type=float, default=0.65,
                        help="Minimum confidence score to keep an extracted edge")
    parser.add_argument("--batch-size",   type=int, default=5,
                        help="Chunk pairs per LLM extraction call")

    # ── Graph building ────────────────────────────────────────────────────────
    parser.add_argument("--k-neighbors",  type=int, default=8,
                        help="Semantic neighbours per chunk for candidate pairs")
    parser.add_argument("--max-pairs",    type=int, default=None,
                        help="Cap total candidate pairs (no limit by default)")
    parser.add_argument("--all-sources",  action="store_true", default=False,
                        help="Include same-source pairs (default: cross-source only)")
    parser.add_argument("--force",        action="store_true", default=False,
                        help="Rebuild even if graph-path already exists")

    args = parser.parse_args()

    # ── Resolve API key from environment if not passed ────────────────────────
    import os
    api_key = args.llm_api_key or os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    # ── Auto-fix extension ────────────────────────────────────────────────────
    graph_path = Path(args.graph_path)
    if graph_path.suffix not in (".gz", ".json"):
        graph_path = Path(str(graph_path) + ".json.gz")

    from .prism import PRISM

    p = PRISM(
        lancedb_path   = args.lancedb_path,
        graph_path     = graph_path,
        table_name     = args.table_name,
        ollama_url     = args.ollama_url,
        embed_model    = args.embed_model,
        llm_base_url   = args.llm_base_url,
        llm_model      = args.llm_model,
        llm_api_key    = api_key,
        min_confidence = args.min_confidence,
        batch_size     = args.batch_size,
    )

    p.build(
        k_neighbors       = args.k_neighbors,
        cross_source_only = not args.all_sources,
        max_pairs         = args.max_pairs,
        force             = args.force,
    )

    stats = p.stats()
    print()
    if "graph" in stats:
        g = stats["graph"]
        print(f"[prism-build] ✓ Graph saved to: {graph_path}")
        print(f"[prism-build]   nodes : {g.get('n_nodes', '?'):,}")
        print(f"[prism-build]   edges : {g.get('n_edges', '?'):,}")
        print(f"[prism-build]   types : {g.get('edge_types', {})}")
    else:
        print(f"[prism-build] ✓ Graph saved to: {graph_path}")

    print()
    print("[prism-build] Next steps:")
    print("[prism-build]   from prism import PRISM")
    print(f"[prism-build]   p = PRISM(lancedb_path=..., graph_path={str(graph_path)!r}, ...)")
    print("[prism-build]   p.load_graph()")
    print("[prism-build]   result = p.retrieve('your question')")
    print("[prism-build]   print(result.format_for_llm())")


if __name__ == "__main__":
    main()
