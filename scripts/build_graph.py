#!/usr/bin/env python3
"""
scripts/build_graph.py
----------------------
CLI tool to build (or rebuild) a PRISM epistemic graph from an existing
LanceDB vector store.

Usage:
    python scripts/build_graph.py \
        --lancedb-path  /path/to/lancedb \
        --graph-path    /path/to/prism_graph.json.gz \
        --llm-api-key   sk-... \
        --max-pairs     50000

    # With a local Ollama model instead of DeepSeek:
    python scripts/build_graph.py \
        --lancedb-path  /path/to/lancedb \
        --graph-path    /path/to/prism_graph.json.gz \
        --llm-base-url  http://localhost:11434/v1 \
        --llm-model     gemma4:27b \
        --max-pairs     50000 \
        --force

Or install prism-rag and use the entry point:
    prism-build --lancedb-path ... --graph-path ... --llm-api-key ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PRISM epistemic graph from a LanceDB vector store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--lancedb-path", required=True,
        help="Path to the LanceDB directory",
    )
    parser.add_argument(
        "--graph-path", required=True,
        help="Output path for the epistemic graph (e.g. prism_graph.json.gz)",
    )

    # ── LanceDB / embedding ───────────────────────────────────────────────────
    parser.add_argument(
        "--table-name", default="knowledge",
        help="LanceDB table name",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama API base URL (used for embeddings)",
    )
    parser.add_argument(
        "--embed-model", default="qwen3-embedding:4b",
        help="Embedding model name (must match the model used at ingest time)",
    )

    # ── LLM extraction ────────────────────────────────────────────────────────
    parser.add_argument(
        "--llm-base-url", default="https://api.deepseek.com",
        help="OpenAI-compatible API base URL for epistemic extraction",
    )
    parser.add_argument(
        "--llm-model", default="deepseek-chat",
        help="LLM model name for epistemic extraction",
    )
    parser.add_argument(
        "--llm-api-key", default="",
        help="API key for the LLM provider",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.65,
        help="Minimum confidence score to include an extracted edge",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Number of chunk pairs per LLM extraction call",
    )

    # ── Graph building ────────────────────────────────────────────────────────
    parser.add_argument(
        "--k-neighbors", type=int, default=8,
        help="Semantic neighbours per chunk to consider for candidate pairs",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap total candidate pairs (useful for testing; default: no limit)",
    )
    parser.add_argument(
        "--all-sources", action="store_true", default=False,
        help="Include same-source pairs (default: cross-source only)",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Rebuild even if graph-path already exists",
    )

    args = parser.parse_args()

    # ── Lazy import (allows --help without installing the package) ────────────
    try:
        from prism import PRISM
    except ImportError:
        # Try adding parent directory to path (running directly from repo)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from prism import PRISM
        except ImportError as e:
            print(f"[prism-build] ERROR: Cannot import prism: {e}")
            print(f"[prism-build] Install with: pip install -e /path/to/prism")
            sys.exit(1)

    graph_path = Path(args.graph_path)
    if graph_path.suffix not in (".gz", ".json"):
        graph_path = graph_path.with_suffix(graph_path.suffix + ".json.gz")

    p = PRISM(
        lancedb_path   = args.lancedb_path,
        graph_path     = graph_path,
        table_name     = args.table_name,
        ollama_url     = args.ollama_url,
        embed_model    = args.embed_model,
        llm_base_url   = args.llm_base_url,
        llm_model      = args.llm_model,
        llm_api_key    = args.llm_api_key,
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
    if "graph" in stats:
        g = stats["graph"]
        print(f"\n[prism-build] Graph saved to: {args.graph_path}")
        print(f"[prism-build]   nodes : {g.get('n_nodes', '?'):,}")
        print(f"[prism-build]   edges : {g.get('n_edges', '?'):,}")
        print(f"[prism-build]   types : {g.get('edge_types', {})}")
    else:
        print(f"\n[prism-build] Graph saved to: {args.graph_path}")


if __name__ == "__main__":
    main()
