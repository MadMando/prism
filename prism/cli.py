"""
prism.cli
---------
Entry point for the `prism-build` CLI command.
Two-stage async pipeline: local Ollama filter → async LLM extraction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    """
    Build (or rebuild) a PRISM epistemic graph from an existing LanceDB.

    Two-stage pipeline (v2):
      Stage 1 — local Ollama model pre-filters candidate pairs (~50% reduction)
      Stage 2 — async LLM classifies surviving pairs (20x concurrent, batch=20)

    Typical build times (30k-chunk corpus, 50k candidate pairs):
      v1 (sync, no filter):                ~40 hours
      v2 (--no-filter, async batch=20):    ~30 minutes
      v2 (with stage-1 filter, default):   ~15-20 minutes

    Examples
    --------
    prism-build \\
        --lancedb-path  /path/to/lancedb \\
        --graph-path    /path/to/prism_graph.json.gz \\
        --llm-api-key   $DEEPSEEK_API_KEY

    # Skip stage-1 filter if Ollama is unavailable
    prism-build \\
        --lancedb-path  /path/to/lancedb \\
        --graph-path    /path/to/prism_graph.json.gz \\
        --llm-api-key   $DEEPSEEK_API_KEY \\
        --no-filter

    # Resume an interrupted build from checkpoint
    prism-build \\
        --lancedb-path  /path/to/lancedb \\
        --graph-path    /path/to/prism_graph.json.gz \\
        --llm-api-key   $DEEPSEEK_API_KEY \\
        --resume
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
                        help="Ollama base URL (used for embeddings and stage-1 filter)")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Embedding model (must match the model used at ingest time)")

    # ── Stage 1: local filter ─────────────────────────────────────────────────
    parser.add_argument("--filter-model", default="llama3.1:8b",
                        help="Ollama model for stage-1 binary pre-filter (use a fast model <5GB; larger models are slower than the API they're filtering for)")
    parser.add_argument("--filter-batch-size", type=int, default=10,
                        help="Pairs per stage-1 Ollama call")
    parser.add_argument("--filter-concurrency", type=int, default=5,
                        help="Concurrent stage-1 Ollama requests")
    parser.add_argument("--no-filter", action="store_true", default=False,
                        help="Skip stage-1 filter (use if Ollama unavailable)")

    # ── Stage 2: LLM extraction ───────────────────────────────────────────────
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com",
                        help="OpenAI-compatible API base URL for epistemic extraction")
    parser.add_argument("--llm-model",    default="deepseek-chat",
                        help="LLM model for epistemic extraction")
    parser.add_argument("--llm-api-key",  default="",
                        help="API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--min-confidence", type=float, default=0.65,
                        help="Minimum confidence score to keep an extracted edge")
    parser.add_argument("--batch-size",   type=int, default=20,
                        help="Pairs per LLM call — v2 default is 20 (was 5 in v1)")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Concurrent stage-2 LLM API requests")

    # ── Graph building ────────────────────────────────────────────────────────
    parser.add_argument("--k-neighbors",  type=int, default=8,
                        help="Semantic neighbours per chunk for candidate pairs")
    parser.add_argument("--max-pairs",    type=int, default=None,
                        help="Cap total candidate pairs (no limit by default)")
    parser.add_argument("--cross-source-only", action="store_true", default=False,
                        help="Only extract inter-source pairs (skips same-source pairs; "
                             "reduces edges by ~50%% but misses within-source relationships)")
    parser.add_argument("--all-sources",  action="store_true", default=False,
                        help=argparse.SUPPRESS)  # deprecated — was the old default-off flag
    parser.add_argument("--force",        action="store_true", default=False,
                        help="Rebuild even if graph-path already exists")
    parser.add_argument("--no-resume",    action="store_true", default=False,
                        help="Ignore any existing checkpoint and start fresh")
    parser.add_argument("--failure-log",  default=None,
                        help="Path to write a JSON log of failed extraction batches")

    args = parser.parse_args()

    # ── Resolve API key from environment if not passed ────────────────────────
    api_key = (
        args.llm_api_key
        or os.environ.get("DEEPSEEK_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )

    # ── Auto-fix extension ────────────────────────────────────────────────────
    graph_path = Path(args.graph_path)
    if graph_path.suffix not in (".gz", ".json"):
        graph_path = Path(str(graph_path) + ".json.gz")

    from .prism import PRISM

    # --all-sources is deprecated; warn and treat as "not --cross-source-only"
    if args.all_sources:
        import warnings
        warnings.warn(
            "--all-sources is deprecated and will be removed in a future release. "
            "The default is now to include all sources. "
            "Use --cross-source-only to restrict to inter-source pairs.",
            DeprecationWarning,
            stacklevel=1,
        )

    cross_source_only = args.cross_source_only  # default False = include all sources

    p = PRISM(
        lancedb_path          = args.lancedb_path,
        graph_path            = graph_path,
        table_name            = args.table_name,
        ollama_url            = args.ollama_url,
        embed_model           = args.embed_model,
        llm_base_url          = args.llm_base_url,
        llm_model             = args.llm_model,
        llm_api_key           = api_key,
        min_confidence        = args.min_confidence,
        batch_size            = args.batch_size,
        max_concurrent        = args.max_concurrent,
        filter_model          = args.filter_model,
        filter_batch_size     = args.filter_batch_size,
        filter_max_concurrent = args.filter_concurrency,
        failure_log_path      = args.failure_log,
    )

    p.build(
        k_neighbors       = args.k_neighbors,
        cross_source_only = cross_source_only,
        max_pairs         = args.max_pairs,
        force             = args.force,
        use_filter        = not args.no_filter,
        resume            = not args.no_resume,
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
