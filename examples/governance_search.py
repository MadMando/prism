"""
examples/governance_search.py
------------------------------
Demonstrates PRISM against a governance corpus in LanceDB.

Configuration via environment variables (recommended) or edit the
constants below directly:

    export LANCEDB_PATH=/path/to/lancedb
    export GRAPH_PATH=/path/to/prism_graph.json.gz
    export OLLAMA_URL=http://localhost:11434      # or remote Ollama host
    export EMBED_MODEL=nomic-embed-text
    export LLM_BASE_URL=https://api.deepseek.com
    export LLM_MODEL=deepseek-chat
    export DEEPSEEK_API_KEY=sk-...

Run:
    python examples/governance_search.py
    python examples/governance_search.py "what is data stewardship"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running directly from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism import PRISM

# ── Configuration — override via environment variables ─────────────────────────

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "/path/to/your/lancedb")
GRAPH_PATH   = os.environ.get("GRAPH_PATH",   "/path/to/your/prism_graph.json.gz")

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "nomic-embed-text")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "deepseek-chat")
LLM_API_KEY  = ""   # Set via env: export DEEPSEEK_API_KEY=sk-...

# ── Queries to demonstrate PRISM ──────────────────────────────────────────────

DEMO_QUERIES = [
    "what is data stewardship accountability",
    "DMBOK data governance framework components",
    "master data management vs reference data",
    "data quality dimensions and metrics",
    "erwin data modeling best practices",
]


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", LLM_API_KEY)

    query = sys.argv[1] if len(sys.argv) > 1 else DEMO_QUERIES[0]

    print("=" * 70)
    print("PRISM — Epistemic Graph RAG")
    print("=" * 70)

    p = PRISM(
        lancedb_path = LANCEDB_PATH,
        graph_path   = GRAPH_PATH,
        ollama_url   = OLLAMA_URL,
        embed_model  = EMBED_MODEL,
        llm_base_url = LLM_BASE_URL,
        llm_model    = LLM_MODEL,
        llm_api_key  = api_key,
        # Extraction settings (only needed for build)
        min_confidence = 0.65,
        batch_size     = 5,
        # Retrieval settings
        hops               = 3,
        decay              = 0.7,
        seed_top_k         = 20,
        convergence_weight = 0.4,
    )

    # ── Load or build graph ────────────────────────────────────────────────────
    graph_path = Path(GRAPH_PATH)
    if graph_path.exists():
        print(f"\n[demo] Loading existing epistemic graph from {GRAPH_PATH}")
        p.load_graph()
        stats = p.stats()
        if "graph" in stats:
            g = stats["graph"]
            print(f"[demo] Graph: {g['n_nodes']:,} nodes, {g['n_edges']:,} edges")
            print(f"[demo] Edge types: {g['edge_types']}")
    else:
        print(f"\n[demo] No graph found at {GRAPH_PATH}")
        print(f"[demo] Falling back to pure vector search.")
        print(f"[demo] To build the graph, run:")
        print(f"[demo]   python scripts/build_graph.py \\")
        print(f"[demo]       --lancedb-path {LANCEDB_PATH} \\")
        print(f"[demo]       --graph-path   {GRAPH_PATH} \\")
        print(f"[demo]       --llm-api-key  $DEEPSEEK_API_KEY \\")
        print(f"[demo]       --max-pairs    50000")
        print()

    # ── Retrieve ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Query: {query!r}")
    print(f"{'─' * 70}\n")

    result = p.retrieve(query, top_k=5)

    # Print structured output
    print(result.format_for_llm())

    # ── Show stats ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Retrieval stats:")
    print(f"  Seeds (vector hits) : {result.n_seeds}")
    print(f"  Graph nodes reached : {result.n_graph_nodes}")
    print(f"  Edges traversed     : {result.n_edges_traversed}")
    print(f"  Graph used          : {result.graph_was_used}")
    print(f"  Primary chunks      : {len(result.primary)}")
    print(f"  Supporting chunks   : {len(result.supporting)}")
    print(f"  Contrasting chunks  : {len(result.contrasting)}")
    print(f"  Qualifying chunks   : {len(result.qualifying)}")
    print(f"  Superseded chunks   : {len(result.superseded)}")

    # ── MCP-style output (what Quinn would see) ────────────────────────────────
    print(f"\n{'─' * 70}")
    print("MCP-style output (as sent to Quinn):")
    print(f"{'─' * 70}")
    print(result.format_mcp())

    # ── Run all demo queries ───────────────────────────────────────────────────
    if len(sys.argv) == 1:
        print(f"\n{'=' * 70}")
        print("Running all demo queries (vector scores only — no graph yet)...")
        print(f"{'=' * 70}")
        for q in DEMO_QUERIES[1:]:
            r = p.retrieve(q, top_k=3)
            top = r.primary[0] if r.primary else None
            if top:
                print(f"\n  Q: {q}")
                print(f"     → [{top.source}] p.{top.page}  score={top.final_score:.3f}")
                print(f"        {top.text[:120]}...")


if __name__ == "__main__":
    main()
