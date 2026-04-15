"""
prism.prism
-----------
PRISM — the main public interface.

Ties together: LanceDBAdapter → EpistemicGraph → EpistemicFilter (Stage 1)
               → EpistemicExtractor (Stage 2, async) → SpreadingActivation
               → PRISMRetriever → EpistemicResult
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .adapters.lancedb import LanceDBAdapter
from .extractor import EpistemicExtractor
from .filter import EpistemicFilter
from .graph import EpistemicGraph
from .result import EpistemicResult
from .retriever import PRISMRetriever


class PRISM:
    """
    Epistemic Graph RAG with Spreading Activation.

    Quick start:
        p = PRISM(
            lancedb_path = "/path/to/lancedb",
            graph_path   = "/path/to/prism_graph.json.gz",
            ollama_url   = "http://localhost:11434",
            embed_model  = "nomic-embed-text",
            llm_base_url = "https://api.deepseek.com",
            llm_model    = "deepseek-chat",
            llm_api_key  = "sk-...",
        )

        # Build the epistemic graph (one-time, run offline)
        # Two-stage pipeline: local filter → async LLM extraction
        # ~30 min for a 30k-chunk corpus (vs 40+ hours in v1)
        p.build()

        # Retrieve with full epistemic structuring
        result = p.retrieve("your question here")
        print(result.format_for_llm())
    """

    def __init__(
        self,
        lancedb_path:   str | Path,
        graph_path:     str | Path,
        table_name:     str = "knowledge",
        # ── Embedding: Option A — Ollama (default) ────────────────────
        ollama_url:     str = "http://localhost:11434",
        embed_model:    str = "nomic-embed-text",
        # ── Embedding: Option B — OpenAI-compatible API ───────────────
        # Set embed_api_key to switch to API mode (ollama_url is ignored)
        embed_api_url:  Optional[str] = None,
        embed_api_key:  Optional[str] = None,
        # ── LLM for graph building ────────────────────────────────────
        llm_base_url:   str = "https://api.openai.com",
        llm_model:      str = "gpt-4o-mini",
        llm_api_key:    str = "",
        # ── Extraction settings ───────────────────────────────────────
        min_confidence: float = 0.65,
        batch_size:     int   = 20,        # v2 default: 20 (was 5)
        max_concurrent: int   = 20,        # v2: async concurrent requests
        # ── Stage 1 filter settings ───────────────────────────────────
        filter_model:         str = "gemma4:31b-cloud",  # local Ollama model
        filter_batch_size:    int = 10,
        filter_max_concurrent:int = 5,
        # ── Retrieval settings ────────────────────────────────────────
        hops:               int   = 3,
        decay:              float = 0.7,
        seed_top_k:         int   = 20,
        convergence_weight: float = 0.4,
    ):
        self.graph_path = Path(graph_path)
        self.ollama_url = ollama_url

        self.adapter = LanceDBAdapter(
            db_path       = lancedb_path,
            table_name    = table_name,
            ollama_url    = ollama_url,
            embed_model   = embed_model,
            embed_api_url = embed_api_url,
            embed_api_key = embed_api_key,
        )

        self._extractor_kwargs = dict(
            base_url         = llm_base_url,
            model            = llm_model,
            api_key          = llm_api_key,
            min_confidence   = min_confidence,
            batch_size       = batch_size,
            max_concurrent   = max_concurrent,
        )

        self._filter_kwargs = dict(
            ollama_url     = ollama_url,
            model          = filter_model,
            batch_size     = filter_batch_size,
            max_concurrent = filter_max_concurrent,
        )

        self._retriever_kwargs = dict(
            hops               = hops,
            decay              = decay,
            seed_top_k         = seed_top_k,
            convergence_weight = convergence_weight,
        )

        self.graph:      Optional[EpistemicGraph] = None
        self._retriever: Optional[PRISMRetriever] = None

    # ── Graph management ──────────────────────────────────────────────────────

    def load_graph(self) -> "PRISM":
        """Load the epistemic graph from disk. Call this before retrieve()."""
        self.graph = EpistemicGraph.load(self.graph_path)
        self._retriever = PRISMRetriever(
            adapter=self.adapter,
            graph=self.graph,
            **self._retriever_kwargs,
        )
        return self

    def build(
        self,
        k_neighbors:       int  = 8,
        cross_source_only: bool = True,
        max_pairs:         Optional[int] = None,
        force:             bool = False,
        use_filter:        bool = True,
        resume:            bool = True,
    ) -> "PRISM":
        """
        Build (or rebuild) the epistemic graph from the LanceDB corpus.

        Uses a two-stage pipeline for dramatically faster build times:

          Stage 1 — Local filter (fast, free):
            A local Ollama model pre-screens candidate pairs for any
            epistemic relationship (binary YES/NO). Filters ~50% of pairs,
            saving ~50% of Stage 2 API cost.

          Stage 2 — Async LLM extraction:
            Surviving pairs are classified with full type+confidence using
            an async OpenAI-compatible API with configurable concurrency
            (default 20 parallel requests, batch_size=20).

        Build time comparison (30k-chunk corpus, 50k candidates):
          v1 (sync, batch=5):               ~40 hours
          v2 (async, batch=20, no filter):  ~30 minutes
          v2 (async + stage-1 filter):      ~15–20 minutes

        Args:
            k_neighbors:       Semantic neighbours per chunk for candidate pairs
            cross_source_only: Only extract inter-source pairs (recommended)
            max_pairs:         Cap total candidates (omit for full build)
            force:             Rebuild even if graph already exists
            use_filter:        Run Stage 1 local pre-filter (default True)
            resume:            Resume from checkpoint if one exists (default True)
        """
        if not force and self.graph_path.exists():
            print(f"[prism] graph already exists at {self.graph_path}")
            print("[prism] use force=True to rebuild, or load_graph() to use existing")
            return self.load_graph()

        checkpoint_path = self.graph_path.with_suffix("").with_suffix(".partial.json.gz")

        print("[prism] ── BUILD START ─────────────────────────────────────")
        self.adapter.connect()

        # Step 1: Populate nodes
        graph = EpistemicGraph()
        self.adapter.populate_graph_nodes(graph)

        # Step 2: Candidate pairs
        candidates = self.adapter.candidate_pairs(
            k_neighbors       = k_neighbors,
            cross_source_only = cross_source_only,
            max_pairs         = max_pairs,
        )
        print(f"[prism] {len(candidates):,} candidate pairs generated")

        # Step 3: Stage 1 — local filter
        if use_filter:
            f = EpistemicFilter(**self._filter_kwargs)
            candidates = f.filter(candidates)
        else:
            print("[prism] stage 1 filter skipped (use_filter=False)")

        # Step 4: Stage 2 — async LLM extraction
        extractor = EpistemicExtractor(**self._extractor_kwargs)
        n_added = extractor.extract_from_candidates(
            candidates,
            graph,
            checkpoint_path = checkpoint_path if resume else None,
        )

        # Step 5: Finalise + save
        graph.meta = {
            "built_at":          datetime.now(timezone.utc).isoformat(),
            "lancedb_path":      str(self.adapter.db_path),
            "embed_model":       self.adapter.embed_model,
            "extraction_model":  self._extractor_kwargs["model"],
            "filter_model":      self._filter_kwargs["model"] if use_filter else None,
            "k_neighbors":       k_neighbors,
            "cross_source_only": cross_source_only,
            "n_candidates":      len(candidates),
            "n_edges_extracted": n_added,
            "pipeline":          "two-stage-async-v2",
        }
        graph.save(self.graph_path)

        # Remove checkpoint on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        self.graph = graph
        self._retriever = PRISMRetriever(
            adapter=self.adapter,
            graph=self.graph,
            **self._retriever_kwargs,
        )

        stats = graph.stats()
        print("[prism] ── BUILD COMPLETE ───────────────────────────────────")
        print(f"[prism]   nodes:  {stats['n_nodes']:,}")
        print(f"[prism]   edges:  {stats['n_edges']:,}")
        print(f"[prism]   types:  {stats['edge_types']}")
        return self

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:         str,
        top_k:         int  = 5,
        source_filter: Optional[str] = None,
        persona:       Optional[str] = None,
    ) -> EpistemicResult:
        """
        Retrieve epistemically-structured results.

        Returns EpistemicResult with primary, supporting, contrasting,
        qualifying, and superseded chunk buckets.

        Falls back to pure vector search if graph is not loaded.
        """
        if self._retriever is None:
            self.adapter.connect()
            self._retriever = PRISMRetriever(
                adapter=self.adapter,
                graph=None,
                **self._retriever_kwargs,
            )
        return self._retriever.retrieve(
            query         = query,
            top_k         = top_k,
            source_filter = source_filter,
            persona       = persona,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        s: dict = {"graph_path": str(self.graph_path)}
        if self.graph_path.exists():
            s["graph_exists"] = True
            if self.graph:
                s["graph"] = self.graph.stats()
        else:
            s["graph_exists"] = False
        try:
            self.adapter.connect()
            s["lancedb"] = self.adapter.stats()
        except Exception as e:
            s["lancedb_error"] = str(e)
        return s
