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

from .adapters.base import VectorAdapter
from .adapters.lancedb import LanceDBAdapter
from .extractor import EpistemicExtractor
from .filter import EpistemicFilter
from .graph import EpistemicGraph
from .result import EpistemicChunk, EpistemicResult
from .retriever import PRISMRetriever, Reranker


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
        graph_path:     str | Path,
        lancedb_path:   Optional[str | Path] = None,
        table_name:     str = "knowledge",
        # ── Custom adapter (alternative to lancedb_path) ──────────────
        adapter:        Optional[VectorAdapter] = None,
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
        max_retries:    int   = 3,
        failure_log_path: Optional[str] = None,
        # ── Stage 1 filter settings ───────────────────────────────────
        filter_model:         str = "llama3.1:8b",  # fast Ollama model for Stage 1 pre-filter
        filter_batch_size:    int = 10,
        filter_max_concurrent:int = 5,
        # ── Retrieval settings ────────────────────────────────────────
        hops:               int   = 3,
        decay:              float = 0.7,
        seed_top_k:         int   = 20,
        convergence_weight: float = 0.4,
        reranker:           Optional[Reranker] = None,
    ):
        self.graph_path = Path(graph_path)
        self.ollama_url = ollama_url
        self._reranker  = reranker

        if adapter is not None:
            self.adapter = adapter
        elif lancedb_path is not None:
            self.adapter = LanceDBAdapter(
                db_path       = lancedb_path,
                table_name    = table_name,
                ollama_url    = ollama_url,
                embed_model   = embed_model,
                embed_api_url = embed_api_url,
                embed_api_key = embed_api_key,
            )
        else:
            raise ValueError(
                "Provide either lancedb_path= or adapter= — both are None."
            )

        self._extractor_kwargs = dict(
            base_url          = llm_base_url,
            model             = llm_model,
            api_key           = llm_api_key,
            min_confidence    = min_confidence,
            batch_size        = batch_size,
            max_concurrent    = max_concurrent,
            max_retries       = max_retries,
            failure_log_path  = failure_log_path,
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
            reranker=self._reranker,
            **self._retriever_kwargs,
        )
        return self

    def build(
        self,
        k_neighbors:       int  = 8,
        cross_source_only: bool = False,
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
            "adapter":           self.adapter.__class__.__name__,
            "embed_model":       getattr(self.adapter, "embed_model", "unknown"),
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
                reranker=self._reranker,
                **self._retriever_kwargs,
            )
        return self._retriever.retrieve(
            query         = query,
            top_k         = top_k,
            source_filter = source_filter,
            persona       = persona,
        )

    # ── Incremental updates ───────────────────────────────────────────────────

    def add_documents(
        self,
        node_ids:          list[str],
        k_neighbors:       int  = 8,
        cross_source_only: bool = False,
        use_filter:        bool = True,
    ) -> int:
        """
        Incrementally update the graph with newly-added chunks.

        After adding new documents to the vector store, call this to extract
        epistemic relationships involving the new chunks and update the graph.
        The existing graph is updated in-place and re-saved to disk.

        Args:
            node_ids:          IDs of the newly-added chunks (must already be
                               in the vector store)
            k_neighbors:       k-NN neighbours to consider for each new chunk
            cross_source_only: Only extract inter-source pairs
            use_filter:        Run Stage 1 local pre-filter

        Returns:
            Number of new edges added
        """
        if self.graph is None:
            if self.graph_path.exists():
                self.load_graph()
            else:
                raise RuntimeError(
                    "No graph loaded and no graph file found. Run build() first."
                )

        self.adapter.connect()

        # Add new nodes to the graph
        chunks = self.adapter.get_chunks(node_ids)
        for nid, data in chunks.items():
            self.graph.add_node(
                nid,
                source       = data.get("source", ""),
                page         = data.get("page", 0),
                section      = data.get("section", ""),
                text_preview = data.get("text", "")[:200],
            )

        # Find candidate pairs for the new nodes
        candidates = self.adapter.candidate_pairs_for(
            node_ids          = node_ids,
            k_neighbors       = k_neighbors,
            cross_source_only = cross_source_only,
        )
        if not candidates:
            print("[prism] add_documents: no candidate pairs found for new chunks")
            return 0

        print(f"[prism] add_documents: {len(candidates):,} candidate pairs for {len(node_ids)} new chunks")

        if use_filter:
            f = EpistemicFilter(**self._filter_kwargs)
            candidates = f.filter(candidates)

        extractor = EpistemicExtractor(**self._extractor_kwargs)
        n_added = extractor.extract_from_candidates(candidates, self.graph)

        # Save updated graph
        self.graph.save(self.graph_path)
        print(f"[prism] add_documents: {n_added} new edges added, graph updated")
        return n_added

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
