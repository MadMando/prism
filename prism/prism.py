"""
prism.prism
-----------
PRISM — the main public interface.

Ties together: LanceDBAdapter → EpistemicGraph → EpistemicExtractor
               → SpreadingActivation → PRISMRetriever → EpistemicResult
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .adapters.lancedb import LanceDBAdapter
from .extractor import EpistemicExtractor
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
            embed_model  = "qwen3-embedding:4b",
            llm_base_url = "https://api.deepseek.com",
            llm_model    = "deepseek-chat",
            llm_api_key  = "sk-...",
        )

        # Build the epistemic graph (one-time, run offline)
        p.build(max_pairs=50_000)

        # Retrieve with full epistemic structuring
        result = p.retrieve("what is data stewardship accountability")
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
        embed_api_url:  Optional[str] = None,   # default: https://api.openai.com/v1/embeddings
        embed_api_key:  Optional[str] = None,   # e.g. "sk-..."
        # ── LLM for graph building ────────────────────────────────────
        llm_base_url:   str = "https://api.openai.com",
        llm_model:      str = "gpt-4o-mini",
        llm_api_key:    str = "",
        # ── Extraction settings ───────────────────────────────────────
        min_confidence: float = 0.65,
        batch_size:     int   = 5,
        # ── Retrieval settings ────────────────────────────────────────
        hops:               int   = 3,
        decay:              float = 0.7,
        seed_top_k:         int   = 20,
        convergence_weight: float = 0.4,
    ):
        self.graph_path = Path(graph_path)

        self.adapter = LanceDBAdapter(
            db_path       = lancedb_path,
            table_name    = table_name,
            ollama_url    = ollama_url,
            embed_model   = embed_model,
            embed_api_url = embed_api_url,
            embed_api_key = embed_api_key,
        )

        self._extractor_kwargs = dict(
            base_url       = llm_base_url,
            model          = llm_model,
            api_key        = llm_api_key,
            min_confidence = min_confidence,
            batch_size     = batch_size,
        )

        self._retriever_kwargs = dict(
            hops               = hops,
            decay              = decay,
            seed_top_k         = seed_top_k,
            convergence_weight = convergence_weight,
        )

        self.graph:    Optional[EpistemicGraph] = None
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
    ) -> "PRISM":
        """
        Build (or rebuild) the epistemic graph from the LanceDB corpus.

        Steps:
          1. Add all LanceDB chunks as graph nodes
          2. Find candidate pairs via vector similarity
          3. Run LLM extraction to identify epistemic triples
          4. Save the graph to graph_path

        This runs offline — expect it to take 30-60 min for a 14k-chunk corpus
        at batch_size=5 with DeepSeek-chat.

        Args:
            k_neighbors:       Number of semantic neighbours per chunk to consider
            cross_source_only: Only extract inter-source relationships (recommended)
            max_pairs:         Cap total candidate pairs (useful for testing)
            force:             Rebuild even if graph_path already exists
        """
        if not force and self.graph_path.exists():
            print(f"[prism] graph already exists at {self.graph_path}")
            print(f"[prism] use force=True to rebuild, or load_graph() to use existing")
            return self.load_graph()

        print(f"[prism] ── BUILD START ─────────────────────────────────────")
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

        # Step 3: LLM extraction
        extractor = EpistemicExtractor(**self._extractor_kwargs)
        n_added = extractor.extract_from_candidates(candidates, graph)

        # Step 4: Finalise + save
        graph.meta = {
            "built_at":         datetime.now(timezone.utc).isoformat(),
            "lancedb_path":     str(self.adapter.db_path),
            "embed_model":      self.adapter.embed_model,
            "extraction_model": self._extractor_kwargs["model"],
            "k_neighbors":      k_neighbors,
            "cross_source_only": cross_source_only,
            "n_candidates":     len(candidates),
            "n_edges_extracted": n_added,
        }
        graph.save(self.graph_path)

        self.graph = graph
        self._retriever = PRISMRetriever(
            adapter=self.adapter,
            graph=self.graph,
            **self._retriever_kwargs,
        )

        stats = graph.stats()
        print(f"[prism] ── BUILD COMPLETE ───────────────────────────────────")
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
            # Auto-connect adapter without graph
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
