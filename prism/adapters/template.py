"""
prism.adapters.template
-----------------------
Copy-paste skeleton for building a custom PRISM vector store adapter.

Steps:
  1. Copy this file to e.g. my_project/my_adapter.py
  2. Replace MyVectorStore with your actual DB client
  3. Implement the six methods below
  4. Pass an instance to PRISM:

        from prism import PRISM
        from my_adapter import MyAdapter

        p = PRISM(
            adapter     = MyAdapter(...),
            graph_path  = "/path/to/prism_graph.json.gz",
            llm_base_url = "...",
            llm_model    = "...",
            llm_api_key  = "...",
        )

Embedding is handled for you by the Embedder helper — you don't need
to implement it unless you want to use a different embedding strategy.

    from prism.adapters.embedder import Embedder
    emb = Embedder(model="nomic-embed-text")              # Ollama
    emb = Embedder(model="text-embedding-3-small",        # OpenAI-compatible
                   api_url="https://api.openai.com/v1/embeddings",
                   api_key="sk-...")
    vector = emb.embed("some text")   # -> list[float]
"""

from __future__ import annotations

from typing import Optional

from .base import VectorAdapter  # noqa: F401  (for isinstance checks)
from .embedder import Embedder


class MyAdapter:
    """
    Template adapter — replace with your vector store.

    This class satisfies the VectorAdapter Protocol as long as all
    seven methods are implemented with the correct signatures.
    """

    def __init__(
        self,
        # --- your DB connection params go here ---
        # db_url: str,
        # collection: str = "knowledge",
        # --- embedding (reuse Embedder or roll your own) ---
        embed_model: str = "nomic-embed-text",
        ollama_url:  str = "http://localhost:11434",
        embed_api_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embed_timeout: int = 60,
    ):
        # self._client = MyVectorStore(db_url)
        # self._collection = collection

        self._embedder = Embedder(
            model      = embed_model,
            api_url    = embed_api_url,
            api_key    = embed_api_key,
            ollama_url = ollama_url,
            timeout    = embed_timeout,
        )

    # ── Required: open the connection ─────────────────────────────────────────

    def connect(self) -> None:
        """Open / initialise the connection to the vector store."""
        raise NotImplementedError

    # ── Required: query-time retrieval ───────────────────────────────────────

    def seed_scores(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Embed the query and return the top-k nearest chunks.

        Must return:
            {chunk_id: similarity_score}  where scores are in [0, 1]

        Tip: similarity = 1 - cosine_distance
        """
        vec = self._embedder.embed(query)
        # results = self._client.search(vec, top_k=top_k * 2)
        # if source_filter:
        #     results = [r for r in results if source_filter in r["source"]]
        # return {r["id"]: round(1.0 - r["distance"], 4) for r in results[:top_k]}
        raise NotImplementedError

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Fetch full chunk data for a list of node IDs.

        Must return:
            {node_id: {"id": str, "source": str, "page": int,
                       "section": str, "text": str}}
        Missing IDs should simply be absent from the dict.
        """
        raise NotImplementedError

    # ── Required: graph-build time ────────────────────────────────────────────

    def populate_graph_nodes(self, graph) -> int:
        """
        Add every chunk in the store as a node in the EpistemicGraph.

        Call graph.add_node(id, source=..., page=..., section=..., text_preview=...)
        for each chunk. Return the count of nodes added.
        """
        # rows = self._client.get_all(fields=["id", "source", "page", "section", "text"])
        # for row in rows:
        #     graph.add_node(
        #         row["id"],
        #         source       = row.get("source", ""),
        #         page         = row.get("page", 0),
        #         section      = row.get("section", ""),
        #         text_preview = row.get("text", "")[:200],
        #     )
        # return len(rows)
        raise NotImplementedError

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = False,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        """
        Generate candidate chunk pairs for epistemic edge extraction.

        Strategy: for each chunk, find its k nearest neighbours,
        yield deduplicated (chunk_a, chunk_b) pairs.

        Each chunk dict must have: id, source, page, section, text
        """
        raise NotImplementedError

    def candidate_pairs_for(
        self,
        node_ids: list[str],
        k_neighbors: int = 8,
        cross_source_only: bool = False,
    ) -> list[tuple[dict, dict]]:
        """
        Like candidate_pairs but restricted to a subset of node IDs.
        Used for incremental graph updates after add_documents().
        """
        raise NotImplementedError

    # ── Required: diagnostics ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return a summary of the vector store.

        Must include at minimum:
            {"n_rows": int, "sources": list[str]}
        Any extra keys are fine (shown in prism-stats output).
        """
        raise NotImplementedError
