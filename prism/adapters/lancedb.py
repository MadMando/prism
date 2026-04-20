"""
prism.adapters.lancedb
----------------------
LanceDB adapter — connects PRISM to an existing LanceDB vector store.

Requires the lancedb extra::

    pip install prism-rag[lancedb]

Responsibilities:
  1. Seed activation:  embed query → vector search → initial {node_id: score} map
  2. Candidate pairs:  for graph building, find semantically similar chunk pairs
  3. Chunk hydration:  given a list of node_ids, fetch full chunk data from LanceDB

Embedding providers supported
------------------------------
Option A — Ollama (local, default):
    LanceDBAdapter(embed_model="nomic-embed-text")

Option B — Any OpenAI-compatible embeddings API:
    LanceDBAdapter(
        embed_api_url = "https://api.openai.com/v1/embeddings",
        embed_api_key = "sk-...",
        embed_model   = "text-embedding-3-small",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import lancedb
except ImportError as _e:
    raise ImportError(
        "LanceDBAdapter requires lancedb. Install it with:\n"
        "    pip install prism-rag[lancedb]"
    ) from _e

from .embedder import Embedder


class LanceDBAdapter:
    """
    Connects PRISM to an existing LanceDB knowledge base.

    Args:
        db_path:        Path to the LanceDB directory
        table_name:     LanceDB table name (default "knowledge")
        embed_model:    Embedding model name
        ollama_url:     Ollama base URL (used when embed_api_key is not set)
        embed_api_url:  Full URL for OpenAI-compatible embeddings endpoint
        embed_api_key:  API key — if set, switches to API mode (ignores ollama_url)
        embed_timeout:  Request timeout in seconds (default 60)
    """

    def __init__(
        self,
        db_path:       str | Path,
        table_name:    str = "knowledge",
        embed_model:   str = "nomic-embed-text",
        # Ollama (default)
        ollama_url:    str = "http://localhost:11434",
        # OpenAI-compatible API (alternative)
        embed_api_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        # Shared
        embed_timeout: int = 60,
    ):
        self.db_path    = Path(db_path)
        self.table_name = table_name

        self._embedder = Embedder(
            model      = embed_model,
            api_url    = embed_api_url,
            api_key    = embed_api_key,
            ollama_url = ollama_url,
            timeout    = embed_timeout,
        )

        self._db:    Optional[lancedb.DBConnection] = None
        self._table = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._db    = lancedb.connect(str(self.db_path))
        tables      = self._db.list_tables()
        names       = tables.tables if hasattr(tables, "tables") else tables
        if self.table_name not in names:
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' not found in {self.db_path}. "
                f"Available: {names}"
            )
        self._table = self._db.open_table(self.table_name)

    def _ensure_connected(self) -> None:
        if self._table is None:
            self.connect()

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Embed text using the configured provider."""
        return self._embedder.embed(text)

    # ── Seed activation ───────────────────────────────────────────────────────

    def seed_scores(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Embed the query and return the top-K nearest chunks as
        {chunk_id: similarity_score} — the seed activation map for PRISM.

        Similarity is (1 - cosine_distance), so higher = more relevant.
        """
        self._ensure_connected()
        vec = self._embedder.embed(query)

        fetch = top_k * 4 if source_filter else top_k * 2
        results = self._table.search(vec).limit(fetch).to_list()

        if source_filter:
            results = [r for r in results if source_filter.lower() in r.get("source", "").lower()]

        results = results[:top_k]

        return {
            r["id"]: max(0.0, round(1.0 - float(r.get("_distance", 1.0)), 4))
            for r in results
        }

    # ── Chunk hydration ───────────────────────────────────────────────────────

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Fetch full chunk data for a list of node IDs.
        Returns {chunk_id: {id, source, page, section, text, ...}}
        Missing IDs are silently skipped.
        """
        self._ensure_connected()
        if not node_ids:
            return {}

        id_set = set(node_ids)
        results = {}

        try:
            safe_ids  = [nid.replace("'", "''") for nid in node_ids[:100]]
            id_list_sql = ", ".join(f"'{nid}'" for nid in safe_ids)
            rows = self._table.search().where(f"id IN ({id_list_sql})").limit(len(node_ids) + 10).to_list()
            for row in rows:
                if row["id"] in id_set:
                    results[row["id"]] = {
                        "id":      row["id"],
                        "source":  row.get("source", ""),
                        "page":    row.get("page", 0),
                        "section": row.get("section", ""),
                        "text":    row.get("text", ""),
                    }
        except Exception:
            pass

        return results

    # ── Candidate pair generation (for graph building) ─────────────────────────

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = True,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        """
        Generate candidate chunk pairs for epistemic extraction.

        Strategy: for each chunk, find its top-K semantic neighbors.
        By default, only cross-source pairs are returned — these have
        the most valuable inter-framework epistemic signal.

        Returns a deduplicated list of (chunk_a, chunk_b) dicts.
        Each dict has: id, source, page, section, text
        """
        self._ensure_connected()

        print("[prism] loading all chunks for candidate generation ...")
        all_rows = self._table.to_pandas()[
            ["id", "source", "page", "section", "text", "vector"]
        ].to_dict("records")
        print(f"[prism] {len(all_rows):,} chunks loaded")

        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        from tqdm import tqdm
        for row in tqdm(all_rows, desc="finding candidate pairs", unit="chunk"):
            vec = row["vector"]
            try:
                neighbors = self._table.search(vec).limit(k_neighbors + 1).to_list()
            except Exception:
                continue

            for nbr in neighbors:
                if nbr["id"] == row["id"]:
                    continue
                if cross_source_only and nbr.get("source") == row.get("source"):
                    continue

                pair_key = frozenset([row["id"], nbr["id"]])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                chunk_a = {
                    "id":      row["id"],
                    "source":  row.get("source", ""),
                    "page":    row.get("page", 0),
                    "section": row.get("section", ""),
                    "text":    row.get("text", ""),
                }
                chunk_b = {
                    "id":      nbr["id"],
                    "source":  nbr.get("source", ""),
                    "page":    nbr.get("page", 0),
                    "section": nbr.get("section", ""),
                    "text":    nbr.get("text", ""),
                }
                candidates.append((chunk_a, chunk_b))

                if max_pairs and len(candidates) >= max_pairs:
                    print(f"[prism] max_pairs={max_pairs} reached, stopping candidate generation")
                    return candidates

        print(f"[prism] {len(candidates):,} candidate pairs generated")
        return candidates

    def candidate_pairs_for(
        self,
        node_ids: list[str],
        k_neighbors: int = 8,
        cross_source_only: bool = False,
    ) -> list[tuple[dict, dict]]:
        """
        Generate candidate pairs for a specific subset of nodes.
        Used for incremental graph updates — only looks at newly-added chunks.

        Returns deduplicated (chunk_a, chunk_b) pairs where at least one
        member is in node_ids.
        """
        self._ensure_connected()
        if not node_ids:
            return []

        id_set = set(node_ids)
        safe_ids    = [nid.replace("'", "''") for nid in list(id_set)[:500]]
        id_list_sql = ", ".join(f"'{nid}'" for nid in safe_ids)
        try:
            target_rows = (
                self._table.search()
                .where(f"id IN ({id_list_sql})")
                .limit(len(id_set) + 10)
                .to_list()
            )
        except Exception:
            return []

        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for row in target_rows:
            vec = row.get("vector")
            if vec is None:
                continue
            try:
                neighbors = self._table.search(vec).limit(k_neighbors + 1).to_list()
            except Exception:
                continue

            for nbr in neighbors:
                if nbr["id"] == row["id"]:
                    continue
                if cross_source_only and nbr.get("source") == row.get("source"):
                    continue

                pair_key = frozenset([row["id"], nbr["id"]])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                chunk_a = {
                    "id":      row["id"],
                    "source":  row.get("source", ""),
                    "page":    row.get("page", 0),
                    "section": row.get("section", ""),
                    "text":    row.get("text", ""),
                }
                chunk_b = {
                    "id":      nbr["id"],
                    "source":  nbr.get("source", ""),
                    "page":    nbr.get("page", 0),
                    "section": nbr.get("section", ""),
                    "text":    nbr.get("text", ""),
                }
                candidates.append((chunk_a, chunk_b))

        return candidates

    def populate_graph_nodes(self, graph) -> int:
        """
        Add all chunks in the LanceDB table as nodes in the epistemic graph.
        Returns the number of nodes added.
        """
        self._ensure_connected()
        rows = self._table.to_pandas()[["id", "source", "page", "section", "text"]].to_dict("records")
        added = 0
        for row in rows:
            graph.add_node(
                row["id"],
                source       = row.get("source", ""),
                page         = row.get("page", 0),
                section      = row.get("section", ""),
                text_preview = row.get("text", "")[:200],
            )
            added += 1
        print(f"[prism] {added:,} nodes added to graph from LanceDB")
        return added

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self._ensure_connected()
        n = self._table.count_rows()
        sources = self._table.to_pandas()[["source"]]["source"].value_counts().to_dict()
        return {
            "total_chunks": n,
            "sources": sources,
            "db_path": str(self.db_path),
            "table": self.table_name,
            "embed_model": self._embedder.model,
        }
