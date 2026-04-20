"""
prism.adapters.qdrant
---------------------
Qdrant adapter — connects PRISM to an existing Qdrant collection.

Requires the qdrant extra::

    pip install prism-rag[qdrant]

Your Qdrant collection must store point payloads with at minimum:
  - "id"      : str  — the chunk ID (used as PRISM node ID)
  - "source"  : str  — source document name
  - "page"    : int  — page number
  - "section" : str  — section title
  - "text"    : str  — chunk text

Qdrant point IDs (integers or UUIDs) are used internally;
PRISM uses the payload "id" field as the canonical chunk identifier.
"""

from __future__ import annotations

from typing import Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchText, SearchRequest
except ImportError as _e:
    raise ImportError(
        "QdrantAdapter requires qdrant-client. Install it with:\n"
        "    pip install prism-rag[qdrant]"
    ) from _e

from .embedder import Embedder


class QdrantAdapter:
    """
    Connects PRISM to an existing Qdrant collection.

    Args:
        collection_name: Qdrant collection name
        url:             Qdrant server URL (e.g. "http://localhost:6333")
        api_key:         Qdrant API key (for Qdrant Cloud)
        path:            Local path for on-disk Qdrant (overrides url)
        vector_name:     Named vector to use (None for the default unnamed vector)
        id_payload_key:  Payload field that holds the PRISM chunk ID (default "id")
        embed_model:     Embedding model name
        ollama_url:      Ollama base URL
        embed_api_url:   OpenAI-compatible embeddings endpoint
        embed_api_key:   API key (switches to API mode when set)
        embed_timeout:   Request timeout in seconds
    """

    def __init__(
        self,
        collection_name: str = "knowledge",
        url:             str = "http://localhost:6333",
        api_key:         Optional[str] = None,
        path:            Optional[str] = None,
        vector_name:     Optional[str] = None,
        id_payload_key:  str = "id",
        # Embedding
        embed_model:     str = "nomic-embed-text",
        ollama_url:      str = "http://localhost:11434",
        embed_api_url:   Optional[str] = None,
        embed_api_key:   Optional[str] = None,
        embed_timeout:   int = 60,
    ):
        self.collection_name  = collection_name
        self.url              = url
        self.qdrant_api_key   = api_key
        self.path             = path
        self.vector_name      = vector_name
        self.id_payload_key   = id_payload_key

        self._embedder = Embedder(
            model      = embed_model,
            api_url    = embed_api_url,
            api_key    = embed_api_key,
            ollama_url = ollama_url,
            timeout    = embed_timeout,
        )

        self._client: Optional[QdrantClient] = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.path:
            self._client = QdrantClient(path=self.path)
        else:
            self._client = QdrantClient(url=self.url, api_key=self.qdrant_api_key)

    def _ensure_connected(self) -> None:
        if self._client is None:
            self.connect()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _point_to_chunk(self, point) -> dict:
        payload = point.payload or {}
        return {
            "id":      payload.get(self.id_payload_key, str(point.id)),
            "source":  payload.get("source", ""),
            "page":    int(payload.get("page", 0)),
            "section": payload.get("section", ""),
            "text":    payload.get("text", ""),
        }

    def _search_vec(self, vec: list[float], limit: int):
        if self.vector_name:
            return self._client.search(
                collection_name = self.collection_name,
                query_vector    = (self.vector_name, vec),
                limit           = limit,
                with_payload    = True,
            )
        return self._client.search(
            collection_name = self.collection_name,
            query_vector    = vec,
            limit           = limit,
            with_payload    = True,
        )

    # ── Seed activation ───────────────────────────────────────────────────────

    def seed_scores(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> dict[str, float]:
        self._ensure_connected()
        vec   = self._embedder.embed(query)
        fetch = top_k * 4 if source_filter else top_k * 2

        results = self._search_vec(vec, limit=fetch)

        scores = {}
        for r in results:
            chunk = self._point_to_chunk(r)
            if source_filter and source_filter.lower() not in chunk["source"].lower():
                continue
            chunk_id = chunk["id"]
            # Qdrant cosine scores are already in [−1, 1]; normalise to [0, 1]
            scores[chunk_id] = max(0.0, round((r.score + 1.0) / 2.0, 4))
            if len(scores) >= top_k:
                break

        return scores

    # ── Chunk hydration ───────────────────────────────────────────────────────

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        self._ensure_connected()
        if not node_ids:
            return {}

        # Scroll all and filter by payload ID (Qdrant has no "get by payload" shortcut)
        id_set = set(node_ids)
        results = {}
        offset  = None

        while True:
            batch, offset = self._client.scroll(
                collection_name = self.collection_name,
                scroll_filter   = Filter(
                    must=[
                        FieldCondition(
                            key   = self.id_payload_key,
                            match = {"any": list(id_set)},
                        )
                    ]
                ),
                limit           = 100,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            for point in batch:
                chunk = self._point_to_chunk(point)
                results[chunk["id"]] = chunk

            if offset is None or len(results) >= len(id_set):
                break

        return results

    # ── Graph building ────────────────────────────────────────────────────────

    def populate_graph_nodes(self, graph) -> int:
        self._ensure_connected()
        print("[prism] loading all chunks from Qdrant ...")
        added  = 0
        offset = None

        while True:
            batch, offset = self._client.scroll(
                collection_name = self.collection_name,
                limit           = 500,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            for point in batch:
                chunk = self._point_to_chunk(point)
                graph.add_node(
                    chunk["id"],
                    source       = chunk["source"],
                    page         = chunk["page"],
                    section      = chunk["section"],
                    text_preview = chunk["text"][:200],
                )
                added += 1
            if offset is None:
                break

        print(f"[prism] {added:,} nodes added to graph from Qdrant")
        return added

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = False,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        self._ensure_connected()
        print("[prism] loading all chunks + vectors from Qdrant ...")

        all_points = []
        offset     = None
        while True:
            batch, offset = self._client.scroll(
                collection_name = self.collection_name,
                limit           = 500,
                offset          = offset,
                with_payload    = True,
                with_vectors    = True,
            )
            all_points.extend(batch)
            if offset is None:
                break

        print(f"[prism] {len(all_points):,} chunks loaded")

        id_to_chunk = {
            (p.payload or {}).get(self.id_payload_key, str(p.id)): self._point_to_chunk(p)
            for p in all_points
        }

        from tqdm import tqdm
        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for point in tqdm(all_points, desc="finding candidate pairs", unit="chunk"):
            vec = point.vector
            if vec is None:
                continue
            if self.vector_name and isinstance(vec, dict):
                vec = vec.get(self.vector_name)
            if vec is None:
                continue

            row_id    = (point.payload or {}).get(self.id_payload_key, str(point.id))
            row_chunk = id_to_chunk.get(row_id)
            if row_chunk is None:
                continue

            neighbors = self._search_vec(list(vec), limit=k_neighbors + 1)
            for nbr in neighbors:
                nbr_id = (nbr.payload or {}).get(self.id_payload_key, str(nbr.id))
                if nbr_id == row_id:
                    continue
                nbr_chunk = id_to_chunk.get(nbr_id)
                if nbr_chunk is None:
                    continue
                if cross_source_only and nbr_chunk["source"] == row_chunk["source"]:
                    continue

                pair_key = frozenset([row_id, nbr_id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append((row_chunk, nbr_chunk))

                if max_pairs and len(candidates) >= max_pairs:
                    print(f"[prism] max_pairs={max_pairs} reached")
                    return candidates

        print(f"[prism] {len(candidates):,} candidate pairs generated")
        return candidates

    def candidate_pairs_for(
        self,
        node_ids: list[str],
        k_neighbors: int = 8,
        cross_source_only: bool = False,
    ) -> list[tuple[dict, dict]]:
        self._ensure_connected()
        if not node_ids:
            return []

        id_set = set(node_ids)
        # Fetch target nodes with vectors
        target_points = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name = self.collection_name,
                scroll_filter   = Filter(
                    must=[
                        FieldCondition(
                            key   = self.id_payload_key,
                            match = {"any": list(id_set)},
                        )
                    ]
                ),
                limit           = 500,
                offset          = offset,
                with_payload    = True,
                with_vectors    = True,
            )
            target_points.extend(batch)
            if offset is None:
                break

        # Also need a lookup for all chunks (for neighbors)
        all_chunks: dict[str, dict] = {}
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name = self.collection_name,
                limit           = 500,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            for p in batch:
                chunk = self._point_to_chunk(p)
                all_chunks[chunk["id"]] = chunk
            if offset is None:
                break

        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for point in target_points:
            vec = point.vector
            if vec is None:
                continue
            if self.vector_name and isinstance(vec, dict):
                vec = vec.get(self.vector_name)
            if vec is None:
                continue

            row_id    = (point.payload or {}).get(self.id_payload_key, str(point.id))
            row_chunk = all_chunks.get(row_id)
            if row_chunk is None:
                continue

            neighbors = self._search_vec(list(vec), limit=k_neighbors + 1)
            for nbr in neighbors:
                nbr_id = (nbr.payload or {}).get(self.id_payload_key, str(nbr.id))
                if nbr_id == row_id:
                    continue
                nbr_chunk = all_chunks.get(nbr_id)
                if nbr_chunk is None:
                    continue
                if cross_source_only and nbr_chunk["source"] == row_chunk["source"]:
                    continue
                pair_key = frozenset([row_id, nbr_id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append((row_chunk, nbr_chunk))

        return candidates

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self._ensure_connected()
        from collections import Counter
        info    = self._client.get_collection(self.collection_name)
        n       = info.points_count
        sources: Counter = Counter()
        offset  = None
        while True:
            batch, offset = self._client.scroll(
                collection_name = self.collection_name,
                limit           = 500,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            for p in batch:
                sources[(p.payload or {}).get("source", "<unknown>")] += 1
            if offset is None:
                break
        return {
            "total_chunks": n,
            "sources": dict(sources),
            "collection": self.collection_name,
            "embed_model": self._embedder.model,
        }
