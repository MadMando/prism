"""
prism.adapters.chroma
---------------------
ChromaDB adapter — connects PRISM to an existing ChromaDB collection.

Requires the chroma extra::

    pip install prism-rag[chroma]

Your ChromaDB collection must store chunks with:
  - ids         : chunk ID strings (used as PRISM node IDs)
  - documents   : chunk text
  - metadatas   : dicts containing at minimum {"source": ..., "page": ..., "section": ...}

Embedding providers supported (same as LanceDB adapter):
  - Ollama (default): pass embed_model="nomic-embed-text"
  - Any OpenAI-compatible API: pass embed_api_url + embed_api_key

Important: configure your ChromaDB collection with cosine distance
(hnsw:space="cosine") to get correct similarity scores. L2 is the
ChromaDB default but gives less interpretable scores for PRISM.
"""

from __future__ import annotations

from typing import Optional

try:
    import chromadb
except ImportError as _e:
    raise ImportError(
        "ChromaAdapter requires chromadb. Install it with:\n"
        "    pip install prism-rag[chroma]"
    ) from _e

from .embedder import Embedder


class ChromaAdapter:
    """
    Connects PRISM to an existing ChromaDB collection.

    Args:
        collection_name: ChromaDB collection name
        host:            ChromaDB HTTP host (default "localhost")
        port:            ChromaDB HTTP port (default 8000)
        path:            Path for persistent local client (overrides host/port)
        distance_metric: "cosine" (recommended) or "l2" — must match the
                         collection's hnsw:space setting
        embed_model:     Embedding model name
        ollama_url:      Ollama base URL
        embed_api_url:   OpenAI-compatible embeddings endpoint
        embed_api_key:   API key (switches to API mode when set)
        embed_timeout:   Request timeout in seconds
    """

    def __init__(
        self,
        collection_name: str = "knowledge",
        host:            str = "localhost",
        port:            int = 8000,
        path:            Optional[str] = None,
        distance_metric: str = "cosine",
        # Embedding
        embed_model:     str = "nomic-embed-text",
        ollama_url:      str = "http://localhost:11434",
        embed_api_url:   Optional[str] = None,
        embed_api_key:   Optional[str] = None,
        embed_timeout:   int = 60,
    ):
        self.collection_name = collection_name
        self.host            = host
        self.port            = port
        self.path            = path
        self.distance_metric = distance_metric.lower()

        self._embedder = Embedder(
            model      = embed_model,
            api_url    = embed_api_url,
            api_key    = embed_api_key,
            ollama_url = ollama_url,
            timeout    = embed_timeout,
        )

        self._client:     Optional[chromadb.ClientAPI] = None
        self._collection = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.path:
            self._client = chromadb.PersistentClient(path=self.path)
        else:
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
        self._collection = self._client.get_collection(self.collection_name)

    def _ensure_connected(self) -> None:
        if self._collection is None:
            self.connect()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _distance_to_score(self, distance: float) -> float:
        if self.distance_metric == "cosine":
            return max(0.0, round(1.0 - distance, 4))
        # L2: no universal mapping; use inverse
        return round(1.0 / (1.0 + distance), 4)

    # ── Seed activation ───────────────────────────────────────────────────────

    def seed_scores(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> dict[str, float]:
        self._ensure_connected()
        vec = self._embedder.embed(query)

        # Chroma's `where` operators are scalar equality only (no $contains on
        # strings), so substring matching has to happen client-side.
        fetch = top_k * 4 if source_filter else top_k * 2
        results = self._collection.query(
            query_embeddings=[vec],
            n_results=fetch,
            include=["distances", "metadatas"],
        )

        ids       = results["ids"][0]
        distances = results["distances"][0]
        metas     = results.get("metadatas", [[]])[0] or [{}] * len(ids)

        if source_filter:
            needle   = source_filter.lower()
            filtered = [
                (i, d) for i, d, m in zip(ids, distances, metas)
                if needle in (m or {}).get("source", "").lower()
            ]
            ids       = [x[0] for x in filtered]
            distances = [x[1] for x in filtered]

        ids       = ids[:top_k]
        distances = distances[:top_k]

        return {
            cid: self._distance_to_score(dist)
            for cid, dist in zip(ids, distances)
        }

    # ── Chunk hydration ───────────────────────────────────────────────────────

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        self._ensure_connected()
        if not node_ids:
            return {}

        result = self._collection.get(
            ids=node_ids,
            include=["documents", "metadatas"],
        )

        chunks = {}
        for cid, doc, meta in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            if meta is None:
                meta = {}
            chunks[cid] = {
                "id":      cid,
                "source":  meta.get("source", ""),
                "page":    int(meta.get("page", 0)),
                "section": meta.get("section", ""),
                "text":    doc or "",
            }
        return chunks

    # ── Graph building ────────────────────────────────────────────────────────

    def populate_graph_nodes(self, graph) -> int:
        self._ensure_connected()
        print("[prism] loading all chunks from ChromaDB ...")
        result = self._collection.get(include=["documents", "metadatas"])
        added = 0
        for cid, doc, meta in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            if meta is None:
                meta = {}
            graph.add_node(
                cid,
                source       = meta.get("source", ""),
                page         = int(meta.get("page", 0)),
                section      = meta.get("section", ""),
                text_preview = (doc or "")[:200],
            )
            added += 1
        print(f"[prism] {added:,} nodes added to graph from ChromaDB")
        return added

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = False,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        self._ensure_connected()
        print("[prism] loading all chunks for candidate generation ...")
        result = self._collection.get(include=["embeddings", "documents", "metadatas"])

        ids   = result.get("ids", [])
        embs  = result.get("embeddings", [])
        docs  = result.get("documents", [])
        metas = result.get("metadatas", [])

        if not ids or embs is None:
            print("[prism] WARNING: no embeddings returned — cannot generate pairs")
            return []

        print(f"[prism] {len(ids):,} chunks loaded")

        id_to_data = {
            cid: {
                "id":      cid,
                "source":  (metas[i] or {}).get("source", ""),
                "page":    int((metas[i] or {}).get("page", 0)),
                "section": (metas[i] or {}).get("section", ""),
                "text":    docs[i] or "",
            }
            for i, cid in enumerate(ids)
        }

        from tqdm import tqdm
        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for i, (cid, vec) in enumerate(tqdm(zip(ids, embs), total=len(ids), desc="finding candidate pairs", unit="chunk")):
            nbr_results = self._collection.query(
                query_embeddings=[list(vec)],
                n_results=k_neighbors + 1,
                include=["metadatas"],
            )
            nbr_ids   = nbr_results["ids"][0]
            nbr_metas = nbr_results["metadatas"][0]

            row_data = id_to_data[cid]

            for nbr_id, nbr_meta in zip(nbr_ids, nbr_metas):
                if nbr_id == cid:
                    continue
                nbr_data = id_to_data.get(nbr_id)
                if nbr_data is None:
                    continue
                if cross_source_only and nbr_data["source"] == row_data["source"]:
                    continue

                pair_key = frozenset([cid, nbr_id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append((row_data, nbr_data))

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

        result = self._collection.get(
            ids=node_ids,
            include=["embeddings", "documents", "metadatas"],
        )
        ids   = result.get("ids", [])
        embs  = result.get("embeddings", [])
        docs  = result.get("documents", [])
        metas = result.get("metadatas", [])

        all_result  = self._collection.get(include=["documents", "metadatas"])
        all_id_data = {
            cid: {
                "id":      cid,
                "source":  (all_result["metadatas"][i] or {}).get("source", ""),
                "page":    int((all_result["metadatas"][i] or {}).get("page", 0)),
                "section": (all_result["metadatas"][i] or {}).get("section", ""),
                "text":    all_result["documents"][i] or "",
            }
            for i, cid in enumerate(all_result.get("ids", []))
        }

        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for i, (cid, vec) in enumerate(zip(ids, embs)):
            if vec is None:
                continue
            row_data = {
                "id":      cid,
                "source":  (metas[i] or {}).get("source", ""),
                "page":    int((metas[i] or {}).get("page", 0)),
                "section": (metas[i] or {}).get("section", ""),
                "text":    docs[i] or "",
            }
            nbr_results = self._collection.query(
                query_embeddings=[list(vec)],
                n_results=k_neighbors + 1,
                include=["metadatas"],
            )
            for nbr_id in nbr_results["ids"][0]:
                if nbr_id == cid:
                    continue
                nbr_data = all_id_data.get(nbr_id)
                if nbr_data is None:
                    continue
                if cross_source_only and nbr_data["source"] == row_data["source"]:
                    continue
                pair_key = frozenset([cid, nbr_id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append((row_data, nbr_data))

        return candidates

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self._ensure_connected()
        from collections import Counter
        n = self._collection.count()
        result = self._collection.get(include=["metadatas"])
        sources: Counter = Counter()
        for meta in result.get("metadatas", []):
            sources[meta.get("source", "<unknown>")] += 1
        return {
            "total_chunks": n,
            "sources": dict(sources),
            "collection": self.collection_name,
            "embed_model": self._embedder.model,
        }
