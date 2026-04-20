"""
prism.adapters.weaviate
-----------------------
Weaviate adapter — connects PRISM to an existing Weaviate collection.

Requires the weaviate extra::

    pip install prism-rag[weaviate]

Uses the Weaviate Python client v4 (weaviate-client>=4.0).

Your Weaviate collection must store objects with properties:
  - chunk_id  : str  — PRISM chunk ID (configurable via id_property)
  - source    : str  — source document name
  - page      : int  — page number
  - section   : str  — section title
  - text      : str  — chunk text

The Weaviate object UUID is used internally; PRISM identifies chunks
by the chunk_id property value.
"""

from __future__ import annotations

from typing import Optional

try:
    import weaviate
    import weaviate.classes as wvc
except ImportError as _e:
    raise ImportError(
        "WeaviateAdapter requires weaviate-client>=4.0. Install it with:\n"
        "    pip install prism-rag[weaviate]"
    ) from _e

from .embedder import Embedder


class WeaviateAdapter:
    """
    Connects PRISM to an existing Weaviate collection (v4 client).

    Args:
        collection_name: Weaviate collection / class name
        host:            Weaviate HTTP host (default "localhost")
        port:            Weaviate HTTP port (default 8080)
        grpc_port:       Weaviate gRPC port (default 50051)
        api_key:         Weaviate API key (for Weaviate Cloud)
        wcs_url:         Weaviate Cloud Services cluster URL (overrides host/port)
        id_property:     Object property holding the PRISM chunk ID (default "chunk_id")
        embed_model:     Embedding model name
        ollama_url:      Ollama base URL
        embed_api_url:   OpenAI-compatible embeddings endpoint
        embed_api_key:   API key (switches to API mode when set)
        embed_timeout:   Request timeout in seconds
    """

    def __init__(
        self,
        collection_name: str = "Knowledge",
        host:            str = "localhost",
        port:            int = 8080,
        grpc_port:       int = 50051,
        api_key:         Optional[str] = None,
        wcs_url:         Optional[str] = None,
        id_property:     str = "chunk_id",
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
        self.grpc_port       = grpc_port
        self.weaviate_api_key = api_key
        self.wcs_url         = wcs_url
        self.id_property     = id_property

        self._embedder = Embedder(
            model      = embed_model,
            api_url    = embed_api_url,
            api_key    = embed_api_key,
            ollama_url = ollama_url,
            timeout    = embed_timeout,
        )

        self._client     = None
        self._collection = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.wcs_url:
            auth = weaviate.auth.AuthApiKey(self.weaviate_api_key) if self.weaviate_api_key else None
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url        = self.wcs_url,
                auth_credentials   = auth,
            )
        else:
            self._client = weaviate.connect_to_local(
                host      = self.host,
                port      = self.port,
                grpc_port = self.grpc_port,
            )
        self._collection = self._client.collections.get(self.collection_name)

    def _ensure_connected(self) -> None:
        if self._collection is None:
            self.connect()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _obj_to_chunk(self, obj) -> dict:
        props = obj.properties or {}
        return {
            "id":      str(props.get(self.id_property, str(obj.uuid))),
            "source":  str(props.get("source", "")),
            "page":    int(props.get("page", 0)),
            "section": str(props.get("section", "")),
            "text":    str(props.get("text", "")),
        }

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

        response = self._collection.query.near_vector(
            near_vector     = vec,
            limit           = fetch,
            return_metadata = wvc.query.MetadataQuery(distance=True),
        )

        scores = {}
        for obj in response.objects:
            chunk = self._obj_to_chunk(obj)
            if source_filter and source_filter.lower() not in chunk["source"].lower():
                continue
            distance = obj.metadata.distance or 0.0
            scores[chunk["id"]] = max(0.0, round(1.0 - distance, 4))
            if len(scores) >= top_k:
                break

        return scores

    # ── Chunk hydration ───────────────────────────────────────────────────────

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        self._ensure_connected()
        if not node_ids:
            return {}

        results = {}
        for chunk_id in node_ids:
            try:
                response = self._collection.query.fetch_objects(
                    filters = wvc.query.Filter.by_property(self.id_property).equal(chunk_id),
                    limit   = 1,
                )
                if response.objects:
                    chunk = self._obj_to_chunk(response.objects[0])
                    results[chunk["id"]] = chunk
            except Exception:
                continue

        return results

    # ── Graph building ────────────────────────────────────────────────────────

    def populate_graph_nodes(self, graph) -> int:
        self._ensure_connected()
        print("[prism] loading all chunks from Weaviate ...")
        added = 0
        cursor = None

        while True:
            response = self._collection.query.fetch_objects(
                limit        = 500,
                after        = cursor,
                return_properties = [self.id_property, "source", "page", "section", "text"],
            )
            if not response.objects:
                break

            for obj in response.objects:
                chunk = self._obj_to_chunk(obj)
                graph.add_node(
                    chunk["id"],
                    source       = chunk["source"],
                    page         = chunk["page"],
                    section      = chunk["section"],
                    text_preview = chunk["text"][:200],
                )
                added += 1

            cursor = response.objects[-1].uuid

        print(f"[prism] {added:,} nodes added to graph from Weaviate")
        return added

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = False,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        self._ensure_connected()
        print("[prism] loading all chunks + vectors from Weaviate ...")

        all_objs = []
        cursor   = None
        while True:
            response = self._collection.query.fetch_objects(
                limit          = 500,
                after          = cursor,
                include_vector = True,
            )
            if not response.objects:
                break
            all_objs.extend(response.objects)
            cursor = response.objects[-1].uuid

        print(f"[prism] {len(all_objs):,} chunks loaded")

        id_to_chunk = {self._obj_to_chunk(obj)["id"]: self._obj_to_chunk(obj) for obj in all_objs}

        from tqdm import tqdm
        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for obj in tqdm(all_objs, desc="finding candidate pairs", unit="chunk"):
            vec = obj.vector
            if vec is None:
                continue
            if isinstance(vec, dict):
                vec = vec.get("default") or next(iter(vec.values()), None)
            if vec is None:
                continue

            row_chunk = self._obj_to_chunk(obj)
            row_id    = row_chunk["id"]

            response = self._collection.query.near_vector(
                near_vector = list(vec),
                limit       = k_neighbors + 1,
            )
            for nbr_obj in response.objects:
                nbr_chunk = self._obj_to_chunk(nbr_obj)
                nbr_id    = nbr_chunk["id"]
                if nbr_id == row_id:
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

        # Build full ID→chunk map first
        all_chunks: dict[str, dict] = {}
        cursor = None
        while True:
            response = self._collection.query.fetch_objects(limit=500, after=cursor)
            if not response.objects:
                break
            for obj in response.objects:
                chunk = self._obj_to_chunk(obj)
                all_chunks[chunk["id"]] = chunk
            cursor = response.objects[-1].uuid

        id_set = set(node_ids)
        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        for chunk_id in id_set:
            try:
                response = self._collection.query.fetch_objects(
                    filters        = wvc.query.Filter.by_property(self.id_property).equal(chunk_id),
                    limit          = 1,
                    include_vector = True,
                )
                if not response.objects:
                    continue
                obj = response.objects[0]
            except Exception:
                continue

            vec = obj.vector
            if vec is None:
                continue
            if isinstance(vec, dict):
                vec = vec.get("default") or next(iter(vec.values()), None)
            if vec is None:
                continue

            row_chunk = self._obj_to_chunk(obj)

            nbr_response = self._collection.query.near_vector(
                near_vector = list(vec),
                limit       = k_neighbors + 1,
            )
            for nbr_obj in nbr_response.objects:
                nbr_chunk = self._obj_to_chunk(nbr_obj)
                nbr_id    = nbr_chunk["id"]
                if nbr_id == chunk_id:
                    continue
                if cross_source_only and nbr_chunk["source"] == row_chunk["source"]:
                    continue
                pair_key = frozenset([chunk_id, nbr_id])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append((row_chunk, nbr_chunk))

        return candidates

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self._ensure_connected()
        from collections import Counter
        info = self._client.collections.get(self.collection_name)
        sources: Counter = Counter()
        cursor = None
        total  = 0
        while True:
            response = self._collection.query.fetch_objects(limit=500, after=cursor)
            if not response.objects:
                break
            for obj in response.objects:
                props = obj.properties or {}
                sources[props.get("source", "<unknown>")] += 1
                total += 1
            cursor = response.objects[-1].uuid
        return {
            "total_chunks": total,
            "sources": dict(sources),
            "collection": self.collection_name,
            "embed_model": self._embedder.model,
        }
