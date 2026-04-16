"""
prism.adapters.base
-------------------
VectorAdapter Protocol — the interface any vector store adapter must satisfy
to be plugged into PRISM.

Implement this protocol to connect PRISM to a vector store other than LanceDB
(e.g. Qdrant, Weaviate, Chroma, pgvector, a custom store, etc.).

Example:

    from prism.adapters.base import VectorAdapter

    class MyAdapter:
        def seed_scores(self, query, top_k=20, source_filter=None):
            ...
        def get_chunks(self, node_ids):
            ...
        def connect(self):
            ...
        def populate_graph_nodes(self, graph):
            ...
        def candidate_pairs(self, k_neighbors=8, cross_source_only=False, max_pairs=None):
            ...
        def candidate_pairs_for(self, node_ids, k_neighbors=8, cross_source_only=False):
            ...
        def stats(self):
            ...

    assert isinstance(MyAdapter(), VectorAdapter)   # passes at runtime
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..graph import EpistemicGraph


@runtime_checkable
class VectorAdapter(Protocol):
    """
    Protocol defining the interface required by PRISM for any vector store.

    All methods are required. Implement them to connect PRISM to a custom store.
    """

    def seed_scores(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Embed query and return top-k chunk IDs with their similarity scores.

        Args:
            query:         Query string to embed and search
            top_k:         Number of seed nodes to return
            source_filter: Optional source name substring to restrict search

        Returns:
            {node_id: similarity_score} dict (scores in [0, 1])
        """
        ...

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Fetch full chunk data for a list of node IDs.

        Args:
            node_ids: List of chunk IDs to fetch

        Returns:
            {node_id: {"id": ..., "source": ..., "page": ..., "section": ..., "text": ...}}
            Missing IDs are simply absent from the returned dict.
        """
        ...

    def connect(self) -> None:
        """Open / initialise the connection to the vector store."""
        ...

    def populate_graph_nodes(self, graph: EpistemicGraph) -> int:
        """
        Add all chunks in the store as nodes to the given graph.

        Args:
            graph: EpistemicGraph to populate

        Returns:
            Number of nodes added
        """
        ...

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = False,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        """
        Generate candidate chunk pairs for graph building via k-NN search.

        Args:
            k_neighbors:       k-NN neighbours per chunk
            cross_source_only: Only return pairs from different sources
            max_pairs:         Cap total pairs returned (None = no limit)

        Returns:
            List of (chunk_a, chunk_b) dicts
        """
        ...

    def candidate_pairs_for(
        self,
        node_ids: list[str],
        k_neighbors: int = 8,
        cross_source_only: bool = False,
    ) -> list[tuple[dict, dict]]:
        """
        Generate candidate pairs for a specific subset of node IDs.
        Used for incremental graph updates after adding new documents.

        Args:
            node_ids:          IDs of newly-added chunks
            k_neighbors:       k-NN neighbours to consider
            cross_source_only: Only return pairs from different sources

        Returns:
            List of (chunk_a, chunk_b) dicts involving at least one new chunk
        """
        ...

    def stats(self) -> dict:
        """
        Return a summary dict about the vector store contents.

        Should include at minimum:
            {"n_rows": int, "sources": list[str]}
        """
        ...
