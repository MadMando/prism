"""
prism.retriever
---------------
Full PRISM retrieval pipeline:
  query → seed activation (LanceDB) → spreading activation (graph)
        → convergence scoring → epistemic bucketing → EpistemicResult

Falls back to pure vector search gracefully when no graph is loaded.
"""

from __future__ import annotations

from typing import Optional

from .activation import SpreadingActivation, classify_activation
from .adapters.lancedb import LanceDBAdapter
from .edges import EdgeValence
from .graph import EpistemicGraph
from .result import EpistemicChunk, EpistemicResult


class PRISMRetriever:
    """
    The core retrieval engine.

    Args:
        adapter:            LanceDBAdapter connected to the vector store
        graph:              EpistemicGraph (can be None for pure vector fallback)
        hops:               Spreading activation depth (default 3)
        decay:              Per-hop activation decay (default 0.7)
        min_activation:     Prune paths below this threshold (default 0.02)
        convergence_weight: Bonus weight for multi-path convergence (default 0.4)
        seed_top_k:         Number of vector-search seed nodes (default 20)
    """

    def __init__(
        self,
        adapter:            LanceDBAdapter,
        graph:              Optional[EpistemicGraph] = None,
        hops:               int   = 3,
        decay:              float = 0.7,
        min_activation:     float = 0.02,
        convergence_weight: float = 0.4,
        seed_top_k:         int   = 20,
    ):
        self.adapter            = adapter
        self.graph              = graph
        self.seed_top_k         = seed_top_k
        self._activation_engine = SpreadingActivation(
            hops               = hops,
            decay              = decay,
            min_activation     = min_activation,
            convergence_weight = convergence_weight,
        )

    # ── Main retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:         str,
        top_k:         int = 5,
        source_filter: Optional[str] = None,
        persona:       Optional[str] = None,
    ) -> EpistemicResult:
        """
        Retrieve epistemically-structured results for a query.

        If a graph is loaded, runs full spreading activation + convergence.
        Otherwise falls back to pure vector search (still returns EpistemicResult).
        """
        result = EpistemicResult(query=query, persona=persona)

        # ── Step 1: Seed activation via vector search ─────────────────────────
        seed_scores = self.adapter.seed_scores(
            query, top_k=self.seed_top_k, source_filter=source_filter
        )
        result.n_seeds = len(seed_scores)

        if not seed_scores:
            return result

        # ── Step 2: Spreading activation through graph ────────────────────────
        if self.graph is not None and self.graph.node_count() > 0:
            result.graph_was_used = True
            activation_state = self._activation_engine.activate(
                self.graph,
                seed_scores,
                source_filter=source_filter,
            )
            n_seeds = len(seed_scores)

            # Score and sort all activated nodes
            scored_nodes = self._activation_engine.score(activation_state, n_seeds)
            result.n_graph_nodes = len(scored_nodes)

            # Count edges traversed
            result.n_edges_traversed = sum(
                len(na.paths) - (1 if na.is_seed else 0)
                for na in activation_state.values()
            )

            # Collect top candidates
            top_node_ids = [nid for nid, _ in scored_nodes[: top_k * 6]]
            scored_map   = {nid: score for nid, score in scored_nodes}

        else:
            # Fallback: use seed scores directly
            activation_state = {}
            scored_map = dict(seed_scores)
            top_node_ids = sorted(seed_scores.keys(), key=lambda k: -seed_scores[k])[: top_k * 2]

        # ── Step 3: Hydrate chunks from LanceDB ───────────────────────────────
        chunks_data = self.adapter.get_chunks(top_node_ids)

        # ── Step 4: Build EpistemicChunk objects ──────────────────────────────
        all_chunks: list[EpistemicChunk] = []
        for node_id in top_node_ids:
            data = chunks_data.get(node_id)
            if not data:
                continue

            na = activation_state.get(node_id)
            if na:
                ec = EpistemicChunk(
                    id           = data["id"],
                    source       = data["source"],
                    page         = data["page"],
                    section      = data["section"],
                    text         = data["text"],
                    vector_score = seed_scores.get(node_id, 0.0),
                    activation   = na.activation,
                    convergence  = len(na.contributing_seeds) / max(len(seed_scores), 1),
                    final_score  = scored_map.get(node_id, 0.0),
                    is_seed      = na.is_seed,
                    paths        = na.paths,
                    via_edge_types = list(na.via_edge_types),
                )
            else:
                # Vector-only fallback chunk
                score = seed_scores.get(node_id, 0.0)
                ec = EpistemicChunk(
                    id           = data["id"],
                    source       = data["source"],
                    page         = data["page"],
                    section      = data["section"],
                    text         = data["text"],
                    vector_score = score,
                    activation   = score,
                    convergence  = 1.0 if node_id in seed_scores else 0.0,
                    final_score  = score,
                    is_seed      = node_id in seed_scores,
                )
            all_chunks.append(ec)

        # ── Step 5: Epistemic bucketing ───────────────────────────────────────
        # Seeds with no incoming epistemic edges → primary
        # Non-seeds classified by dominant edge valence
        primary: list[EpistemicChunk]    = []
        supporting: list[EpistemicChunk] = []
        contrasting: list[EpistemicChunk] = []
        qualifying: list[EpistemicChunk] = []
        superseded: list[EpistemicChunk] = []

        for ec in all_chunks:
            na = activation_state.get(ec.id)
            if na is None or na.is_seed or not na.via_edge_types:
                primary.append(ec)
                continue

            valence = classify_activation(ec.id, na, seed_scores)

            if valence == EdgeValence.POSITIVE:
                supporting.append(ec)
            elif valence == EdgeValence.QUALIFYING:
                qualifying.append(ec)
            elif valence == EdgeValence.DIALECTICAL:
                contrasting.append(ec)
            elif valence == EdgeValence.TEMPORAL:
                superseded.append(ec)
            else:
                primary.append(ec)

        # Sort each bucket by final_score, truncate to top_k
        def _sort(lst: list[EpistemicChunk]) -> list[EpistemicChunk]:
            return sorted(lst, key=lambda c: c.final_score, reverse=True)

        result.primary     = _sort(primary)[:top_k]
        result.supporting  = _sort(supporting)[:max(2, top_k // 2)]
        result.contrasting = _sort(contrasting)[:2]
        result.qualifying  = _sort(qualifying)[:2]
        result.superseded  = _sort(superseded)[:2]

        return result
