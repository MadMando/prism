"""
prism.activation
----------------
Spreading activation over the epistemic graph.

Inspired by Collins & Loftus (1975) spreading activation theory — the same
mechanism the human brain uses to retrieve associated memories. A query
activates seed nodes; activation propagates through epistemic edges; nodes
activated by multiple independent paths (high convergence) rank highest.

This is fundamentally different from standard RAG retrieval:
  Standard RAG:   query → similarity → ranked list  (single signal)
  PRISM:          query → seeds → propagation → convergence  (multi-signal)

The key insight: a chunk reached by 4 independent activation paths is almost
certainly relevant. A chunk reached by 1 path at high similarity might just
be a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .edges import EpistemicEdgeType, EDGE_VALENCE, EdgeValence
from .graph import EpistemicGraph
from .result import ActivationPath


# ── Activation state ──────────────────────────────────────────────────────────

@dataclass
class NodeActivation:
    """Activation state for a single graph node."""
    node_id:          str
    activation:       float                       = 0.0
    contributing_seeds: set[str]                  = field(default_factory=set)
    paths:            list[ActivationPath]        = field(default_factory=list)
    via_edge_types:   set[EpistemicEdgeType]      = field(default_factory=set)
    is_seed:          bool                        = False

    @property
    def convergence(self) -> float:
        """Fraction of total seeds that independently contributed to this node."""
        # Set externally after full propagation (needs total seed count)
        return len(self.contributing_seeds)

    def final_score(self, n_total_seeds: int, convergence_weight: float = 0.4) -> float:
        """
        Final ranking score combining activation level and convergence.
        convergence_weight: how much extra weight multi-path convergence adds.
        """
        if n_total_seeds == 0:
            return self.activation
        convergence_ratio = len(self.contributing_seeds) / n_total_seeds
        return self.activation * (1.0 + convergence_weight * convergence_ratio)


# ── Spreading activation engine ───────────────────────────────────────────────

class SpreadingActivation:
    """
    Propagates activation from seed nodes through the epistemic graph.

    Args:
        hops:               Maximum propagation depth (default 3)
        decay:              Per-hop decay multiplier (default 0.7)
        min_activation:     Prune paths below this threshold (default 0.02)
        convergence_weight: Bonus weight for multi-path convergence (default 0.4)
        use_reverse_edges:  Also propagate backwards through incoming edges (default True)
    """

    def __init__(
        self,
        hops: int = 3,
        decay: float = 0.7,
        min_activation: float = 0.02,
        convergence_weight: float = 0.4,
        use_reverse_edges: bool = True,
    ):
        self.hops               = hops
        self.decay              = decay
        self.min_activation     = min_activation
        self.convergence_weight = convergence_weight
        self.use_reverse_edges  = use_reverse_edges

    def activate(
        self,
        graph: EpistemicGraph,
        seed_scores: dict[str, float],         # {node_id: initial_activation}
        source_filter: Optional[str] = None,   # only propagate to nodes from this source
    ) -> dict[str, NodeActivation]:
        """
        Run spreading activation from seed nodes.

        Returns a dict of {node_id: NodeActivation} for all activated nodes,
        including seeds.
        """
        state: dict[str, NodeActivation] = {}

        # Initialise seed nodes
        for node_id, score in seed_scores.items():
            if not graph.has_node(node_id):
                continue
            na = NodeActivation(
                node_id          = node_id,
                activation       = score,
                is_seed          = True,
            )
            na.contributing_seeds.add(node_id)
            na.paths.append(ActivationPath(
                from_node  = node_id,
                edge_type  = None,
                step       = 0,
                propagated = score,
            ))
            state[node_id] = na

        # Propagate hop by hop
        current_frontier: dict[str, float] = dict(seed_scores)

        for step in range(1, self.hops + 1):
            next_frontier: dict[str, float] = {}
            hop_decay = self.decay ** (step - 1)

            for node_id, frontier_activation in current_frontier.items():
                if node_id not in state:
                    continue

                seed_origins = state[node_id].contributing_seeds

                # Forward edges
                for nbr, edge_type, weight, rationale in graph.neighbors(node_id):
                    propagated = frontier_activation * weight * hop_decay
                    if propagated < self.min_activation:
                        continue
                    if source_filter and _node_source(graph, nbr) != source_filter:
                        pass  # source filter does not block — it's optional hint
                    self._accumulate(state, next_frontier, nbr, node_id, edge_type,
                                     propagated, step, seed_origins)

                # Reverse edges (optional — catches "B supports A" when traversing from B)
                if self.use_reverse_edges:
                    for src, edge_type, weight, rationale in graph.incoming(node_id):
                        propagated = frontier_activation * weight * hop_decay * 0.6  # reverse penalty
                        if propagated < self.min_activation:
                            continue
                        self._accumulate(state, next_frontier, src, node_id, edge_type,
                                         propagated, step, seed_origins)

            current_frontier = next_frontier
            if not current_frontier:
                break  # activation died out

        return state

    def _accumulate(
        self,
        state: dict[str, NodeActivation],
        frontier: dict[str, float],
        node_id: str,
        from_node: str,
        edge_type: EpistemicEdgeType,
        propagated: float,
        step: int,
        seed_origins: set[str],
    ) -> None:
        if node_id not in state:
            state[node_id] = NodeActivation(node_id=node_id)

        na = state[node_id]
        na.activation += propagated
        na.contributing_seeds.update(seed_origins)
        na.via_edge_types.add(edge_type)
        na.paths.append(ActivationPath(
            from_node  = from_node,
            edge_type  = edge_type,
            step       = step,
            propagated = propagated,
        ))

        frontier[node_id] = frontier.get(node_id, 0.0) + propagated

    def score(
        self,
        state: dict[str, NodeActivation],
        n_seeds: int,
        exclude_seeds: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Return (node_id, final_score) sorted descending.
        Seeds are included by default (they always rank high).
        """
        scored = []
        for node_id, na in state.items():
            if exclude_seeds and na.is_seed:
                continue
            scored.append((node_id, na.final_score(n_seeds, self.convergence_weight)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def _node_source(graph: EpistemicGraph, node_id: str) -> str:
    """Safely get the source of a graph node."""
    if graph.has_node(node_id):
        return graph._g.nodes[node_id].get("source", "")
    return ""


# ── Epistemic bucket classifier ───────────────────────────────────────────────

def classify_activation(
    node_id: str,
    na: NodeActivation,
    seed_scores: dict[str, float],
) -> EdgeValence:
    """
    Classify a non-seed activated node into an epistemic bucket
    based on the dominant edge types used to reach it.
    """
    if not na.via_edge_types:
        return EdgeValence.POSITIVE  # default for seeds / unknowns

    valence_counts: dict[EdgeValence, float] = {}
    for path in na.paths:
        if path.edge_type is None:
            continue
        v = EDGE_VALENCE.get(path.edge_type, EdgeValence.POSITIVE)
        valence_counts[v] = valence_counts.get(v, 0.0) + path.propagated

    if not valence_counts:
        return EdgeValence.POSITIVE

    return max(valence_counts, key=lambda k: valence_counts[k])
