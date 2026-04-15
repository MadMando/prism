"""
prism.edges
-----------
Epistemic edge type taxonomy — the core differentiator of PRISM.

Standard knowledge graphs model what things ARE (is-a, has-a, related-to).
Epistemic graphs model what knowledge MEANS TO OTHER KNOWLEDGE:
  - Does this chunk support, refute, supersede, or derive from another?
  - Is this a specific case of a general principle, or the reverse?

This is the thing that doesn't exist in any current RAG library.
"""

from enum import Enum


class EpistemicEdgeType(str, Enum):
    # ── Reinforcing ────────────────────────────────────────────────────────────
    SUPPORTS      = "supports"       # A provides evidence/reasoning for B's claims
    DERIVES_FROM  = "derives_from"   # A is logically/conceptually derived from B
    SPECIALIZES   = "specializes"    # A is a specific case or application of B
    IMPLEMENTS    = "implements"     # A is the practical tool/method for B's concept
    EXEMPLIFIES   = "exemplifies"    # A is a concrete example illustrating B
    GENERALIZES   = "generalizes"    # A is a broader abstraction of which B is a case

    # ── Modifying ─────────────────────────────────────────────────────────────
    QUALIFIES     = "qualifies"      # A adds conditions, exceptions, or nuance to B

    # ── Dialectical ───────────────────────────────────────────────────────────
    CONTRASTS_WITH = "contrasts_with"  # A and B take different positions (both valid)
    REFUTES       = "refutes"          # A directly contradicts or undermines B

    # ── Temporal ──────────────────────────────────────────────────────────────
    SUPERSEDES    = "supersedes"     # A replaces B (A is newer or more authoritative)


# ── Propagation weights ───────────────────────────────────────────────────────
# How strongly activation spreads through each edge type.
# Reinforcing edges propagate strongly. Dialectical edges propagate but flag
# the node as counter-evidence. Temporal edges propagate with a deprecation flag.

PROPAGATION_WEIGHTS: dict[EpistemicEdgeType, float] = {
    EpistemicEdgeType.SUPPORTS:        0.90,
    EpistemicEdgeType.DERIVES_FROM:    0.85,
    EpistemicEdgeType.SPECIALIZES:     0.80,
    EpistemicEdgeType.IMPLEMENTS:      0.75,
    EpistemicEdgeType.EXEMPLIFIES:     0.75,
    EpistemicEdgeType.GENERALIZES:     0.70,
    EpistemicEdgeType.QUALIFIES:       0.65,
    EpistemicEdgeType.CONTRASTS_WITH:  0.55,  # propagates — dialectical context matters
    EpistemicEdgeType.REFUTES:         0.50,  # propagates — knowing what refutes is valuable
    EpistemicEdgeType.SUPERSEDES:      0.40,  # propagates weakly — historical context only
}

# ── Valence ───────────────────────────────────────────────────────────────────
# Used to structure EpistemicResult into sections: primary, supporting,
# contrasting, qualifying, superseded.

class EdgeValence(str, Enum):
    POSITIVE    = "positive"    # reinforces the answer
    QUALIFYING  = "qualifying"  # nuances the answer
    DIALECTICAL = "dialectical" # challenges or contrasts the answer
    TEMPORAL    = "temporal"    # historical / superseded context

EDGE_VALENCE: dict[EpistemicEdgeType, EdgeValence] = {
    EpistemicEdgeType.SUPPORTS:        EdgeValence.POSITIVE,
    EpistemicEdgeType.DERIVES_FROM:    EdgeValence.POSITIVE,
    EpistemicEdgeType.SPECIALIZES:     EdgeValence.POSITIVE,
    EpistemicEdgeType.IMPLEMENTS:      EdgeValence.POSITIVE,
    EpistemicEdgeType.EXEMPLIFIES:     EdgeValence.POSITIVE,
    EpistemicEdgeType.GENERALIZES:     EdgeValence.POSITIVE,
    EpistemicEdgeType.QUALIFIES:       EdgeValence.QUALIFYING,
    EpistemicEdgeType.CONTRASTS_WITH:  EdgeValence.DIALECTICAL,
    EpistemicEdgeType.REFUTES:         EdgeValence.DIALECTICAL,
    EpistemicEdgeType.SUPERSEDES:      EdgeValence.TEMPORAL,
}

# ── Human-readable labels ─────────────────────────────────────────────────────
EDGE_LABELS: dict[EpistemicEdgeType, str] = {
    EpistemicEdgeType.SUPPORTS:        "supports",
    EpistemicEdgeType.DERIVES_FROM:    "derived from",
    EpistemicEdgeType.SPECIALIZES:     "specializes",
    EpistemicEdgeType.IMPLEMENTS:      "implements",
    EpistemicEdgeType.EXEMPLIFIES:     "exemplifies",
    EpistemicEdgeType.GENERALIZES:     "generalizes",
    EpistemicEdgeType.QUALIFIES:       "qualifies",
    EpistemicEdgeType.CONTRASTS_WITH:  "contrasts with",
    EpistemicEdgeType.REFUTES:         "refutes",
    EpistemicEdgeType.SUPERSEDES:      "supersedes",
}

# ── Edge directionality for reverse traversal ─────────────────────────────────
# When traversing BACKWARDS through the graph, which edge types should also
# propagate activation (i.e. the relationship is meaningful in both directions)?
BIDIRECTIONAL_EDGE_TYPES: frozenset[EpistemicEdgeType] = frozenset({
    EpistemicEdgeType.CONTRASTS_WITH,
    EpistemicEdgeType.REFUTES,
})
