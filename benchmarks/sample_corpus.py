"""
benchmarks/sample_corpus.py
----------------------------
A self-contained benchmark that demonstrates PRISM's epistemic bucketing
against a small synthetic corpus — no real vector store or LLM required.

What this shows
---------------
- Standard vector search returns a flat list ordered by similarity only.
- PRISM with a pre-built graph returns the same passages structured
  by their epistemic role: what supports the answer, what qualifies it,
  what contradicts it.

Run it:
    python benchmarks/sample_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.graph import EpistemicGraph
from prism.edges import EpistemicEdgeType
from prism.result import EpistemicChunk, EpistemicResult


# ── Synthetic corpus ──────────────────────────────────────────────────────────
# 8 short passages on "accountability in distributed systems"
# drawn from three imaginary sources: framework-A, framework-B, standard-C

CORPUS = [
    {
        "id": "fa-001",
        "source": "framework-a",
        "page": 14,
        "section": "2.1 Ownership",
        "text": "Accountability in distributed systems requires explicit ownership assignment. "
                "Every data asset must have a named owner responsible for its quality and lifecycle.",
    },
    {
        "id": "fa-002",
        "source": "framework-a",
        "page": 22,
        "section": "3.0 Enforcement",
        "text": "Ownership without enforcement mechanisms is symbolic. Accountability requires "
                "tooling to detect violations and workflows to escalate unresolved issues.",
    },
    {
        "id": "fb-001",
        "source": "framework-b",
        "page": 8,
        "section": "1.2 Stewardship",
        "text": "Data stewardship operationalises accountability. A steward is the individual "
                "or team who exercises day-to-day accountability for a defined data domain.",
    },
    {
        "id": "fb-002",
        "source": "framework-b",
        "page": 31,
        "section": "4.1 Federated Model",
        "text": "In federated architectures, accountability may be distributed across domain "
                "owners rather than centralised in a single governance office. This reduces "
                "bottlenecks but requires strong coordination protocols.",
    },
    {
        "id": "fb-003",
        "source": "framework-b",
        "page": 45,
        "section": "5.0 Critique",
        "text": "Strict centralised accountability models can create governance theatre: "
                "nominal ownership with no real authority. Framework-A's enforcement model "
                "assumes organisational authority that rarely exists in practice.",
    },
    {
        "id": "sc-001",
        "source": "standard-c",
        "page": 3,
        "section": "Scope",
        "text": "This standard defines accountability roles for data management. "
                "Superseded by revision 2.0 (2024). Retained for historical reference only.",
    },
    {
        "id": "sc-002",
        "source": "standard-c",
        "page": 7,
        "section": "4. Definitions",
        "text": "Accountability: the obligation to accept responsibility for actions, "
                "decisions, and their consequences within an assigned domain.",
    },
    {
        "id": "sc-003",
        "source": "standard-c",
        "page": 12,
        "section": "6. Implementation",
        "text": "Implementation of accountability frameworks should begin with a pilot "
                "domain to test ownership assignment before full rollout.",
    },
]


# ── Pre-built epistemic graph ─────────────────────────────────────────────────
# In a real deployment this graph is built once offline by EpistemicExtractor.
# Here we define it manually so the benchmark runs without an LLM.

def build_sample_graph() -> EpistemicGraph:
    g = EpistemicGraph()

    for doc in CORPUS:
        g.add_node(
            doc["id"],
            source=doc["source"],
            page=doc["page"],
            section=doc["section"],
            text_preview=doc["text"][:200],
        )

    # Framework-B's stewardship definition specialises Framework-A's ownership concept
    g.add_edge("fb-001", "fa-001", EpistemicEdgeType.SPECIALIZES,
               confidence=0.88,
               rationale="Stewardship (fb-001) is the practical specialisation of ownership accountability (fa-001)")

    # Framework-B's enforcement enforcement observation supports Framework-A's claim
    g.add_edge("fb-002", "fa-001", EpistemicEdgeType.QUALIFIES,
               confidence=0.82,
               rationale="Federated model (fb-002) qualifies centralised ownership (fa-001) with conditions")

    # Framework-B's critique directly refutes Framework-A's enforcement model
    g.add_edge("fb-003", "fa-002", EpistemicEdgeType.REFUTES,
               confidence=0.79,
               rationale="Critique (fb-003) challenges the authority assumption in fa-002's enforcement model")

    # Standard-C's definition derives from the same concept as Framework-A
    g.add_edge("sc-002", "fa-001", EpistemicEdgeType.DERIVES_FROM,
               confidence=0.75,
               rationale="Standard-C's accountability definition derives from the same ownership principle")

    # Standard-C v1 has been superseded
    g.add_edge("sc-001", "sc-002", EpistemicEdgeType.SUPERSEDES,
               confidence=0.95,
               rationale="sc-001 notes it is superseded; sc-002 is the active definition")

    # Implementation note exemplifies the ownership assignment concept
    g.add_edge("sc-003", "fa-001", EpistemicEdgeType.EXEMPLIFIES,
               confidence=0.71,
               rationale="sc-003's pilot guidance exemplifies how fa-001's ownership assignment works in practice")

    return g


# ── Simulated retrieval results ───────────────────────────────────────────────

def simulate_vector_search(query_topic: str, top_k: int = 5) -> list[dict]:
    """
    Simulate vector search results (ordered by mock similarity score).
    In production these come from an actual embedding + ANN search.
    """
    # Scores hand-tuned to reflect semantic relevance to "accountability ownership"
    mock_scores = {
        "fa-001": 0.93,   # most relevant — core ownership concept
        "fb-001": 0.88,   # stewardship ≈ operationalised ownership
        "sc-002": 0.85,   # definition of accountability
        "fa-002": 0.80,   # enforcement = related to ownership
        "fb-002": 0.74,   # federated model — related but tangential
        "fb-003": 0.68,   # critique — topically similar
        "sc-003": 0.62,   # implementation — relevant
        "sc-001": 0.55,   # superseded standard — low relevance signal
    }
    ranked = sorted(CORPUS, key=lambda d: -mock_scores.get(d["id"], 0))
    return [(d, mock_scores[d["id"]]) for d in ranked[:top_k]]


def build_prism_result(query: str, seed_scores: dict[str, float], graph: EpistemicGraph, top_k: int = 5) -> EpistemicResult:
    """
    Simulate PRISM retrieval using the pre-built graph + mock seed scores.
    (In production, PRISMRetriever does this automatically.)
    """
    from prism.activation import SpreadingActivation, classify_activation
    from prism.edges import EdgeValence

    sa = SpreadingActivation(hops=3, decay=0.7, convergence_weight=0.4)
    state = sa.activate(graph, seed_scores)
    scored = sa.score(state, n_seeds=len(seed_scores))
    scored_map = dict(scored)

    corpus_map = {d["id"]: d for d in CORPUS}

    result = EpistemicResult(query=query)
    result.graph_was_used = True
    result.n_seeds = len(seed_scores)
    result.n_graph_nodes = len(state)

    primary, supporting, contrasting, qualifying, superseded = [], [], [], [], []

    for node_id, score in scored[:top_k * 4]:
        doc = corpus_map.get(node_id)
        if not doc:
            continue
        na = state.get(node_id)
        if na is None:
            continue

        chunk = EpistemicChunk(
            id=node_id,
            source=doc["source"],
            page=doc["page"],
            section=doc["section"],
            text=doc["text"],
            vector_score=seed_scores.get(node_id, 0.0),
            activation=na.activation,
            convergence=len(na.contributing_seeds) / max(len(seed_scores), 1),
            final_score=score,
            is_seed=na.is_seed,
            paths=na.paths,
            via_edge_types=list(na.via_edge_types),
        )

        if na.is_seed or not na.via_edge_types:
            primary.append(chunk)
        else:
            valence = classify_activation(node_id, na, seed_scores)
            if valence == EdgeValence.POSITIVE:
                supporting.append(chunk)
            elif valence == EdgeValence.QUALIFYING:
                qualifying.append(chunk)
            elif valence == EdgeValence.DIALECTICAL:
                contrasting.append(chunk)
            elif valence == EdgeValence.TEMPORAL:
                superseded.append(chunk)
            else:
                primary.append(chunk)

    def _sort(lst):
        return sorted(lst, key=lambda c: c.final_score, reverse=True)

    result.primary     = _sort(primary)[:top_k]
    result.supporting  = _sort(supporting)[:3]
    result.contrasting = _sort(contrasting)[:2]
    result.qualifying  = _sort(qualifying)[:2]
    result.superseded  = _sort(superseded)[:2]
    return result


# ── Main comparison ───────────────────────────────────────────────────────────

def main():
    query = "accountability and ownership in distributed systems"
    top_k = 5

    print("=" * 70)
    print("PRISM Benchmark — Sample Corpus")
    print(f"Query: {query!r}")
    print("=" * 70)

    # ── Standard RAG (vector search only) ────────────────────────────────────
    print("\n── STANDARD RAG (vector similarity only) ──────────────────────────\n")
    vector_results = simulate_vector_search(query, top_k=top_k)
    for i, (doc, score) in enumerate(vector_results, 1):
        print(f"[{i}] ({score:.2f})  [{doc['source']}  p.{doc['page']}]")
        print(f"     {doc['text'][:120]}...")
        print()

    print("Result: a flat list. No indication of which passages support,")
    print("contradict, qualify, or supersede each other.\n")

    # ── PRISM (epistemic graph + spreading activation) ────────────────────────
    print("── PRISM (epistemic graph + spreading activation) ──────────────────\n")

    graph = build_sample_graph()
    print(f"Graph: {graph}")
    print(f"  Edge types: {graph.stats()['edge_types']}\n")

    seed_scores = {doc["id"]: score for doc, score in simulate_vector_search(query, top_k=20)}
    result = build_prism_result(query, seed_scores, graph, top_k=top_k)
    print(result.format_for_llm())

    # ── Head-to-head comparison ───────────────────────────────────────────────
    print("\n── Head-to-head comparison ─────────────────────────────────────────\n")
    print("Standard RAG top-5:  same 5 passages, ranked by cosine similarity")
    print("                     → LLM gets a flat context with no relational signal")
    print()
    print("PRISM top-5:         passages bucketed by epistemic role")
    print(f"  PRIMARY      : {len(result.primary)} chunks  (core answer)")
    print(f"  SUPPORTING   : {len(result.supporting)} chunks  (reinforcing evidence)")
    print(f"  CONTRASTING  : {len(result.contrasting)} chunks  (challenges the answer)")
    print(f"  QUALIFYING   : {len(result.qualifying)} chunks  (conditions / nuances)")
    print(f"  SUPERSEDED   : {len(result.superseded)} chunks  (historical, now outdated)")
    print()
    print("  → LLM context explicitly signals that fb-003 REFUTES fa-002's enforcement")
    print("    model, and sc-001 is superseded. The model can reason about this rather")
    print("    than treating all passages as equally authoritative.")
    print()
    print(f"Graph stats: {result.n_seeds} seeds → {result.n_graph_nodes} nodes activated")


if __name__ == "__main__":
    main()
