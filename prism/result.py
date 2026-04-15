"""
prism.result
------------
EpistemicResult — structured retrieval output.

Unlike plain RAG that returns a flat list of chunks, PRISM returns knowledge
organised by its epistemic role: what the primary answer is, what supports it,
what challenges it, what qualifies it, and what historical context has been
superseded. This structure is the visible product of the epistemic graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .edges import EpistemicEdgeType, EDGE_LABELS


@dataclass
class ActivationPath:
    """Records how a chunk was reached during spreading activation."""
    from_node:  str
    edge_type:  Optional[EpistemicEdgeType]  # None for seed nodes
    step:       int                          # 0 = seed, 1+ = propagated
    propagated: float                        # activation contributed


@dataclass
class EpistemicChunk:
    """A retrieved chunk with full epistemic context."""
    id:           str
    source:       str
    page:         int
    section:      str
    text:         str

    # Scores
    vector_score:    float  # raw semantic similarity (0-1, higher = closer)
    activation:      float  # total activation received through the graph
    convergence:     float  # fraction of seeds that independently activated this
    final_score:     float  # activation * (1 + convergence_bonus)

    # How it was reached
    is_seed:         bool = False  # directly activated by vector search
    paths:           list[ActivationPath] = field(default_factory=list)
    via_edge_types:  list[EpistemicEdgeType] = field(default_factory=list)

    @property
    def source_short(self) -> str:
        return self.source.replace(".pdf", "").replace("_compress", "")

    @property
    def citation(self) -> str:
        parts = [self.source_short]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(f"§ {self.section}")
        return "  ".join(parts)

    def format_text(self, max_chars: int = 600) -> str:
        return self.text[:max_chars].replace("\n", " ").strip()


@dataclass
class EpistemicResult:
    """
    Structured retrieval result from PRISM.

    Results are grouped by epistemic role, not just ranked by score:
      primary    — the core relevant chunks (high convergence, positive valence)
      supporting — chunks that reinforce/extend the primary answer
      contrasting — chunks that challenge or offer a different perspective
      qualifying  — chunks that add nuance, conditions, or exceptions
      superseded  — historically relevant but now outdated context
    """
    query:       str
    persona:     Optional[str] = None

    primary:     list[EpistemicChunk] = field(default_factory=list)
    supporting:  list[EpistemicChunk] = field(default_factory=list)
    contrasting: list[EpistemicChunk] = field(default_factory=list)
    qualifying:  list[EpistemicChunk] = field(default_factory=list)
    superseded:  list[EpistemicChunk] = field(default_factory=list)

    # Stats
    n_seeds:           int   = 0
    n_graph_nodes:     int   = 0
    n_edges_traversed: int   = 0
    graph_was_used:    bool  = False

    @property
    def all_chunks(self) -> list[EpistemicChunk]:
        return self.primary + self.supporting + self.contrasting + self.qualifying + self.superseded

    @property
    def has_dialectical_context(self) -> bool:
        return bool(self.contrasting)

    @property
    def has_temporal_context(self) -> bool:
        return bool(self.superseded)

    def format_for_llm(self, max_primary: int = 5, max_per_section: int = 2) -> str:
        """
        Format the result for injection into an LLM context window.
        Structured so the model understands the epistemic landscape, not just
        a list of relevant chunks.
        """
        lines: list[str] = []
        lines.append(f'PRISM retrieval for: "{self.query}"')
        if self.persona:
            lines.append(f"Persona: {self.persona}")
        lines.append("─" * 60)

        def _fmt_section(title: str, chunks: list[EpistemicChunk], limit: int):
            if not chunks:
                return
            lines.append(f"\n## {title}")
            for i, c in enumerate(chunks[:limit], 1):
                via = ""
                if c.via_edge_types:
                    labels = [EDGE_LABELS[e] for e in c.via_edge_types[:2]]
                    via = f"  [via: {', '.join(labels)}]"
                lines.append(f"[{i}] {c.citation}  (score: {c.final_score:.3f}{via})")
                lines.append(f"    {c.format_text(500)}")

        _fmt_section("PRIMARY", self.primary, max_primary)
        _fmt_section("SUPPORTING EVIDENCE", self.supporting, max_per_section)
        _fmt_section("CONTRASTING PERSPECTIVES", self.contrasting, max_per_section)
        _fmt_section("QUALIFICATIONS & NUANCES", self.qualifying, max_per_section)
        _fmt_section("HISTORICAL CONTEXT (superseded)", self.superseded, max_per_section)

        lines.append(f"\n─ {len(self.primary)} primary · "
                     f"{len(self.supporting)} supporting · "
                     f"{len(self.contrasting)} contrasting · "
                     f"{len(self.qualifying)} qualifying · "
                     f"{len(self.superseded)} superseded ─")
        if self.graph_was_used:
            lines.append(f"  graph: {self.n_seeds} seeds → "
                         f"{self.n_graph_nodes} nodes activated "
                         f"({self.n_edges_traversed} edges traversed)")
        return "\n".join(lines)

    def format_mcp(self) -> str:
        """Compact format for MCP tool output (knowledge_search compatible)."""
        return self.format_for_llm(max_primary=5, max_per_section=2)

    def to_dict(self) -> dict:
        def chunk_dict(c: EpistemicChunk) -> dict:
            return {
                "id": c.id, "source": c.source, "page": c.page,
                "section": c.section, "text": c.text,
                "scores": {
                    "vector": round(c.vector_score, 4),
                    "activation": round(c.activation, 4),
                    "convergence": round(c.convergence, 4),
                    "final": round(c.final_score, 4),
                },
                "is_seed": c.is_seed,
                "via_edge_types": [e.value for e in c.via_edge_types],
            }
        return {
            "query": self.query,
            "persona": self.persona,
            "primary":     [chunk_dict(c) for c in self.primary],
            "supporting":  [chunk_dict(c) for c in self.supporting],
            "contrasting": [chunk_dict(c) for c in self.contrasting],
            "qualifying":  [chunk_dict(c) for c in self.qualifying],
            "superseded":  [chunk_dict(c) for c in self.superseded],
            "stats": {
                "n_seeds": self.n_seeds,
                "n_graph_nodes": self.n_graph_nodes,
                "n_edges_traversed": self.n_edges_traversed,
                "graph_was_used": self.graph_was_used,
            },
        }
