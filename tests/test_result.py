"""
Tests for prism.result — EpistemicChunk, EpistemicResult, formatting, bucketing.
"""

from prism.result import EpistemicResult, EpistemicChunk
from prism.edges import EpistemicEdgeType


# ── helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    id_="c1",
    source="doc-alpha",
    score=0.9,
    via=None,
    is_seed=None,
    text="A passage about the topic.",
    page=1,
):
    if is_seed is None:
        is_seed = via is None
    return EpistemicChunk(
        id=id_,
        source=source,
        page=page,
        section="1.0",
        text=text,
        vector_score=score,
        activation=score,
        convergence=0.5,
        final_score=score,
        is_seed=is_seed,
        via_edge_types=via or [],
    )


def make_full_result():
    r = EpistemicResult(query="test query")
    r.primary     = [make_chunk("p1", "doc-a", 0.95)]
    r.supporting  = [make_chunk("s1", "doc-b", 0.80, [EpistemicEdgeType.SUPPORTS])]
    r.contrasting = [make_chunk("c1", "doc-c", 0.70, [EpistemicEdgeType.REFUTES])]
    r.qualifying  = [make_chunk("q1", "doc-d", 0.65, [EpistemicEdgeType.QUALIFIES])]
    r.superseded  = [make_chunk("u1", "doc-e", 0.50, [EpistemicEdgeType.SUPERSEDES])]
    r.n_seeds = 5
    r.n_graph_nodes = 42
    r.graph_was_used = True
    return r


# ── EpistemicChunk ────────────────────────────────────────────────────────────

def test_chunk_source_short_strips_pdf():
    c = make_chunk(source="my-doc.pdf")
    assert c.source_short == "my-doc"


def test_chunk_citation_with_page_and_section():
    c = make_chunk(source="doc.pdf", page=42)
    c.section = "4.2 Governance"
    assert "42" in c.citation
    assert "4.2" in c.citation


def test_chunk_citation_no_page():
    c = EpistemicChunk(
        id="x", source="doc", page=0, section="",
        text="text", vector_score=0.5, activation=0.5,
        convergence=0.5, final_score=0.5,
    )
    assert "doc" in c.citation


def test_chunk_format_text_truncates():
    long_text = "word " * 300  # 1500 chars
    c = make_chunk(text=long_text)
    formatted = c.format_text(max_chars=100)
    assert len(formatted) <= 100


def test_chunk_format_text_strips_newlines():
    c = make_chunk(text="line one\nline two\nline three")
    formatted = c.format_text()
    assert "\n" not in formatted


# ── EpistemicResult — basic ───────────────────────────────────────────────────

def test_empty_result_no_query_error():
    r = EpistemicResult(query="anything")
    assert r.query == "anything"
    assert r.primary == []


def test_all_chunks_aggregates_all_buckets():
    r = make_full_result()
    assert len(r.all_chunks) == 5


def test_has_dialectical_context_false_when_empty():
    r = EpistemicResult(query="q")
    assert not r.has_dialectical_context


def test_has_dialectical_context_true_when_contrasting():
    r = EpistemicResult(query="q")
    r.contrasting = [make_chunk("c1", "doc", 0.7, [EpistemicEdgeType.REFUTES])]
    assert r.has_dialectical_context


def test_has_temporal_context():
    r = EpistemicResult(query="q")
    assert not r.has_temporal_context
    r.superseded = [make_chunk("u1", "doc", 0.5, [EpistemicEdgeType.SUPERSEDES])]
    assert r.has_temporal_context


def test_persona_optional():
    r = EpistemicResult(query="q")
    assert r.persona is None
    r2 = EpistemicResult(query="q", persona="expert")
    assert r2.persona == "expert"


# ── format_for_llm ────────────────────────────────────────────────────────────

def test_format_for_llm_contains_query():
    r = EpistemicResult(query="my specific question")
    out = r.format_for_llm()
    assert "my specific question" in out


def test_format_for_llm_no_sections_when_empty():
    r = EpistemicResult(query="q")
    out = r.format_for_llm()
    assert "PRIMARY" not in out


def test_format_for_llm_shows_primary_section():
    r = EpistemicResult(query="q")
    r.primary = [make_chunk("p1", "my-source", 0.9)]
    out = r.format_for_llm()
    assert "PRIMARY" in out
    assert "my-source" in out


def test_format_for_llm_all_sections_present():
    r = make_full_result()
    out = r.format_for_llm()
    assert "PRIMARY" in out
    assert "SUPPORTING" in out
    assert "CONTRASTING" in out
    assert "QUALIFICATIONS" in out   # rendered as "QUALIFICATIONS & NUANCES"
    assert "superseded" in out.lower()


def test_format_for_llm_shows_persona():
    r = EpistemicResult(query="q", persona="data engineer")
    r.primary = [make_chunk()]
    out = r.format_for_llm()
    assert "data engineer" in out


def test_format_for_llm_shows_stats_line():
    r = make_full_result()
    out = r.format_for_llm()
    # Stats footer should show bucket counts
    assert "primary" in out.lower()
    assert "supporting" in out.lower()


# ── format_mcp ────────────────────────────────────────────────────────────────

def test_format_mcp_returns_string():
    r = make_full_result()
    out = r.format_mcp()
    assert isinstance(out, str)
    assert len(out) > 0


def test_format_mcp_contains_text():
    r = EpistemicResult(query="q")
    r.primary = [make_chunk(text="The answer is here.", source="source-a")]
    out = r.format_mcp()
    assert "source-a" in out or "The answer" in out


# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_structure():
    r = make_full_result()
    d = r.to_dict()
    assert d["query"] == "test query"
    assert "primary" in d
    assert "supporting" in d
    assert "contrasting" in d
    assert "qualifying" in d
    assert "superseded" in d


def test_to_dict_chunk_fields():
    r = EpistemicResult(query="q")
    r.primary = [make_chunk("chunk-1", "my-doc", 0.88)]
    d = r.to_dict()
    chunk = d["primary"][0]
    assert chunk["id"] == "chunk-1"
    assert chunk["source"] == "my-doc"
    assert chunk["scores"]["final"] == 0.88
