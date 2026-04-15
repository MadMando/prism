"""
Tests for prism.extractor — JSON parsing, confidence filtering, batch handling.

These tests cover the parsing and validation logic only — no live API calls.
"""

from prism.extractor import EpistemicExtractor
from prism.edges import EpistemicEdgeType


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_extractor(min_confidence=0.65):
    return EpistemicExtractor(
        base_url="https://api.example.com",
        model="test-model",
        api_key="test-key",
        min_confidence=min_confidence,
        batch_size=3,
    )


def make_pair(id_a="a1", id_b="b1", src_a="source-x", src_b="source-y"):
    return (
        {"id": id_a, "text": "Text for chunk A", "source": src_a, "section": "1.0"},
        {"id": id_b, "text": "Text for chunk B", "source": src_b, "section": "2.0"},
    )


# ── _parse_single ─────────────────────────────────────────────────────────────

def test_parse_single_valid_relationship():
    ex = make_extractor()
    raw = '{"has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.9, "rationale": "A supports B"}'
    result = ex._parse_single(raw, "a1", "b1")
    assert result is not None
    assert result.edge_type == EpistemicEdgeType.SUPPORTS
    assert result.confidence == 0.9
    assert result.source_id == "a1"
    assert result.target_id == "b1"
    assert result.rationale == "A supports B"


def test_parse_single_no_relationship():
    ex = make_extractor()
    raw = '{"has_relationship": false}'
    assert ex._parse_single(raw, "a1", "b1") is None


def test_parse_single_below_confidence_threshold():
    ex = make_extractor(min_confidence=0.65)
    raw = '{"has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.3, "rationale": "weak"}'
    assert ex._parse_single(raw, "a1", "b1") is None


def test_parse_single_at_confidence_boundary():
    ex = make_extractor(min_confidence=0.65)
    raw = '{"has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.65, "rationale": "borderline"}'
    result = ex._parse_single(raw, "a1", "b1")
    assert result is not None  # exactly at threshold = accepted


def test_parse_single_invalid_json():
    ex = make_extractor()
    assert ex._parse_single("this is not json", "a1", "b1") is None


def test_parse_single_empty_string():
    ex = make_extractor()
    assert ex._parse_single("", "a1", "b1") is None


def test_parse_single_unknown_edge_type():
    ex = make_extractor()
    raw = '{"has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "invented_type_xyz", "confidence": 0.9, "rationale": "test"}'
    assert ex._parse_single(raw, "a1", "b1") is None


def test_parse_single_json_embedded_in_surrounding_text():
    """LLMs often prepend/append text — the parser must extract the JSON."""
    ex = make_extractor()
    raw = 'Sure! Here is my analysis:\n{"has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "refutes", "confidence": 0.85, "rationale": "A refutes B"}\nHope that helps!'
    result = ex._parse_single(raw, "a1", "b1")
    assert result is not None
    assert result.edge_type == EpistemicEdgeType.REFUTES


def test_parse_single_reversed_direction_respected():
    """The LLM can flip source/target — should honour what it returns."""
    ex = make_extractor()
    raw = '{"has_relationship": true, "source_id": "b1", "target_id": "a1", "edge_type": "derives_from", "confidence": 0.8, "rationale": "B derives from A"}'
    result = ex._parse_single(raw, "a1", "b1")
    assert result is not None
    assert result.source_id == "b1"
    assert result.target_id == "a1"


def test_parse_single_all_edge_types():
    """All valid edge types should parse without error."""
    ex = make_extractor()
    for etype in EpistemicEdgeType:
        raw = f'{{"has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "{etype.value}", "confidence": 0.9, "rationale": "test"}}'
        result = ex._parse_single(raw, "a1", "b1")
        assert result is not None, f"Failed to parse edge type: {etype.value}"
        assert result.edge_type == etype


# ── _parse_batch ──────────────────────────────────────────────────────────────

def test_parse_batch_all_relationships():
    ex = make_extractor()
    pairs = [make_pair("a1", "b1"), make_pair("a2", "b2")]
    raw = '[{"pair_index": 0, "has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.8, "rationale": "ok"}, {"pair_index": 1, "has_relationship": true, "source_id": "a2", "target_id": "b2", "edge_type": "refutes", "confidence": 0.75, "rationale": "ok"}]'
    results = ex._parse_batch(raw, pairs)
    assert len(results) == 2
    assert results[0] is not None
    assert results[0].edge_type == EpistemicEdgeType.SUPPORTS
    assert results[1] is not None
    assert results[1].edge_type == EpistemicEdgeType.REFUTES


def test_parse_batch_mixed_results():
    ex = make_extractor()
    pairs = [make_pair("a1", "b1"), make_pair("a2", "b2")]
    raw = '[{"pair_index": 0, "has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.8, "rationale": "ok"}, {"pair_index": 1, "has_relationship": false}]'
    results = ex._parse_batch(raw, pairs)
    assert results[0] is not None
    assert results[1] is None


def test_parse_batch_malformed_json_returns_nones():
    ex = make_extractor()
    pairs = [make_pair()]
    results = ex._parse_batch("not json at all", pairs)
    assert results == [None]


def test_parse_batch_empty_array_returns_nones():
    ex = make_extractor()
    pairs = [make_pair()]
    results = ex._parse_batch("[]", pairs)
    assert results == [None]


def test_parse_batch_out_of_bounds_index_ignored():
    ex = make_extractor()
    pairs = [make_pair("a1", "b1")]  # only index 0 valid
    raw = '[{"pair_index": 99, "has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.9, "rationale": "r"}]'
    results = ex._parse_batch(raw, pairs)
    assert results == [None]


def test_parse_batch_confidence_filter_applied():
    ex = make_extractor(min_confidence=0.65)
    pairs = [make_pair()]
    raw = '[{"pair_index": 0, "has_relationship": true, "source_id": "a1", "target_id": "b1", "edge_type": "supports", "confidence": 0.2, "rationale": "weak"}]'
    results = ex._parse_batch(raw, pairs)
    assert results == [None]


def test_parse_batch_preserves_correct_length():
    ex = make_extractor()
    pairs = [make_pair(f"a{i}", f"b{i}") for i in range(5)]
    raw = '[{"pair_index": 2, "has_relationship": true, "source_id": "a2", "target_id": "b2", "edge_type": "supports", "confidence": 0.8, "rationale": "ok"}]'
    results = ex._parse_batch(raw, pairs)
    assert len(results) == 5
    assert results[2] is not None
    for i in [0, 1, 3, 4]:
        assert results[i] is None
