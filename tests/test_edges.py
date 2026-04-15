"""
Tests for prism.edges — epistemic edge taxonomy, propagation weights, valence.
"""

from prism.edges import EpistemicEdgeType, PROPAGATION_WEIGHTS, EDGE_VALENCE, EdgeValence


def test_all_edge_types_have_propagation_weight():
    """Every edge type must have a propagation weight defined."""
    for etype in EpistemicEdgeType:
        assert etype in PROPAGATION_WEIGHTS, f"{etype} missing from PROPAGATION_WEIGHTS"


def test_all_propagation_weights_are_valid():
    """Propagation weights must be strictly between 0 and 1."""
    for etype, weight in PROPAGATION_WEIGHTS.items():
        assert 0.0 < weight <= 1.0, f"{etype} weight {weight} out of range (0, 1]"


def test_all_edge_types_have_valence():
    """Every edge type must map to an EdgeValence bucket."""
    for etype in EpistemicEdgeType:
        assert etype in EDGE_VALENCE, f"{etype} missing from EDGE_VALENCE"


def test_all_valences_are_valid_enum_members():
    """All valence mappings must be valid EdgeValence enum values."""
    for etype, valence in EDGE_VALENCE.items():
        assert isinstance(valence, EdgeValence)


def test_supports_has_highest_weight():
    """SUPPORTS should carry activation most strongly."""
    supports_w = PROPAGATION_WEIGHTS[EpistemicEdgeType.SUPPORTS]
    for etype, weight in PROPAGATION_WEIGHTS.items():
        assert supports_w >= weight, f"supports ({supports_w}) < {etype} ({weight})"


def test_supersedes_has_lowest_weight():
    """SUPERSEDES should have the lowest propagation weight (historical, less relevant)."""
    supersedes_w = PROPAGATION_WEIGHTS[EpistemicEdgeType.SUPERSEDES]
    for etype, weight in PROPAGATION_WEIGHTS.items():
        assert supersedes_w <= weight, f"supersedes ({supersedes_w}) > {etype} ({weight})"


def test_refutes_has_dialectical_valence():
    assert EDGE_VALENCE[EpistemicEdgeType.REFUTES] == EdgeValence.DIALECTICAL


def test_contrasts_with_has_dialectical_valence():
    assert EDGE_VALENCE[EpistemicEdgeType.CONTRASTS_WITH] == EdgeValence.DIALECTICAL


def test_supersedes_has_temporal_valence():
    assert EDGE_VALENCE[EpistemicEdgeType.SUPERSEDES] == EdgeValence.TEMPORAL


def test_supports_has_positive_valence():
    assert EDGE_VALENCE[EpistemicEdgeType.SUPPORTS] == EdgeValence.POSITIVE


def test_qualifies_has_qualifying_valence():
    assert EDGE_VALENCE[EpistemicEdgeType.QUALIFIES] == EdgeValence.QUALIFYING


def test_edge_type_values_are_strings():
    """Edge types should be usable as plain strings for JSON serialisation."""
    for etype in EpistemicEdgeType:
        assert isinstance(etype.value, str)
        assert etype.value == etype.value.lower()


def test_edge_type_roundtrip_from_string():
    """Every edge type string value should reconstruct the enum."""
    for etype in EpistemicEdgeType:
        reconstructed = EpistemicEdgeType(etype.value)
        assert reconstructed == etype
