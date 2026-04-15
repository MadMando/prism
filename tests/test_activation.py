"""
Tests for prism.activation — SpreadingActivation engine and convergence scoring.
"""

from prism.graph import EpistemicGraph
from prism.edges import EpistemicEdgeType, EdgeValence
from prism.activation import SpreadingActivation, classify_activation, NodeActivation
from prism.result import ActivationPath


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_linear_graph():
    """seed1 → mid → leaf  (all SUPPORTS)"""
    g = EpistemicGraph()
    for n in ["seed1", "mid", "leaf"]:
        g.add_node(n, source=f"doc_{n}", page=1)
    g.add_edge("seed1", "mid",  EpistemicEdgeType.SUPPORTS, confidence=1.0)
    g.add_edge("mid",   "leaf", EpistemicEdgeType.SUPPORTS, confidence=1.0)
    return g


def make_convergence_graph():
    """
    seed1 ──[supports]──▶ target
    seed2 ──[supports]──▶ target   ← target should get convergence from 2 seeds
    """
    g = EpistemicGraph()
    for n in ["seed1", "seed2", "target", "peripheral"]:
        g.add_node(n, source=f"doc_{n}", page=1)
    g.add_edge("seed1",  "target",     EpistemicEdgeType.SUPPORTS, confidence=0.9)
    g.add_edge("seed2",  "target",     EpistemicEdgeType.SUPPORTS, confidence=0.9)
    g.add_edge("target", "peripheral", EpistemicEdgeType.DERIVES_FROM)
    return g


# ── seed initialisation ───────────────────────────────────────────────────────

def test_seeds_are_in_state():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 0.9})
    assert "seed1" in state
    assert state["seed1"].is_seed


def test_seed_activation_equals_input_score():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 0.85})
    assert abs(state["seed1"].activation - 0.85) < 1e-6


def test_unknown_seed_nodes_are_skipped():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 0.9, "ghost_node": 0.8})
    assert "ghost_node" not in state
    assert "seed1" in state


# ── propagation ───────────────────────────────────────────────────────────────

def test_propagation_reaches_direct_neighbor():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 1.0})
    assert "mid" in state
    assert state["mid"].activation > 0


def test_propagation_does_not_exceed_hops():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 1.0})
    # leaf is 2 hops away — should NOT be reached with hops=1
    assert "leaf" not in state


def test_propagation_reaches_two_hops():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=2)
    state = sa.activate(g, {"seed1": 1.0})
    assert "leaf" in state


def test_activation_decays_per_hop():
    g = make_linear_graph()
    decay = 0.5
    sa = SpreadingActivation(hops=2, decay=decay)
    state = sa.activate(g, {"seed1": 1.0})
    # leaf is 2 hops; activation should be less than mid's
    assert state["leaf"].activation < state["mid"].activation


def test_min_activation_prunes_weak_paths():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=3, decay=0.1, min_activation=0.5)
    state = sa.activate(g, {"seed1": 0.6})
    # After heavy decay, leaf should be pruned
    assert "leaf" not in state


# ── convergence ───────────────────────────────────────────────────────────────

def test_convergence_tracked_for_multiple_seeds():
    g = make_convergence_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 0.9, "seed2": 0.8})
    assert "target" in state
    assert len(state["target"].contributing_seeds) == 2


def test_single_seed_no_convergence_bonus():
    g = make_convergence_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 0.9})
    assert len(state["target"].contributing_seeds) == 1


def test_convergence_boosts_final_score():
    """A node with convergence=2 should outscore an equally-activated node with convergence=1."""
    g = make_convergence_graph()
    sa = SpreadingActivation(hops=1, convergence_weight=0.4)
    state = sa.activate(g, {"seed1": 0.9, "seed2": 0.8})

    n_seeds = 2
    target_score     = state["target"].final_score(n_seeds, 0.4)
    target_act_only  = state["target"].activation

    # Convergence bonus: score > raw activation
    assert target_score > target_act_only


def test_zero_seeds_returns_empty_state():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=3)
    state = sa.activate(g, {})
    assert state == {}


# ── scoring ───────────────────────────────────────────────────────────────────

def test_score_returns_sorted_descending():
    g = make_convergence_graph()
    sa = SpreadingActivation(hops=2)
    state = sa.activate(g, {"seed1": 0.9, "seed2": 0.8})
    scored = sa.score(state, n_seeds=2)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_includes_seeds_by_default():
    g = make_linear_graph()
    sa = SpreadingActivation(hops=1)
    state = sa.activate(g, {"seed1": 0.9})
    scored = sa.score(state, n_seeds=1)
    node_ids = [nid for nid, _ in scored]
    assert "seed1" in node_ids


# ── classify_activation ───────────────────────────────────────────────────────

def test_classify_positive_valence():
    na = NodeActivation(node_id="x", activation=0.5, is_seed=False)
    na.via_edge_types.add(EpistemicEdgeType.SUPPORTS)
    na.paths.append(ActivationPath(from_node="s", edge_type=EpistemicEdgeType.SUPPORTS, step=1, propagated=0.5))
    result = classify_activation("x", na, {"s": 0.9})
    assert result == EdgeValence.POSITIVE


def test_classify_dialectical_valence():
    na = NodeActivation(node_id="x", activation=0.5, is_seed=False)
    na.via_edge_types.add(EpistemicEdgeType.REFUTES)
    na.paths.append(ActivationPath(from_node="s", edge_type=EpistemicEdgeType.REFUTES, step=1, propagated=0.5))
    result = classify_activation("x", na, {"s": 0.9})
    assert result == EdgeValence.DIALECTICAL


def test_classify_temporal_valence():
    na = NodeActivation(node_id="x", activation=0.5, is_seed=False)
    na.via_edge_types.add(EpistemicEdgeType.SUPERSEDES)
    na.paths.append(ActivationPath(from_node="s", edge_type=EpistemicEdgeType.SUPERSEDES, step=1, propagated=0.5))
    result = classify_activation("x", na, {"s": 0.9})
    assert result == EdgeValence.TEMPORAL


def test_classify_no_edge_types_defaults_positive():
    na = NodeActivation(node_id="x", activation=0.5, is_seed=False)
    result = classify_activation("x", na, {"s": 0.9})
    assert result == EdgeValence.POSITIVE


def test_classify_dominant_valence_wins():
    """When multiple edge types present, the one with most propagated activation wins."""
    na = NodeActivation(node_id="x", activation=1.0, is_seed=False)
    na.via_edge_types.update([EpistemicEdgeType.SUPPORTS, EpistemicEdgeType.REFUTES])
    na.paths.append(ActivationPath(from_node="s1", edge_type=EpistemicEdgeType.SUPPORTS,  step=1, propagated=0.8))
    na.paths.append(ActivationPath(from_node="s2", edge_type=EpistemicEdgeType.REFUTES,   step=1, propagated=0.2))
    result = classify_activation("x", na, {"s1": 0.9, "s2": 0.7})
    assert result == EdgeValence.POSITIVE  # SUPPORTS carried more activation
