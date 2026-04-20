"""
Tests for prism.adapters.pgvector.PgvectorAdapter.

Mocks the psycopg2 connection / cursor so no real Postgres is required.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg2")

from prism.adapters.pgvector import PgvectorAdapter


class FakeCursor:
    """
    Minimal context-manager cursor. `fetchalls` is a list where each element
    is the list of rows returned by one `fetchall()` call (in order).
    """

    def __init__(self, fetchalls):
        self._fetchalls = list(fetchalls)
        self.executes   = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executes.append((sql, params))

    def fetchall(self):
        if not self._fetchalls:
            return []
        return self._fetchalls.pop(0)

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None


class FakeConn:
    """
    `cursor_plan` is a list where each element is itself a list of fetchall
    results for the N-th cursor opened. Order matters.
    """

    closed = False

    def __init__(self, cursor_plan):
        self._cursor_plan    = list(cursor_plan)
        self.cursors_created = []

    def cursor(self, name=None):
        fetchalls = self._cursor_plan.pop(0) if self._cursor_plan else []
        cur = FakeCursor(fetchalls)
        self.cursors_created.append(cur)
        return cur


def _make_adapter(cursor_plan=None):
    adapter = PgvectorAdapter(dsn="postgresql://fake")
    adapter._embedder = MagicMock()
    adapter._embedder.embed.return_value = [0.1] * 4
    adapter._embedder.model = "mock-embed"
    adapter._conn = FakeConn(cursor_plan or [])
    return adapter


# ── seed_scores ───────────────────────────────────────────────────────────────

def test_seed_scores_returns_scores():
    rows = [
        ("a", "s1", 0.9),
        ("b", "s2", 0.8),
    ]
    adapter = _make_adapter(cursor_plan=[[rows]])
    scores  = adapter.seed_scores("q", top_k=2)
    assert scores == {"a": 0.9, "b": 0.8}


def test_seed_scores_applies_source_filter_client_side():
    rows = [
        ("a", "dmbok", 0.9),
        ("b", "nist",  0.8),
    ]
    adapter = _make_adapter(cursor_plan=[[rows]])
    scores  = adapter.seed_scores("q", top_k=5, source_filter="dmbok")
    assert set(scores.keys()) == {"a"}


# ── get_chunks ────────────────────────────────────────────────────────────────

def test_get_chunks_empty_input():
    adapter = _make_adapter()
    assert adapter.get_chunks([]) == {}


def test_get_chunks_builds_dict():
    rows = [
        ("a", "s1", 1, "§1", "text A"),
        ("b", "s2", 2, "§2", "text B"),
    ]
    adapter = _make_adapter(cursor_plan=[[rows]])
    chunks = adapter.get_chunks(["a", "b"])
    assert chunks["a"]["text"] == "text A"
    assert chunks["b"]["page"] == 2


# ── candidate_pairs_for cursor-split fix ──────────────────────────────────────

def test_candidate_pairs_for_uses_two_cursors():
    """
    Regression test: fetch_sql and nbr_sql must use separate cursors so the
    neighbour query cannot invalidate the target-rows fetch buffer.
    """
    target_rows = [
        ("a", "s1", 1, "§", "text", "[0.1,0.2]"),
    ]
    neighbor_rows = [
        ("b", "s2", 1, "§", "text"),
    ]
    # Two cursors: first fetches target_rows, second fetches neighbor_rows.
    adapter = _make_adapter(cursor_plan=[[target_rows], [neighbor_rows]])
    pairs = adapter.candidate_pairs_for(["a"], k_neighbors=2)

    assert len(adapter._conn.cursors_created) == 2
    assert len(pairs) == 1
    assert pairs[0][0]["id"] == "a"
    assert pairs[0][1]["id"] == "b"


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_counts():
    """
    PgvectorAdapter.stats() runs two queries on the same cursor: a COUNT (read
    via fetchone) then a GROUP BY source (read via fetchall). Stage the fake
    cursor to return the right shape for each.
    """
    class StatsCursor(FakeCursor):
        def __init__(self):
            super().__init__([])
            self._stage = 0
        def execute(self, sql, params=None):
            self._stage += 1
        def fetchone(self):
            return (3,) if self._stage == 1 else None
        def fetchall(self):
            return [("x", 2), ("y", 1)] if self._stage == 2 else []

    class OneCursorConn:
        closed = False
        def __init__(self):
            self.cur = StatsCursor()
        def cursor(self, name=None):
            return self.cur

    adapter = _make_adapter()
    adapter._conn = OneCursorConn()

    s = adapter.stats()
    assert s["total_chunks"] == 3
    assert s["sources"]      == {"x": 2, "y": 1}
