"""
prism.graph
-----------
EpistemicGraph — the heterogeneous typed graph at the heart of PRISM.

Wraps NetworkX MultiDiGraph so multiple epistemic edges can exist between
the same pair of nodes (e.g. chunk A can both SUPPORTS and SPECIALIZES chunk B).
Serialises to/from plain JSON for portability — no graph DB required.
"""

from __future__ import annotations

import json
import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx

from .edges import EpistemicEdgeType, PROPAGATION_WEIGHTS


class EpistemicGraph:
    """
    In-memory epistemic knowledge graph.

    Nodes  = chunk IDs (strings matching the 'id' field in LanceDB)
    Edges  = typed epistemic relationships with confidence scores

    Usage:
        g = EpistemicGraph()
        g.add_node("chunk_123", source="dmbok.pdf", page=120, section="4.2 Stewardship")
        g.add_edge("chunk_123", "chunk_456", EpistemicEdgeType.SUPPORTS, confidence=0.92)
        g.save("/path/to/graph.json.gz")

        g2 = EpistemicGraph.load("/path/to/graph.json.gz")
    """

    def __init__(self):
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.meta: dict = {}

    # ── Node management ───────────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        source: str = "",
        page: int = 0,
        section: str = "",
        text_preview: str = "",
    ) -> None:
        self._g.add_node(
            node_id,
            source=source,
            page=page,
            section=section,
            text_preview=text_preview[:200],
        )

    def has_node(self, node_id: str) -> bool:
        return self._g.has_node(node_id)

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    # ── Edge management ───────────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EpistemicEdgeType,
        confidence: float = 1.0,
        rationale: str = "",
    ) -> None:
        """Add a directed epistemic edge. Auto-adds missing nodes."""
        if not self._g.has_node(source_id):
            self._g.add_node(source_id)
        if not self._g.has_node(target_id):
            self._g.add_node(target_id)

        self._g.add_edge(
            source_id,
            target_id,
            type=edge_type.value,
            weight=PROPAGATION_WEIGHTS[edge_type] * confidence,
            confidence=confidence,
            rationale=rationale,
        )

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def has_edge(self, source_id: str, target_id: str) -> bool:
        return self._g.has_edge(source_id, target_id)

    # ── Traversal ─────────────────────────────────────────────────────────────

    def neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[EpistemicEdgeType]] = None,
        min_confidence: float = 0.0,
    ) -> Iterator[tuple[str, EpistemicEdgeType, float, str]]:
        """
        Yield (neighbor_id, edge_type, weight, rationale) for all outgoing
        edges from node_id, optionally filtered by edge type and confidence.
        """
        if not self._g.has_node(node_id):
            return
        allowed = {e.value for e in edge_types} if edge_types else None
        for _, nbr, data in self._g.out_edges(node_id, data=True):
            etype = data.get("type", "")
            conf  = data.get("confidence", 1.0)
            if allowed and etype not in allowed:
                continue
            if conf < min_confidence:
                continue
            try:
                yield nbr, EpistemicEdgeType(etype), data.get("weight", 0.5), data.get("rationale", "")
            except ValueError:
                continue  # unknown edge type — skip

    def incoming(
        self,
        node_id: str,
        edge_types: Optional[list[EpistemicEdgeType]] = None,
        min_confidence: float = 0.0,
    ) -> Iterator[tuple[str, EpistemicEdgeType, float, str]]:
        """Yield (source_id, edge_type, weight, rationale) for all incoming edges."""
        if not self._g.has_node(node_id):
            return
        allowed = {e.value for e in edge_types} if edge_types else None
        for src, _, data in self._g.in_edges(node_id, data=True):
            etype = data.get("type", "")
            conf  = data.get("confidence", 1.0)
            if allowed and etype not in allowed:
                continue
            if conf < min_confidence:
                continue
            try:
                yield src, EpistemicEdgeType(etype), data.get("weight", 0.5), data.get("rationale", "")
            except ValueError:
                continue

    def edges_between(
        self, source_id: str, target_id: str
    ) -> list[tuple[EpistemicEdgeType, float]]:
        """Return all edge types (and weights) between two nodes."""
        result: list[tuple[EpistemicEdgeType, float]] = []
        if not (self._g.has_node(source_id) and self._g.has_node(target_id)):
            return result
        for _, data in self._g.get_edge_data(source_id, target_id, default={}).items():
            try:
                result.append((EpistemicEdgeType(data["type"]), data.get("weight", 0.5)))
            except (ValueError, KeyError):
                continue
        return result

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path, compress: bool = True) -> None:
        """Save graph to JSON (optionally gzipped)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "version": "0.1.0",
            "meta": {
                **self.meta,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "n_nodes": self.node_count(),
                "n_edges": self.edge_count(),
            },
            "nodes": {},
            "edges": [],
        }

        for node_id, attrs in self._g.nodes(data=True):
            data["nodes"][node_id] = {
                "source":       attrs.get("source", ""),
                "page":         attrs.get("page", 0),
                "section":      attrs.get("section", ""),
                "text_preview": attrs.get("text_preview", ""),
            }

        for src, tgt, attrs in self._g.edges(data=True):
            data["edges"].append({
                "source":     src,
                "target":     tgt,
                "type":       attrs.get("type", ""),
                "confidence": attrs.get("confidence", 1.0),
                "rationale":  attrs.get("rationale", ""),
            })

        payload = json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")

        if compress or str(path).endswith(".gz"):
            if not str(path).endswith(".gz"):
                path = Path(str(path) + ".gz")
            with gzip.open(path, "wb") as f:
                f.write(payload)
        else:
            path.write_bytes(payload)

        print(f"[prism] graph saved → {path}  ({data['meta']['n_nodes']:,} nodes, {data['meta']['n_edges']:,} edges)")

    @classmethod
    def load(cls, path: str | Path) -> "EpistemicGraph":
        """Load graph from JSON or gzipped JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PRISM graph not found: {path}")

        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))

        g = cls()
        g.meta = data.get("meta", {})

        for node_id, attrs in data.get("nodes", {}).items():
            g.add_node(
                node_id,
                source=attrs.get("source", ""),
                page=attrs.get("page", 0),
                section=attrs.get("section", ""),
                text_preview=attrs.get("text_preview", ""),
            )

        skip = 0
        for edge in data.get("edges", []):
            try:
                g.add_edge(
                    edge["source"],
                    edge["target"],
                    EpistemicEdgeType(edge["type"]),
                    confidence=edge.get("confidence", 1.0),
                    rationale=edge.get("rationale", ""),
                )
            except (ValueError, KeyError):
                skip += 1

        if skip:
            print(f"[prism] skipped {skip} edges with unknown types during load")

        n, e = g.node_count(), g.edge_count()
        print(f"[prism] graph loaded ← {path}  ({n:,} nodes, {e:,} edges)")
        return g

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        from collections import Counter
        edge_type_counts: Counter = Counter()
        for _, _, data in self._g.edges(data=True):
            edge_type_counts[data.get("type", "unknown")] += 1

        return {
            "n_nodes": self.node_count(),
            "n_edges": self.edge_count(),
            "edge_types": dict(edge_type_counts),
            "avg_out_degree": (
                sum(d for _, d in self._g.out_degree()) / max(self.node_count(), 1)
            ),
            "meta": self.meta,
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def to_networkx(self) -> "nx.MultiDiGraph":
        """Return a copy of the underlying NetworkX MultiDiGraph.

        All node and edge attributes are preserved.  Callers can run any
        NetworkX algorithm (centrality, community detection, shortest paths,
        etc.) directly on the returned graph without affecting the PRISM graph.

        Example::

            import networkx as nx
            G = graph.to_networkx()
            pr = nx.pagerank(G, weight="weight")
        """
        return self._g.copy()

    def to_cypher(self, path: str | Path, batch_size: int = 500) -> Path:
        """Write a Cypher script that recreates the graph in Neo4j.

        Generates ``CREATE`` statements for all nodes and edges, batched into
        transactions of *batch_size* operations.  Run the output file with
        ``cypher-shell`` or paste it into Neo4j Browser::

            cypher-shell -u neo4j -p secret < graph.cypher

        Each chunk node gets the label ``:Chunk`` with properties
        ``id``, ``source``, ``page``, ``section``, and ``text_preview``.
        Each edge gets a relationship type matching the epistemic type in
        upper-case (e.g. ``:SUPPORTS``, ``:REFUTES``) with ``weight`` and
        ``confidence`` properties.

        Args:
            path: Output ``.cypher`` file path.
            batch_size: Number of CREATE statements per transaction block.

        Returns:
            The resolved output path.
        """
        path = Path(path)

        def _esc(v: object) -> str:
            if v is None:
                return "null"
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"

        lines: list[str] = [
            "// PRISM Epistemic Graph — generated by prism-rag",
            f"// {self.node_count():,} nodes  {self.edge_count():,} edges",
            "",
        ]

        nodes = list(self._g.nodes(data=True))
        for i in range(0, len(nodes), batch_size):
            lines.append(":begin")
            for node_id, attrs in nodes[i : i + batch_size]:
                props = (
                    f"id: {_esc(node_id)}, "
                    f"source: {_esc(attrs.get('source', ''))}, "
                    f"page: {_esc(attrs.get('page', 0))}, "
                    f"section: {_esc(attrs.get('section', ''))}, "
                    f"text_preview: {_esc(attrs.get('text_preview', ''))}"
                )
                lines.append(f"CREATE (:Chunk {{{props}}});")
            lines.append(":commit")
            lines.append("")

        lines.append("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id);")
        lines.append("")

        edges = list(self._g.edges(data=True))
        for i in range(0, len(edges), batch_size):
            lines.append(":begin")
            for u, v, data in edges[i : i + batch_size]:
                rel = data.get("type", "related").upper().replace(" ", "_")
                props = (
                    f"weight: {_esc(round(float(data.get('weight', 0.5)), 4))}, "
                    f"confidence: {_esc(round(float(data.get('confidence', 1.0)), 4))}"
                )
                lines.append(
                    f"MATCH (a:Chunk {{id: {_esc(u)}}}), (b:Chunk {{id: {_esc(v)}}})"
                    f" CREATE (a)-[:{rel} {{{props}}}]->(b);"
                )
            lines.append(":commit")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def to_neo4j(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        batch_size: int = 500,
        clear_existing: bool = False,
    ) -> dict:
        """Push the graph directly into a running Neo4j instance.

        Requires the ``neo4j`` Python driver (``pip install prism-rag[neo4j]``).

        Args:
            uri: Bolt URI, e.g. ``"bolt://localhost:7687"``.
            user: Neo4j username.
            password: Neo4j password.
            database: Target database name (default ``"neo4j"``).
            batch_size: Nodes / edges per transaction.
            clear_existing: If ``True``, delete all ``:Chunk`` nodes and their
                relationships before importing (use with care).

        Returns:
            ``{"nodes_created": int, "edges_created": int}``
        """
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "Neo4j export requires the neo4j driver.\n"
                "Install it with:  pip install prism-rag[neo4j]"
            ) from e

        driver = GraphDatabase.driver(uri, auth=(user, password))
        nodes_created = edges_created = 0

        with driver.session(database=database) as session:
            if clear_existing:
                session.run("MATCH (c:Chunk) DETACH DELETE c")

            session.run(
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
            )

            nodes = list(self._g.nodes(data=True))
            for i in range(0, len(nodes), batch_size):
                batch = [
                    {
                        "id": nid,
                        "source": attrs.get("source", ""),
                        "page": attrs.get("page", 0),
                        "section": attrs.get("section", ""),
                        "text_preview": attrs.get("text_preview", ""),
                    }
                    for nid, attrs in nodes[i : i + batch_size]
                ]
                result = session.run(
                    "UNWIND $rows AS row "
                    "CREATE (:Chunk {id: row.id, source: row.source, "
                    "page: row.page, section: row.section, "
                    "text_preview: row.text_preview})",
                    rows=batch,
                )
                nodes_created += len(batch)

            edges = list(self._g.edges(data=True))
            for i in range(0, len(edges), batch_size):
                for u, v, data in edges[i : i + batch_size]:
                    rel = data.get("type", "related").upper().replace(" ", "_")
                    session.run(
                        f"MATCH (a:Chunk {{id: $u}}), (b:Chunk {{id: $v}}) "
                        f"CREATE (a)-[:{rel} {{weight: $w, confidence: $c}}]->(b)",
                        u=u, v=v,
                        w=round(float(data.get("weight", 0.5)), 4),
                        c=round(float(data.get("confidence", 1.0)), 4),
                    )
                    edges_created += 1

        driver.close()
        return {"nodes_created": nodes_created, "edges_created": edges_created}

    def __repr__(self) -> str:
        return (f"EpistemicGraph(nodes={self.node_count():,}, "
                f"edges={self.edge_count():,})")
