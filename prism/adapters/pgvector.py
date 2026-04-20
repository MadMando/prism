"""
prism.adapters.pgvector
-----------------------
pgvector adapter — connects PRISM to a PostgreSQL table with pgvector embeddings.

Requires the pgvector extra::

    pip install prism-rag[pgvector]

Your PostgreSQL table must have at minimum these columns:

    CREATE TABLE chunks (
        id        TEXT PRIMARY KEY,
        source    TEXT,
        page      INTEGER,
        section   TEXT,
        text      TEXT,
        embedding vector(N)   -- N = your embedding dimension
    );

The table name and column names are configurable via constructor args.
"""

from __future__ import annotations

from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError as _e:
    raise ImportError(
        "PgvectorAdapter requires psycopg2. Install it with:\n"
        "    pip install prism-rag[pgvector]"
    ) from _e

from .embedder import Embedder


class PgvectorAdapter:
    """
    Connects PRISM to a PostgreSQL table using pgvector for vector search.

    Args:
        dsn:         PostgreSQL DSN string, e.g.
                     "postgresql://user:pass@localhost:5432/mydb"
        table:       Table name (default "chunks")
        id_col:      Column name for the chunk ID (default "id")
        source_col:  Column for source document name (default "source")
        page_col:    Column for page number (default "page")
        section_col: Column for section title (default "section")
        text_col:    Column for chunk text (default "text")
        vector_col:  Column for the pgvector embedding (default "embedding")
        embed_model:     Embedding model name
        ollama_url:      Ollama base URL
        embed_api_url:   OpenAI-compatible embeddings endpoint
        embed_api_key:   API key (switches to API mode when set)
        embed_timeout:   Request timeout in seconds
    """

    def __init__(
        self,
        dsn:         str,
        table:       str = "chunks",
        id_col:      str = "id",
        source_col:  str = "source",
        page_col:    str = "page",
        section_col: str = "section",
        text_col:    str = "text",
        vector_col:  str = "embedding",
        # Embedding
        embed_model:     str = "nomic-embed-text",
        ollama_url:      str = "http://localhost:11434",
        embed_api_url:   Optional[str] = None,
        embed_api_key:   Optional[str] = None,
        embed_timeout:   int = 60,
    ):
        self.dsn         = dsn
        self.table       = table
        self.id_col      = id_col
        self.source_col  = source_col
        self.page_col    = page_col
        self.section_col = section_col
        self.text_col    = text_col
        self.vector_col  = vector_col

        self._embedder = Embedder(
            model      = embed_model,
            api_url    = embed_api_url,
            api_key    = embed_api_key,
            ollama_url = ollama_url,
            timeout    = embed_timeout,
        )

        self._conn = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._conn = psycopg2.connect(self.dsn)
        self._conn.autocommit = True

    def _ensure_connected(self) -> None:
        if self._conn is None or self._conn.closed:
            self.connect()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vec_literal(self, vec: list[float]) -> str:
        """Format a vector as a pgvector literal string."""
        return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

    def _row_to_chunk(self, row: tuple) -> dict:
        return {
            "id":      str(row[0]),
            "source":  str(row[1] or ""),
            "page":    int(row[2] or 0),
            "section": str(row[3] or ""),
            "text":    str(row[4] or ""),
        }

    # ── Seed activation ───────────────────────────────────────────────────────

    def seed_scores(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> dict[str, float]:
        self._ensure_connected()
        vec     = self._embedder.embed(query)
        vec_lit = self._vec_literal(vec)
        fetch   = top_k * 4 if source_filter else top_k * 2

        sql = (
            f"SELECT {self.id_col}, {self.source_col}, "
            f"1 - ({self.vector_col} <=> %s::vector) AS score "
            f"FROM {self.table} "
            f"ORDER BY {self.vector_col} <=> %s::vector "
            f"LIMIT %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (vec_lit, vec_lit, fetch))
            rows = cur.fetchall()

        scores = {}
        for row in rows:
            chunk_id, source, score = row[0], row[1] or "", float(row[2])
            if source_filter and source_filter.lower() not in source.lower():
                continue
            scores[str(chunk_id)] = max(0.0, round(score, 4))
            if len(scores) >= top_k:
                break

        return scores

    # ── Chunk hydration ───────────────────────────────────────────────────────

    def get_chunks(self, node_ids: list[str]) -> dict[str, dict]:
        self._ensure_connected()
        if not node_ids:
            return {}

        sql = (
            f"SELECT {self.id_col}, {self.source_col}, {self.page_col}, "
            f"{self.section_col}, {self.text_col} "
            f"FROM {self.table} "
            f"WHERE {self.id_col} = ANY(%s)"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (list(node_ids),))
            rows = cur.fetchall()

        return {str(row[0]): self._row_to_chunk(row) for row in rows}

    # ── Graph building ────────────────────────────────────────────────────────

    def populate_graph_nodes(self, graph) -> int:
        self._ensure_connected()
        print("[prism] loading all chunks from PostgreSQL ...")
        sql = (
            f"SELECT {self.id_col}, {self.source_col}, {self.page_col}, "
            f"{self.section_col}, {self.text_col} "
            f"FROM {self.table}"
        )
        with self._conn.cursor(name="prism_populate") as cur:
            cur.itersize = 1000
            cur.execute(sql)
            added = 0
            for row in cur:
                chunk = self._row_to_chunk(row)
                graph.add_node(
                    chunk["id"],
                    source       = chunk["source"],
                    page         = chunk["page"],
                    section      = chunk["section"],
                    text_preview = chunk["text"][:200],
                )
                added += 1

        print(f"[prism] {added:,} nodes added to graph from PostgreSQL")
        return added

    def candidate_pairs(
        self,
        k_neighbors: int = 8,
        cross_source_only: bool = False,
        max_pairs: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        self._ensure_connected()
        print("[prism] loading all chunks + vectors from PostgreSQL ...")

        sql = (
            f"SELECT {self.id_col}, {self.source_col}, {self.page_col}, "
            f"{self.section_col}, {self.text_col}, {self.vector_col}::text "
            f"FROM {self.table}"
        )

        all_rows = []
        with self._conn.cursor(name="prism_pairs") as cur:
            cur.itersize = 500
            cur.execute(sql)
            for row in cur:
                all_rows.append(row)

        print(f"[prism] {len(all_rows):,} chunks loaded")

        id_to_chunk = {str(row[0]): self._row_to_chunk(row) for row in all_rows}

        from tqdm import tqdm
        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        nbr_sql = (
            f"SELECT {self.id_col}, {self.source_col}, {self.page_col}, "
            f"{self.section_col}, {self.text_col} "
            f"FROM {self.table} "
            f"ORDER BY {self.vector_col} <=> %s::vector "
            f"LIMIT %s"
        )

        with self._conn.cursor() as cur:
            for row in tqdm(all_rows, desc="finding candidate pairs", unit="chunk"):
                row_id    = str(row[0])
                row_chunk = id_to_chunk[row_id]
                vec_text  = row[5]  # already a string from ::text cast
                if not vec_text:
                    continue

                cur.execute(nbr_sql, (vec_text, k_neighbors + 1))
                neighbors = cur.fetchall()

                for nbr_row in neighbors:
                    nbr_id = str(nbr_row[0])
                    if nbr_id == row_id:
                        continue
                    nbr_chunk = id_to_chunk.get(nbr_id)
                    if nbr_chunk is None:
                        nbr_chunk = self._row_to_chunk(nbr_row)

                    if cross_source_only and nbr_chunk["source"] == row_chunk["source"]:
                        continue

                    pair_key = frozenset([row_id, nbr_id])
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    candidates.append((row_chunk, nbr_chunk))

                    if max_pairs and len(candidates) >= max_pairs:
                        print(f"[prism] max_pairs={max_pairs} reached")
                        return candidates

        print(f"[prism] {len(candidates):,} candidate pairs generated")
        return candidates

    def candidate_pairs_for(
        self,
        node_ids: list[str],
        k_neighbors: int = 8,
        cross_source_only: bool = False,
    ) -> list[tuple[dict, dict]]:
        self._ensure_connected()
        if not node_ids:
            return []

        fetch_sql = (
            f"SELECT {self.id_col}, {self.source_col}, {self.page_col}, "
            f"{self.section_col}, {self.text_col}, {self.vector_col}::text "
            f"FROM {self.table} "
            f"WHERE {self.id_col} = ANY(%s)"
        )
        nbr_sql = (
            f"SELECT {self.id_col}, {self.source_col}, {self.page_col}, "
            f"{self.section_col}, {self.text_col} "
            f"FROM {self.table} "
            f"ORDER BY {self.vector_col} <=> %s::vector "
            f"LIMIT %s"
        )

        seen_pairs: set[frozenset] = set()
        candidates: list[tuple[dict, dict]] = []

        with self._conn.cursor() as fetch_cur:
            fetch_cur.execute(fetch_sql, (list(node_ids),))
            target_rows = fetch_cur.fetchall()

        with self._conn.cursor() as nbr_cur:
            for row in target_rows:
                row_id    = str(row[0])
                row_chunk = self._row_to_chunk(row)
                vec_text  = row[5]
                if not vec_text:
                    continue

                nbr_cur.execute(nbr_sql, (vec_text, k_neighbors + 1))
                neighbors = nbr_cur.fetchall()

                for nbr_row in neighbors:
                    nbr_id    = str(nbr_row[0])
                    nbr_chunk = self._row_to_chunk(nbr_row)
                    if nbr_id == row_id:
                        continue
                    if cross_source_only and nbr_chunk["source"] == row_chunk["source"]:
                        continue
                    pair_key = frozenset([row_id, nbr_id])
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    candidates.append((row_chunk, nbr_chunk))

        return candidates

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            n = cur.fetchone()[0]

            cur.execute(
                f"SELECT {self.source_col}, COUNT(*) FROM {self.table} "
                f"GROUP BY {self.source_col} ORDER BY COUNT(*) DESC"
            )
            sources = {row[0] or "<unknown>": row[1] for row in cur.fetchall()}

        return {
            "total_chunks": n,
            "sources": sources,
            "table": self.table,
            "dsn": self.dsn.split("@")[-1] if "@" in self.dsn else self.dsn,
            "embed_model": self._embedder.model,
        }
