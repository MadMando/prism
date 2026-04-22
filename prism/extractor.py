"""
prism.extractor
---------------
Stage 2: Async LLM epistemic triple extraction with checkpointing.

v2 improvements over the original synchronous implementation:
  - Async HTTP with configurable concurrency (default 20 parallel requests)
  - Larger default batch size (20 pairs vs 5) — fewer API round trips
  - Checkpoint / resume: saves progress every N batches to a .partial file
  - ~40–80x faster wall time than v1 on large corpora

Typical performance:
  30k-chunk corpus, 25k pairs (post-filter), batch=20, concurrency=20:
  → ~1,250 batches / 20 concurrent = ~63 rounds × ~15s = ~16 minutes

Compatible with any OpenAI-compatible API (DeepSeek, OpenAI, etc.).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx
from tqdm import tqdm

from .edges import EpistemicEdgeType
from .graph import EpistemicGraph


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert knowledge analyst identifying epistemic relationships "
    "between text passages. You classify how one passage relates to another "
    "in terms of evidence, logic, or epistemic stance — not just topic similarity. "
    "Be precise and conservative: only assert a relationship when it is clear "
    "and meaningful. Topical overlap alone is not sufficient."
)

_BATCH_PROMPT = """\
Analyse the following {n} chunk pairs and classify their epistemic relationship.

RELATIONSHIP TYPES (choose exactly one, or none):

  Reinforcing — A strengthens or elaborates B:
    supports      : A provides direct evidence or reasoning that B's claim is correct
    derives_from  : A is logically or conceptually derived from B (B is the foundation)
    specializes   : A is a specific case or application of the broader concept in B
    implements    : A is the concrete tool/method that realises B's abstract concept
    exemplifies   : A is a concrete example that illustrates B's general point
    generalizes   : A is a broader abstraction of which B is one instance

  Modifying — A nuances but does not negate B:
    qualifies     : A adds conditions, exceptions, or scope limits to B's claim

  Dialectical — A challenges B:
    contrasts_with: A and B take different positions on the same topic (both may be valid)
    refutes       : A directly contradicts or undermines B's central claim

  Temporal — A supersedes B:
    supersedes    : A replaces B because A is newer, corrected, or more authoritative

DIRECTIONALITY:
  source_id = the chunk that MAKES the relationship active (the acting chunk)
  target_id = the chunk whose claim is being acted upon
  Example: if A cites B as evidence → A supports B → source=A, target=B
  Example: if A is a specific case of B → A specializes B → source=A, target=B

RULES:
- Assign a relationship only if it is CLEAR and UNAMBIGUOUS
- Topical similarity alone → has_relationship: false
- Pick the MOST SPECIFIC type that fits (prefer specializes over supports when both apply)
- Each response object must include pair_index even when has_relationship is false

OUTPUT FORMAT — a JSON array of exactly {n} objects in order:
  {{"pair_index": N, "has_relationship": true, "source_id": "...", "target_id": "...", "edge_type": "...", "confidence": 0.0–1.0, "rationale": "one sentence max"}}
  {{"pair_index": N, "has_relationship": false}}

PAIRS:
{pairs_text}

Respond with ONLY the JSON array — no preamble, no explanation."""


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    source_id:  str
    target_id:  str
    edge_type:  EpistemicEdgeType
    confidence: float
    rationale:  str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "ExtractionResult":
        return ExtractionResult(
            source_id  = d["source_id"],
            target_id  = d["target_id"],
            edge_type  = EpistemicEdgeType(d["edge_type"]),
            confidence = d["confidence"],
            rationale  = d.get("rationale", ""),
        )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> tuple[int, list[ExtractionResult]]:
    """Returns (pairs_completed, edges_so_far)."""
    if not path.exists():
        return 0, []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        edges = [ExtractionResult.from_dict(e) for e in data.get("edges", [])]
        return int(data.get("pairs_completed", 0)), edges
    except Exception:
        return 0, []


def _save_checkpoint(
    path: Path,
    pairs_completed: int,
    edges: list[ExtractionResult],
) -> None:
    data = {
        "pairs_completed": pairs_completed,
        "edges": [e.to_dict() for e in edges],
    }
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


# ── Extractor ─────────────────────────────────────────────────────────────────

class EpistemicExtractor:
    """
    Async LLM-based epistemic triple extractor (Stage 2).

    Args:
        base_url:         OpenAI-compatible API base URL
        model:            Model name (e.g. "deepseek-chat")
        api_key:          API key
        min_confidence:   Minimum confidence to keep an edge (default 0.65)
        batch_size:       Pairs per LLM call (default 20)
        max_concurrent:   Concurrent API requests (default 20)
        timeout:          Per-request timeout in seconds (default 90)
        checkpoint_every: Save checkpoint every N batches (default 100)
    """

    def __init__(
        self,
        base_url:         str,
        model:            str,
        api_key:          str,
        min_confidence:   float = 0.65,
        batch_size:       int   = 20,
        max_concurrent:   int   = 20,
        timeout:          int   = 90,
        checkpoint_every: int   = 100,
        max_retries:      int   = 3,
        retry_base_delay: float = 1.0,
        failure_log_path: Optional[str] = None,
        # legacy compat — ignored in v2 (async replaces delay)
        delay_s:          float = 0.0,
    ):
        self.base_url         = base_url.rstrip("/")
        self.model            = model
        self.api_key          = api_key
        self.min_confidence   = min_confidence
        self.batch_size       = batch_size
        self.max_concurrent   = max_concurrent
        self.timeout          = timeout
        self.checkpoint_every = checkpoint_every
        self.max_retries      = max_retries
        self.retry_base_delay = retry_base_delay
        self.failure_log_path = Path(failure_log_path) if failure_log_path else None
        self._failures: list[dict] = []

    # ── Single-pair parsing (used by tests and single-pair fallback) ─────────

    def _parse_single(
        self, raw: str, id_a: str, id_b: str
    ) -> Optional[ExtractionResult]:
        """Parse a single-object JSON response (non-batch mode)."""
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
        if not data.get("has_relationship"):
            return None
        try:
            edge_type  = EpistemicEdgeType(data["edge_type"])
            confidence = float(data.get("confidence", 0.0))
        except (ValueError, KeyError):
            return None
        if confidence < self.min_confidence:
            return None
        return ExtractionResult(
            source_id  = data.get("source_id", id_a),
            target_id  = data.get("target_id", id_b),
            edge_type  = edge_type,
            confidence = confidence,
            rationale  = data.get("rationale", ""),
        )

    # ── Single async batch ────────────────────────────────────────────────────

    async def _call_llm(
        self,
        client:   httpx.AsyncClient,
        sem:      asyncio.Semaphore,
        messages: list[dict],
    ) -> str:
        async with sem:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       self.model,
                    "messages":    messages,
                    "temperature": 0.0,
                    "max_tokens":  1024,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _extract_batch(
        self,
        client: httpx.AsyncClient,
        sem:    asyncio.Semaphore,
        batch:  list[tuple[dict, dict]],
    ) -> list[Optional[ExtractionResult]]:
        parts = []
        for i, (a, b) in enumerate(batch):
            parts.append(
                f"PAIR {i}\n"
                f"  A: id={a['id']} | {a.get('source','')} § {a.get('section','')}\n"
                f"     {a['text'][:400]}\n"
                f"  B: id={b['id']} | {b.get('source','')} § {b.get('section','')}\n"
                f"     {b['text'][:400]}"
            )

        prompt = _BATCH_PROMPT.format(n=len(batch), pairs_text="\n\n".join(parts))
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = await self._call_llm(client, sem, messages)
                return self._parse_batch(raw, batch)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_base_delay * (2 ** attempt))

        # All retries exhausted — log and return empty
        self._failures.append({
            "pair_ids": [[a["id"], b["id"]] for a, b in batch],
            "error": last_error or "unknown",
        })
        return [None] * len(batch)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_batch(
        self,
        raw:   str,
        batch: list[tuple[dict, dict]],
    ) -> list[Optional[ExtractionResult]]:
        results: list[Optional[ExtractionResult]] = [None] * len(batch)
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            return results
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return results

        for item in items:
            idx = item.get("pair_index")
            if idx is None or not (0 <= idx < len(batch)):
                continue
            if not item.get("has_relationship"):
                continue
            try:
                edge_type  = EpistemicEdgeType(item["edge_type"])
                confidence = float(item.get("confidence", 0.0))
            except (ValueError, KeyError):
                continue
            if confidence < self.min_confidence:
                continue
            a, b = batch[idx]
            results[idx] = ExtractionResult(
                source_id  = item.get("source_id", a["id"]),
                target_id  = item.get("target_id", b["id"]),
                edge_type  = edge_type,
                confidence = confidence,
                rationale  = item.get("rationale", ""),
            )
        return results

    # ── Full async extraction ─────────────────────────────────────────────────

    async def _run(
        self,
        pairs:           list[tuple[dict, dict]],
        graph:           EpistemicGraph,
        checkpoint_path: Optional[Path],
        show_progress:   bool,
    ) -> int:
        # Resume from checkpoint if available
        resume_from, prior_edges = (
            _load_checkpoint(checkpoint_path) if checkpoint_path else (0, [])
        )
        if resume_from:
            print(f"[prism] resuming from checkpoint: {resume_from:,} pairs done, "
                  f"{len(prior_edges):,} edges already extracted")

        # Apply prior edges to graph
        accumulated: list[ExtractionResult] = list(prior_edges)
        added = 0
        for result in accumulated:
            if graph.has_node(result.source_id) and graph.has_node(result.target_id):
                graph.add_edge(
                    result.source_id, result.target_id, result.edge_type,
                    confidence=result.confidence, rationale=result.rationale,
                )
                added += 1

        # Slice pairs to resume position
        remaining = pairs[resume_from:]
        if not remaining:
            return added

        batches = [
            remaining[i: i + self.batch_size]
            for i in range(0, len(remaining), self.batch_size)
        ]

        sem     = asyncio.Semaphore(self.max_concurrent)
        limits  = httpx.Limits(max_connections=self.max_concurrent + 4)
        bar     = tqdm(
            total=len(batches),
            desc="[stage 2] extracting triples",
            unit="batch",
            disable=not show_progress,
        )

        chunk_size = self.max_concurrent   # process this many batches at once
        pairs_done = resume_from

        async with httpx.AsyncClient(limits=limits) as client:
            for i in range(0, len(batches), chunk_size):
                chunk   = batches[i: i + chunk_size]
                tasks   = [self._extract_batch(client, sem, b) for b in chunk]
                results = await asyncio.gather(*tasks)

                for batch, batch_results in zip(chunk, results):
                    pairs_done += len(batch)
                    for result in batch_results:
                        if result is None:
                            continue
                        if not (graph.has_node(result.source_id)
                                and graph.has_node(result.target_id)):
                            continue
                        graph.add_edge(
                            result.source_id, result.target_id, result.edge_type,
                            confidence=result.confidence, rationale=result.rationale,
                        )
                        accumulated.append(result)
                        added += 1

                bar.update(len(chunk))
                bar.set_postfix({"edges": added})

                # Checkpoint
                if (checkpoint_path
                        and (i // chunk_size + 1) % self.checkpoint_every == 0):
                    _save_checkpoint(checkpoint_path, pairs_done, accumulated)

        bar.close()

        # Final checkpoint
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, pairs_done, accumulated)

        return added

    # ── Public interface ──────────────────────────────────────────────────────

    def extract_from_candidates(
        self,
        candidate_pairs: list[tuple[dict, dict]],
        graph:           EpistemicGraph,
        checkpoint_path: Optional[Path | str] = None,
        show_progress:   bool = True,
    ) -> int:
        """
        Run async extraction over all candidate pairs and add edges to graph.

        Args:
            candidate_pairs: List of (chunk_a, chunk_b) dicts
            graph:           EpistemicGraph to add edges to
            checkpoint_path: Path to save/resume progress (.partial.json.gz)
            show_progress:   Show tqdm progress bars

        Returns:
            Number of edges added to the graph.
        """
        cp = Path(checkpoint_path) if checkpoint_path else None
        n_added = asyncio.run(self._run(candidate_pairs, graph, cp, show_progress))

        if self._failures:
            n_failed = sum(len(f["pair_ids"]) for f in self._failures)
            print(f"[prism] WARNING: {n_failed:,} pairs failed after {self.max_retries} retries")
            if self.failure_log_path:
                self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.failure_log_path, "w", encoding="utf-8") as fh:
                    json.dump(self._failures, fh, indent=2)
                print(f"[prism]   failures logged → {self.failure_log_path}")

        return n_added
