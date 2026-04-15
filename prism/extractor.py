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
    "You are an expert knowledge analyst specialising in identifying epistemic "
    "relationships between text passages. You are precise, concise, and only "
    "assert relationships with high confidence."
)

_BATCH_PROMPT = """\
Analyse the following {n} chunk pairs and identify epistemic relationships.

Epistemic relationship types:
  supports | refutes | supersedes | derives_from | specializes |
  contrasts_with | implements | generalizes | exemplifies | qualifies

Rules:
- Only assert if the relationship is CLEAR and MEANINGFUL — topical similarity alone is not enough
- Specify directionality: which chunk is source (makes the claim), which is target (claim is about)
- Respond with a JSON array of EXACTLY {n} objects, one per pair, in order

Each object:
  {{"pair_index": N, "has_relationship": true,  "source_id": "...", "target_id": "...", "edge_type": "...", "confidence": 0.0–1.0, "rationale": "one sentence"}}
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
        results: list[Optional[ExtractionResult]] = [None] * len(batch)

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

        try:
            raw = await self._call_llm(client, sem, messages)
            results = self._parse_batch(raw, batch)
        except Exception:
            # Batch failed entirely — return all None (silent skip)
            pass

        return results

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
        return asyncio.run(self._run(candidate_pairs, graph, cp, show_progress))
