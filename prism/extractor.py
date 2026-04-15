"""
prism.extractor
---------------
LLM-based epistemic triple extraction.

Given a pair of text chunks, asks an LLM to identify if an epistemic
relationship exists between them and what type it is.

Strategy:
  1. Use existing vector embeddings to find top-K nearest neighbours
     per chunk — these are the candidate pairs most likely to have
     epistemic relationships.
  2. Prioritise cross-source pairs (inter-framework relationships are
     the most valuable epistemic signal for governance/domain corpora).
  3. Run LLM extraction in batches to amortise API round-trips.
  4. Filter by confidence threshold before adding to the graph.

Compatible with any OpenAI-compatible API (DeepSeek, Ollama, etc.).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from tqdm import tqdm

from .edges import EpistemicEdgeType
from .graph import EpistemicGraph


# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert knowledge analyst specialising in identifying epistemic relationships between text passages. You are precise, concise, and only assert relationships with high confidence."""

_PAIR_PROMPT = """Analyse the following two text chunks and determine if there is a meaningful epistemic relationship between them.

CHUNK A
id: {id_a}
source: {source_a}  |  section: {section_a}
---
{text_a}

CHUNK B
id: {id_b}
source: {source_b}  |  section: {section_b}
---
{text_b}

Epistemic relationship types:
- supports        : A provides evidence or reasoning that reinforces the claims in B
- refutes         : A directly contradicts or undermines the claims in B
- supersedes      : A replaces or updates B (A is newer or more authoritative)
- derives_from    : A is logically or conceptually derived from B
- specializes     : A is a specific application, instance, or case of the general principle in B
- contrasts_with  : A and B take different but not mutually exclusive positions or approaches
- implements      : A is a practical tool or method that puts the abstract concept of B into practice
- generalizes     : A is a broader abstraction or principle of which B is a specific case
- exemplifies     : A is a concrete example illustrating the concept described in B
- qualifies       : A adds important conditions, exceptions, or nuances to the claims in B

Rules:
- Only assert a relationship if it is clear and meaningful, not just topical similarity
- The relationship must be DIRECTIONAL: specify which chunk is source and which is target
- Respond ONLY with valid JSON, no explanation outside the JSON object

If a meaningful epistemic relationship exists:
{{"has_relationship": true, "source_id": "<id of chunk making the epistemic claim>", "target_id": "<id of chunk being claimed about>", "edge_type": "<type from list above>", "confidence": <0.0-1.0>, "rationale": "<one concise sentence>"}}

If no meaningful epistemic relationship (mere topical similarity is NOT enough):
{{"has_relationship": false}}"""

_BATCH_PROMPT = """Analyse the following {n} chunk pairs and identify epistemic relationships.

For each pair, determine if an epistemic relationship exists.

Epistemic relationship types:
supports | refutes | supersedes | derives_from | specializes | contrasts_with | implements | generalizes | exemplifies | qualifies

Rules:
- Only assert if the relationship is clear and meaningful, not mere topical similarity
- Specify directionality: which chunk is source, which is target
- Respond with a JSON array of exactly {n} objects, one per pair

Each object must be either:
  {{"pair_index": N, "has_relationship": true, "source_id": "...", "target_id": "...", "edge_type": "...", "confidence": 0.0-1.0, "rationale": "one sentence"}}
  or:
  {{"pair_index": N, "has_relationship": false}}

PAIRS:
{pairs_text}

Respond with ONLY the JSON array."""


@dataclass
class ExtractionResult:
    source_id:  str
    target_id:  str
    edge_type:  EpistemicEdgeType
    confidence: float
    rationale:  str


class EpistemicExtractor:
    """
    Extracts epistemic triples from chunk pairs using an LLM.

    Args:
        base_url:  OpenAI-compatible API base (e.g. "https://api.deepseek.com")
        model:     Model name (e.g. "deepseek-chat")
        api_key:   API key
        min_confidence: Minimum confidence to include an extracted edge (default 0.65)
        batch_size:     Pairs per LLM call in batch mode (default 5)
        delay_s:        Delay between API calls in seconds (default 0.3)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        min_confidence: float = 0.65,
        batch_size: int = 5,
        delay_s: float = 0.3,
        timeout: int = 60,
    ):
        self.base_url       = base_url.rstrip("/")
        self.model          = model
        self.api_key        = api_key
        self.min_confidence = min_confidence
        self.batch_size     = batch_size
        self.delay_s        = delay_s
        self.timeout        = timeout

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 512,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Single pair extraction ────────────────────────────────────────────────

    def extract_pair(
        self,
        id_a: str, text_a: str, source_a: str, section_a: str,
        id_b: str, text_b: str, source_b: str, section_b: str,
    ) -> Optional[ExtractionResult]:
        """Extract epistemic relationship from a single chunk pair."""
        prompt = _PAIR_PROMPT.format(
            id_a=id_a, text_a=text_a[:600], source_a=source_a, section_a=section_a,
            id_b=id_b, text_b=text_b[:600], source_b=source_b, section_b=section_b,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = self._chat(messages)
            return self._parse_single(raw, id_a, id_b)
        except Exception as e:
            return None

    def _parse_single(
        self, raw: str, id_a: str, id_b: str
    ) -> Optional[ExtractionResult]:
        raw = raw.strip()
        # Extract JSON even if there's surrounding text
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

    # ── Batch extraction ──────────────────────────────────────────────────────

    def extract_batch(
        self, pairs: list[tuple[dict, dict]]
    ) -> list[Optional[ExtractionResult]]:
        """
        Extract epistemic relationships from a batch of chunk pairs.
        pairs: list of (chunk_a_dict, chunk_b_dict) where each dict has
               keys: id, text, source, section
        """
        if not pairs:
            return []

        pairs_text_parts = []
        for i, (a, b) in enumerate(pairs):
            pairs_text_parts.append(
                f"PAIR {i}\n"
                f"  A: id={a['id']} | {a['source']} § {a.get('section','')}\n"
                f"     {a['text'][:400]}\n"
                f"  B: id={b['id']} | {b['source']} § {b.get('section','')}\n"
                f"     {b['text'][:400]}"
            )

        prompt = _BATCH_PROMPT.format(
            n=len(pairs),
            pairs_text="\n\n".join(pairs_text_parts),
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            raw = self._chat(messages, temperature=0.0)
            return self._parse_batch(raw, pairs)
        except Exception:
            # Batch failed — fall back to individual extraction
            results = []
            for a, b in pairs:
                r = self.extract_pair(
                    a["id"], a["text"], a["source"], a.get("section", ""),
                    b["id"], b["text"], b["source"], b.get("section", ""),
                )
                results.append(r)
                time.sleep(self.delay_s)
            return results

    def _parse_batch(
        self, raw: str, pairs: list[tuple[dict, dict]]
    ) -> list[Optional[ExtractionResult]]:
        results: list[Optional[ExtractionResult]] = [None] * len(pairs)
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            return results
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return results

        for item in items:
            idx = item.get("pair_index")
            if idx is None or not (0 <= idx < len(pairs)):
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
            a, b = pairs[idx]
            results[idx] = ExtractionResult(
                source_id  = item.get("source_id", a["id"]),
                target_id  = item.get("target_id", b["id"]),
                edge_type  = edge_type,
                confidence = confidence,
                rationale  = item.get("rationale", ""),
            )
        return results

    # ── Full corpus extraction ─────────────────────────────────────────────────

    def extract_from_candidates(
        self,
        candidate_pairs: list[tuple[dict, dict]],
        graph: EpistemicGraph,
        show_progress: bool = True,
    ) -> int:
        """
        Run extraction over all candidate pairs and add edges to the graph.
        Returns the number of edges added.
        """
        added = 0
        batches = [
            candidate_pairs[i: i + self.batch_size]
            for i in range(0, len(candidate_pairs), self.batch_size)
        ]

        it = tqdm(batches, desc="extracting epistemic triples", unit="batch") if show_progress else batches

        for batch in it:
            results = self.extract_batch(batch)
            for result in results:
                if result is None:
                    continue
                # Ensure both nodes are in the graph
                if not graph.has_node(result.source_id):
                    continue
                if not graph.has_node(result.target_id):
                    continue
                graph.add_edge(
                    result.source_id,
                    result.target_id,
                    result.edge_type,
                    confidence=result.confidence,
                    rationale=result.rationale,
                )
                added += 1

            if show_progress and hasattr(it, "set_postfix"):
                it.set_postfix({"edges_added": added})

            time.sleep(self.delay_s)

        return added
