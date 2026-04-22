"""
prism.filter
------------
Stage 1 of the two-stage build pipeline.

Uses a fast LLM (local Ollama or any OpenAI-compatible API) to pre-filter
candidate pairs, discarding those with no meaningful epistemic relationship
before the expensive Stage 2 classification step.

Typical yield: keeps ~45-55% of pairs, saving ~50% of Stage 2 cost
with <10% loss of true epistemic edges (conservative YES bias on error).

Supports any OpenAI-compatible endpoint:
  - Ollama:   base_url="http://localhost:11434/v1", api_key="ollama"
  - DeepSeek: base_url="https://api.deepseek.com/v1", api_key="sk-..."
  - OpenAI:   base_url="https://api.openai.com/v1",  api_key="sk-..."
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
from tqdm import tqdm


_FILTER_SYSTEM = (
    "You are a fast binary classifier. Given pairs of text passages, "
    "determine only whether a meaningful directional epistemic relationship "
    "likely exists between them. Be strict: topical similarity alone is NOT enough."
)

_FILTER_PROMPT = """\
For each pair below, answer YES if a meaningful epistemic relationship likely exists \
(one passage supports, refutes, qualifies, supersedes, specializes, derives from, \
implements, or exemplifies the other), or NO if they are merely topically similar.

{pairs_text}

Respond with ONLY a JSON array — no explanation:
[{{"pair_index": 0, "has_relationship": true}}, {{"pair_index": 1, "has_relationship": false}}, ...]"""


class EpistemicFilter:
    """
    Stage 1 fast pre-filter using any OpenAI-compatible LLM API.

    Reduces the candidate pair set by ~50% before expensive Stage 2 classification.
    Falls back to keeping all pairs if the API is unavailable.

    Args:
        base_url:       OpenAI-compatible API base URL
                        (default: http://localhost:11434/v1 for Ollama)
        model:          Model name (e.g. "llama3.1:8b", "deepseek-chat")
        api_key:        API key (use any value for Ollama, e.g. "ollama")
        batch_size:     Pairs per API call (default 10)
        max_concurrent: Concurrent API requests (default 20)
        timeout:        Per-request timeout in seconds (default 60)
    """

    def __init__(
        self,
        base_url:       str = "http://localhost:11434/v1",
        model:          str = "llama3.1:8b",
        api_key:        str = "ollama",
        batch_size:     int = 10,
        max_concurrent: int = 20,
        timeout:        int = 60,
        # Legacy compat — ignored if base_url is set explicitly
        ollama_url:     str = "",
    ):
        # Back-compat: if caller passes ollama_url but not base_url
        if ollama_url and base_url == "http://localhost:11434/v1":
            base_url = ollama_url.rstrip("/") + "/v1"

        self.base_url       = base_url.rstrip("/")
        self.model          = model
        self.api_key        = api_key
        self.batch_size     = batch_size
        self.max_concurrent = max_concurrent
        self.timeout        = timeout

    # ── Internal async implementation ────────────────────────────────────────

    async def _filter_batch(
        self,
        client: httpx.AsyncClient,
        sem:    asyncio.Semaphore,
        batch:  list[tuple[dict, dict]],
    ) -> list[bool]:
        """Returns True for each pair that likely has an epistemic relationship."""
        keep = [True] * len(batch)  # conservative default: keep all on failure

        parts = []
        for i, (a, b) in enumerate(batch):
            parts.append(
                f"PAIR {i}:\n"
                f"  A [{a.get('source','')}]: {a['text'][:280]}\n"
                f"  B [{b.get('source','')}]: {b['text'][:280]}"
            )

        prompt = _FILTER_PROMPT.format(pairs_text="\n\n".join(parts))

        async with sem:
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type":  "application/json",
                    },
                    json={
                        "model":       self.model,
                        "messages": [
                            {"role": "system", "content": _FILTER_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                        "temperature": 0,
                        "max_tokens":  300,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()

                m = re.search(r'\[.*\]', raw, re.DOTALL)
                if m:
                    items = json.loads(m.group())
                    for item in items:
                        idx = item.get("pair_index")
                        if idx is not None and 0 <= idx < len(batch):
                            keep[idx] = bool(item.get("has_relationship", True))
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass  # API not available — keep all pairs silently
            except Exception:
                pass  # Any other error: keep all (conservative)

        return keep

    async def _run(
        self,
        pairs:         list[tuple[dict, dict]],
        show_progress: bool,
    ) -> list[tuple[dict, dict]]:
        batches = [
            pairs[i: i + self.batch_size]
            for i in range(0, len(pairs), self.batch_size)
        ]

        sem    = asyncio.Semaphore(self.max_concurrent)
        kept:  list[tuple[dict, dict]] = []
        limits = httpx.Limits(max_connections=self.max_concurrent + 4)

        async with httpx.AsyncClient(limits=limits) as client:
            chunk_size = self.max_concurrent
            bar = tqdm(
                total=len(batches),
                desc="[stage 1] filtering pairs",
                unit="batch",
                disable=not show_progress,
            )
            for i in range(0, len(batches), chunk_size):
                chunk   = batches[i: i + chunk_size]
                tasks   = [self._filter_batch(client, sem, b) for b in chunk]
                results = await asyncio.gather(*tasks)

                for batch, keep_flags in zip(chunk, results):
                    for pair, keep in zip(batch, keep_flags):
                        if keep:
                            kept.append(pair)
                bar.update(len(chunk))

            bar.close()

        return kept

    # ── Public interface ──────────────────────────────────────────────────────

    def filter(
        self,
        pairs:         list[tuple[dict, dict]],
        show_progress: bool = True,
    ) -> list[tuple[dict, dict]]:
        """
        Filter candidate pairs, keeping only those likely to have
        an epistemic relationship.

        Args:
            pairs:         List of (chunk_a, chunk_b) dicts
            show_progress: Show tqdm progress bar

        Returns:
            Filtered subset of pairs (full set returned on API failure).
        """
        if not pairs:
            return pairs

        n_in = len(pairs)
        kept = asyncio.run(self._run(pairs, show_progress))
        n_out = len(kept)
        reduction = (1 - n_out / n_in) * 100 if n_in else 0

        if show_progress:
            print(
                f"[prism] stage 1 complete: {n_in:,} → {n_out:,} pairs "
                f"({reduction:.0f}% filtered out)"
            )
        return kept
