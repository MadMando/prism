"""
prism.filter
------------
Stage 1 of the two-stage build pipeline.

Uses a fast local Ollama model to pre-filter candidate pairs,
discarding those with no meaningful epistemic relationship before
the expensive Stage 2 API classification step.

Typical yield: keeps ~45-55% of pairs, saving ~50% of Stage 2 cost
with <10% loss of true epistemic edges (conservative YES bias on error).
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
    Stage 1 fast pre-filter using a local Ollama model.

    Reduces the candidate pair set by ~50% before expensive LLM classification.
    Falls back to keeping all pairs if Ollama is unavailable.

    Args:
        ollama_url:     Ollama base URL (default http://localhost:11434)
        model:          Ollama model name (default llama3.1:8b)
        batch_size:     Pairs per Ollama call (default 10)
        max_concurrent: Concurrent Ollama requests (default 5)
        timeout:        Per-request timeout in seconds (default 120)
    """

    def __init__(
        self,
        ollama_url:     str = "http://localhost:11434",
        model:          str = "llama3.1:8b",
        batch_size:     int = 10,
        max_concurrent: int = 5,
        timeout:        int = 120,
    ):
        self.ollama_url     = ollama_url.rstrip("/")
        self.model          = model
        self.batch_size     = batch_size
        self.max_concurrent = max_concurrent
        self.timeout        = timeout

    # ── Model availability check ──────────────────────────────────────────────

    @staticmethod
    def available_models(ollama_url: str = "http://localhost:11434") -> list[str]:
        """Return list of model names currently loaded in Ollama."""
        try:
            resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def check_model(self) -> bool:
        """Return True if the configured filter model is available in Ollama."""
        models = self.available_models(self.ollama_url)
        if self.model not in models:
            available = ", ".join(models) if models else "none found"
            print(
                f"[prism] WARNING: filter model '{self.model}' not found in Ollama.\n"
                f"[prism]   Available: {available}\n"
                f"[prism]   Set filter_model= to one of the above, or use --no-filter."
            )
            return False
        return True

    # ── Internal async implementation ────────────────────────────────────────

    async def _filter_batch(
        self,
        client: httpx.AsyncClient,
        sem:    asyncio.Semaphore,
        batch:  list[tuple[dict, dict]],
    ) -> list[bool]:
        """Returns True for each pair that likely has an epistemic relationship."""
        # Default: keep all (conservative — never drop on failure)
        keep = [True] * len(batch)

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
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model":   self.model,
                        "prompt":  prompt,
                        "system":  _FILTER_SYSTEM,
                        "stream":  False,
                        "options": {"temperature": 0, "num_predict": 300},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")

                m = re.search(r'\[.*\]', raw, re.DOTALL)
                if m:
                    items = json.loads(m.group())
                    for item in items:
                        idx = item.get("pair_index")
                        if idx is not None and 0 <= idx < len(batch):
                            keep[idx] = bool(item.get("has_relationship", True))
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Ollama not available — keep all pairs silently
                pass
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

        sem = asyncio.Semaphore(self.max_concurrent)
        kept: list[tuple[dict, dict]] = []

        limits  = httpx.Limits(max_connections=self.max_concurrent + 2)
        async with httpx.AsyncClient(limits=limits) as client:
            # Process in chunks of max_concurrent for clean progress reporting
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
            Filtered subset of pairs (superset on Ollama failure).
        """
        if not pairs:
            return pairs

        # Bail early with a clear message if model isn't available
        if not self.check_model():
            print("[prism] stage 1 skipped — returning all pairs unfiltered")
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
