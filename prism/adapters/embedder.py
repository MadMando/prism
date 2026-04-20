"""
prism.adapters.embedder
-----------------------
Standalone embedding helper — Ollama or any OpenAI-compatible API.

Used by LanceDBAdapter and available for any custom adapter so you
don't have to re-implement embedding yourself.

Usage::

    from prism.adapters.embedder import Embedder

    # Ollama (local, default)
    emb = Embedder(model="nomic-embed-text")

    # OpenAI-compatible API
    emb = Embedder(
        model     = "text-embedding-3-small",
        api_url   = "https://api.openai.com/v1/embeddings",
        api_key   = "sk-...",
    )

    vector = emb.embed("some text")   # -> list[float]
"""

from __future__ import annotations

from typing import Optional

import requests


class Embedder:
    """
    Embed text via Ollama (default) or any OpenAI-compatible embeddings API.

    Args:
        model:    Embedding model name (must match what was used to build the corpus)
        api_url:  Full URL for an OpenAI-compatible embeddings endpoint.
                  If omitted, Ollama is used.
        api_key:  Bearer token for the API. Setting this switches to API mode.
        ollama_url: Ollama base URL (used only when api_key is not set).
        timeout:  Request timeout in seconds.
    """

    def __init__(
        self,
        model:      str = "nomic-embed-text",
        api_url:    Optional[str] = None,
        api_key:    Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        timeout:    int = 60,
    ):
        self.model      = model
        self.api_key    = api_key
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout    = timeout
        self._use_api   = bool(api_key)

        if self._use_api:
            self.api_url = api_url or "https://api.openai.com/v1/embeddings"
        else:
            self.api_url = api_url  # unused in Ollama mode

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""
        if self._use_api:
            return self._embed_api(text)
        return self._embed_ollama(text)

    def _embed_ollama(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _embed_api(self, text: str) -> list[float]:
        resp = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
