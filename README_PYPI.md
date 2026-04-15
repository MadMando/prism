# PRISM — Epistemic Graph RAG with Spreading Activation

**Propagation & Retrieval via Informed Semantic Mapping**

[![PyPI](https://img.shields.io/pypi/v/prism-rag?style=flat-square&color=blue)](https://pypi.org/project/prism-rag/)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/MadMando/prism/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-MadMando%2Fprism-black?style=flat-square&logo=github)](https://github.com/MadMando/prism)

PRISM layers a **typed epistemic knowledge graph** over your existing vector store, then uses **spreading activation** to surface knowledge structured by *how it relates* — not just *how similar it is*.

---

## The Problem with Standard RAG

Standard RAG returns a flat ranked list. Every chunk gets a similarity score and nothing else:

```
query → embed → similarity → [chunk, chunk, chunk, ...]   ← no structure
```

Chunk B may *refute* Chunk A. Chunk C may *specialise* a principle in Chunk D. An older document may have been *superseded*. Standard RAG can't express any of this.

---

## What PRISM Does

PRISM builds a graph where edges carry **epistemic type**:

```
Doc A  ──[supports]──▶  Doc B
Doc C  ──[refutes]───▶  Doc D
Doc E  ──[supersedes]▶  Doc F
```

Retrieval uses **spreading activation**: a query fires seed nodes via vector search, activation propagates through typed edges, and nodes reached by *multiple independent paths* (convergence) rank highest.

The result is a **structured epistemic answer** with five buckets:

| Bucket | Contents |
|--------|----------|
| **PRIMARY** | Core relevant chunks, highest convergence |
| **SUPPORTING** | Chunks that reinforce or extend the primary answer |
| **CONTRASTING** | Chunks that challenge or take a different position |
| **QUALIFYING** | Chunks that add conditions, exceptions, or nuances |
| **SUPERSEDED** | Historically relevant context now replaced by newer work |

---

## Installation

```bash
pip install prism-rag
```

Requires Python 3.11+, an existing LanceDB vector store, and an embedding provider.

---

## Quick Start

### 1. Build the epistemic graph (one-time)

```python
from prism import PRISM

# Ollama embeddings (local)
p = PRISM(
    lancedb_path = "/path/to/your/lancedb",
    graph_path   = "/path/to/prism_graph.json.gz",
    ollama_url   = "http://localhost:11434",
    embed_model  = "nomic-embed-text",
    llm_base_url = "https://api.openai.com",
    llm_model    = "gpt-4o-mini",
    llm_api_key  = "sk-...",
)

# Or OpenAI-compatible API embeddings
p = PRISM(
    lancedb_path  = "/path/to/your/lancedb",
    graph_path    = "/path/to/prism_graph.json.gz",
    embed_api_url = "https://api.openai.com/v1/embeddings",
    embed_api_key = "sk-...",
    embed_model   = "text-embedding-3-small",
    llm_base_url  = "https://api.openai.com",
    llm_model     = "gpt-4o-mini",
    llm_api_key   = "sk-...",
)

p.build(k_neighbors=8, cross_source_only=True)
```

Or via the CLI:

```bash
prism-build \
    --lancedb-path /path/to/lancedb \
    --graph-path   /path/to/prism_graph.json.gz \
    --llm-api-key  $OPENAI_API_KEY
```

### 2. Retrieve

```python
p.load_graph()
result = p.retrieve("your question here", top_k=5)
print(result.format_for_llm())
```

Output:

```
PRISM retrieval for: "your question here"
────────────────────────────────────────────────────────────

## PRIMARY
[1] source-a  p.14  § 2.1  (score: 0.923)
    The core relevant passage...

## SUPPORTING EVIDENCE
[1] source-c  p.201  § 8.2  (score: 0.841  [via: specializes])
    A passage that extends the primary answer...

## QUALIFICATIONS & NUANCES
[1] source-d  p.38  § 3.1  (score: 0.712  [via: qualifies])
    A passage adding conditions or exceptions...

─ 1 primary · 1 supporting · 0 contrasting · 1 qualifying · 0 superseded ─
```

### 3. Access results programmatically

```python
for chunk in result.primary:
    print(chunk.source, chunk.page, chunk.final_score, chunk.text)

for chunk in result.contrasting:
    print("Contrasting view:", chunk.text[:200])

# Feed structured context directly into your LLM
context = result.format_for_llm()
```

---

## Epistemic Edge Types

```
supports        — A provides evidence reinforcing B
refutes         — A directly contradicts B
supersedes      — A replaces or updates B
derives_from    — A is logically derived from B
specializes     — A is a specific instance of B
contrasts_with  — A and B take different, non-exclusive positions
implements      — A is a concrete method putting B into practice
generalizes     — A is a broader abstraction of which B is a case
exemplifies     — A is a concrete example illustrating B
qualifies       — A adds conditions, exceptions, or nuances to B
```

Each edge carries a **propagation weight** (0.40–0.90) and a **valence** that determines which result bucket its target lands in.

---

## No Re-embedding Required

PRISM works **on top of your existing vector store**. If you have a LanceDB corpus with embeddings, you don't need to re-index anything.

- Existing vectors → used as-is for seed activation
- Epistemic graph → built from text via LLM, stored as a separate `.json.gz` file
- Fallback → if no graph exists, PRISM automatically falls back to pure vector search

---

## Embedding Providers

**Ollama (local):**
```python
PRISM(ollama_url="http://localhost:11434", embed_model="nomic-embed-text", ...)
```

**OpenAI-compatible API** (OpenAI, Azure, Together, Jina, Mistral, etc.):
```python
PRISM(embed_api_url="https://api.openai.com/v1/embeddings", embed_api_key="sk-...", ...)
```

---

## Links

- **Full documentation & architecture:** [github.com/MadMando/prism](https://github.com/MadMando/prism)
- **Issues:** [github.com/MadMando/prism/issues](https://github.com/MadMando/prism/issues)

---

## License

MIT
