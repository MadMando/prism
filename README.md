<div align="center">

# PRISM

**Propagation & Retrieval via Informed Semantic Mapping**

*Epistemic Graph RAG with Spreading Activation*

[![PyPI](https://img.shields.io/badge/pypi-coming_soon-orange?style=flat-square)](https://pypi.org/project/prism-rag/)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

</div>

---

PRISM is a retrieval library that layers a **typed epistemic knowledge graph** over your existing vector store, then uses **spreading activation** to surface knowledge structured by *how it relates* — not just *how similar it is*.

---

## The Problem

Standard RAG returns a flat ranked list. Every chunk is treated the same — a similarity score and nothing else.

```
query → embed → similarity → [chunk, chunk, chunk, ...]   ← no structure
```

This loses the relational fabric of your knowledge. Chunk B may *refute* Chunk A. Chunk C may *specialize* a principle in Chunk D. An older document may have been *superseded* by a newer one. Standard RAG can't express any of this — and neither can the LLM reasoning over it.

---

## PRISM's Approach

PRISM builds a graph where edges carry **epistemic type**:

```
Doc A  ──[supports]──▶  Doc B
Doc C  ──[refutes]───▶  Doc D
Doc E  ──[supersedes]▶  Doc F
```

Retrieval then uses **spreading activation**: a query fires seed nodes via vector search, activation propagates through typed edges, and nodes reached by *multiple independent paths* (convergence) rank highest.

The result is a **structured epistemic answer** — not a ranked list:

| Bucket | Contents |
|--------|----------|
| **PRIMARY** | Core relevant chunks, highest convergence |
| **SUPPORTING** | Chunks that reinforce or extend the primary answer |
| **CONTRASTING** | Chunks that challenge or take a different position |
| **QUALIFYING** | Chunks that add conditions, exceptions, nuances |
| **SUPERSEDED** | Historically relevant context now replaced by newer work |

---

## Architecture

![PRISM Architecture](docs/architecture.svg)

Three stages:
1. **Seed** — embed the query, vector-search your corpus, get top-K scored chunks as activation seeds
2. **Activate** — propagate activation through the epistemic graph; track which seeds independently reach each node (convergence)
3. **Structure** — bucket results by epistemic role based on dominant edge valence

---

## Why This Is Novel

| System | Vector Search | Knowledge Graph | **Epistemic Edge Typing** | Spreading Activation |
|--------|:---:|:---:|:---:|:---:|
| Standard RAG | ✅ | ❌ | ❌ | ❌ |
| GraphRAG (Microsoft, 2024) | ✅ | ✅ | ❌ | ❌ |
| SYNAPSE (2026) | ✅ | ✅ | ❌ | ✅ |
| **PRISM** | ✅ | ✅ | ✅ | ✅ |

To our knowledge, no open-source retrieval library combines all four of these signals. If you know of one, please open an issue — we'd genuinely like to know.

---

## Epistemic Edge Types

```
supports        — A provides evidence reinforcing B
refutes         — A directly contradicts B
supersedes      — A replaces or updates B (A is newer / more authoritative)
derives_from    — A is logically or conceptually derived from B
specializes     — A is a specific instance of the general principle in B
contrasts_with  — A and B take different but non-exclusive positions
implements      — A is a concrete method that puts the abstract concept of B into practice
generalizes     — A is a broader abstraction of which B is a specific case
exemplifies     — A is a concrete example illustrating the concept in B
qualifies       — A adds conditions, exceptions, or nuances to B
```

Each edge has:
- A **propagation weight** — how strongly it carries activation (0.40–0.90)
- A **valence** — determines which result bucket the target lands in (positive / qualifying / dialectical / temporal)

---

## Installation

```bash
pip install prism-rag
```

From source:

```bash
git clone https://github.com/MadMando/prism
cd prism
pip install -e .
```

**Requirements:** Python 3.11+, an existing LanceDB vector store, and an embedding provider (Ollama local *or* any OpenAI-compatible API).

---

## Quick Start

### 1. Build the epistemic graph (one-time)

```python
from prism import PRISM

# Using Ollama for embeddings (local)
p = PRISM(
    lancedb_path = "/path/to/your/lancedb",
    graph_path   = "/path/to/prism_graph.json.gz",
    ollama_url   = "http://localhost:11434",
    embed_model  = "nomic-embed-text",
    llm_base_url = "https://api.openai.com",
    llm_model    = "gpt-4o-mini",
    llm_api_key  = "sk-...",
)

# Using an API for embeddings instead
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

p.build(
    k_neighbors       = 8,     # semantic neighbours per chunk to examine
    cross_source_only = True,  # only extract inter-document relationships (recommended)
    max_pairs         = 50_000 # cap for testing; omit for full build
)
```

Or via CLI:

```bash
prism-build \
    --lancedb-path /path/to/lancedb \
    --graph-path   /path/to/prism_graph.json.gz \
    --llm-api-key  $OPENAI_API_KEY \
    --max-pairs    50000
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
    The core relevant passage from your corpus...

[2] source-b  p.67  § 5.0  (score: 0.891)
    Another highly activated chunk...

## SUPPORTING EVIDENCE
[1] source-c  p.201  § 8.2  (score: 0.841  [via: specializes])
    A passage that specializes or extends the primary answer...

## QUALIFICATIONS & NUANCES
[1] source-d  p.38  § 3.1  (score: 0.712  [via: qualifies])
    A passage adding conditions or exceptions...

─ 2 primary · 1 supporting · 0 contrasting · 1 qualifying · 0 superseded ─
```

### 3. Use the result programmatically

```python
# Access buckets directly
for chunk in result.primary:
    print(chunk.source, chunk.page, chunk.final_score)
    print(chunk.text)

for chunk in result.contrasting:
    print("Contrasting view:", chunk.text[:200])

# Feed into your LLM
context = result.format_for_llm()
# ... pass `context` to your LLM system prompt

# Stats
print(f"Seeds: {result.n_seeds}")
print(f"Graph nodes reached: {result.n_graph_nodes}")
print(f"Graph used: {result.graph_was_used}")
```

---

## Embedding Providers

PRISM supports two embedding modes — use whichever matches how your corpus was built.

### Ollama (local or self-hosted)

```python
PRISM(
    ollama_url  = "http://localhost:11434",  # your Ollama instance
    embed_model = "nomic-embed-text",        # any model loaded in Ollama
    ...
)
```

Popular Ollama embedding models: `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`

### OpenAI-compatible API

```python
PRISM(
    embed_api_url = "https://api.openai.com/v1/embeddings",  # or any compatible endpoint
    embed_api_key = "sk-...",
    embed_model   = "text-embedding-3-small",
    ...
)
```

Works with OpenAI, Azure OpenAI, Together AI, Jina, Cohere, Mistral, or any endpoint that accepts `{"model": ..., "input": ...}` and returns `{"data": [{"embedding": [...]}]}`.

> **Important:** The embedding model at retrieval time must match the model used when your LanceDB corpus was originally indexed. Dimensions must be identical.

---

## No Re-embedding Required

PRISM works **on top of your existing vector store**. If you already have a LanceDB corpus with embeddings, you do not need to re-index anything.

- Existing vectors → used as-is for seed activation
- Epistemic graph → built from text via LLM, stored separately as a `.json.gz` file
- Build is a one-time offline step

---

## Graph Building: What Happens

The build phase extracts epistemic relationships from your corpus:

1. **Candidate pairs** — for each chunk, find top-K semantic neighbours via vector search. Filter to cross-document pairs (recommended — inter-document signal is most valuable).

2. **LLM extraction** — send batches of 5 pairs to any OpenAI-compatible LLM. Ask: *does an epistemic relationship exist between these two passages, and what type?*

3. **Graph construction** — confirmed relationships (above a confidence threshold) become typed, weighted edges.

4. **Save** — graph serialised to gzipped JSON (`prism_graph.json.gz`), loaded instantly at retrieval time.

**Cost estimate (typical 30k-chunk corpus):**

| Item | Estimate |
|------|----------|
| Candidate pairs generated | ~80k–120k |
| LLM calls (batch=5) | ~16k–24k |
| Input tokens | ~25M–35M |
| Output tokens | ~5M–7M |
| Cost with `gpt-4o-mini` | ~$3–6 |
| Cost with `deepseek-chat` | ~$5–9 |
| Build time | 6–14 hours |

---

## Fallback Behaviour

If no graph file exists (or graph loading fails), PRISM automatically falls back to **pure vector search** and still returns a valid `EpistemicResult` — just without epistemic bucketing. All chunks land in `primary`.

This means PRISM is a safe drop-in replacement for any standard vector retriever from day one.

---

## Full Configuration

```python
PRISM(
    # ── Storage ───────────────────────────────────────────────────
    lancedb_path  = "/path/to/lancedb",
    graph_path    = "/path/to/prism_graph.json.gz",
    table_name    = "knowledge",          # LanceDB table name

    # ── Embedding: Ollama (default) ───────────────────────────────
    ollama_url    = "http://localhost:11434",
    embed_model   = "nomic-embed-text",

    # ── Embedding: API (set embed_api_key to activate) ────────────
    embed_api_url = "https://api.openai.com/v1/embeddings",
    embed_api_key = None,                 # set to switch from Ollama

    # ── LLM for graph building ────────────────────────────────────
    llm_base_url  = "https://api.openai.com",
    llm_model     = "gpt-4o-mini",
    llm_api_key   = "sk-...",
    min_confidence = 0.65,               # edge confidence threshold
    batch_size    = 5,                   # pairs per LLM call

    # ── Retrieval tuning ──────────────────────────────────────────
    hops               = 3,             # spreading activation depth
    decay              = 0.7,           # per-hop decay factor
    seed_top_k         = 20,            # vector search seed count
    convergence_weight = 0.4,           # bonus weight for convergence
)
```

---

## Project Structure

```
prism/
├── prism/
│   ├── __init__.py         public API
│   ├── prism.py            PRISM — main entry point
│   ├── edges.py            epistemic edge taxonomy + propagation weights
│   ├── graph.py            EpistemicGraph (networkx MultiDiGraph + JSON serialisation)
│   ├── extractor.py        LLM-based triple extraction (batch mode)
│   ├── activation.py       SpreadingActivation engine + convergence scoring
│   ├── retriever.py        PRISMRetriever — the 5-step pipeline
│   ├── result.py           EpistemicResult + EpistemicChunk dataclasses
│   ├── cli.py              prism-build CLI
│   └── adapters/
│       └── lancedb.py      LanceDB adapter (Ollama + API embedding)
├── scripts/
│   └── build_graph.py      standalone build script
├── examples/
│   └── governance_search.py
├── docs/
│   └── architecture.svg
└── pyproject.toml
```

---

## Limitations

PRISM is alpha software. These are known constraints you should understand before using it in production:

**Graph quality depends on your corpus and LLM.**
The epistemic graph is built by asking an LLM to classify relationships between chunk pairs. The LLM can misclassify — a chunk may be tagged `supports` when it actually only tangentially relates, or `refutes` when it merely offers a different framing. Graph quality is correlated with chunk quality, chunking strategy, and the capability of the extraction model. Review extracted edges before trusting them.

**Confidence scores are not ground truth.**
The `confidence` values returned by the LLM are self-reported estimates, not calibrated probabilities. They are used as a filter (default threshold: 0.65) and as edge weights, but should not be treated as precise measures of relationship strength.

**Retrieval quality can degrade with noisy edges.**
If the graph contains many false-positive edges, spreading activation will propagate to irrelevant nodes. This produces worse results than pure vector search. Monitor your graph's edge yield rate and edge type distribution after building.

**Cross-source assumption.**
PRISM is designed for corpora with multiple distinct sources (documents, books, standards, frameworks). The `cross_source_only=True` default optimises for inter-document relationships. Single-source corpora will see fewer extracted edges and less epistemic structure.

**Build is slow and LLM-dependent.**
The one-time graph build requires thousands of LLM API calls and takes hours for large corpora. There is currently no incremental update — adding new documents requires a rebuild (or manual edge addition).

**No evaluation benchmark yet.**
We do not currently publish retrieval quality metrics comparing PRISM against standard RAG on standardised QA datasets. The `benchmarks/` directory contains a structural demonstration only.

---

## Roadmap

- [ ] Async LLM extraction (parallel API calls → 10× faster build)
- [ ] Incremental graph updates (add new docs without full rebuild)
- [ ] Additional vector store adapters (Chroma, Qdrant, Weaviate, pgvector)
- [ ] Graph visualisation (`prism-viz` CLI — exports to Gephi / D3)
- [ ] Export to Neo4j / NetworkX formats
- [ ] PyPI release (`pip install prism-rag`)

---

## References

- Collins, A.M. & Loftus, E.F. (1975). *A spreading-activation theory of semantic processing.* Psychological Review, 82(6), 407–428.
- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research.

---

## License

MIT
