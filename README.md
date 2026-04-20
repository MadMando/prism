<div align="center">

# PRISM

**Propagation & Retrieval via Informed Semantic Mapping**

*Epistemic Graph RAG with Spreading Activation*

[![PyPI](https://img.shields.io/pypi/v/prism-rag?style=flat-square&color=blue)](https://pypi.org/project/prism-rag/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/prism-rag?style=flat-square&color=blue)](https://pypi.org/project/prism-rag/)
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
# Core — bring your own vector store adapter
pip install prism-rag

# With a built-in vector store adapter:
pip install prism-rag[lancedb]    # LanceDB
pip install prism-rag[chroma]     # ChromaDB
pip install prism-rag[qdrant]     # Qdrant
pip install prism-rag[weaviate]   # Weaviate (v4 client)
pip install prism-rag[pgvector]   # PostgreSQL + pgvector
```

From source:

```bash
git clone https://github.com/MadMando/prism
cd prism
pip install -e .[lancedb]   # or [chroma], [qdrant], etc.
```

**Requirements:** Python 3.11+, an embedding provider (Ollama local *or* any OpenAI-compatible API), and a vector store.

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
    k_neighbors       = 8,      # semantic neighbours per chunk to examine
    cross_source_only = False,  # False recommended — see note below
    max_pairs         = 50_000  # cap for testing; omit for full build
)
```

Or via CLI:

```bash
prism-build \
    --lancedb-path /path/to/lancedb \
    --graph-path   /path/to/prism_graph.json.gz \
    --llm-api-key  $OPENAI_API_KEY
```

> **`cross_source_only` — default is `False` (include all sources)**
>
> PRISM defaults to extracting edges from *all* chunk pairs — same-source and cross-source alike. This produces richer graphs. On a 30k-chunk governance corpus:
>
> | Setting | Edges | Avg deg | Connected chunks |
> |---------|-------|---------|-----------------|
> | `cross_source_only=True` | 3,571 | 0.23 | ~49% |
> | `cross_source_only=False` (default) | 9,989 | 0.64 | ~82% |
>
> Pass `--cross-source-only` to restrict to inter-source pairs only. The old `--all-sources` flag is deprecated (it was the opt-in to current default behaviour).

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

The build phase uses a **two-stage pipeline** to extract epistemic relationships efficiently:

**Stage 1 — Ollama pre-filter (fast, free)**

An Ollama model screens every candidate pair with a binary question: *does any epistemic relationship exist here at all?* About half of all candidate pairs are topically similar but not epistemically related — Stage 1 discards these before any API call is made. Runs via Ollama, costs nothing.

**Stage 2 — Async LLM classification**

Surviving pairs are sent to an OpenAI-compatible LLM in batches of 20, with 20 concurrent requests running in parallel. For each pair that passes Stage 1, the LLM determines the relationship type, direction, and confidence score. The async architecture means 20 batches complete in the time v1 took to complete one.

**Stage 3 — Graph construction**

Confirmed relationships (above confidence threshold) become typed, weighted edges saved to `prism_graph.json.gz`.

**Build time comparison (30k-chunk corpus, ~50k candidate pairs):**

| Pipeline | Batches | Wall Time |
|----------|---------|-----------|
| v1 — sync, batch=5 | ~10,000 | **~40 hours** |
| v2 — async, batch=20, no filter | ~2,500 | **~30 minutes** |
| v2 — async + stage-1 filter (fast model) | ~1,250 | **~15–20 minutes** |

**Cost estimate (30k-chunk corpus, v2 pipeline):**

| Item | Estimate |
|------|----------|
| Stage 1 Ollama calls (free) | ~5,000 |
| Stage 2 LLM calls after filter (batch=20) | ~1,250 |
| Input tokens | ~12M–18M |
| Output tokens | ~2M–4M |
| Cost with `deepseek-chat` | ~$2–4 |
| Cost with `gpt-4o-mini` | ~$1–2 |
| Total wall time | ~15–20 min |

**Checkpoint & resume** — if the build is interrupted, PRISM saves a `.partial.json.gz` checkpoint and resumes automatically from where it left off.

**Stage 1 model check** — PRISM verifies the filter model is available in Ollama before starting. If it isn't, it prints a clear warning listing available models and skips Stage 1 rather than silently doing nothing. Pass `--filter-model <name>` or `filter_model=` to override.

---

## Choosing a Stage 1 Filter Model

Stage 1 only pays off if the filter model is **fast**. The goal is a binary yes/no decision per pair in well under a second — if your model is slower than the API you're filtering for, skip Stage 1 entirely with `--no-filter`.

**Rule of thumb: use a model under ~10 GB.** On typical hardware these complete in 0.2–0.8 seconds per call. Models above 20–30 GB (especially cloud-streamed ones) can take 2–4 seconds per call, making Stage 1 slower than it saves.

**Good filter model choices:**

| Model | Size | Notes |
|-------|------|-------|
| `llama3.1:8b` | 4.9 GB | **Recommended default** — fast, widely available, strong instruction following |
| `llama3.2:3b` | 2.0 GB | Faster still, good for binary yes/no |
| `qwen3.5:latest` | 6.6 GB | Strong instruction following |
| `gemma3:4b` | 3.3 GB | Lightweight, accurate |

**Avoid as filter models:** models above ~6 GB — at that size, per-call latency often exceeds the API savings, especially over a network connection. Cloud-streamed models (`*:cloud`, `*:1t`) are never suitable.

**Stage 1 requires true parallel GPU inference.** `OLLAMA_NUM_PARALLEL` only helps if your GPU has enough VRAM to hold multiple simultaneous model instances. If the GPU is already near capacity running a single 8B model, Ollama will queue extra requests and concurrency has no effect. If Stage 1 isn't reducing your wall time, use `--no-filter` — Stage 2 alone typically finishes in ~30 minutes on a 50k-pair corpus.

**If your Ollama server is remote** (not localhost), point `ollama_url` at it:

```python
p = PRISM(
    ollama_url    = "http://your-ollama-host:11434",  # remote Ollama
    embed_model   = "qwen3-embedding:4b",             # must match your ingest-time model
    filter_model  = "gemma4:latest",                  # fast model for Stage 1
    ...
)
```

Or via CLI:
```bash
prism-build \
    --ollama-url    http://your-ollama-host:11434 \
    --filter-model  gemma4:latest \
    ...
```

**No suitable model? Skip Stage 1:**
```bash
prism-build --no-filter ...   # async Stage 2 only, ~30 min for 50k pairs
```

---

## Fallback Behaviour

If no graph file exists (or graph loading fails), PRISM automatically falls back to **pure vector search** and still returns a valid `EpistemicResult` — just without epistemic bucketing. All chunks land in `primary`.

This means PRISM is a safe drop-in replacement for any standard vector retriever from day one.

---

## Full Configuration

```python
PRISM(
    # ── Storage ───────────────────────────────────────────────────
    graph_path    = "/path/to/prism_graph.json.gz",
    lancedb_path  = "/path/to/lancedb",   # OR use adapter= below
    table_name    = "knowledge",           # LanceDB table name

    # ── Custom adapter (alternative to lancedb_path) ─────────────
    adapter       = None,                  # pass a VectorAdapter instance

    # ── Embedding: Ollama (default) ───────────────────────────────
    ollama_url    = "http://localhost:11434",
    embed_model   = "nomic-embed-text",

    # ── Embedding: API (set embed_api_key to activate) ────────────
    embed_api_url = "https://api.openai.com/v1/embeddings",
    embed_api_key = None,                  # set to switch from Ollama

    # ── LLM for graph building ────────────────────────────────────
    llm_base_url      = "https://api.openai.com",
    llm_model         = "gpt-4o-mini",
    llm_api_key       = "sk-...",
    min_confidence    = 0.65,             # edge confidence threshold
    batch_size        = 20,              # pairs per LLM call (v2 default)
    max_concurrent    = 20,             # concurrent LLM requests
    max_retries       = 3,              # retries per failed batch (exp backoff)
    failure_log_path  = None,           # path to write failed-batch JSON log

    # ── Stage 1 filter ────────────────────────────────────────────
    filter_model          = "llama3.1:8b",
    filter_batch_size     = 10,
    filter_max_concurrent = 5,

    # ── Retrieval tuning ──────────────────────────────────────────
    hops               = 3,             # spreading activation depth
    decay              = 0.7,           # per-hop decay factor
    seed_top_k         = 20,            # vector search seed count
    convergence_weight = 0.4,           # bonus weight for convergence
    reranker           = None,          # optional Reranker callable
)
```

---

## Project Structure

```
prism/
├── prism/
│   ├── __init__.py         public API (PRISM, VectorAdapter, Embedder, Reranker, ...)
│   ├── prism.py            PRISM — main entry point + add_documents()
│   ├── edges.py            epistemic edge taxonomy + propagation weights
│   ├── graph.py            EpistemicGraph (networkx MultiDiGraph + JSON serialisation)
│   ├── extractor.py        LLM-based triple extraction (async, retries, failure log)
│   ├── filter.py           Stage 1 Ollama pre-filter
│   ├── activation.py       SpreadingActivation engine + convergence scoring
│   ├── retriever.py        PRISMRetriever — 5-step pipeline + reranker hook
│   ├── result.py           EpistemicResult + EpistemicChunk dataclasses
│   ├── cli.py              prism-build CLI
│   ├── inspect_cli.py      prism-stats + prism-inspect diagnostic CLIs
│   ├── viz_cli.py          prism-viz — export to Gephi GEXF or D3 JSON
│   └── adapters/
│       ├── base.py         VectorAdapter Protocol (@runtime_checkable)
│       ├── embedder.py     Embedder — reusable Ollama + OpenAI-compatible embedding helper
│       ├── lancedb.py      LanceDB adapter (requires prism-rag[lancedb])
│       ├── chroma.py       ChromaDB adapter (requires prism-rag[chroma])
│       ├── qdrant.py       Qdrant adapter (requires prism-rag[qdrant])
│       ├── weaviate.py     Weaviate v4 adapter (requires prism-rag[weaviate])
│       ├── pgvector.py     pgvector/PostgreSQL adapter (requires prism-rag[pgvector])
│       └── template.py     copy-paste skeleton for building a custom adapter
├── scripts/
│   └── build_graph.py      standalone build script
├── examples/
│   └── governance_search.py
├── docs/
│   └── architecture.svg
└── pyproject.toml
```

---

## Incremental Updates

After your graph is built, add new documents without a full rebuild:

```python
# 1. Ingest new docs into your vector store first (LanceDB, etc.)
# 2. Collect the new chunk IDs
new_ids = ["chunk_abc", "chunk_def", "chunk_ghi"]

# 3. Update the graph — finds neighbours of new chunks, runs pipeline, saves
p.load_graph()
n_new_edges = p.add_documents(new_ids)
print(f"Added {n_new_edges} new edges")
```

`add_documents()` runs the same two-stage pipeline but only over candidate pairs involving the new chunks. The existing graph is updated in-place and re-saved.

---

## Built-in Vector Store Adapters

PRISM ships adapters for four popular vector stores in addition to LanceDB. All share the same interface and support both Ollama and OpenAI-compatible embedding providers.

### ChromaDB

```python
from prism import PRISM
from prism.adapters.chroma import ChromaAdapter

adapter = ChromaAdapter(
    collection_name = "knowledge",
    host            = "localhost",
    port            = 8000,
    embed_model     = "nomic-embed-text",
)
p = PRISM(graph_path="prism_graph.json.gz", adapter=adapter, ...)
```

> Your collection should be configured with `hnsw:space="cosine"` for correct similarity scores.

### Qdrant

```python
from prism.adapters.qdrant import QdrantAdapter

adapter = QdrantAdapter(
    collection_name = "knowledge",
    url             = "http://localhost:6333",
    embed_model     = "nomic-embed-text",
)
```

> Chunk IDs are read from the `"id"` key in the point payload (configurable via `id_payload_key`).

### Weaviate

```python
from prism.adapters.weaviate import WeaviateAdapter

adapter = WeaviateAdapter(
    collection_name = "Knowledge",
    host            = "localhost",
    port            = 8080,
    id_property     = "chunk_id",  # property holding the PRISM chunk ID
    embed_model     = "nomic-embed-text",
)
```

> Uses the Weaviate v4 Python client. Objects must have a `chunk_id` property (or whatever `id_property` you configure) to be retrievable by ID.

### pgvector (PostgreSQL)

```python
from prism.adapters.pgvector import PgvectorAdapter

adapter = PgvectorAdapter(
    dsn         = "postgresql://user:pass@localhost:5432/mydb",
    table       = "chunks",   # table with id, source, page, section, text, embedding columns
    embed_model = "nomic-embed-text",
)
```

> Requires a table with a `vector(N)` column (pgvector extension). Column names are configurable.

---

## Custom Vector Stores

Implement the `VectorAdapter` Protocol to connect PRISM to any store not listed above.

**Quickest path:** copy `prism/adapters/template.py` from the repo — it's a fully-commented skeleton with all 7 methods stubbed and the `Embedder` helper already wired in.

```python
from prism import PRISM, VectorAdapter
from prism.adapters.embedder import Embedder

class MyAdapter:
    def __init__(self, ...):
        self._embedder = Embedder(model="nomic-embed-text")

    def seed_scores(self, query, top_k=20, source_filter=None) -> dict[str, float]: ...
    def get_chunks(self, node_ids) -> dict[str, dict]: ...
    def connect(self) -> None: ...
    def populate_graph_nodes(self, graph) -> int: ...
    def candidate_pairs(self, k_neighbors=8, cross_source_only=False, max_pairs=None): ...
    def candidate_pairs_for(self, node_ids, k_neighbors=8, cross_source_only=False): ...
    def stats(self) -> dict: ...

assert isinstance(MyAdapter(), VectorAdapter)  # Protocol is @runtime_checkable

p = PRISM(graph_path="prism_graph.json.gz", adapter=MyAdapter(), ...)
```

---

## Reranker Hook

Plug in any reranker (cross-encoder, LLM-based, Cohere Rerank, etc.) to post-process retrieval results:

```python
from prism import PRISM, Reranker, EpistemicChunk

def my_reranker(query: str, chunks: list[EpistemicChunk]) -> list[EpistemicChunk]:
    # Call your reranker API, return chunks in new order
    scores = cohere_client.rerank(query=query, documents=[c.text for c in chunks])
    return [chunks[r.index] for r in scores.results]

p = PRISM(
    lancedb_path = "/path/to/lancedb",
    graph_path   = "/path/to/prism_graph.json.gz",
    reranker     = my_reranker,
    ...
)

result = p.retrieve("your question")
# Chunks in result.primary are in reranker order, with final_score updated to reflect rank
```

The reranker runs after epistemic bucketing as a final pass over all retrieved chunks. If the reranker raises an exception, PRISM falls back to the original spreading-activation order.

---

## Graph Visualisation

Export the epistemic graph for visual exploration in Gephi or any D3.js-based tool:

```bash
# D3 JSON (force-directed graph) — default
prism-viz prism_graph.json.gz --output graph.json

# Gephi GEXF — open directly in Gephi for layout and clustering
prism-viz prism_graph.json.gz --format gexf --output graph.gexf

# Filter: only high-confidence supporting/refuting edges
prism-viz prism_graph.json.gz --format d3 --edge-types supports,refutes --min-confidence 0.8

# Filter: only chunks from one source
prism-viz prism_graph.json.gz --format d3 --source-filter "dmbok"

# Cap size for large graphs (keeps top-N nodes by degree)
prism-viz prism_graph.json.gz --format d3 --max-nodes 500
```

**D3 JSON format** — nodes have `id`, `source`, `group` (= source, for colour), and `degree`; links have `source`, `target`, `type`, `weight`, and `confidence`. Load directly with `d3.json()` and `d3.forceSimulation`.

**GEXF format** — standard Gephi format. Edge type, weight, confidence, and rationale are exported as edge attributes. Open in Gephi: `File → Open → graph.gexf`.

---

## Diagnostic CLIs

Two diagnostic commands are included after `pip install prism-rag`:

```bash
# Graph and store statistics
prism-stats /path/to/prism_graph.json.gz
prism-stats /path/to/prism_graph.json.gz --lancedb-path /path/to/lancedb  # include store stats
prism-stats /path/to/prism_graph.json.gz --json  # machine-readable output

# Inspect a specific node — its edges, neighbours, rationales
prism-inspect /path/to/prism_graph.json.gz --node chunk_abc123
prism-inspect /path/to/prism_graph.json.gz --node chunk_abc123 --json
```

Example `prism-stats` output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PRISM Graph Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  File     : prism_graph.json.gz
  Nodes    : 15,426
  Edges    : 9,989
  Avg deg  : 0.64 edges/node

  Edge type distribution:
    exemplifies          4,183  41.9%  ████████████
    supports             2,710  27.1%  ████████
    specializes          1,289  12.9%  ████
    ...
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

**`cross_source_only=True` produces sparse graphs on multi-book corpora.**
Setting `cross_source_only=True` skips all chunk pairs from the same source document. This seems conservative and clean, but in practice it misses the majority of valuable epistemic relationships. A large book like DMBOK (5,940 chunks) contains hundreds of intra-chapter `supports`, `derives_from`, and `specializes` relationships that are never extracted when same-source pairs are excluded.

In a real-world test on a 30k-chunk governance corpus (DMBOK + 2 governance books + erwin docs):
- `cross_source_only=True`, k=8: **3,571 edges**, ~51% of chunks unconnected — graph rarely fires on queries
- `cross_source_only=False`, k=8: **9,989 edges**, same corpus — graph activates supporting/qualifying buckets on most queries

**Recommendation: use `cross_source_only=False` for most corpora.** Only use `True` if your sources are genuinely independent and you specifically want to suppress intra-document structure.

**Build is slow and LLM-dependent.**
The one-time graph build requires thousands of LLM API calls and takes hours for large corpora. Incremental updates via `add_documents()` are supported for adding new chunks without a full rebuild.

**No evaluation benchmark yet.**
We do not currently publish retrieval quality metrics comparing PRISM against standard RAG on standardised QA datasets. The `benchmarks/` directory contains a structural demonstration only.

---

## Roadmap

- [x] Async LLM extraction (parallel API calls → 80× faster build)
- [x] Two-stage pipeline: local pre-filter + async classification
- [x] Checkpoint / resume for interrupted builds
- [x] Incremental graph updates via `add_documents()`
- [x] Pluggable `VectorAdapter` Protocol for custom vector stores
- [x] Reranker hook for cross-encoder / LLM-based reranking
- [x] `prism-stats` and `prism-inspect` diagnostic CLIs
- [x] Additional vector store adapters (Chroma, Qdrant, Weaviate, pgvector)
- [x] Graph visualisation (`prism-viz` CLI — exports to Gephi GEXF / D3 JSON)
- [ ] Export to Neo4j / NetworkX formats
- [ ] Retrieval quality benchmarks on standard QA datasets

---

## References

- Collins, A.M. & Loftus, E.F. (1975). *A spreading-activation theory of semantic processing.* Psychological Review, 82(6), 407–428.
- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research.

---

## License

MIT
