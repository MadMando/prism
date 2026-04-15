# PRISM — Epistemic Graph RAG with Spreading Activation

> **P**ropagation and **R**etrieval via **I**nformed **S**emantic **M**apping

PRISM is a novel RAG retrieval library that layers a **typed epistemic graph** over an existing vector store, then uses **spreading activation** (Collins & Loftus, 1975) to retrieve knowledge that is not just semantically similar — but epistemically structured.

---

## The Problem with Standard RAG

Standard RAG treats all retrieved chunks the same: a flat ranked list by cosine similarity.

```
query → embed → similarity → [chunk1, chunk2, chunk3, ...]
```

This loses critical structure. Consider a data governance corpus:

- DMBOK *defines* data stewardship accountability
- Ladley *specializes* that definition for practical roles
- Plotkin *qualifies* it with exceptions for federated orgs
- An older standard *has been superseded* by DMBOK

Standard RAG returns all four at the same level. An LLM given this context can't tell what to trust, what to weight, or what's outdated.

---

## PRISM's Approach

PRISM builds a **knowledge graph where edges are epistemic relationships**, not generic "related_to" links:

```
DMBOK chunk ──[specializes]──▶ Ladley chunk
DMBOK chunk ──[qualifies]────▶ Plotkin chunk  
Old standard ──[superseded by]▶ DMBOK chunk
```

Then retrieval uses **spreading activation**: a query activates seed nodes via vector search, activation propagates through typed edges, and nodes reached by *multiple independent paths* (convergence) rank highest.

```
query → seeds (vector) → spreading activation → convergence scoring
      → epistemic bucketing → structured result
```

The result isn't a ranked list — it's a structured answer:

```
PRIMARY:    The core relevant chunks
SUPPORTING: Chunks that reinforce/extend the primary answer  
CONTRASTING: Chunks that challenge or offer a different view
QUALIFYING: Chunks that add conditions, exceptions, nuances
SUPERSEDED: Historically relevant but now outdated context
```

---

## Why This Is Novel

| Approach | Vector Search | Graph | Epistemic Typing | Spreading Activation |
|----------|:---:|:---:|:---:|:---:|
| Standard RAG | ✅ | ❌ | ❌ | ❌ |
| GraphRAG (Microsoft) | ✅ | ✅ | ❌ | ❌ |
| SYNAPSE (2026) | ✅ | ✅ | ❌ | ✅ |
| **PRISM** | ✅ | ✅ | ✅ | ✅ |

The key gap: no existing system classifies graph edges by *epistemic relationship type*. PRISM is the first to combine all four signals.

---

## Epistemic Edge Types

```
supports        — A provides evidence reinforcing B
refutes         — A directly contradicts B  
supersedes      — A replaces/updates B (A is newer/more authoritative)
derives_from    — A is logically derived from B
specializes     — A is a specific application of the general principle in B
contrasts_with  — A and B take different but non-exclusive positions
implements      — A is a practical tool putting the abstract concept of B into practice
generalizes     — A is a broader abstraction of which B is a specific case
exemplifies     — A is a concrete example illustrating the concept in B
qualifies       — A adds conditions, exceptions, or nuances to B
```

Each edge has a **propagation weight** (how strongly it carries activation) and a **valence** (positive / qualifying / dialectical / temporal) that determines which bucket a retrieved chunk lands in.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         PRISM                               │
│                                                             │
│  LanceDB Vector Store                                       │
│  ┌─────────────────┐     ┌───────────────────────────┐     │
│  │  Chunk Corpus   │────▶│  EpistemicGraph           │     │
│  │  (embeddings)   │     │  (typed edge graph)       │     │
│  └────────┬────────┘     └────────────┬──────────────┘     │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌─────────────────┐     ┌───────────────────────────┐     │
│  │  Seed Activation│     │  SpreadingActivation      │     │
│  │  (vector search)│────▶│  (hop propagation +       │     │
│  └─────────────────┘     │   convergence scoring)    │     │
│                          └────────────┬──────────────┘     │
│                                       ▼                     │
│                          ┌───────────────────────────┐     │
│                          │  EpistemicResult          │     │
│                          │  primary / supporting /   │     │
│                          │  contrasting / qualifying │     │
│                          │  / superseded             │     │
│                          └───────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install prism-rag
```

Or from source:

```bash
git clone https://github.com/MadMando/prism
cd prism
pip install -e .
```

**Requirements:** Python 3.11+, an existing LanceDB vector store, an Ollama instance (for embeddings), and an OpenAI-compatible LLM API for the one-time graph build.

---

## Quick Start

### 1. Build the epistemic graph (one-time, offline)

```python
from prism import PRISM

p = PRISM(
    lancedb_path = "/path/to/lancedb",
    graph_path   = "/path/to/prism_graph.json.gz",
    ollama_url   = "http://localhost:11434",
    embed_model  = "qwen3-embedding:4b",       # must match ingest-time model
    llm_base_url = "https://api.deepseek.com",  # any OpenAI-compatible API
    llm_model    = "deepseek-chat",
    llm_api_key  = "sk-...",
)

p.build(
    k_neighbors       = 8,      # semantic neighbours per chunk
    cross_source_only = True,   # only extract inter-source relationships
    max_pairs         = 50_000, # cap for testing; remove for full build
)
```

Or via CLI:

```bash
prism-build \
    --lancedb-path /path/to/lancedb \
    --graph-path   /path/to/prism_graph.json.gz \
    --llm-api-key  $DEEPSEEK_API_KEY \
    --max-pairs    50000
```

### 2. Retrieve with epistemic structure

```python
p.load_graph()

result = p.retrieve("what is data stewardship accountability")

print(result.format_for_llm())
# PRISM retrieval for: "what is data stewardship accountability"
# ────────────────────────────────────────────────────────────
#
# ## PRIMARY
# [1] dmbok  p.120  § 4.2 Stewardship  (score: 0.923)
#     Data stewardship is the formal accountability for business and technical ...
#
# ## SUPPORTING EVIDENCE
# [1] ladley  p.44  § 2.1 Roles  (score: 0.841  [via: specializes])
#     A data steward owns the quality and fitness-for-use of data within ...
#
# ## QUALIFICATIONS & NUANCES
# [1] plotkin  p.201  § 9.3 Federated  (score: 0.712  [via: qualifies])
#     In federated architectures, stewardship accountability may be distributed ...
#
# ─ 3 primary · 2 supporting · 0 contrasting · 1 qualifying · 0 superseded ─
```

### 3. Inspect the result

```python
for chunk in result.primary:
    print(chunk.source, chunk.page, chunk.final_score)

for chunk in result.contrasting:
    print("CONTRASTS:", chunk.text[:200])

print("Graph used:", result.graph_was_used)
print("Seeds activated:", result.n_seeds)
print("Nodes reached:", result.n_graph_nodes)
```

---

## Graph Building: How It Works

The build phase scans all chunk pairs in the vector store and uses an LLM to identify epistemic relationships:

1. **Candidate pairs** — for each chunk, find its top-K semantic neighbours via vector search. Filter to cross-source pairs only (inter-framework signal is most valuable).

2. **LLM extraction** — send batches of 5 pairs to an OpenAI-compatible LLM. Ask: *"does an epistemic relationship exist between these two chunks, and if so, what type?"*

3. **Graph construction** — confirmed relationships (above a confidence threshold) become typed, weighted edges in the graph.

4. **One-time cost** — for a 30k-chunk corpus: ~$5–8 with DeepSeek-chat, ~8–12 hours. The graph is then saved as a gzipped JSON file and loaded instantly at query time.

---

## No Re-embedding Required

PRISM works **on top of your existing vector store**. If you already have a LanceDB (or compatible) corpus with embeddings, you do not need to re-embed anything. The epistemic graph is built from text, not vectors.

---

## Configuration

```python
PRISM(
    # Vector store
    lancedb_path   = "/path/to/lancedb",
    table_name     = "knowledge",          # LanceDB table name
    ollama_url     = "http://localhost:11434",
    embed_model    = "qwen3-embedding:4b",

    # Graph file
    graph_path     = "/path/to/prism_graph.json.gz",

    # LLM for graph building (OpenAI-compatible)
    llm_base_url   = "https://api.deepseek.com",
    llm_model      = "deepseek-chat",
    llm_api_key    = "sk-...",
    min_confidence = 0.65,   # edge confidence threshold
    batch_size     = 5,      # pairs per LLM call

    # Retrieval tuning
    hops               = 3,    # spreading activation depth
    decay              = 0.7,  # per-hop activation decay
    seed_top_k         = 20,   # vector search seed count
    convergence_weight = 0.4,  # bonus for multi-path convergence
)
```

---

## Fallback Behaviour

If no graph is loaded (or the graph file doesn't exist yet), PRISM falls back to pure vector search and still returns an `EpistemicResult` — just without the epistemic bucketing. This means you can use PRISM as a drop-in replacement for a standard vector retriever from day one.

---

## Project Structure

```
prism/
├── prism/
│   ├── __init__.py        # public API
│   ├── prism.py           # PRISM main class
│   ├── edges.py           # epistemic edge taxonomy
│   ├── graph.py           # EpistemicGraph (networkx MultiDiGraph)
│   ├── extractor.py       # LLM triple extraction
│   ├── activation.py      # SpreadingActivation engine
│   ├── retriever.py       # PRISMRetriever pipeline
│   ├── result.py          # EpistemicResult dataclasses
│   ├── cli.py             # prism-build CLI entry point
│   └── adapters/
│       └── lancedb.py     # LanceDB adapter
├── scripts/
│   └── build_graph.py     # standalone build script
├── examples/
│   └── governance_search.py
└── pyproject.toml
```

---

## Roadmap

- [ ] Additional vector store adapters (Chroma, Qdrant, Weaviate, pgvector)
- [ ] Async extraction with concurrent LLM calls (10x build speedup)
- [ ] Incremental graph updates (add new chunks without full rebuild)
- [ ] Graph visualisation (`prism-viz` CLI)
- [ ] Export to Neo4j / NetworkX formats
- [ ] PyPI release

---

## Citation / Inspiration

- Collins, A.M. & Loftus, E.F. (1975). *A spreading-activation theory of semantic processing.* Psychological Review, 82(6), 407–428.
- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research.

---

## License

MIT
