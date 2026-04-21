# PRISM — Epistemic Graph RAG with Spreading Activation

**Propagation & Retrieval via Informed Semantic Mapping**

[![PyPI](https://img.shields.io/pypi/v/prism-rag?style=flat-square&color=blue)](https://pypi.org/project/prism-rag/)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/MadMando/prism/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-MadMando%2Fprism-black?style=flat-square&logo=github)](https://github.com/MadMando/prism)
[![AI Attribution](https://img.shields.io/badge/AI%20attribution-disclosed-blueviolet?style=flat-square)](https://aiattribution.github.io/)

<table><tr><td bgcolor="#f3e8ff"><a href="https://aiattribution.github.io/statements/AIA-HAb-SeCeNc-Hin-R-?model=Claude%20Sonnet%204.6-v1.0">AIA HAb SeCeNc Hin R Claude Sonnet 4.6 v1.0</a> <img src="https://raw.githubusercontent.com/MadMando/prism/main/docs/attribution/human-ai-blend.png" height="22" alt="Human-AI blend"> <img src="https://raw.githubusercontent.com/MadMando/prism/main/docs/attribution/stylistic-edits.png" height="22" alt="Stylistic edits"> <img src="https://raw.githubusercontent.com/MadMando/prism/main/docs/attribution/content-edits.png" height="22" alt="Content edits"> <img src="https://raw.githubusercontent.com/MadMando/prism/main/docs/attribution/new-content.png" height="22" alt="New content"> <img src="https://raw.githubusercontent.com/MadMando/prism/main/docs/attribution/human-initiated.png" height="22" alt="Human-initiated"> <img src="https://raw.githubusercontent.com/MadMando/prism/main/docs/attribution/reviewed.png" height="22" alt="Reviewed"></td></tr></table>

> This project was designed and directed by a human author. Code, documentation, and the research paper were substantially drafted with the assistance of [Claude Sonnet 4.6](https://anthropic.com/claude) (Anthropic). Architecture decisions, domain framing, and editorial judgement remain the author's own. Disclosed using the [AI Attribution Toolkit](https://aiattribution.github.io/).

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
# Core — bring your own vector store adapter
pip install prism-rag

# With a built-in adapter:
pip install prism-rag[lancedb]    # LanceDB
pip install prism-rag[chroma]     # ChromaDB
pip install prism-rag[qdrant]     # Qdrant
pip install prism-rag[weaviate]   # Weaviate (v4 client)
pip install prism-rag[pgvector]   # PostgreSQL + pgvector

# With the interactive explorer:
pip install prism-rag[lancedb,explorer]

# With Neo4j Bolt export:
pip install prism-rag[neo4j]
```

Requires Python 3.11+ and an embedding provider (Ollama or any OpenAI-compatible API).

---

## CLI Quickstart

This is the typical end-to-end workflow from an existing vector store to structured retrieval.

### Step 1 — Build the graph

Point PRISM at your existing LanceDB store. The graph is built once and saved to a `.json.gz` file.

```bash
prism-build \
  --lancedb-path /path/to/lancedb \
  --graph-path   prism_graph.json.gz \
  --llm-api-key  $OPENAI_API_KEY
```

With Ollama embeddings and a local filter model:

```bash
prism-build \
  --lancedb-path  /path/to/lancedb \
  --graph-path    prism_graph.json.gz \
  --ollama-url    http://localhost:11434 \
  --embed-model   nomic-embed-text \
  --filter-model  llama3.1:8b \
  --llm-base-url  https://api.openai.com \
  --llm-model     gpt-4o-mini \
  --llm-api-key   $OPENAI_API_KEY
```

If interrupted, the build **resumes automatically from its checkpoint** — just re-run the same command.

### Step 2 — Verify the graph

```bash
prism-stats prism_graph.json.gz
```

Also show LanceDB stats alongside:

```bash
prism-stats prism_graph.json.gz --lancedb-path /path/to/lancedb
```

Output JSON for scripting:

```bash
prism-stats prism_graph.json.gz --json
```

### Step 3 — Explore interactively

```bash
prism-explore \
  --lancedb-path /path/to/lancedb \
  --graph-path   prism_graph.json.gz \
  --embed-model  nomic-embed-text
```

Open `http://localhost:7860` to get a force-directed graph with semantic search. Type a question and watch activation spread — nodes glow by bucket (primary / supporting / contrasting / qualifying / superseded).

---

## CLI Reference

### `prism-build` — Build the epistemic graph

```
prism-build --lancedb-path PATH --graph-path PATH [options]
```

**Graph shape:**

| Flag | Default | Description |
|------|---------|-------------|
| `--k-neighbors` | `8` | Semantic neighbours per chunk used as candidate pairs |
| `--cross-source-only` | off | Only extract edges between different source documents |
| `--min-confidence` | `0.65` | Drop edges below this confidence score |
| `--max-pairs` | unlimited | Cap candidate pairs (useful for large corpora) |
| `--force` | off | Rebuild even if the graph file already exists |
| `--no-resume` | off | Ignore checkpoint, start from scratch |

**Embeddings:**

| Flag | Default | Description |
|------|---------|-------------|
| `--table-name` | `knowledge` | LanceDB table name |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |
| `--embed-model` | `nomic-embed-text` | Embedding model (must match ingest-time model) |

**Stage 1 — local filter (fast, free):**

| Flag | Default | Description |
|------|---------|-------------|
| `--filter-model` | `llama3.1:8b` | Ollama model for binary pre-filter |
| `--filter-batch-size` | `10` | Pairs per Ollama call |
| `--filter-concurrency` | `5` | Concurrent Ollama requests |
| `--no-filter` | off | Skip Stage 1 (if Ollama is unavailable) |

**Stage 2 — LLM extraction:**

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-base-url` | `https://api.deepseek.com` | OpenAI-compatible API base URL |
| `--llm-model` | `deepseek-chat` | Model for epistemic extraction |
| `--llm-api-key` | `""` | API key (or set `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`) |
| `--batch-size` | `20` | Pairs per LLM call |
| `--max-concurrent` | `20` | Concurrent Stage 2 API requests |
| `--failure-log` | none | Path to write JSON log of failed extraction batches |

---

### `prism-stats` — Graph and store statistics

```
prism-stats GRAPH_PATH [--lancedb-path PATH] [--table-name NAME] [--json]
```

Prints node count, edge count, edge-type breakdown, and density. Add `--lancedb-path` to also report vector store chunk count and source breakdown.

---

### `prism-inspect` — Inspect a single node

```
prism-inspect GRAPH_PATH --node NODE_ID [--max-edges N] [--json]
```

Shows the node's metadata and all its incoming and outgoing edges with types and confidence scores. Useful for debugging why a chunk is or isn't appearing in retrieval.

```bash
prism-inspect prism_graph.json.gz --node "chunk_abc123" --max-edges 30
```

---

### `prism-explore` — Interactive web explorer

```
prism-explore --lancedb-path PATH --graph-path PATH [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--table-name` | `knowledge` | LanceDB table name |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |
| `--embed-model` | `nomic-embed-text` | Embedding model |
| `--embed-api-url` | none | OpenAI-compatible embedding API URL |
| `--embed-api-key` | none | Embedding API key |
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `7860` | Bind port |

With an OpenAI-compatible embedding API instead of Ollama:

```bash
prism-explore \
  --lancedb-path  /path/to/lancedb \
  --graph-path    prism_graph.json.gz \
  --embed-api-url https://api.openai.com/v1/embeddings \
  --embed-api-key $OPENAI_API_KEY \
  --embed-model   text-embedding-3-small
```

The explorer lets you:
- Browse the force-directed epistemic graph (15k+ nodes)
- Toggle edge types on/off, filter by source, set a confidence floor
- Type a question and watch spreading activation colour nodes by result bucket
- Click any node to see its connections in a side panel
- Export the current layout as a standalone interactive HTML file

---

### `prism-viz` — Export for Gephi or D3

```
prism-viz GRAPH_PATH [--format gexf|d3] [--output PATH] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `d3` | `d3` (JSON for D3.js) or `gexf` (Gephi) |
| `--output`, `-o` | auto | Output file; use `-` to write D3 JSON to stdout |
| `--edge-types` | all | Comma-separated list: `supports,refutes,supersedes,…` |
| `--min-confidence` | `0.0` | Drop edges below this threshold |
| `--source-filter` | none | Only include nodes whose source contains this string |
| `--max-nodes` | unlimited | Keep the top-N highest-degree nodes only |

```bash
# Export high-confidence supports/refutes edges for one document set
prism-viz prism_graph.json.gz \
  --format gexf \
  --edge-types supports,refutes \
  --min-confidence 0.8 \
  --output review_graph.gexf

# Pipe D3 JSON into another tool
prism-viz prism_graph.json.gz --output - | jq '.nodes | length'
```

---

### `prism-export` — Export to Neo4j

```
prism-export GRAPH_PATH [--format cypher|neo4j] [options]
```

**Write a Cypher script** (no Neo4j required):

```bash
prism-export prism_graph.json.gz --format cypher --output graph.cypher

# Load it:
cypher-shell -u neo4j -p secret < graph.cypher
```

**Push directly via Bolt:**

```bash
pip install prism-rag[neo4j]

prism-export prism_graph.json.gz \
  --format neo4j \
  --uri      bolt://localhost:7687 \
  --user     neo4j \
  --password secret \
  --clear           # wipe existing :Chunk nodes first
```

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | `500` | Nodes/edges per transaction |
| `--database` | `neo4j` | Target Neo4j database name |
| `--clear` | off | Delete all `:Chunk` nodes before import |

---

## Python API Quick Start

### Build the graph

```python
from prism import PRISM

p = PRISM(
    lancedb_path = "/path/to/lancedb",
    graph_path   = "prism_graph.json.gz",
    ollama_url   = "http://localhost:11434",
    embed_model  = "nomic-embed-text",
    llm_base_url = "https://api.openai.com",
    llm_model    = "gpt-4o-mini",
    llm_api_key  = "sk-...",
)

p.build(k_neighbors=8, cross_source_only=False)
```

### Retrieve

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

### Access results programmatically

```python
for chunk in result.primary:
    print(chunk.source, chunk.page, chunk.final_score, chunk.text)

for chunk in result.contrasting:
    print("Contrasting view:", chunk.text[:200])

# Feed structured context directly into your LLM
context = result.format_for_llm()
```

### Export the graph

```python
import networkx as nx

# NetworkX — use any graph algorithm
G = graph.to_networkx()          # returns nx.MultiDiGraph copy
pr = nx.pagerank(G, weight="weight")
communities = nx.community.greedy_modularity_communities(G.to_undirected())

# Cypher script
graph.to_cypher("graph.cypher")

# Neo4j Bolt
graph.to_neo4j("bolt://localhost:7687", user="neo4j", password="secret")
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

## Build Performance

The graph is built once offline. PRISM uses a **two-stage pipeline** that makes large-corpus builds practical:

**Stage 1 — Ollama pre-filter (fast, free)**
An Ollama model screens candidate pairs with a binary yes/no question. ~50% of pairs are discarded before any API call. Runs via Ollama, costs nothing.

**Stage 2 — Async LLM classification**
Surviving pairs are classified with full type + confidence using 20 concurrent API requests, in batches of 20 pairs each.

**Build time — 30k-chunk corpus, ~50k candidate pairs:**

| Pipeline | Wall Time |
|----------|-----------|
| v1 — sync, batch=5 | ~40 hours |
| v2 — async only, batch=20 | ~30 minutes |
| v2 — async + stage-1 filter | **~15–20 minutes** |

**Checkpoint / resume** — if interrupted, the build saves progress automatically and resumes from where it left off.

**`cross_source_only=False` produces significantly richer graphs.** On a 30k-chunk governance corpus: `True` = 3,571 edges (graph rarely fires); `False` = 9,989 edges (supporting/qualifying buckets activate on most queries). Use `False` unless your sources are genuinely independent.

**Choosing a Stage 1 filter model — use a model under ~5 GB.** Small, fast models (`llama3.1:8b`, `llama3.2:3b`, `gemma3:4b`) complete each binary call in under a second. Models above ~6 GB can take 2–4 seconds per call and negate the benefit of filtering entirely. If no fast model is available, use `--no-filter` and rely on Stage 2 alone (~30 min).

---

## No Re-embedding Required

PRISM works **on top of your existing vector store**. If you have an existing corpus with embeddings, you don't need to re-index anything.

- Existing vectors → used as-is for seed activation
- Epistemic graph → built from text via LLM, stored as a separate `.json.gz` file
- Fallback → if no graph exists, PRISM automatically falls back to pure vector search

---

## Vector Store Adapters

PRISM ships adapters for LanceDB, ChromaDB, Qdrant, Weaviate, and pgvector. All share the same interface:

```python
from prism.adapters.chroma   import ChromaAdapter
from prism.adapters.qdrant   import QdrantAdapter
from prism.adapters.weaviate import WeaviateAdapter
from prism.adapters.pgvector import PgvectorAdapter

adapter = QdrantAdapter(collection_name="knowledge", url="http://localhost:6333")
p = PRISM(graph_path="prism_graph.json.gz", adapter=adapter, ...)
```

To connect a different store, implement the `VectorAdapter` Protocol — copy `prism/adapters/template.py` for a fully-commented skeleton.

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

## Changelog

### 0.2.7 — NetworkX and Neo4j export

```python
G = graph.to_networkx()          # nx.MultiDiGraph copy — use any nx algorithm
graph.to_cypher("graph.cypher")  # write Cypher script
graph.to_neo4j("bolt://localhost:7687", user="neo4j", password="secret")
```

CLI:

```bash
prism-export graph.json.gz --format cypher --output graph.cypher
prism-export graph.json.gz --format neo4j --uri bolt://localhost:7687 \
  --user neo4j --password secret
```

Install: `pip install prism-rag[neo4j]` for direct Bolt push.

### 0.2.6 — Local interactive graph explorer

```bash
pip install prism-rag[lancedb,explorer]

prism-explore \
  --lancedb-path /path/to/lancedb \
  --graph-path   prism_graph.json.gz \
  --embed-model  nomic-embed-text
```

Open `http://localhost:7860` — force-directed graph, edge-type toggles, confidence slider, semantic query mode, Export HTML.

### 0.2.5 — Adapter bug fixes

- **LanceDB:** `get_chunks` no longer silently drops node IDs past the first 100.
- **ChromaDB:** dropped the invalid `$contains` `where` filter; `source_filter` now applies client-side.
- **Weaviate:** vectors cached in initial scan — no more N+1 round-trips in `candidate_pairs_for`.
- **pgvector:** separate cursors for fetch and neighbour queries (avoids psycopg2 buffer-invalidation).
- Tests added for all five adapters and `prism-viz`. CI now runs a matrix over all extras with a 50% coverage floor.

---

## License

MIT
