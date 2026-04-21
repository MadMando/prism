# FEVER Benchmark for PRISM

Evaluates PRISM's epistemic bucketing against the FEVER fact-verification dataset.

## What FEVER provides

185,441 Wikipedia-derived claims, each labelled:
- **SUPPORTS** — evidence confirms the claim
- **REFUTES** — evidence contradicts the claim
- **NOT ENOUGH INFO** — evidence is insufficient

Each verified claim comes with the gold Wikipedia sentence(s) that justify the label.

## How we map FEVER → PRISM buckets

| FEVER label | Expected PRISM bucket |
|-------------|----------------------|
| SUPPORTS    | SUPPORTING (or PRIMARY) |
| REFUTES     | CONTRASTING |
| NOT ENOUGH INFO | QUALIFYING |

A retrieval is considered correct if the gold evidence sentence lands in the
expected bucket. We use `copenlu/fever_gold_evidence` (HuggingFace) which
ships only the verified claims with clean gold evidence sentences.

## Metrics

- **Bucket accuracy** — % of gold evidence sentences routed to the correct bucket
- **Bucket precision / recall** per label
- **Baseline** — flat vector search (no graph), which has no bucket concept and
  is scored by whether the gold sentence appears anywhere in the top-K results

## Running

```bash
pip install prism-rag[lancedb] datasets

# Build a FEVER mini-corpus in LanceDB, then run the benchmark
python benchmarks/fever/build_corpus.py \
    --output-dir /tmp/fever_lancedb \
    --n-claims 500          # subset size for quick eval; omit for full run
    --ollama-url http://localhost:11434 \
    --embed-model nomic-embed-text

python benchmarks/fever/run_benchmark.py \
    --lancedb-dir /tmp/fever_lancedb \
    --graph-path  /tmp/fever_graph.json.gz \
    --n-claims 500 \
    --ollama-url http://localhost:11434 \
    --embed-model nomic-embed-text \
    --llm-api-key $OPENAI_API_KEY \
    --output results.json
```

## Files

| File | Purpose |
|------|---------|
| `build_corpus.py` | Download FEVER gold evidence, embed into LanceDB, build PRISM graph |
| `run_benchmark.py` | Run PRISM retrieval on each claim, score bucket routing |
| `score.py` | Standalone scorer — given a results JSON, print metrics table |
| `results/` | Saved benchmark runs (gitignored except summaries) |
