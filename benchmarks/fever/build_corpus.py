"""
Build a LanceDB corpus and PRISM graph from FEVER gold-evidence claims.

Usage:
    python benchmarks/fever/build_corpus.py \
        --output-dir /tmp/fever_lancedb \
        --n-claims 500 \
        --ollama-url http://localhost:11434 \
        --embed-model nomic-embed-text
"""
import argparse
import json
import hashlib
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",  required=True, help="LanceDB output directory")
    p.add_argument("--graph-path",  default=None,  help="Graph output path (default: <output-dir>/fever_graph.json.gz)")
    p.add_argument("--n-claims",    type=int, default=None, help="Limit number of claims (None = all ~9k verifiable)")
    p.add_argument("--split",       default="train", choices=["train", "validation", "test"])
    p.add_argument("--ollama-url",  default="http://localhost:11434")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--embed-api-url", default=None, help="OpenAI-compatible embedding API URL")
    p.add_argument("--embed-api-key", default=None)
    p.add_argument("--llm-base-url",  default="https://api.openai.com")
    p.add_argument("--llm-model",     default="gpt-4o-mini")
    p.add_argument("--llm-api-key",   default="")
    p.add_argument("--table-name",    default="fever")
    p.add_argument("--k-neighbors",   type=int, default=8)
    p.add_argument("--no-build-graph", action="store_true", help="Skip graph build (corpus only)")
    return p.parse_args()


def chunk_id(claim_id: str, wiki_page: str, sent_idx: int) -> str:
    raw = f"{claim_id}|{wiki_page}|{sent_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def load_fever_claims(split: str, n: int | None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install the 'datasets' package: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading FEVER gold evidence ({split})…")
    ds = load_dataset("copenlu/fever_gold_evidence", split=split)

    # Only keep verifiable claims (SUPPORTS or REFUTES) — they have gold evidence
    ds = ds.filter(lambda x: x["verifiable"] == "VERIFIABLE")

    if n:
        ds = ds.select(range(min(n, len(ds))))

    print(f"  {len(ds)} verifiable claims loaded")
    return ds


def build_lancedb_corpus(ds, output_dir: str, table_name: str, ollama_url: str,
                         embed_model: str, embed_api_url: str | None, embed_api_key: str | None):
    try:
        import lancedb
    except ImportError:
        print("ERROR: install lancedb: pip install prism-rag[lancedb]", file=sys.stderr)
        sys.exit(1)

    from prism.embedder import Embedder

    embedder = Embedder(
        ollama_url=ollama_url,
        embed_model=embed_model,
        embed_api_url=embed_api_url,
        embed_api_key=embed_api_key,
    )

    # Collect all unique evidence sentences as chunks
    chunks = {}  # chunk_id -> dict
    claim_map = []  # [{claim_id, claim_text, label, chunk_ids}]

    for row in ds:
        cid = row["id"]
        c_chunks = []
        for ev in row["evidence"]:
            # ev = [wiki_page, sent_id, sentence_text]
            if len(ev) < 3:
                continue
            wiki_page, sent_idx, sent_text = ev[0], ev[1], ev[2]
            ckey = chunk_id(cid, wiki_page, int(sent_idx) if sent_idx else 0)
            if ckey not in chunks:
                chunks[ckey] = {
                    "id":           ckey,
                    "source":       wiki_page,
                    "page":         int(sent_idx) if sent_idx else 0,
                    "section":      "",
                    "text":         sent_text,
                    "text_preview": sent_text[:200],
                    "fever_label":  row["label"],
                    "claim_id":     cid,
                }
            c_chunks.append(ckey)
        claim_map.append({
            "claim_id":   cid,
            "claim":      row["claim"],
            "label":      row["label"],
            "chunk_ids":  c_chunks,
        })

    print(f"  {len(chunks)} unique evidence sentences → embedding…")

    texts = [c["text"] for c in chunks.values()]
    chunk_list = list(chunks.values())

    batch = 64
    embeddings = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        embeddings.extend(embedder.embed(batch_texts))
        if i % 640 == 0:
            print(f"    embedded {i+len(batch_texts)}/{len(texts)}")

    db = lancedb.connect(output_dir)
    if table_name in db.table_names():
        db.drop_table(table_name)

    import pyarrow as pa
    dim = len(embeddings[0])
    schema = pa.schema([
        pa.field("id",           pa.string()),
        pa.field("source",       pa.string()),
        pa.field("page",         pa.int32()),
        pa.field("section",      pa.string()),
        pa.field("text",         pa.string()),
        pa.field("text_preview", pa.string()),
        pa.field("fever_label",  pa.string()),
        pa.field("claim_id",     pa.string()),
        pa.field("vector",       pa.list_(pa.float32(), dim)),
    ])

    rows = []
    for chunk, vec in zip(chunk_list, embeddings):
        rows.append({**chunk, "vector": vec})

    tbl = db.create_table(table_name, data=rows, schema=schema)
    print(f"  Wrote {len(rows)} rows to LanceDB table '{table_name}'")

    # Save claim map for the benchmark runner
    claim_map_path = Path(output_dir) / "claim_map.jsonl"
    with open(claim_map_path, "w") as f:
        for c in claim_map:
            f.write(json.dumps(c) + "\n")
    print(f"  Claim map → {claim_map_path}")

    return claim_map_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = args.graph_path or str(output_dir / "fever_graph.json.gz")

    ds = load_fever_claims(args.split, args.n_claims)

    claim_map_path = build_lancedb_corpus(
        ds,
        output_dir=str(output_dir),
        table_name=args.table_name,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        embed_api_url=args.embed_api_url,
        embed_api_key=args.embed_api_key,
    )

    if args.no_build_graph:
        print("Skipping graph build (--no-build-graph).")
        print(f"\nCorpus ready. Claim map: {claim_map_path}")
        return

    print("\nBuilding PRISM epistemic graph…")
    from prism import PRISM

    p = PRISM(
        lancedb_path=str(output_dir),
        graph_path=graph_path,
        table_name=args.table_name,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        embed_api_url=args.embed_api_url,
        embed_api_key=args.embed_api_key,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )
    p.build(k_neighbors=args.k_neighbors, cross_source_only=False)

    print(f"\nDone.")
    print(f"  LanceDB corpus : {output_dir}")
    print(f"  PRISM graph    : {graph_path}")
    print(f"  Claim map      : {claim_map_path}")
    print(f"\nNext: python benchmarks/fever/run_benchmark.py --lancedb-dir {output_dir} --graph-path {graph_path}")


if __name__ == "__main__":
    main()
