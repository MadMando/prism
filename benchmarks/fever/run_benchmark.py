"""
Run PRISM retrieval against FEVER claims and score bucket routing.

Usage:
    python benchmarks/fever/run_benchmark.py \
        --lancedb-dir /tmp/fever_lancedb \
        --graph-path  /tmp/fever_graph.json.gz \
        --n-claims 500 \
        --ollama-url http://localhost:11434 \
        --embed-model nomic-embed-text \
        --output results.json
"""
import argparse
import json
import sys
from pathlib import Path


# FEVER label → expected PRISM bucket(s) (order = preference)
LABEL_TO_BUCKETS = {
    "SUPPORTS":        ["supporting", "primary"],
    "REFUTES":         ["contrasting"],
    "NOT ENOUGH INFO": ["qualifying"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lancedb-dir",   required=True)
    p.add_argument("--graph-path",    required=True)
    p.add_argument("--claim-map",     default=None, help="Path to claim_map.jsonl (default: <lancedb-dir>/claim_map.jsonl)")
    p.add_argument("--n-claims",      type=int, default=None, help="Limit claims evaluated")
    p.add_argument("--top-k",         type=int, default=10)
    p.add_argument("--table-name",    default="fever")
    p.add_argument("--ollama-url",    default="http://localhost:11434")
    p.add_argument("--embed-model",   default="nomic-embed-text")
    p.add_argument("--embed-api-url", default=None)
    p.add_argument("--embed-api-key", default=None)
    p.add_argument("--output",        default="benchmarks/fever/results/latest.json")
    p.add_argument("--baseline-only", action="store_true", help="Score flat vector search only (no graph)")
    return p.parse_args()


def bucket_of(chunk_id: str, result) -> str | None:
    """Return the PRISM bucket name a chunk landed in, or None if not retrieved."""
    buckets = {
        "primary":     [c.chunk_id for c in result.primary],
        "supporting":  [c.chunk_id for c in result.supporting],
        "contrasting": [c.chunk_id for c in result.contrasting],
        "qualifying":  [c.chunk_id for c in result.qualifying],
        "superseded":  [c.chunk_id for c in result.superseded],
    }
    for name, ids in buckets.items():
        if chunk_id in ids:
            return name
    return None


def score_result(label: str, chunk_ids: list[str], result) -> dict:
    """
    For a single claim, check whether any gold chunk landed in the expected bucket.
    Returns a dict with hit/miss details.
    """
    expected = LABEL_TO_BUCKETS.get(label, [])
    found_buckets = {cid: bucket_of(cid, result) for cid in chunk_ids}
    retrieved = {cid: b for cid, b in found_buckets.items() if b is not None}

    # Bucket hit: at least one gold chunk in an expected bucket
    bucket_hit = any(b in expected for b in retrieved.values())

    # Retrieval hit: at least one gold chunk retrieved at all (any bucket)
    retrieval_hit = len(retrieved) > 0

    return {
        "label":          label,
        "expected_buckets": expected,
        "gold_chunks":    chunk_ids,
        "retrieved":      retrieved,
        "bucket_hit":     bucket_hit,
        "retrieval_hit":  retrieval_hit,
    }


def main():
    args = parse_args()

    claim_map_path = args.claim_map or str(Path(args.lancedb_dir) / "claim_map.jsonl")
    if not Path(claim_map_path).exists():
        print(f"ERROR: claim map not found at {claim_map_path}", file=sys.stderr)
        print("Run build_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    claims = []
    with open(claim_map_path) as f:
        for line in f:
            claims.append(json.loads(line))
    if args.n_claims:
        claims = claims[:args.n_claims]

    print(f"Evaluating {len(claims)} claims (top_k={args.top_k})…")

    from prism import PRISM

    p = PRISM(
        lancedb_path=args.lancedb_dir,
        graph_path=args.graph_path,
        table_name=args.table_name,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        embed_api_url=args.embed_api_url,
        embed_api_key=args.embed_api_key,
    )
    p.load_graph()

    records = []
    for i, claim in enumerate(claims):
        if i % 50 == 0:
            print(f"  {i}/{len(claims)}")

        result = p.retrieve(claim["claim"], top_k=args.top_k)
        scored = score_result(claim["label"], claim["chunk_ids"], result)
        scored["claim_id"] = claim["claim_id"]
        scored["claim"]    = claim["claim"]
        records.append(scored)

    # Aggregate
    from collections import defaultdict
    label_stats = defaultdict(lambda: {"total": 0, "bucket_hits": 0, "retrieval_hits": 0})
    for r in records:
        s = label_stats[r["label"]]
        s["total"]          += 1
        s["bucket_hits"]    += int(r["bucket_hit"])
        s["retrieval_hits"] += int(r["retrieval_hit"])

    print("\n── FEVER Benchmark Results ──────────────────────────────")
    print(f"{'Label':<20} {'N':>5} {'Retrieval%':>11} {'Bucket%':>9}")
    print("─" * 50)
    totals = {"total": 0, "bucket_hits": 0, "retrieval_hits": 0}
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        s = label_stats[label]
        if s["total"] == 0:
            continue
        ret_pct    = 100 * s["retrieval_hits"] / s["total"]
        bucket_pct = 100 * s["bucket_hits"]    / s["total"]
        print(f"{label:<20} {s['total']:>5} {ret_pct:>10.1f}% {bucket_pct:>8.1f}%")
        for k in totals:
            totals[k] += s[k]
    print("─" * 50)
    overall_ret    = 100 * totals["retrieval_hits"] / totals["total"] if totals["total"] else 0
    overall_bucket = 100 * totals["bucket_hits"]    / totals["total"] if totals["total"] else 0
    print(f"{'OVERALL':<20} {totals['total']:>5} {overall_ret:>10.1f}% {overall_bucket:>8.1f}%")
    print()
    print("Retrieval% = gold evidence sentence appeared in any retrieved bucket")
    print("Bucket%    = gold evidence sentence landed in the EXPECTED bucket")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": vars(args),
        "summary": {label: dict(s) for label, s in label_stats.items()},
        "overall": {
            "total":           totals["total"],
            "retrieval_hits":  totals["retrieval_hits"],
            "bucket_hits":     totals["bucket_hits"],
            "retrieval_pct":   round(overall_ret,    2),
            "bucket_pct":      round(overall_bucket, 2),
        },
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
