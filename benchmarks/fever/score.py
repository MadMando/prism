"""
Standalone scorer — given a results JSON from run_benchmark.py, print metrics table.

Usage:
    python benchmarks/fever/score.py benchmarks/fever/results/latest.json
    python benchmarks/fever/score.py results1.json results2.json  # compare runs
"""
import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_table(data: dict, label: str = ""):
    if label:
        print(f"\n── {label} ──")
    summary = data.get("summary", {})
    overall = data.get("overall", {})

    print(f"{'Label':<20} {'N':>5} {'Retrieval%':>11} {'Bucket%':>9}")
    print("─" * 50)
    for lbl in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        s = summary.get(lbl)
        if not s or s["total"] == 0:
            continue
        ret_pct    = 100 * s["retrieval_hits"] / s["total"]
        bucket_pct = 100 * s["bucket_hits"]    / s["total"]
        print(f"{lbl:<20} {s['total']:>5} {ret_pct:>10.1f}% {bucket_pct:>8.1f}%")
    print("─" * 50)
    total = overall.get("total", 0)
    if total:
        print(f"{'OVERALL':<20} {total:>5} {overall.get('retrieval_pct', 0):>10.1f}% {overall.get('bucket_pct', 0):>8.1f}%")
    print()
    cfg = data.get("config", {})
    print(f"top_k={cfg.get('top_k', '?')}  embed={cfg.get('embed_model', '?')}  graph={cfg.get('graph_path', '?')}")


def compare_tables(paths: list[str]):
    runs = [(Path(p).name, load_results(p)) for p in paths]

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "OVERALL"]
    print(f"\n{'Label':<22}", end="")
    for name, _ in runs:
        print(f"  {name[:18]:>18}", end="")
    print()
    print("─" * (22 + 20 * len(runs)))

    for lbl in labels:
        print(f"{lbl:<22}", end="")
        for _, data in runs:
            if lbl == "OVERALL":
                ov = data.get("overall", {})
                ret = ov.get("retrieval_pct", 0)
                bkt = ov.get("bucket_pct", 0)
            else:
                s = data.get("summary", {}).get(lbl, {})
                if not s or s.get("total", 0) == 0:
                    print(f"  {'—':>18}", end="")
                    continue
                ret = 100 * s["retrieval_hits"] / s["total"]
                bkt = 100 * s["bucket_hits"]    / s["total"]
            print(f"  {f'{ret:.1f}%/{bkt:.1f}%':>18}", end="")
        print()
    print()
    print("Format: Retrieval%/Bucket%")


def main():
    p = argparse.ArgumentParser(description="Score FEVER benchmark results")
    p.add_argument("results", nargs="+", help="Path(s) to results JSON file(s)")
    p.add_argument("--compare", action="store_true", help="Side-by-side comparison of multiple runs")
    args = p.parse_args()

    for path in args.results:
        if not Path(path).exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)

    if len(args.results) > 1 or args.compare:
        compare_tables(args.results)
    else:
        data = load_results(args.results[0])
        print_table(data, label=Path(args.results[0]).name)


if __name__ == "__main__":
    main()
