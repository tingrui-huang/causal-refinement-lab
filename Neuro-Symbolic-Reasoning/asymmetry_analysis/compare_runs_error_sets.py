"""
Compare Phase1 error sets between two runs (e.g., random_prior vs llm_prior).

Given per_pair_{agg}.csv from Phase1 for two runs, compare:
  - wrong GT-true edges (directed predicted edges that are incorrect)
  - correct GT-true edges
  - fixed_by_B: wrong in A, correct in B
  - regressed_in_B: correct in A, wrong in B

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/compare_runs_error_sets.py ^
    --run_a results/phase1_error_diagnosis/andes/random_prior ^
    --run_b results/phase1_error_diagnosis/andes/llm_prior ^
    --aggs mean nuc --topk 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class RunSets:
    correct_pairs: Set[Tuple[str, str]]  # undirected pair key (min(var_a,var_b), max(...))
    wrong_pairs: Set[Tuple[str, str]]
    margin_by_pair: Dict[Tuple[str, str], float]
    pred_dir_by_pair: Dict[Tuple[str, str], Tuple[str, str]]
    gt_dir_by_pair: Dict[Tuple[str, str], Tuple[str, str]]


def load_run(csv_path: Path) -> RunSets:
    df = pd.read_csv(csv_path)
    df = df[(df["edge_type"] == "gt_true_edge") & (df["is_correct"].astype(str) != "")]

    correct_pairs: Set[Tuple[str, str]] = set()
    wrong_pairs: Set[Tuple[str, str]] = set()
    margin_by_pair: Dict[Tuple[str, str], float] = {}
    pred_dir_by_pair: Dict[Tuple[str, str], Tuple[str, str]] = {}
    gt_dir_by_pair: Dict[Tuple[str, str], Tuple[str, str]] = {}

    for _, r in df.iterrows():
        a = str(r["var_a"])
        b = str(r["var_b"])
        pair = (a, b) if a < b else (b, a)
        pred = (str(r["pred_src"]), str(r["pred_dst"]))
        gt = (str(r["gt_src"]), str(r["gt_dst"])) if str(r["gt_src"]) and str(r["gt_dst"]) else ("", "")

        margin_by_pair[pair] = float(r["margin"])
        pred_dir_by_pair[pair] = pred
        gt_dir_by_pair[pair] = gt

        if int(r["is_correct"]) == 1:
            correct_pairs.add(pair)
        else:
            wrong_pairs.add(pair)

    return RunSets(
        correct_pairs=correct_pairs,
        wrong_pairs=wrong_pairs,
        margin_by_pair=margin_by_pair,
        pred_dir_by_pair=pred_dir_by_pair,
        gt_dir_by_pair=gt_dir_by_pair,
    )


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def top_k(keys: Iterable[Tuple[str, str]], margin: Dict[Tuple[str, str], float], k: int) -> List[Tuple[float, Tuple[str, str]]]:
    items = [(margin.get(e, float("nan")), e) for e in keys]
    # Put NaNs at end
    items.sort(key=lambda x: (x[0] != x[0], -x[0] if x[0] == x[0] else 0.0))
    return items[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", type=str, required=True, help="Directory containing per_pair_{agg}.csv (baseline)")
    ap.add_argument("--run_b", type=str, required=True, help="Directory containing per_pair_{agg}.csv (comparison)")
    ap.add_argument("--aggs", nargs="+", default=["mean", "nuc"], choices=["mean", "nuc"])
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    topk = int(args.topk)

    print("=" * 80)
    print("COMPARE RUNS: ERROR SETS (Phase1)")
    print("=" * 80)
    print(f"run_a: {run_a}")
    print(f"run_b: {run_b}")

    for agg in args.aggs:
        csv_a = run_a / f"per_pair_{agg}.csv"
        csv_b = run_b / f"per_pair_{agg}.csv"
        if not csv_a.exists():
            raise FileNotFoundError(csv_a)
        if not csv_b.exists():
            raise FileNotFoundError(csv_b)

        A = load_run(csv_a)
        B = load_run(csv_b)

        wrongA = A.wrong_pairs
        wrongB = B.wrong_pairs
        correctA = A.correct_pairs
        correctB = B.correct_pairs

        fixed = wrongA & correctB
        regressed = correctA & wrongB

        print("\n" + "-" * 80)
        print(f"[agg={agg}]")
        print(f"  GT-true pairs evaluated: {len(wrongA | correctA)}")
        print(f"  wrong(A): {len(wrongA)}   wrong(B): {len(wrongB)}   delta_wrong(B-A): {len(wrongB) - len(wrongA):+d}")
        print(f"  jaccard(wrongA, wrongB): {jaccard(wrongA, wrongB):.4f}")
        print(f"  fixed_by_B (wrong->correct): {len(fixed)}")
        print(f"  regressed_in_B (correct->wrong): {len(regressed)}")

        if topk > 0:
            print(f"\n  Top {topk} fixed_by_B edges (sorted by A.margin):")
            for m, pair in top_k(fixed, A.margin_by_pair, k=topk):
                pred_a = A.pred_dir_by_pair.get(pair, ("", ""))
                pred_b = B.pred_dir_by_pair.get(pair, ("", ""))
                gt = A.gt_dir_by_pair.get(pair, ("", ""))
                print(
                    f"    A.margin={m:.6f}  pair={pair[0]}--{pair[1]}  "
                    f"A.pred={pred_a[0]}->{pred_a[1]}  B.pred={pred_b[0]}->{pred_b[1]}  gt={gt[0]}->{gt[1]}"
                )

            print(f"\n  Top {topk} regressed_in_B edges (sorted by B.margin):")
            for m, pair in top_k(regressed, B.margin_by_pair, k=topk):
                pred_a = A.pred_dir_by_pair.get(pair, ("", ""))
                pred_b = B.pred_dir_by_pair.get(pair, ("", ""))
                gt = B.gt_dir_by_pair.get(pair, ("", ""))
                print(
                    f"    B.margin={m:.6f}  pair={pair[0]}--{pair[1]}  "
                    f"A.pred={pred_a[0]}->{pred_a[1]}  B.pred={pred_b[0]}->{pred_b[1]}  gt={gt[0]}->{gt[1]}"
                )


if __name__ == "__main__":
    main()

