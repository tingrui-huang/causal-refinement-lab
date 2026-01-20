"""
Compare error sets between per_pair_mean.csv and per_pair_nuc.csv

Question:
  "Loss 犯错的边，和 Rank 犯错的边，是不是同一批？互补吗？"

In our Phase1 outputs:
  - per_pair_mean.csv: uses block strength aggregation S(i->j) = mean(block_ij)
  - per_pair_nuc.csv:  uses block strength aggregation S(i->j) = nuclear_norm(block_ij)

Both CSVs contain per FCI-skeleton pair:
  - predicted direction (pred_src->pred_dst)
  - GT direction (gt_src->gt_dst) if GT has an oriented edge
  - is_correct (1/0) for GT-true edges

This script compares the sets of "wrong GT-true edges" under the two aggregations.

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/compare_error_sets_mean_vs_nuc.py ^
    --mean_csv results/phase1_error_diagnosis/andes/random_prior/per_pair_mean.csv ^
    --nuc_csv  results/phase1_error_diagnosis/andes/random_prior/per_pair_nuc.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class ErrorSet:
    wrong: Set[Tuple[str, str]]          # directed (src, dst)
    correct: Set[Tuple[str, str]]        # directed (src, dst)
    margins: Dict[Tuple[str, str], float]


def load_error_set(csv_path: Path) -> ErrorSet:
    df = pd.read_csv(csv_path)

    # Keep only GT-true edges where correctness is defined
    df = df[(df["edge_type"] == "gt_true_edge") & (df["is_correct"].astype(str) != "")]

    wrong: Set[Tuple[str, str]] = set()
    correct: Set[Tuple[str, str]] = set()
    margins: Dict[Tuple[str, str], float] = {}

    for _, r in df.iterrows():
        pred = (str(r["pred_src"]), str(r["pred_dst"]))
        is_correct = int(r["is_correct"])
        margin = float(r["margin"])
        margins[pred] = margin
        if is_correct == 1:
            correct.add(pred)
        else:
            wrong.add(pred)

    return ErrorSet(wrong=wrong, correct=correct, margins=margins)


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def top_k_by_margin(keys: Iterable[Tuple[str, str]], margins: Dict[Tuple[str, str], float], k: int = 20) -> List[Tuple[float, Tuple[str, str]]]:
    items = [(margins.get(edge, float("nan")), edge) for edge in keys]
    items.sort(key=lambda x: (float("-inf") if (x[0] != x[0]) else x[0]), reverse=True)  # NaN last-ish
    return items[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mean_csv", type=str, required=True)
    p.add_argument("--nuc_csv", type=str, required=True)
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()

    mean_csv = Path(args.mean_csv)
    nuc_csv = Path(args.nuc_csv)
    if not mean_csv.exists():
        raise FileNotFoundError(mean_csv)
    if not nuc_csv.exists():
        raise FileNotFoundError(nuc_csv)

    mean_set = load_error_set(mean_csv)
    nuc_set = load_error_set(nuc_csv)

    wrong_mean = mean_set.wrong
    wrong_nuc = nuc_set.wrong

    inter = wrong_mean & wrong_nuc
    only_mean = wrong_mean - wrong_nuc
    only_nuc = wrong_nuc - wrong_mean
    union = wrong_mean | wrong_nuc

    # Confusion counts (on shared universe of predictions)
    # Note: we compare by predicted direction edges; since both CSVs are for the same skeleton pairs,
    # sizes should be comparable.
    print("=" * 80)
    print("COMPARE WRONG-EDGE SETS (MEAN vs NUC)")
    print("=" * 80)
    print(f"mean_csv: {mean_csv}")
    print(f"nuc_csv:  {nuc_csv}")
    print()
    print("[Wrong GT-true edges]")
    print(f"  wrong_mean: {len(wrong_mean)}")
    print(f"  wrong_nuc:  {len(wrong_nuc)}")
    print(f"  overlap:    {len(inter)}")
    print(f"  only_mean:  {len(only_mean)}")
    print(f"  only_nuc:   {len(only_nuc)}")
    print(f"  union:      {len(union)}")
    print(f"  jaccard:    {jaccard(wrong_mean, wrong_nuc):.4f}")

    # Complementarity score: fraction of union that is XOR
    xor_frac = (len(only_mean) + len(only_nuc)) / len(union) if union else 0.0
    print(f"  xor_frac (complementarity): {xor_frac:.4f}")

    # List top-k by margin for each category
    topk = int(args.topk)
    if topk > 0:
        print()
        print(f"Top {topk} overlap-wrong edges by margin (mean):")
        for m, (s, t) in top_k_by_margin(inter, mean_set.margins, k=topk):
            print(f"  margin={m:.6f}  {s}->{t}")

        print()
        print(f"Top {topk} only-mean-wrong edges by margin (mean):")
        for m, (s, t) in top_k_by_margin(only_mean, mean_set.margins, k=topk):
            print(f"  margin={m:.6f}  {s}->{t}")

        print()
        print(f"Top {topk} only-nuc-wrong edges by margin (nuc):")
        for m, (s, t) in top_k_by_margin(only_nuc, nuc_set.margins, k=topk):
            print(f"  margin={m:.6f}  {s}->{t}")


if __name__ == "__main__":
    main()

