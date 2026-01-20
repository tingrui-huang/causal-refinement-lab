"""
Stability-based orientation (no LLM required)

Idea:
  Train the same model multiple times with different random seeds.
  For each FCI-skeleton pair {i,j}, each run votes a direction (i->j or j->i).
  Use agreement across runs as a "stability" score.

  - If a pair's direction is unstable (votes split), leave it UNORIENTED.
  - If stable (agreement >= threshold), output oriented direction.

This is a classic stability-selection trick:
  it reduces "confident-but-wrong" orientations and gives you calibrated subsets.

Inputs:
  - A list of adjacency files (adjacency.pt or complete_adjacency.pt) from multiple runs/seeds
  - metadata.json for var->states block structure
  - FCI skeleton edges_FCI_*.csv (to define candidate pairs)
  - GT edge list (optional, for evaluation)

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/stability_orientation.py ^
    --dataset andes ^
    --adjacency_paths "path1.pt" "path2.pt" "path3.pt" ^
    --agreement 0.8

Or auto-collect from a directory:
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/stability_orientation.py ^
    --dataset andes ^
    --search_dir Neuro-Symbolic-Reasoning/results/andes ^
    --agreement 0.8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
NSR_DIR = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NSR_DIR) not in sys.path:
    sys.path.insert(0, str(NSR_DIR))

# Robustly import unified config from repo root (avoid accidentally importing Neuro-Symbolic-Reasoning/config.py)
import importlib.util

_spec = importlib.util.spec_from_file_location("unified_config", REPO_ROOT / "config.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to import unified config.py via importlib")
uconfig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uconfig)

from modules.data_loader import CausalDataLoader
from modules.ground_truth_loader import GroundTruthLoader


def auto_detect_latest_fci_csv(dataset: str) -> Path:
    auto = uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / dataset)
    if not auto:
        raise FileNotFoundError(f"No edges_FCI_*.csv found under {uconfig.FCI_OUTPUT_DIR / dataset}")
    return Path(auto)


def load_fci_pairs(fci_csv_path: Path) -> Set[Tuple[str, str]]:
    import pandas as pd

    df = pd.read_csv(fci_csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    src_col = None
    dst_col = None
    for cand in ["source", "from", "var1", "x", "node1"]:
        if cand in cols_lower:
            src_col = cols_lower[cand]
            break
    for cand in ["target", "to", "var2", "y", "node2"]:
        if cand in cols_lower:
            dst_col = cols_lower[cand]
            break
    if src_col is None or dst_col is None:
        if len(df.columns) >= 2:
            src_col, dst_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(f"Unexpected FCI CSV format: {fci_csv_path} columns={list(df.columns)}")

    pairs: Set[Tuple[str, str]] = set()
    for a, b in zip(df[src_col].astype(str).tolist(), df[dst_col].astype(str).tolist()):
        if not a or not b:
            continue
        x, y = (a, b) if a < b else (b, a)
        pairs.add((x, y))
    return pairs


def block_strength_mean(adjacency: torch.Tensor, idx_a: Sequence[int], idx_b: Sequence[int]) -> float:
    return float(adjacency[idx_a][:, idx_b].mean().item())


def collect_adjacency_files(search_dir: Path) -> List[Path]:
    candidates: List[Tuple[float, Path]] = []
    for name in ["adjacency.pt", "complete_adjacency.pt"]:
        for p in search_dir.rglob(name):
            try:
                candidates.append((p.stat().st_mtime, p))
            except Exception:
                continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def vote_direction_for_pairs(
    adjacency_path: Path,
    var_structure: Dict,
    fci_pairs: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """
    Returns:
      pair -> (dir, margin)
      where dir is 'a->b' or 'b->a' using lex-ordered pair (a<b).
      margin is |S(a->b) - S(b->a)|.
    """
    adjacency = torch.load(adjacency_path, map_location="cpu")
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.tensor(adjacency)

    vt = var_structure["var_to_states"]
    votes: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for a, b in fci_pairs:
        if a not in vt or b not in vt:
            continue
        idx_a = vt[a]
        idx_b = vt[b]
        s_ab = block_strength_mean(adjacency, idx_a, idx_b)
        s_ba = block_strength_mean(adjacency, idx_b, idx_a)
        if s_ab >= s_ba:
            direction = f"{a}->{b}"
        else:
            direction = f"{b}->{a}"
        votes[(a, b)] = (direction, abs(s_ab - s_ba))
    return votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="andes")
    ap.add_argument("--agreement", type=float, default=0.8, help="Single stability threshold (fraction of runs agreeing)")
    ap.add_argument(
        "--agreement_sweep",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of agreement thresholds to sweep (e.g., 0.6 0.7 0.8 0.9). If provided, overrides --agreement for reporting.",
    )
    ap.add_argument("--adjacency_paths", nargs="*", default=None, help="Explicit list of adjacency .pt files")
    ap.add_argument("--search_dir", type=str, default=None, help="If set, auto-collect adjacency files under this dir")
    ap.add_argument("--max_runs", type=int, default=10, help="Max number of runs to use from search_dir (most recent)")
    ap.add_argument("--fci_skeleton_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    dataset = args.dataset
    if dataset not in uconfig.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'")
    ds_cfg = uconfig.DATASET_CONFIGS[dataset]

    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "edge_list")

    # Resolve adjacency paths
    adjacency_paths: List[Path] = []
    if args.adjacency_paths:
        adjacency_paths = [Path(p) for p in args.adjacency_paths]
    elif args.search_dir:
        found = collect_adjacency_files(Path(args.search_dir))
        adjacency_paths = found[: max(1, int(args.max_runs))]
    else:
        raise ValueError("Provide either --adjacency_paths or --search_dir")

    adjacency_paths = [p for p in adjacency_paths if p.exists()]
    if len(adjacency_paths) < 2:
        raise ValueError(f"Need >=2 adjacency files for stability, got {len(adjacency_paths)}")

    # FCI skeleton
    fci_path = Path(args.fci_skeleton_path) if args.fci_skeleton_path else auto_detect_latest_fci_csv(dataset)
    fci_pairs = load_fci_pairs(fci_path)

    # Load var_structure (metadata-only; don't call load_data)
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()

    # GT edges for evaluation (optional)
    gt_edges = None
    if gt_path.exists():
        gt_loader = GroundTruthLoader(str(gt_path), ground_truth_type=gt_type)
        gt_edges = gt_loader.get_edges()

    # Vote per run
    per_run_votes: List[Dict[Tuple[str, str], Tuple[str, float]]] = []
    for p in adjacency_paths:
        per_run_votes.append(vote_direction_for_pairs(p, var_structure, fci_pairs))

    # Aggregate votes
    rows = []

    for pair in sorted(fci_pairs):
        # collect votes for this pair across runs
        dirs = []
        margins = []
        for v in per_run_votes:
            if pair in v:
                d, m = v[pair]
                dirs.append(d)
                margins.append(m)
        if not dirs:
            continue
        # majority direction
        vals, counts = np.unique(np.asarray(dirs), return_counts=True)
        best_idx = int(np.argmax(counts))
        maj_dir = str(vals[best_idx])
        agree = float(counts[best_idx] / len(dirs))
        mean_margin = float(np.mean(margins))
        std_margin = float(np.std(margins))

        # GT correctness if available and GT has this edge (either direction)
        gt_true = 0
        gt_correct = ""
        if gt_edges is not None:
            a, b = pair
            gt_has_ab = (a, b) in gt_edges
            gt_has_ba = (b, a) in gt_edges
            if gt_has_ab or gt_has_ba:
                gt_true = 1
                gt_dir = f"{a}->{b}" if gt_has_ab else f"{b}->{a}"
                gt_correct = int(maj_dir == gt_dir)
        rows.append(
            {
                "var_a": pair[0],
                "var_b": pair[1],
                "maj_dir": maj_dir,
                "agree": agree,
                "n_runs": len(dirs),
                "mean_margin": mean_margin,
                "std_margin": std_margin,
                "gt_true": gt_true,
                "gt_correct": gt_correct,
            }
        )

    # Output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT / "results" / "stability_orientation" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stability_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plots
    agrees = np.asarray([r["agree"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(agrees, bins=20, alpha=0.85)
    thresh_for_line = float(args.agreement_sweep[0]) if args.agreement_sweep else float(args.agreement)
    ax.axvline(thresh_for_line, color="black", linewidth=1)
    ax.set_title(f"Agreement histogram ({dataset})")
    ax.set_xlabel("agreement (fraction of runs voting majority)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "agreement_hist.png", dpi=200)
    plt.close(fig)

    # Sweep thresholds (coverage vs accuracy)
    thresholds = args.agreement_sweep if args.agreement_sweep else [float(args.agreement)]
    thresholds = [float(t) for t in thresholds]
    thresholds = sorted(set(thresholds))

    def compute_stats(th: float) -> Dict[str, float]:
        stable = [r for r in rows if float(r["agree"]) >= th]
        stable_oriented = len(stable)
        gt_true = [r for r in stable if int(r["gt_true"]) == 1 and str(r["gt_correct"]) != ""]
        acc = float(np.mean([int(r["gt_correct"]) for r in gt_true])) if gt_true else float("nan")
        coverage = stable_oriented / max(1, len(rows))
        gt_coverage = len(gt_true) / max(1, sum(1 for r in rows if int(r["gt_true"]) == 1 and str(r["gt_correct"]) != ""))
        return {
            "threshold": th,
            "stable_oriented_pairs": stable_oriented,
            "coverage_pairs": coverage,
            "stable_gt_true_pairs": len(gt_true),
            "coverage_gt_true_pairs": gt_coverage,
            "stable_orientation_accuracy": acc,
        }

    sweep = [compute_stats(th) for th in thresholds]

    sweep_csv = out_dir / "agreement_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    # Plot: coverage vs accuracy
    xs = [d["coverage_gt_true_pairs"] for d in sweep]
    ys = [d["stable_orientation_accuracy"] for d in sweep]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(xs, ys, marker="o")
    for d in sweep:
        ax.annotate(f"{d['threshold']:.2f}", (d["coverage_gt_true_pairs"], d["stable_orientation_accuracy"]), fontsize=8)
    ax.set_xlabel("coverage on GT-true skeleton pairs")
    ax.set_ylabel("orientation accuracy (stable subset)")
    ax.set_title(f"Stability sweep ({dataset})")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_vs_accuracy.png", dpi=200)
    plt.close(fig)

    # Summary
    summary_lines = []
    summary_lines.append("STABILITY ORIENTATION SUMMARY")
    summary_lines.append(f"dataset={dataset}")
    summary_lines.append(f"n_adjacency_files={len(adjacency_paths)}")
    summary_lines.append(f"fci_skeleton_path={fci_path}")
    summary_lines.append("")
    summary_lines.append(f"pairs_in_skeleton={len(fci_pairs)}")
    summary_lines.append(f"pairs_scored={len(rows)}")
    summary_lines.append(f"thresholds_swept={thresholds}")
    summary_lines.append("")
    summary_lines.append("Sweep results (threshold, stable_gt_true_pairs, coverage_gt_true, accuracy):")
    for d in sweep:
        summary_lines.append(
            f"  {d['threshold']:.2f}  n={d['stable_gt_true_pairs']:4d}  cov={d['coverage_gt_true_pairs']:.3f}  acc={d['stable_orientation_accuracy']:.4f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'agreement_hist.png'}")
    print(f"  {sweep_csv}")
    print(f"  {out_dir / 'coverage_vs_accuracy.png'}")
    print(f"  {out_dir / 'summary.txt'}")
    print("\nAdjacency files used:")
    for p in adjacency_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

