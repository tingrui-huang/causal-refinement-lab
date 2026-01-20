"""
Phase 1: Error diagnosis on existing training outputs (no new data)

Implements 3 diagnostics (on FCI skeleton pairs by default):
  1) margin vs correctness:
       margin_ij = |S(i->j) - S(j->i)|
       correctness = whether predicted direction matches GT (only for GT-true edges)
  2) edge-type grouping:
       - GT true edge (GT contains i->j or j->i)
       - skeleton false positive (skeleton has {i,j} but GT has neither direction)
  3) structural position grouping:
       group by node degree (GT degree and skeleton degree) and see where errors concentrate

Supports strength aggregation S(i->j) from state-level adjacency blocks:
  - mean (stable default)
  - nuc  (block nuclear norm sensitivity)

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/phase1_error_diagnosis.py ^
    --dataset andes ^
    --adjacency_path "Neuro-Symbolic-Reasoning/results/experiment_llm_vs_random/andes/random_prior/complete_adjacency.pt" ^
    --aggs mean nuc
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Inputs:
    dataset: str
    adjacency_path: Path
    run_dir: Path
    data_path: Path
    metadata_path: Path
    ground_truth_path: Path
    ground_truth_type: str
    fci_skeleton_path: Path


def _auto_fci_skeleton_csv(dataset: str) -> Path:
    auto = uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / dataset)
    if not auto:
        raise FileNotFoundError(f"No edges_FCI_*.csv found under {uconfig.FCI_OUTPUT_DIR / dataset}")
    return Path(auto)


def resolve_inputs(dataset: str, adjacency_path: Path, fci_skeleton_path: Optional[Path]) -> Inputs:
    if dataset not in uconfig.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'")
    ds_cfg = uconfig.DATASET_CONFIGS[dataset]

    adjacency_path = Path(adjacency_path)
    if not adjacency_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adjacency_path}")

    run_dir = adjacency_path.parent
    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "edge_list")

    skel_path = Path(fci_skeleton_path) if fci_skeleton_path else _auto_fci_skeleton_csv(dataset)
    if not skel_path.exists():
        raise FileNotFoundError(f"FCI skeleton CSV not found: {skel_path}")

    return Inputs(
        dataset=dataset,
        adjacency_path=adjacency_path,
        run_dir=run_dir,
        data_path=data_path,
        metadata_path=metadata_path,
        ground_truth_path=gt_path,
        ground_truth_type=gt_type,
        fci_skeleton_path=skel_path,
    )


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


def undirected_degree(pairs: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for a, b in pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def block_strength(block: torch.Tensor, agg: str) -> float:
    agg = agg.lower().strip()
    if agg == "mean":
        return float(block.mean().item())
    if agg == "nuc":
        return float(torch.norm(block, p="nuc").item())
    raise ValueError(f"Unsupported agg '{agg}'. Use mean or nuc.")


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation without scipy: corr(rank(x), rank(y))
    """
    if x.size == 0:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def compute_per_pair_table(
    adjacency: torch.Tensor,
    var_structure: Dict,
    *,
    fci_pairs: Set[Tuple[str, str]],
    gt_edges: Set[Tuple[str, str]],
    deg_gt: Dict[str, int],
    deg_skel: Dict[str, int],
    agg: str,
) -> List[Dict]:
    var_to_states = var_structure["var_to_states"]

    rows: List[Dict] = []
    for a, b in sorted(fci_pairs):
        if a not in var_to_states or b not in var_to_states:
            continue
        idx_a = var_to_states[a]
        idx_b = var_to_states[b]

        block_ab = adjacency[idx_a][:, idx_b]
        block_ba = adjacency[idx_b][:, idx_a]
        s_ab = block_strength(block_ab, agg)
        s_ba = block_strength(block_ba, agg)
        margin = abs(s_ab - s_ba)
        pred_dir = (a, b) if s_ab >= s_ba else (b, a)

        gt_has_ab = (a, b) in gt_edges
        gt_has_ba = (b, a) in gt_edges
        gt_true_edge = bool(gt_has_ab or gt_has_ba)

        edge_type = "gt_true_edge" if gt_true_edge else "skeleton_false_positive"

        gt_dir = None
        is_correct = None
        if gt_true_edge:
            if gt_has_ab and not gt_has_ba:
                gt_dir = (a, b)
            elif gt_has_ba and not gt_has_ab:
                gt_dir = (b, a)
            else:
                # Unexpected (cycle) -> mark ambiguous
                gt_dir = None
            if gt_dir is not None:
                is_correct = int(pred_dir == gt_dir)

        rows.append(
            {
                "var_a": a,
                "var_b": b,
                "k_a": len(idx_a),
                "k_b": len(idx_b),
                "s_a_to_b": s_ab,
                "s_b_to_a": s_ba,
                "margin": margin,
                "pred_src": pred_dir[0],
                "pred_dst": pred_dir[1],
                "edge_type": edge_type,
                "gt_src": (gt_dir[0] if gt_dir else ""),
                "gt_dst": (gt_dir[1] if gt_dir else ""),
                "is_correct": ("" if is_correct is None else int(is_correct)),
                "deg_gt_a": deg_gt.get(a, 0),
                "deg_gt_b": deg_gt.get(b, 0),
                "deg_skel_a": deg_skel.get(a, 0),
                "deg_skel_b": deg_skel.get(b, 0),
                "deg_gt_max": max(deg_gt.get(a, 0), deg_gt.get(b, 0)),
                "deg_skel_max": max(deg_skel.get(a, 0), deg_skel.get(b, 0)),
            }
        )

    return rows


def plot_margin_vs_correctness(rows: List[Dict], out_path: Path, title: str) -> Dict:
    # Only GT-true edges with defined correctness
    xs = []
    ys = []
    for r in rows:
        if r["edge_type"] != "gt_true_edge":
            continue
        if r["is_correct"] == "":
            continue
        xs.append(float(r["margin"]))
        ys.append(int(r["is_correct"]))

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.int32)
    if x.size == 0:
        return {"n": 0, "spearman": float("nan")}

    # Bin by margin quantiles (10 bins)
    q = np.quantile(x, np.linspace(0, 1, 11))
    # Ensure monotonic (handle many zeros)
    q = np.unique(q)
    if q.size < 3:
        # fall back: fixed bins
        q = np.asarray([x.min(), np.median(x), x.max()])

    bin_ids = np.digitize(x, q[1:-1], right=True)
    acc = []
    centers = []
    counts = []
    for b in range(int(bin_ids.min()), int(bin_ids.max()) + 1):
        m = bin_ids == b
        if not np.any(m):
            continue
        acc.append(float(y[m].mean()))
        centers.append(float(np.median(x[m])))
        counts.append(int(m.sum()))

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))
    ax1.plot(centers, acc, marker="o")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("margin (binned by quantiles; x=median margin in bin)")
    ax1.set_ylabel("accuracy (GT-true edges)")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    sp = spearman_corr(x, y.astype(np.float64))
    return {"n": int(x.size), "spearman": sp}


def plot_margin_by_edge_type(rows: List[Dict], out_path: Path, title: str) -> None:
    margins_true = [float(r["margin"]) for r in rows if r["edge_type"] == "gt_true_edge"]
    margins_fp = [float(r["margin"]) for r in rows if r["edge_type"] == "skeleton_false_positive"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.boxplot([margins_true, margins_fp], tick_labels=["GT true", "Skeleton FP"], showmeans=True)
    ax.set_ylabel("margin = |S(i->j) - S(j->i)|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_vs_degree(rows: List[Dict], out_path: Path, title: str, deg_key: str = "deg_gt_max") -> Dict:
    # Only GT-true edges with defined correctness
    deg = []
    err = []
    for r in rows:
        if r["edge_type"] != "gt_true_edge":
            continue
        if r["is_correct"] == "":
            continue
        deg.append(float(r[deg_key]))
        err.append(1.0 - float(r["is_correct"]))
    d = np.asarray(deg, dtype=np.float64)
    e = np.asarray(err, dtype=np.float64)
    if d.size == 0:
        return {"n": 0}

    # Bin degrees into 6 quantile bins
    q = np.quantile(d, np.linspace(0, 1, 7))
    q = np.unique(q)
    if q.size < 3:
        q = np.asarray([d.min(), np.median(d), d.max()])
    bin_ids = np.digitize(d, q[1:-1], right=True)

    rates = []
    centers = []
    counts = []
    for b in range(int(bin_ids.min()), int(bin_ids.max()) + 1):
        m = bin_ids == b
        if not np.any(m):
            continue
        rates.append(float(e[m].mean()))
        centers.append(float(np.median(d[m])))
        counts.append(int(m.sum()))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(centers, rates, marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{deg_key} (binned by quantiles; x=median degree in bin)")
    ax.set_ylabel("error rate (1-accuracy) on GT-true edges")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"n": int(d.size)}


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="andes")
    parser.add_argument("--adjacency_path", type=str, required=True)
    parser.add_argument("--fci_skeleton_path", type=str, default=None, help="Optional override for edges_FCI_*.csv")
    parser.add_argument("--aggs", nargs="+", default=["mean", "nuc"], choices=["mean", "nuc"])
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    inputs = resolve_inputs(
        dataset=args.dataset,
        adjacency_path=Path(args.adjacency_path),
        fci_skeleton_path=Path(args.fci_skeleton_path) if args.fci_skeleton_path else None,
    )

    # Load var_structure (metadata-only; don't call load_data)
    loader = CausalDataLoader(data_path=str(inputs.data_path), metadata_path=str(inputs.metadata_path))
    var_structure = loader.get_variable_structure()

    # Load adjacency
    adjacency = torch.load(inputs.adjacency_path, map_location="cpu")
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.tensor(adjacency)

    # Load GT
    gt_loader = GroundTruthLoader(str(inputs.ground_truth_path), ground_truth_type=inputs.ground_truth_type)
    gt_edges = gt_loader.get_edges()
    if not gt_edges:
        raise RuntimeError(f"Failed to load GT edges from {inputs.ground_truth_path} (type={inputs.ground_truth_type})")
    gt_undirected = {(a, b) if a < b else (b, a) for (a, b) in gt_edges}

    # Load skeleton pairs
    fci_pairs = load_fci_pairs(inputs.fci_skeleton_path)

    # Degrees
    deg_gt = undirected_degree(gt_undirected)
    deg_skel = undirected_degree(fci_pairs)

    # Output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT / "results" / "phase1_error_diagnosis" / inputs.dataset / inputs.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PHASE 1 ERROR DIAGNOSIS")
    print("=" * 80)
    print(f"dataset:           {inputs.dataset}")
    print(f"run_dir:           {inputs.run_dir}")
    print(f"adjacency_path:    {inputs.adjacency_path}")
    print(f"metadata_path:     {inputs.metadata_path}")
    print(f"gt_path:           {inputs.ground_truth_path} (type={inputs.ground_truth_type})")
    print(f"fci_skeleton_path: {inputs.fci_skeleton_path}")
    print(f"out_dir:           {out_dir}")
    print(f"aggs:              {args.aggs}")
    print(f"pairs in skeleton: {len(fci_pairs)}")

    for agg in args.aggs:
        rows = compute_per_pair_table(
            adjacency,
            var_structure,
            fci_pairs=fci_pairs,
            gt_edges=gt_edges,
            deg_gt=deg_gt,
            deg_skel=deg_skel,
            agg=agg,
        )
        if not rows:
            raise RuntimeError("No rows produced (variable name mismatch?)")

        csv_path = out_dir / f"per_pair_{agg}.csv"
        write_csv(rows, csv_path)

        # Plots
        info1 = plot_margin_vs_correctness(
            rows,
            out_dir / f"margin_vs_correctness_{agg}.png",
            title=f"Margin vs Correctness (agg={agg}) [{inputs.dataset}]",
        )
        plot_margin_by_edge_type(
            rows,
            out_dir / f"margin_by_edge_type_{agg}.png",
            title=f"Margin by Edge Type (agg={agg}) [{inputs.dataset}]",
        )
        plot_error_vs_degree(
            rows,
            out_dir / f"error_vs_degree_gt_{agg}.png",
            title=f"Error vs GT Degree (agg={agg}) [{inputs.dataset}]",
            deg_key="deg_gt_max",
        )
        plot_error_vs_degree(
            rows,
            out_dir / f"error_vs_degree_skeleton_{agg}.png",
            title=f"Error vs Skeleton Degree (agg={agg}) [{inputs.dataset}]",
            deg_key="deg_skel_max",
        )

        # Summaries
        n_true = sum(1 for r in rows if r["edge_type"] == "gt_true_edge")
        n_fp = sum(1 for r in rows if r["edge_type"] == "skeleton_false_positive")
        correct = [int(r["is_correct"]) for r in rows if r["edge_type"] == "gt_true_edge" and r["is_correct"] != ""]
        acc = float(np.mean(correct)) if correct else float("nan")

        margins_true = np.asarray([float(r["margin"]) for r in rows if r["edge_type"] == "gt_true_edge"], dtype=np.float64)
        margins_fp = np.asarray([float(r["margin"]) for r in rows if r["edge_type"] == "skeleton_false_positive"], dtype=np.float64)

        # Margin bucket accuracy (GT-true edges only)
        gt_rows = [r for r in rows if r["edge_type"] == "gt_true_edge" and r["is_correct"] != ""]
        gt_margins = np.asarray([float(r["margin"]) for r in gt_rows], dtype=np.float64)
        gt_correct = np.asarray([int(r["is_correct"]) for r in gt_rows], dtype=np.int32)

        margin_bucket_lines: List[str] = []
        if gt_margins.size > 0:
            q25, q50, q75 = np.quantile(gt_margins, [0.25, 0.5, 0.75])
            for name, lo, hi in [
                ("low (<=Q25)", -np.inf, q25),
                ("mid (Q25-Q75)", q25, q75),
                ("high (>=Q75)", q75, np.inf),
            ]:
                m = (gt_margins >= lo) & (gt_margins <= hi) if math.isfinite(hi) else (gt_margins >= lo)
                if m.sum() == 0:
                    continue
                margin_bucket_lines.append(
                    f"  {name:14s}: n={int(m.sum()):3d}, acc={float(gt_correct[m].mean()):.4f}, median_margin={float(np.median(gt_margins[m])):.6f}"
                )

        # Degree vs error (GT-true edges only)
        deg_gt_max = np.asarray([float(r["deg_gt_max"]) for r in gt_rows], dtype=np.float64)
        deg_skel_max = np.asarray([float(r["deg_skel_max"]) for r in gt_rows], dtype=np.float64)
        gt_error = 1.0 - gt_correct.astype(np.float64)
        sp_deg_gt = spearman_corr(deg_gt_max, gt_error) if deg_gt_max.size else float("nan")
        sp_deg_skel = spearman_corr(deg_skel_max, gt_error) if deg_skel_max.size else float("nan")

        # Top "confident wrong" edges (largest margin among incorrect GT edges)
        wrong_edges = [
            (float(r["margin"]), f"{r['pred_src']}->{r['pred_dst']}", f"{r['gt_src']}->{r['gt_dst']}")
            for r in rows
            if r["edge_type"] == "gt_true_edge" and r["is_correct"] != "" and int(r["is_correct"]) == 0
        ]
        wrong_edges.sort(key=lambda x: x[0], reverse=True)
        top_wrong = wrong_edges[:20]

        summary_lines = []
        summary_lines.append(f"PHASE 1 SUMMARY (agg={agg})")
        summary_lines.append(f"dataset={inputs.dataset}")
        summary_lines.append(f"run_dir={inputs.run_dir}")
        summary_lines.append(f"adjacency_path={inputs.adjacency_path}")
        summary_lines.append(f"fci_skeleton_path={inputs.fci_skeleton_path}")
        summary_lines.append("")
        summary_lines.append(f"skeleton_pairs_total={len(rows)}")
        summary_lines.append(f"skeleton_pairs_gt_true={n_true}")
        summary_lines.append(f"skeleton_pairs_fp={n_fp}")
        summary_lines.append(f"orientation_accuracy_on_gt_true={acc:.4f}")
        summary_lines.append(f"spearman(margin, correct)={info1.get('spearman', float('nan')):.4f} (n={info1.get('n', 0)})")
        summary_lines.append(f"spearman(deg_gt_max, error)={sp_deg_gt:.4f}")
        summary_lines.append(f"spearman(deg_skel_max, error)={sp_deg_skel:.4f}")
        summary_lines.append("")
        if margins_true.size:
            summary_lines.append(
                f"margin(GT-true): mean={margins_true.mean():.6f}, median={np.median(margins_true):.6f}"
            )
        if margins_fp.size:
            summary_lines.append(
                f"margin(Skeleton-FP): mean={margins_fp.mean():.6f}, median={np.median(margins_fp):.6f}"
            )
        if margin_bucket_lines:
            summary_lines.append("")
            summary_lines.append("Accuracy by margin bucket (GT-true edges):")
            summary_lines.extend(margin_bucket_lines)
        summary_lines.append("")
        summary_lines.append("Top confident-wrong GT edges (largest margin among wrong predictions):")
        for m, pred, gt in top_wrong:
            summary_lines.append(f"  margin={m:.6f}  pred={pred:>25s}  gt={gt}")
        (out_dir / f"summary_{agg}.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        print(f"\n[agg={agg}] saved:")
        print(f"  {csv_path}")
        print(f"  {out_dir / f'summary_{agg}.txt'}")


if __name__ == "__main__":
    main()

