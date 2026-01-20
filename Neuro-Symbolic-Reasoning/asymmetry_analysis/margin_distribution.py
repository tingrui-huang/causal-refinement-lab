"""
Micro-asymmetry diagnosis: margin distribution of direction strengths

For each unordered variable pair (i, j), compute direction "strengths":
  S_{i->j}, S_{j->i}
and margin:
  Delta_{ij} = |S_{i->j} - S_{j->i}|

This is intended as a LOW-cost diagnostic to assess whether the trained model's
orientation signal is strong/weak on a dataset (e.g., Andes).

Inputs:
  - adjacency.pt (or complete_adjacency.pt): learned state-level adjacency in [0,1]
  - metadata.json: defines variable->state indices (block structure)
Optionally:
  - FCI skeleton CSV: to restrict to candidate pairs in the skeleton (edge_set=fci_skeleton)

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/margin_distribution.py --dataset andes

Key options:
  --run_dir <dir>: manually pick a run dir containing adjacency.pt
  --edge_set all | fci_skeleton
  --agg mean | max | fro | nuc
"""

from __future__ import annotations

import argparse
import csv
import re
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


@dataclass(frozen=True)
class RunInputs:
    dataset: str
    adjacency_path: Path
    run_dir: Path
    metadata_path: Path
    data_path: Path
    fci_skeleton_path: Optional[Path]


def _iter_adjacency_candidates(run_dir: Path) -> Iterable[Path]:
    # Prefer adjacency.pt (used in many per-dataset runs), otherwise complete_adjacency.pt
    for name in ["adjacency.pt", "complete_adjacency.pt"]:
        p = run_dir / name
        if p.exists():
            yield p


def _find_latest_run_dir(dataset: str) -> Path:
    """
    Search typical result locations and pick the directory containing adjacency
    with the most recent modification time.
    """
    candidates: List[Tuple[float, Path]] = []

    # Common pattern: Neuro-Symbolic-Reasoning/results/<dataset>/<run_id>/adjacency.pt
    base = NSR_DIR / "results" / dataset
    if base.exists():
        for p in base.rglob("adjacency.pt"):
            candidates.append((p.stat().st_mtime, p.parent))
        for p in base.rglob("complete_adjacency.pt"):
            candidates.append((p.stat().st_mtime, p.parent))

    # Also check experiment outputs
    base2 = NSR_DIR / "results" / "experiment_llm_vs_random" / dataset
    if base2.exists():
        for p in base2.rglob("adjacency.pt"):
            candidates.append((p.stat().st_mtime, p.parent))
        for p in base2.rglob("complete_adjacency.pt"):
            candidates.append((p.stat().st_mtime, p.parent))

    if not candidates:
        raise FileNotFoundError(f"No adjacency.pt found under {base} or {base2}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_fci_skeleton_path_from_eval_txt(eval_txt: str) -> Optional[Path]:
    # Example line:
    # fci_skeleton_path        : D:\...\edges_FCI_....csv
    m = re.search(r"fci_skeleton_path\s*:\s*(.+)", eval_txt)
    if not m:
        return None
    raw = m.group(1).strip()
    p = Path(raw)
    return p if p.exists() else None


def _load_fci_skeleton_pairs(fci_csv_path: Path) -> Set[Tuple[str, str]]:
    """
    Return undirected variable pairs from FCI CSV (treat as skeleton).
    Supports the common format used in this repo: columns typically include
    Source/Target (names vary), so we fall back to scanning strings containing two vars.
    """
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
        # Last-resort: try first two columns
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


def resolve_inputs(dataset: str, run_dir: Optional[Path]) -> RunInputs:
    if dataset not in uconfig.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'")
    ds_cfg = uconfig.DATASET_CONFIGS[dataset]

    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])

    if run_dir is None:
        run_dir = _find_latest_run_dir(dataset)
    run_dir = Path(run_dir)

    adjacency_path = None
    for p in _iter_adjacency_candidates(run_dir):
        adjacency_path = p
        break
    if adjacency_path is None:
        raise FileNotFoundError(f"No adjacency.pt or complete_adjacency.pt found in {run_dir}")

    # If evaluation_results.txt exists, try to auto-detect the FCI skeleton path
    fci_skeleton_path = None
    eval_txt_path = run_dir / "evaluation_results.txt"
    if eval_txt_path.exists():
        try:
            fci_skeleton_path = _parse_fci_skeleton_path_from_eval_txt(eval_txt_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            fci_skeleton_path = None

    return RunInputs(
        dataset=dataset,
        adjacency_path=adjacency_path,
        run_dir=run_dir,
        metadata_path=metadata_path,
        data_path=data_path,
        fci_skeleton_path=fci_skeleton_path,
    )


def block_strength(block: torch.Tensor, agg: str) -> float:
    agg = agg.lower().strip()
    if agg == "mean":
        return float(block.mean().item())
    if agg == "max":
        return float(block.max().item())
    if agg == "fro":
        return float(torch.norm(block, p="fro").item())
    if agg == "nuc":
        return float(torch.norm(block, p="nuc").item())
    raise ValueError(f"Unknown agg='{agg}'. Use mean|max|fro|nuc")


def compute_margins(
    adjacency: torch.Tensor,
    var_structure: Dict,
    *,
    agg: str,
    edge_set: str,
    fci_pairs: Optional[Set[Tuple[str, str]]],
) -> List[Dict]:
    var_names: List[str] = list(var_structure["variable_names"])

    rows: List[Dict] = []
    edge_set = edge_set.lower().strip()
    if edge_set not in {"all", "fci_skeleton"}:
        raise ValueError("edge_set must be 'all' or 'fci_skeleton'")
    if edge_set == "fci_skeleton" and not fci_pairs:
        raise ValueError("edge_set=fci_skeleton requires a valid fci_skeleton_path (or pass --fci_skeleton_path)")

    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            a = var_names[i]
            b = var_names[j]

            if edge_set == "fci_skeleton":
                x, y = (a, b) if a < b else (b, a)
                if (x, y) not in fci_pairs:
                    continue

            idx_a = var_structure["var_to_states"][a]
            idx_b = var_structure["var_to_states"][b]
            block_ab = adjacency[idx_a][:, idx_b]
            block_ba = adjacency[idx_b][:, idx_a]

            s_ab = block_strength(block_ab, agg)
            s_ba = block_strength(block_ba, agg)
            margin = abs(s_ab - s_ba)
            direction = f"{a}->{b}" if s_ab >= s_ba else f"{b}->{a}"

            rows.append(
                {
                    "var_a": a,
                    "var_b": b,
                    "k_a": len(idx_a),
                    "k_b": len(idx_b),
                    "s_a_to_b": s_ab,
                    "s_b_to_a": s_ba,
                    "margin": margin,
                    "pred_direction": direction,
                }
            )

    return rows


def summarize_margins(margins: np.ndarray) -> str:
    qs = np.quantile(margins, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    frac_small = float((margins < 1e-3).mean())
    return (
        f"n_pairs={margins.size}\n"
        f"mean={margins.mean():.6f}\n"
        f"median={np.median(margins):.6f}\n"
        f"frac(margin<1e-3)={frac_small:.3f}\n"
        f"quantiles [0,10,25,50,75,90,100]%: {', '.join(f'{q:.6f}' for q in qs)}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="andes")
    parser.add_argument("--run_dir", type=str, default=None, help="Run dir containing adjacency.pt; otherwise auto-pick latest")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max", "fro", "nuc"])
    parser.add_argument("--edge_set", type=str, default="all", choices=["all", "fci_skeleton"])
    parser.add_argument("--fci_skeleton_path", type=str, default=None, help="Optional override for FCI skeleton CSV path")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional output directory; default is results/margin_distribution/<dataset>/")
    args = parser.parse_args()

    inputs = resolve_inputs(args.dataset, Path(args.run_dir) if args.run_dir else None)
    # FCI skeleton path resolution (priority order):
    # 1) CLI override
    # 2) parsed from run_dir/evaluation_results.txt
    # 3) auto-detect latest edges_FCI_*.csv from refactored/output/<dataset> via unified config
    fci_path = Path(args.fci_skeleton_path) if args.fci_skeleton_path else inputs.fci_skeleton_path
    if fci_path is None and args.edge_set == "fci_skeleton":
        try:
            auto = uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / args.dataset)
            if auto:
                fci_path = Path(auto)
        except Exception:
            fci_path = None

    print("\n" + "=" * 80)
    print("MARGIN DISTRIBUTION DIAGNOSTIC")
    print("=" * 80)
    print(f"Dataset:       {inputs.dataset}")
    print(f"Run dir:       {inputs.run_dir}")
    print(f"Adjacency:     {inputs.adjacency_path}")
    print(f"Metadata:      {inputs.metadata_path}")
    print(f"agg:           {args.agg}")
    print(f"edge_set:      {args.edge_set}")
    print(f"FCI skeleton:  {str(fci_path) if fci_path else '(none)'}")

    # Load var_structure (metadata-only; don't call load_data)
    loader = CausalDataLoader(data_path=str(inputs.data_path), metadata_path=str(inputs.metadata_path))
    var_structure = loader.get_variable_structure()

    # Load adjacency
    adjacency = torch.load(inputs.adjacency_path, map_location="cpu")
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.tensor(adjacency)

    fci_pairs = None
    if args.edge_set == "fci_skeleton":
        if not fci_path or not fci_path.exists():
            raise FileNotFoundError(
                "edge_set=fci_skeleton but fci_skeleton_path not found.\n"
                "Fix: pass --fci_skeleton_path <path/to/edges_FCI_*.csv>, or run the FCI step to generate it."
            )
        fci_pairs = _load_fci_skeleton_pairs(fci_path)

    rows = compute_margins(adjacency, var_structure, agg=args.agg, edge_set=args.edge_set, fci_pairs=fci_pairs)
    if not rows:
        raise RuntimeError("No pairs selected (check edge_set and skeleton path).")

    margins = np.asarray([r["margin"] for r in rows], dtype=np.float64)

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT / "results" / "margin_distribution" / inputs.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / f"margins_{args.edge_set}_{args.agg}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save summary
    summary = (
        f"MARGIN DISTRIBUTION SUMMARY\n"
        f"dataset={inputs.dataset}\n"
        f"run_dir={inputs.run_dir}\n"
        f"adjacency_path={inputs.adjacency_path}\n"
        f"agg={args.agg}\n"
        f"edge_set={args.edge_set}\n"
        f"fci_skeleton_path={str(fci_path) if fci_path else ''}\n\n"
        + summarize_margins(margins)
    )
    summary_path = out_dir / f"summary_{args.edge_set}_{args.agg}.txt"
    summary_path.write_text(summary, encoding="utf-8")

    # Plots
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(margins, bins=40, alpha=0.85)
    ax.set_title(f"Margin distribution ({inputs.dataset})\nedge_set={args.edge_set}, agg={args.agg}")
    ax.set_xlabel("|S(i->j) - S(j->i)|")
    ax.set_ylabel("count")
    fig.tight_layout()
    hist_path = out_dir / f"margin_hist_{args.edge_set}_{args.agg}.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.boxplot([margins], tick_labels=["margin"], showmeans=True)
    ax.set_title(f"Margin boxplot ({inputs.dataset})\nedge_set={args.edge_set}, agg={args.agg}")
    fig.tight_layout()
    box_path = out_dir / f"margin_box_{args.edge_set}_{args.agg}.png"
    fig.savefig(box_path, dpi=200)
    plt.close(fig)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {hist_path}")
    print(f"  {box_path}")


if __name__ == "__main__":
    main()

