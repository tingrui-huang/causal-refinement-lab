"""
Aligned evaluation for Andes (strictly comparable SHD definition)

This script evaluates several methods using the SAME evaluator code:
  - CausalGraphEvaluator.evaluate(learned_edges)
which defines:
  skeleton_shd = undirected_fp + undirected_fn
  full_shd     = undirected_fp + undirected_fn + reversals

We report (at least):
  - skeleton_shd, full_shd
  - orientation_accuracy
  - learned_edges count
  - edge_precision/recall (undirected) for context

Methods included:
  1) ours_random_prior: read complete_edges.txt from experiment_llm_vs_random/andes/random_prior
  2) ours_llm_prior:    read complete_edges.txt from experiment_llm_vs_random/andes/llm_prior
  3) five_seed_stability: build edges by stability voting across seed_0..4 complete_adjacency.pt
       - agreement threshold configurable (default 0.9)
  4) edge_kfold_internal: read assembled_edges.txt from andes_internal_edge_cv output dir

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/andes_aligned_metrics.py
"""

from __future__ import annotations

import re
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


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
from modules.evaluator import CausalGraphEvaluator


# -----------------------
# Defaults (edit if needed)
# -----------------------
DATASET = "andes"
AGREEMENT_THRESHOLD = 0.90

RUN_RANDOM_PRIOR = NSR_DIR / "results" / "experiment_llm_vs_random" / "andes" / "random_prior"
RUN_LLM_PRIOR = NSR_DIR / "results" / "experiment_llm_vs_random" / "andes" / "llm_prior"

SEED_RUNS_DIR = NSR_DIR / "results" / "stability_runs" / "andes"
SEED_ADJ_NAME = "complete_adjacency.pt"
N_SEEDS = 5  # seed_0..seed_4

# Latest edge-CV output (you can change to another directory if needed)
EDGE_CV_DIR = REPO_ROOT / "results" / "andes_internal_edge_cv" / "andes_20260120_020930"


def parse_edges_txt(path: Path) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("=") or "Learned Causal Edges" in line:
            continue
        # Handles both formats:
        #  - "A -> B (strength: ...)"
        #  - "A -> B"
        if "->" in line:
            left, right = line.split("->", 1)
            src = left.strip()
            dst = right.split("(")[0].strip()
            if src and dst:
                edges.add((src, dst))
    return edges


def load_seed_adjacencies() -> List[Path]:
    paths = []
    for i in range(N_SEEDS):
        p = SEED_RUNS_DIR / f"seed_{i}" / SEED_ADJ_NAME
        if p.exists():
            paths.append(p)
    if len(paths) < 2:
        raise FileNotFoundError(f"Need >=2 seed adjacencies under {SEED_RUNS_DIR}, found {len(paths)}")
    return paths


def block_mean_strength(A: torch.Tensor, idx_a: Sequence[int], idx_b: Sequence[int]) -> float:
    return float(A[idx_a][:, idx_b].mean().item())


def stability_edges(
    adjacency_paths: List[Path],
    var_structure: Dict,
    skeleton_pairs: Set[Tuple[str, str]],
    *,
    agreement_threshold: float,
) -> Set[Tuple[str, str]]:
    vt = var_structure["var_to_states"]

    # votes per pair: list of 1/0 where 1 means a->b (a<b)
    votes: Dict[Tuple[str, str], List[int]] = {p: [] for p in skeleton_pairs}

    for ap in adjacency_paths:
        A = torch.load(ap, map_location="cpu")
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A)
        for a, b in skeleton_pairs:
            if a not in vt or b not in vt:
                continue
            ia = vt[a]
            ib = vt[b]
            d = block_mean_strength(A, ia, ib) - block_mean_strength(A, ib, ia)
            votes[(a, b)].append(1 if d >= 0 else 0)

    learned: Set[Tuple[str, str]] = set()
    for a, b in skeleton_pairs:
        vs = votes.get((a, b), [])
        if len(vs) < 2:
            continue
        vs_arr = np.asarray(vs, dtype=np.int32)
        maj = 1 if vs_arr.mean() >= 0.5 else 0
        agree = float(np.mean(vs_arr == maj))
        if agree < agreement_threshold:
            continue
        if maj == 1:
            learned.add((a, b))
        else:
            learned.add((b, a))
    return learned


def load_fci_skeleton_pairs(dataset: str) -> Set[Tuple[str, str]]:
    # Use the latest FCI CSV as skeleton definition
    fci_csv = uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / dataset)
    if not fci_csv:
        raise FileNotFoundError(f"No edges_FCI_*.csv found under {uconfig.FCI_OUTPUT_DIR / dataset}")
    import pandas as pd

    df = pd.read_csv(fci_csv)
    cols = {c.lower(): c for c in df.columns}
    src = cols.get("source", df.columns[0])
    dst = cols.get("target", df.columns[1] if len(df.columns) > 1 else df.columns[0])
    pairs: Set[Tuple[str, str]] = set()
    for a, b in zip(df[src].astype(str).tolist(), df[dst].astype(str).tolist()):
        x, y = (a, b) if a < b else (b, a)
        pairs.add((x, y))
    return pairs


def evaluate_named(evaluator: CausalGraphEvaluator, name: str, edges: Set[Tuple[str, str]]) -> Dict:
    m = evaluator.evaluate(edges)
    # Attach name + edges count
    m = dict(m)
    m["name"] = name
    m["learned_edges_count"] = len(edges)
    return m


def main():
    ds_cfg = uconfig.DATASET_CONFIGS[DATASET]
    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "edge_list")

    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()
    evaluator = CausalGraphEvaluator(str(gt_path), var_structure, ground_truth_type=gt_type)

    results: List[Dict] = []

    # 1) ours random prior
    edges_random = parse_edges_txt(RUN_RANDOM_PRIOR / "complete_edges.txt")
    results.append(evaluate_named(evaluator, "ours_random_prior", edges_random))

    # 2) ours llm prior
    edges_llm = parse_edges_txt(RUN_LLM_PRIOR / "complete_edges.txt")
    results.append(evaluate_named(evaluator, "ours_llm_prior", edges_llm))

    # 3) 5-seed stability subset (agreement threshold)
    skel_pairs = load_fci_skeleton_pairs(DATASET)
    seed_adj = load_seed_adjacencies()
    edges_stable = stability_edges(seed_adj, var_structure, skel_pairs, agreement_threshold=AGREEMENT_THRESHOLD)
    results.append(evaluate_named(evaluator, f"stability_{AGREEMENT_THRESHOLD:.2f}", edges_stable))

    # 4) edge-kfold internal assembled DAG
    edges_cv = parse_edges_txt(EDGE_CV_DIR / "assembled_edges.txt")
    results.append(evaluate_named(evaluator, "edge_kfold_internal", edges_cv))

    # Print compact aligned table
    print("\n" + "=" * 100)
    print("ALIGNED METRICS (same evaluator SHD definition)")
    print("=" * 100)
    print("name                 learned  skeleton_shd  full_shd  orient_acc  edge_f1  directed_f1")
    for r in results:
        print(
            f"{r['name']:<20s} {r['learned_edges_count']:>7d} "
            f"{int(r.get('skeleton_shd', -1)):>11d} {int(r.get('full_shd', -1)):>8d} "
            f"{float(r.get('orientation_accuracy', 0.0)):.4f}   "
            f"{float(r.get('edge_f1', 0.0)):.4f}    {float(r.get('directed_f1', 0.0)):.4f}"
        )

    out_dir = REPO_ROOT / "results" / "aligned_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"andes_aligned_metrics_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    import time
    main()

