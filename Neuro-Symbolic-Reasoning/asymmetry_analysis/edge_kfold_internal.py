"""
Dataset-internal edge-KFold scorer + greedy DAG assembly (dataset-generic).

This is the generalized version of the earlier Andes-only script.

Timed sections (written to outputs):
  - feature_compute_seconds: reading multi-seed adjacencies and building edge features
  - edge_kfold_train_seconds: KFold training + OOF scoring on GT-true skeleton edges
  - final_fit_and_score_seconds: final model fit + scoring all skeleton pairs
  - assembly_seconds: greedy DAG assembly with acyclicity checks
  - total_seconds: whole script runtime (this analysis only; excludes seed training time)

Run (repo root):
  1) Edit DEFAULT_* below
  2) python Neuro-Symbolic-Reasoning/asymmetry_analysis/edge_kfold_internal.py
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
from modules.evaluator import CausalGraphEvaluator
from modules.metrics import compute_unresolved_ratio
from modules.prior_builder import PriorBuilder


# --------------------------------------------------------------------------------------
# DEFAULTS (edit like config.py style)
# --------------------------------------------------------------------------------------
DEFAULT_DATASET = "andes"  # set to "hailfinder" or "win95pts"
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_SYMMETRY_THRESHOLD = None  # if None: andes=0.08 else 0.1

# Multi-seed adjacency inputs from run_multi_seed_random_prior.py
DEFAULT_SEED_RUNS_DIR = NSR_DIR / "results" / "stability_runs" / DEFAULT_DATASET
DEFAULT_ADJ_FILENAME = "complete_adjacency.pt"

# FCI skeleton CSV: auto-detect latest under unified config output dir
DEFAULT_FCI_CSV = None  # if set to a Path string, overrides auto-detect

# Output base
OUT_BASE = REPO_ROOT / "results" / "edge_kfold_internal"


def auto_detect_latest_fci_csv(dataset: str) -> Path:
    auto = uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / dataset)
    if not auto:
        raise FileNotFoundError(f"No edges_FCI_*.csv found under {uconfig.FCI_OUTPUT_DIR / dataset}")
    return Path(auto)


def load_fci_pairs(fci_csv: Path) -> Set[Tuple[str, str]]:
    df = pd.read_csv(fci_csv)
    cols = {c.lower(): c for c in df.columns}
    if "source" not in cols or "target" not in cols:
        raise ValueError(f"FCI CSV missing source/target columns: {list(df.columns)}")
    src = cols["source"]
    dst = cols["target"]
    pairs: Set[Tuple[str, str]] = set()
    for a, b in zip(df[src].astype(str).tolist(), df[dst].astype(str).tolist()):
        x, y = (a, b) if a < b else (b, a)
        pairs.add((x, y))
    return pairs


def skeleton_degree(pairs: Set[Tuple[str, str]]) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for a, b in pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def list_seed_adjacencies(seed_runs_dir: Path, filename: str) -> List[Path]:
    paths: List[Path] = []
    if not seed_runs_dir.exists():
        return paths
    for p in sorted(seed_runs_dir.glob("seed_*")):
        cand = p / filename
        if cand.exists():
            paths.append(cand)
    return paths


def block_mean_strength(A: torch.Tensor, idx_a: Sequence[int], idx_b: Sequence[int]) -> float:
    return float(A[idx_a][:, idx_b].mean().item())


def compute_multiseed_features(
    adjacency_paths: List[Path],
    var_structure: Dict,
    pairs: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    vt = var_structure["var_to_states"]
    feats: Dict[Tuple[str, str], Dict[str, float]] = {}

    per_run_diff: List[Dict[Tuple[str, str], float]] = []
    for ap in adjacency_paths:
        A = torch.load(ap, map_location="cpu")
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A)
        d: Dict[Tuple[str, str], float] = {}
        for a, b in pairs:
            if a not in vt or b not in vt:
                continue
            ia = vt[a]
            ib = vt[b]
            s_ab = block_mean_strength(A, ia, ib)
            s_ba = block_mean_strength(A, ib, ia)
            d[(a, b)] = s_ab - s_ba
        per_run_diff.append(d)

    for pair in pairs:
        diffs = []
        votes = []
        for d in per_run_diff:
            if pair not in d:
                continue
            dv = float(d[pair])
            diffs.append(dv)
            votes.append(1 if dv >= 0 else 0)  # 1 means a->b (a<b)
        if not diffs:
            continue
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        votes_arr = np.asarray(votes, dtype=np.int32)
        maj = 1 if votes_arr.mean() >= 0.5 else 0
        agree = float(np.mean(votes_arr == maj))
        feats[pair] = {
            "mean_diff": float(diffs_arr.mean()),
            "abs_mean_diff": float(np.abs(diffs_arr.mean())),
            "std_diff": float(diffs_arr.std()),
            "agreement": agree,
            "n_runs": float(len(diffs_arr)),
        }
    return feats


def greedy_dag_assembly(
    pairs: List[Tuple[str, str]],
    prob_a_to_b: Dict[Tuple[str, str], float],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    g = nx.DiGraph()
    unoriented: Set[Tuple[str, str]] = set()
    oriented: Set[Tuple[str, str]] = set()

    def conf(pair: Tuple[str, str]) -> float:
        p = float(prob_a_to_b.get(pair, 0.5))
        return abs(p - 0.5)

    for a, b in sorted(pairs, key=conf, reverse=True):
        p = float(prob_a_to_b.get((a, b), 0.5))
        src, dst = (a, b) if p >= 0.5 else (b, a)
        g.add_node(a)
        g.add_node(b)
        g.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(src, dst)
            unoriented.add((a, b))
            continue
        oriented.add((src, dst))
    return oriented, unoriented


def main():
    total_start = time.time()

    ap = argparse.ArgumentParser()
    # CLI is optional; defaults are editable at top of file
    ap.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    ap.add_argument("--seed_runs_dir", type=str, default=str(DEFAULT_SEED_RUNS_DIR))
    ap.add_argument("--adj_filename", type=str, default=str(DEFAULT_ADJ_FILENAME))
    ap.add_argument("--fci_csv", type=str, default=str(DEFAULT_FCI_CSV) if DEFAULT_FCI_CSV else "")
    ap.add_argument("--n_splits", type=int, default=int(DEFAULT_N_SPLITS))
    ap.add_argument("--random_state", type=int, default=int(DEFAULT_RANDOM_STATE))
    ap.add_argument("--symmetry_threshold", type=float, default=DEFAULT_SYMMETRY_THRESHOLD)
    args = ap.parse_args()

    dataset = str(args.dataset)
    n_splits = int(args.n_splits)
    random_state = int(args.random_state)
    seed_runs_dir = Path(args.seed_runs_dir)
    adj_filename = str(args.adj_filename)

    ds_cfg = uconfig.DATASET_CONFIGS[dataset]
    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "edge_list")

    if args.fci_csv.strip():
        fci_csv = Path(args.fci_csv)
    else:
        fci_csv = auto_detect_latest_fci_csv(dataset)
    if not fci_csv.exists():
        raise FileNotFoundError(fci_csv)

    adjacency_paths = list_seed_adjacencies(seed_runs_dir, adj_filename)
    if len(adjacency_paths) < 2:
        raise RuntimeError(f"Need >=2 seed adjacencies under {seed_runs_dir}, found {len(adjacency_paths)}")

    # Var structure + evaluator
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()
    vt = var_structure["var_to_states"]
    evaluator = CausalGraphEvaluator(str(gt_path), var_structure, ground_truth_type=gt_type)

    # Build training-equivalent block structure (for Symmetry-Unresolved)
    sym_threshold = args.symmetry_threshold
    if sym_threshold is None:
        sym_threshold = 0.08 if dataset == "andes" else 0.1
    prior_builder = PriorBuilder(var_structure, dataset_name=dataset)
    skeleton_mask = prior_builder.build_skeleton_mask_from_fci(str(fci_csv))
    blocks = prior_builder.build_block_structure(skeleton_mask)

    # Skeleton pairs
    skel_pairs = load_fci_pairs(fci_csv)
    deg = skeleton_degree(skel_pairs)

    # GT edges
    gt_loader = GroundTruthLoader(str(gt_path), ground_truth_type=gt_type)
    gt_edges = gt_loader.get_edges()
    if not gt_edges:
        raise RuntimeError(f"Failed to load GT edges: {gt_path}")
    gt_undirected = {(a, b) if a < b else (b, a) for (a, b) in gt_edges}

    # FCI skeleton SHD (context)
    fci_add = len(skel_pairs - gt_undirected)
    fci_del = len(gt_undirected - skel_pairs)
    fci_skeleton_shd = fci_add + fci_del

    # --- Feature compute ---
    t0 = time.time()
    ms = compute_multiseed_features(adjacency_paths, var_structure, skel_pairs)
    feature_compute_seconds = time.time() - t0

    # Build edge table
    rows = []
    for a, b in sorted(skel_pairs):
        if a not in vt or b not in vt:
            continue
        feat = ms.get((a, b))
        if feat is None:
            continue
        gt_has_ab = (a, b) in gt_edges
        gt_has_ba = (b, a) in gt_edges
        gt_true = int(gt_has_ab or gt_has_ba)
        label = ""
        if gt_true:
            label = 1 if gt_has_ab else 0
        rows.append(
            {
                "var_a": a,
                "var_b": b,
                "k_a": len(vt[a]),
                "k_b": len(vt[b]),
                "deg_skel_a": deg.get(a, 0),
                "deg_skel_b": deg.get(b, 0),
                "deg_skel_max": max(deg.get(a, 0), deg.get(b, 0)),
                **feat,
                "gt_true": gt_true,
                "label": label,
            }
        )
    df = pd.DataFrame(rows)
    gt_df = df[df["gt_true"] == 1].copy()
    gt_df["label"] = gt_df["label"].astype(int)

    # --- KFold train (GT-true only) ---
    feature_cols = ["mean_diff", "agreement", "abs_mean_diff", "std_diff", "deg_skel_max", "k_a", "k_b"]
    X = gt_df[feature_cols].to_numpy(dtype=np.float64)
    y = gt_df["label"].to_numpy(dtype=np.int32)

    t1 = time.time()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_prob = np.zeros_like(y, dtype=np.float64)
    fold_metrics = []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]
        )
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        oof_prob[te] = prob
        acc = accuracy_score(y[te], (prob >= 0.5).astype(int))
        try:
            auc = roc_auc_score(y[te], prob)
        except Exception:
            auc = float("nan")
        fold_metrics.append({"fold": fold, "acc": float(acc), "auc": float(auc), "n_test": int(te.size)})
    edge_kfold_train_seconds = time.time() - t1

    gt_df["oof_score"] = oof_prob

    # --- Final fit + score all pairs ---
    t2 = time.time()
    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )
    final_model.fit(X, y)
    all_X = df[feature_cols].to_numpy(dtype=np.float64)
    all_prob = final_model.predict_proba(all_X)[:, 1]
    df["score_fullfit"] = all_prob
    final_fit_and_score_seconds = time.time() - t2

    # Score map: OOF for GT-true, fullfit for others
    score_map: Dict[Tuple[str, str], float] = {}
    for _, r in gt_df.iterrows():
        score_map[(r["var_a"], r["var_b"])] = float(r["oof_score"])
    for _, r in df.iterrows():
        key = (r["var_a"], r["var_b"])
        if key not in score_map:
            score_map[key] = float(r["score_fullfit"])

    # --- Assembly ---
    t3 = time.time()
    pairs_list = [(str(r["var_a"]), str(r["var_b"])) for _, r in df.iterrows()]
    oriented_edges, unoriented_pairs = greedy_dag_assembly(pairs_list, score_map)
    assembly_seconds = time.time() - t3

    # Evaluator metrics (aligned)
    eval_metrics = evaluator.evaluate(oriented_edges)

    # Unresolved ratios (two different notions!)
    # 1) Acyclicity-unresolved: pairs skipped during DAG assembly because they would form a cycle.
    total_pairs = len(pairs_list)
    acyclicity_unresolved_ratio_skeleton = len(unoriented_pairs) / max(1, total_pairs)
    gt_pairs = {(a, b) for (a, b) in skel_pairs if (a, b) in gt_undirected}
    gt_pairs_eval = {(a, b) for (a, b) in gt_pairs if a in vt and b in vt}
    unoriented_gt = {(a, b) for (a, b) in unoriented_pairs if (a, b) in gt_pairs_eval}
    acyclicity_unresolved_ratio_gt_true = len(unoriented_gt) / max(1, len(gt_pairs_eval))

    # 2) Symmetry-unresolved: bidirectional/symmetric strong weights in the learned adjacency.
    # We compute this on each seed adjacency and report mean/min/max, plus the best-seed value (by evaluator full_shd).
    per_seed_sym = []
    per_seed_full = []
    for apath in adjacency_paths:
        A = torch.load(apath, map_location="cpu")
        sym = compute_unresolved_ratio(A, blocks, threshold=float(sym_threshold))
        per_seed_sym.append(float(sym["unresolved_ratio"]))
        # compute full_shd for this seed adjacency (aligned evaluator) to pick best seed
        # Use dataset-default edge threshold for extracting "active" edges from adjacency.
        # (This affects only best-seed selection for reporting symmetry stats, not the scorer itself.)
        edge_threshold = 0.08 if dataset == "andes" else 0.1
        learned_edges_seed = evaluator.extract_learned_edges(A, threshold=edge_threshold)
        m_seed = evaluator.evaluate(learned_edges_seed)
        per_seed_full.append((int(m_seed.get("full_shd", 1e9)), apath))
    symmetry_unresolved_ratio_mean = float(np.mean(per_seed_sym)) if per_seed_sym else None
    symmetry_unresolved_ratio_min = float(np.min(per_seed_sym)) if per_seed_sym else None
    symmetry_unresolved_ratio_max = float(np.max(per_seed_sym)) if per_seed_sym else None
    best_seed_path = min(per_seed_full, key=lambda x: x[0])[1] if per_seed_full else None
    symmetry_unresolved_ratio_best_seed = None
    if best_seed_path is not None:
        A_best = torch.load(best_seed_path, map_location="cpu")
        sym_best = compute_unresolved_ratio(A_best, blocks, threshold=float(sym_threshold))
        symmetry_unresolved_ratio_best_seed = float(sym_best["unresolved_ratio"])

    # Output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / dataset / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "edge_table_all_pairs.csv", index=False)
    gt_df.to_csv(out_dir / "edge_table_gt_true_oof.csv", index=False)
    (out_dir / "assembled_edges.txt").write_text(
        "".join([f"{s} -> {t}\n" for s, t in sorted(oriented_edges)]), encoding="utf-8"
    )

    total_seconds = time.time() - total_start
    runtime = {
        "what_total": "edge_kfold_internal.py end-to-end (analysis only; excludes seed training)",
        "feature_compute_seconds": float(feature_compute_seconds),
        "edge_kfold_train_seconds": float(edge_kfold_train_seconds),
        "final_fit_and_score_seconds": float(final_fit_and_score_seconds),
        "assembly_seconds": float(assembly_seconds),
        "total_seconds": float(total_seconds),
    }

    metrics = {
        "dataset": dataset,
        "n_seed_runs": len(adjacency_paths),
        "n_skeleton_pairs": int(total_pairs),
        "n_gt_true_pairs_in_skeleton": int(len(gt_pairs_eval)),
        "cv_acc_mean": float(np.mean([m["acc"] for m in fold_metrics])),
        "cv_auc_mean": float(np.nanmean([m["auc"] for m in fold_metrics])),
        "cv_folds": fold_metrics,
        # Acyclicity-unresolved (global assembly)
        "acyclicity_unresolved_ratio_skeleton": float(acyclicity_unresolved_ratio_skeleton),
        "acyclicity_unresolved_ratio_gt_true": float(acyclicity_unresolved_ratio_gt_true),
        # Back-compat keys (older name)
        "unresolved_ratio_skeleton": float(acyclicity_unresolved_ratio_skeleton),
        "unresolved_ratio_gt_true": float(acyclicity_unresolved_ratio_gt_true),
        # Symmetry-unresolved (training metric)
        "symmetry_unresolved_threshold": float(sym_threshold),
        "symmetry_unresolved_ratio_mean_seed": symmetry_unresolved_ratio_mean,
        "symmetry_unresolved_ratio_min_seed": symmetry_unresolved_ratio_min,
        "symmetry_unresolved_ratio_max_seed": symmetry_unresolved_ratio_max,
        "symmetry_unresolved_ratio_best_seed": symmetry_unresolved_ratio_best_seed,
        "fci_skeleton_fp_additions": int(fci_add),
        "fci_skeleton_fn_deletions": int(fci_del),
        "fci_skeleton_shd": int(fci_skeleton_shd),
        "evaluator_skeleton_shd": int(eval_metrics.get("skeleton_shd", -1)),
        "evaluator_full_shd": int(eval_metrics.get("full_shd", -1)),
        "evaluator_orientation_accuracy": float(eval_metrics.get("orientation_accuracy", 0.0)),
        "evaluator_edge_f1": float(eval_metrics.get("edge_f1", 0.0)),
        "evaluator_directed_f1": float(eval_metrics.get("directed_f1", 0.0)),
        "runtime": runtime,
        "paths": {
            "fci_csv": str(fci_csv),
            "gt_path": str(gt_path),
            "seed_runs_dir": str(seed_runs_dir),
            "adjacency_files": [str(p) for p in adjacency_paths],
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_lines = [
        "EDGE-KFOLD INTERNAL (DATASET-GENERIC)",
        f"dataset={dataset}",
        f"seed_runs={len(adjacency_paths)}",
        f"FCI skeleton pairs={total_pairs} (FP add={fci_add}, FN del={fci_del}, fci_skeleton_shd={fci_skeleton_shd})",
        "",
        "Edge-CV (GT-true pairs only):",
        f"  mean_acc={metrics['cv_acc_mean']:.4f}  mean_auc={metrics['cv_auc_mean']:.4f}  folds={n_splits}",
        "",
        "Assembly outputs:",
        f"  acyclicity_unresolved_ratio_skeleton={acyclicity_unresolved_ratio_skeleton:.4f}",
        f"  acyclicity_unresolved_ratio_gt_true={acyclicity_unresolved_ratio_gt_true:.4f}",
        "",
        "Symmetry-Unresolved (training metric; from complete_adjacency.pt):",
        f"  threshold={float(sym_threshold):.4f}",
        f"  mean_seed={symmetry_unresolved_ratio_mean:.4f}  min_seed={symmetry_unresolved_ratio_min:.4f}  max_seed={symmetry_unresolved_ratio_max:.4f}",
        f"  best_seed={symmetry_unresolved_ratio_best_seed:.4f}",
        "",
        "Evaluator-aligned metrics (modules/evaluator.py):",
        f"  skeleton_shd={metrics['evaluator_skeleton_shd']}  full_shd={metrics['evaluator_full_shd']}  orientation_accuracy={metrics['evaluator_orientation_accuracy']:.4f}",
        "",
        "Runtime (this script only; excludes seed training):",
        f"  feature_compute_seconds={runtime['feature_compute_seconds']:.4f}",
        f"  edge_kfold_train_seconds={runtime['edge_kfold_train_seconds']:.4f}",
        f"  final_fit_and_score_seconds={runtime['final_fit_and_score_seconds']:.4f}",
        f"  assembly_seconds={runtime['assembly_seconds']:.4f}",
        f"  total_seconds={runtime['total_seconds']:.4f}",
        "",
        f"Outputs: {out_dir}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

