"""
Andes internal generalization via Edge-KFold CV (no new data, edge split not sample split)

Pipeline:
  1) Build an "edge table" over FCI skeleton undirected pairs {a,b}.
     Features are computed from multi-seed adjacencies (no labels used):
       - mean_diff: mean over seeds of (S(a->b) - S(b->a)) using block mean strength
       - abs_mean_diff
       - std_diff
       - agreement: fraction of seeds voting the majority direction
       - deg_skel_max: max degree of (a,b) in FCI skeleton graph
       - k_a, k_b: number of states per variable
     Labels (only for GT-true pairs):
       y = 1 if GT is a->b (with a<b), else 0 if GT is b->a

  2) 5-fold CV on GT-true edges (edge rows), train LogisticRegression to predict y.
     Produce out-of-fold probabilities for GT-true edges.

  3) Fit a final model on all GT-true edges and score ALL skeleton pairs.

  4) Greedy DAG assembly from scores (highest confidence first), skipping edges that create cycles.

  5) Report:
       - edge-CV metrics (acc/auc)
       - unresolved ratio on skeleton + on GT-true pairs
       - orientation accuracy on oriented GT-true pairs
       - SHD-style stats relative to GT:
           skeleton_shd = additions(FP in skeleton) + deletions(FN missing from skeleton)
           full_shd_strict = skeleton_shd + reversals(wrong oriented) + unresolved_as_errors(unoriented GT-true)

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/andes_edge_kfold_internal.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


# --------------------------------------------------------------------------------------
# DEFAULTS (edit if needed)
# --------------------------------------------------------------------------------------
DATASET = "andes"
N_SPLITS = 5
RANDOM_STATE = 42

# Multi-seed adjacency inputs (produced by run_multi_seed_random_prior.py)
SEED_RUNS_DIR = NSR_DIR / "results" / "stability_runs" / DATASET
ADJ_FILENAME = "complete_adjacency.pt"

# FCI skeleton CSV
FCI_CSV = Path(r"Neuro-Symbolic-Reasoning/data/andes/edges_FCI_20260108_212351.csv")

# Output directory (timestamped)
OUT_BASE = REPO_ROOT / "results" / "andes_internal_edge_cv"


def load_fci_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "source" not in cols or "target" not in cols:
        raise ValueError(f"FCI CSV missing source/target: {list(df.columns)}")
    if "edge_type" not in cols:
        raise ValueError(f"FCI CSV missing edge_type: {list(df.columns)}")
    df = df.rename(columns={cols["source"]: "source", cols["target"]: "target", cols["edge_type"]: "edge_type"})
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["edge_type"] = df["edge_type"].astype(str)
    return df[["source", "target", "edge_type"]]


def skeleton_pairs_from_fci(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for s, t in zip(df["source"].tolist(), df["target"].tolist()):
        a, b = (s, t) if s < t else (t, s)
        pairs.add((a, b))
    return pairs


def skeleton_degree(pairs: Set[Tuple[str, str]]) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for a, b in pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def list_seed_adjacencies(seed_runs_dir: Path, filename: str) -> List[Path]:
    paths = []
    if not seed_runs_dir.exists():
        return paths
    for p in sorted(seed_runs_dir.glob("seed_*")):
        cand = p / filename
        if cand.exists():
            paths.append(cand)
    return paths


def block_mean_strength(adjacency: torch.Tensor, idx_a: Sequence[int], idx_b: Sequence[int]) -> float:
    return float(adjacency[idx_a][:, idx_b].mean().item())


def compute_multiseed_features(
    adjacency_paths: List[Path],
    var_structure: Dict,
    pairs: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    vt = var_structure["var_to_states"]
    feats: Dict[Tuple[str, str], Dict[str, float]] = {}

    # For each run, compute signed diffs
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

    n_runs = len(per_run_diff)
    for pair in pairs:
        diffs = []
        votes = []
        for d in per_run_diff:
            if pair not in d:
                continue
            dv = float(d[pair])
            diffs.append(dv)
            votes.append(1 if dv >= 0 else 0)  # 1 means a->b (a<b), 0 means b->a
        if not diffs:
            continue
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        votes_arr = np.asarray(votes, dtype=np.int32)
        maj = int(np.round(votes_arr.mean()))  # 1 if >=0.5
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
    """
    Build a DAG by greedily adding high-confidence oriented edges, skipping those that create cycles.
    Returns:
      oriented_edges: set of directed edges (src,dst)
      unoriented_pairs: set of pairs left unoriented (due to cycle/skip)
    """
    g = nx.DiGraph()
    unoriented: Set[Tuple[str, str]] = set()
    oriented: Set[Tuple[str, str]] = set()

    # sort by confidence
    def conf(pair: Tuple[str, str]) -> float:
        p = float(prob_a_to_b.get(pair, 0.5))
        return abs(p - 0.5)

    for a, b in sorted(pairs, key=conf, reverse=True):
        p = float(prob_a_to_b.get((a, b), 0.5))
        if p >= 0.5:
            src, dst = a, b
        else:
            src, dst = b, a
        g.add_node(a)
        g.add_node(b)
        g.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(src, dst)
            unoriented.add((a, b))
            continue
        oriented.add((src, dst))
    # any pair not oriented because confidence low? we still attempted all; unoriented are those skipped due to cycles
    return oriented, unoriented


def main():
    start = time.time()
    ds_cfg = uconfig.DATASET_CONFIGS[DATASET]
    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "edge_list")

    if not FCI_CSV.exists():
        raise FileNotFoundError(FCI_CSV)

    adjacency_paths = list_seed_adjacencies(SEED_RUNS_DIR, ADJ_FILENAME)
    if len(adjacency_paths) < 2:
        raise RuntimeError(f"Need >=2 seed adjacencies under {SEED_RUNS_DIR}, found {len(adjacency_paths)}")

    # Load var_structure (metadata-only; don't call load_data)
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()
    vt = var_structure["var_to_states"]
    evaluator = CausalGraphEvaluator(str(gt_path), var_structure, ground_truth_type=gt_type)

    # Load FCI skeleton pairs
    fci_df = load_fci_df(FCI_CSV)
    skel_pairs = skeleton_pairs_from_fci(fci_df)
    deg = skeleton_degree(skel_pairs)

    # Load GT edges
    gt_loader = GroundTruthLoader(str(gt_path), ground_truth_type=gt_type)
    gt_edges = gt_loader.get_edges()
    if not gt_edges:
        raise RuntimeError(f"Failed to load GT edges: {gt_path}")
    gt_undirected = {(a, b) if a < b else (b, a) for (a, b) in gt_edges}

    # Skeleton FP/FN (about the FCI skeleton itself; kept for context only)
    skel_additions = len(skel_pairs - gt_undirected)  # FP in skeleton
    skel_deletions = len(gt_undirected - skel_pairs)  # FN missing from skeleton
    skeleton_shd_from_skeleton = skel_additions + skel_deletions

    # Multi-seed features
    ms = compute_multiseed_features(adjacency_paths, var_structure, skel_pairs)

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

    # CV on edges
    feature_cols = ["mean_diff", "agreement", "abs_mean_diff", "std_diff", "deg_skel_max", "k_a", "k_b"]
    X = gt_df[feature_cols].to_numpy(dtype=np.float64)
    y = gt_df["label"].to_numpy(dtype=np.int32)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
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

    gt_df["oof_score"] = oof_prob

    # Fit final model on all GT-true edges, score all skeleton pairs
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

    # Use OOF score for GT-true pairs, fullfit for non-GT-true (for assembly completeness)
    score_map: Dict[Tuple[str, str], float] = {}
    # Map GT-true pairs to oof
    for _, r in gt_df.iterrows():
        a = r["var_a"]
        b = r["var_b"]
        score_map[(a, b)] = float(r["oof_score"])
    # Fill others
    for _, r in df.iterrows():
        a = r["var_a"]
        b = r["var_b"]
        if (a, b) not in score_map:
            score_map[(a, b)] = float(r["score_fullfit"])

    # Greedy assembly
    pairs_list = [(str(r["var_a"]), str(r["var_b"])) for _, r in df.iterrows()]
    oriented_edges, unoriented_pairs = greedy_dag_assembly(pairs_list, score_map)

    # Aligned evaluation (exactly matches project's evaluator SHD definition)
    eval_metrics = evaluator.evaluate(oriented_edges)

    # Compute unresolved ratios
    total_pairs = len(pairs_list)
    oriented_pairs = total_pairs - len(unoriented_pairs)
    unresolved_ratio_skeleton = len(unoriented_pairs) / max(1, total_pairs)

    # GT-true unresolved + accuracy on oriented subset
    gt_pairs = {(a, b) for (a, b) in skel_pairs if (a, b) in gt_undirected}
    gt_pairs_eval = {(a, b) for (a, b) in gt_pairs if a in vt and b in vt}
    unoriented_gt = {(a, b) for (a, b) in unoriented_pairs if (a, b) in gt_pairs_eval}
    unresolved_ratio_gt_true = len(unoriented_gt) / max(1, len(gt_pairs_eval))

    # Orientation accuracy among oriented GT-true pairs
    correct_oriented = 0
    total_oriented_gt = 0
    reversals = 0
    for a, b in gt_pairs_eval:
        if (a, b) in unoriented_pairs:
            continue
        # find which direction was chosen in oriented_edges
        if (a, b) in oriented_edges:
            pred = (a, b)
        elif (b, a) in oriented_edges:
            pred = (b, a)
        else:
            # shouldn't happen
            continue
        total_oriented_gt += 1
        if pred in gt_edges:
            correct_oriented += 1
        else:
            reversals += 1

    orientation_accuracy_oriented = correct_oriented / max(1, total_oriented_gt)
    coverage_gt_true = total_oriented_gt / max(1, len(gt_pairs_eval))

    # Output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"{DATASET}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "edge_table_all_pairs.csv", index=False)
    gt_df.to_csv(out_dir / "edge_table_gt_true_oof.csv", index=False)
    with open(out_dir / "assembled_edges.txt", "w", encoding="utf-8") as f:
        for s, t in sorted(oriented_edges):
            f.write(f"{s} -> {t}\n")

    metrics = {
        "dataset": DATASET,
        "n_seed_runs": len(adjacency_paths),
        "n_skeleton_pairs": int(total_pairs),
        "n_gt_true_pairs_in_skeleton": int(len(gt_pairs_eval)),
        "cv_folds": fold_metrics,
        "cv_acc_mean": float(np.mean([m["acc"] for m in fold_metrics])),
        "cv_auc_mean": float(np.nanmean([m["auc"] for m in fold_metrics])),
        # Context about the *FCI skeleton* itself (not the learned graph)
        "fci_skeleton_fp_additions": int(skel_additions),
        "fci_skeleton_fn_deletions": int(skel_deletions),
        "fci_skeleton_shd": int(skeleton_shd_from_skeleton),
        "dag_oriented_pairs": int(oriented_pairs),
        "dag_unoriented_pairs": int(len(unoriented_pairs)),
        "unresolved_ratio_skeleton": float(unresolved_ratio_skeleton),
        "gt_true_oriented_pairs": int(total_oriented_gt),
        "gt_true_unoriented_pairs": int(len(unoriented_gt)),
        "unresolved_ratio_gt_true": float(unresolved_ratio_gt_true),
        "coverage_gt_true": float(coverage_gt_true),
        "orientation_accuracy_oriented_gt_true": float(orientation_accuracy_oriented),
        "reversals_on_oriented_gt_true": int(reversals),
        # Aligned evaluator metrics (same as modules/evaluator.py)
        "evaluator_skeleton_shd": int(eval_metrics.get("skeleton_shd", -1)),
        "evaluator_full_shd": int(eval_metrics.get("full_shd", -1)),
        "evaluator_orientation_accuracy": float(eval_metrics.get("orientation_accuracy", 0.0)),
        "evaluator_edge_f1": float(eval_metrics.get("edge_f1", 0.0)),
        "evaluator_directed_f1": float(eval_metrics.get("directed_f1", 0.0)),
        "paths": {
            "fci_csv": str(FCI_CSV),
            "gt_path": str(gt_path),
            "seed_runs_dir": str(SEED_RUNS_DIR),
            "adjacency_files": [str(p) for p in adjacency_paths],
        },
        "runtime_sec": float(time.time() - start),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Human-readable summary (include unresolved ratio prominently)
    summary = []
    summary.append("ANDES EDGE-KFOLD INTERNAL GENERALIZATION")
    summary.append(f"dataset={DATASET}")
    summary.append(f"seed_runs={len(adjacency_paths)}")
    summary.append(
        f"FCI skeleton pairs={total_pairs}  "
        f"(FP add={skel_additions}, FN del={skel_deletions}, fci_skeleton_shd={skeleton_shd_from_skeleton})"
    )
    summary.append("")
    summary.append("Edge-CV (GT-true pairs only):")
    summary.append(f"  mean_acc={metrics['cv_acc_mean']:.4f}  mean_auc={metrics['cv_auc_mean']:.4f}  folds={N_SPLITS}")
    summary.append("")
    summary.append("Assembly outputs:")
    summary.append(f"  unresolved_ratio_skeleton={unresolved_ratio_skeleton:.4f}")
    summary.append(f"  unresolved_ratio_gt_true={unresolved_ratio_gt_true:.4f}  coverage_gt_true={coverage_gt_true:.4f}")
    summary.append(f"  orientation_accuracy_oriented_gt_true={orientation_accuracy_oriented:.4f}  (n_oriented_gt_true={total_oriented_gt})")
    summary.append("")
    summary.append("Evaluator-aligned metrics (same definition as modules/evaluator.py):")
    summary.append(
        f"  skeleton_shd={metrics['evaluator_skeleton_shd']}  full_shd={metrics['evaluator_full_shd']}  "
        f"orientation_accuracy={metrics['evaluator_orientation_accuracy']:.4f}"
    )
    summary.append("")
    summary.append(f"Outputs: {out_dir}")
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("\n".join(summary))


if __name__ == "__main__":
    main()

