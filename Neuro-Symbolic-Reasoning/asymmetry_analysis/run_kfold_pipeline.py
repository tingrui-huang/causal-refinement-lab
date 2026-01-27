"""
One-click (no CLI flags) pipeline runner:
  1) Multi-seed random-prior training for each dataset
  2) Per-seed SHD summary (aligned evaluator metrics) + per-seed training runtime
  3) Edge-KFold internal scorer + greedy DAG assembly (aligned SHD + unresolved ratio + analysis runtime)

Usage (repo root):
  1) Edit DEFAULTS below (datasets / epochs / seeds / etc.)
  2) Run:
       python Neuro-Symbolic-Reasoning/asymmetry_analysis/run_kfold_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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

from train_complete import train_complete
from modules.data_loader import CausalDataLoader
from modules.evaluator import CausalGraphEvaluator
from modules.metrics import compute_unresolved_ratio
from modules.prior_builder import PriorBuilder

from asymmetry_analysis.edge_kfold_internal import (
    auto_detect_latest_fci_csv,
    block_mean_strength,
    compute_multiseed_features,
    greedy_dag_assembly,
    list_seed_adjacencies,
    load_fci_pairs,
    skeleton_degree,
)


# --------------------------------------------------------------------------------------
# DEFAULTS (edit like config.py style)
# --------------------------------------------------------------------------------------
DEFAULT_DATASETS = ["hailfinder", "win95pts"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_EPOCHS = 1500
DEFAULT_LR = 0.01

# If None, dataset-specific fallbacks are used
DEFAULT_LAMBDA_GROUP = None
DEFAULT_LAMBDA_CYCLE = None
DEFAULT_EDGE_THRESHOLD = None

# File naming
ADJ_FILENAME = "complete_adjacency.pt"

# Output base for pipeline summaries
PIPELINE_OUT_DIR = REPO_ROOT / "results" / "kfold_pipeline_runs"


def parse_edges_txt(path: Path) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("=") or "Learned Causal Edges" in line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            src = left.strip()
            dst = right.split("(")[0].strip()
            if src and dst:
                edges.add((src, dst))
    return edges


def read_runtime_seconds(seed_dir: Path) -> float | None:
    p = seed_dir / "runtime_seconds.txt"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("runtime_seconds="):
            try:
                return float(line.split("=", 1)[1].strip())
            except Exception:
                return None
    return None


def dataset_fallbacks(dataset: str) -> tuple[float, float, float]:
    # Match run_multi_seed_random_prior.py defaults
    if dataset == "andes":
        return 0.05, 0.05, 0.08
    return 0.01, 0.001, 0.1


def run_multi_seed_training(dataset: str, seeds: List[int], epochs: int, lr: float) -> Dict:
    ds_cfg = uconfig.DATASET_CONFIGS[dataset]

    lambda_group = DEFAULT_LAMBDA_GROUP
    lambda_cycle = DEFAULT_LAMBDA_CYCLE
    edge_threshold = DEFAULT_EDGE_THRESHOLD
    if lambda_group is None or lambda_cycle is None or edge_threshold is None:
        lg, lc, et = dataset_fallbacks(dataset)
        lambda_group = lg if lambda_group is None else lambda_group
        lambda_cycle = lc if lambda_cycle is None else lambda_cycle
        edge_threshold = et if edge_threshold is None else edge_threshold

    total_start = time.time()
    per_seed = []

    for seed in seeds:
        out_dir = NSR_DIR / "results" / "stability_runs" / dataset / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "data_path": str(ds_cfg["data_path"]),
            "metadata_path": str(ds_cfg["metadata_path"]),
            "ground_truth_path": str(ds_cfg["ground_truth_path"]),
            "ground_truth_type": ds_cfg.get("ground_truth_type", "edge_list"),
            "n_epochs": int(epochs),
            "learning_rate": float(lr),
            "n_hops": 1,
            "lambda_group": float(lambda_group),
            "lambda_cycle": float(lambda_cycle),
            "monitor_interval": max(1, min(20, int(epochs))),
            "edge_threshold": float(edge_threshold),
            "use_llm_prior": False,
            "use_random_prior": True,
            "llm_direction_path": None,
            "fci_skeleton_path": uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / dataset),
            "random_seed": int(seed),
            "output_dir": str(out_dir),
        }

        print("\n" + "=" * 80)
        print(f"[PIPELINE] TRAIN seed={seed} dataset={dataset} epochs={epochs}")
        print("=" * 80)

        s0 = time.time()
        train_complete(cfg)
        s_sec = time.time() - s0
        per_seed.append({"seed": int(seed), "train_complete_seconds": float(s_sec)})
        (out_dir / "runtime_seconds.txt").write_text(
            f"dataset={dataset}\nseed={seed}\nwhat=train_complete(cfg)\nruntime_seconds={s_sec:.4f}\n",
            encoding="utf-8",
        )

    total_sec = time.time() - total_start
    return {
        "dataset": dataset,
        "what": "multi-seed training (sum of train_complete over seeds)",
        "seeds": list(map(int, seeds)),
        "epochs": int(epochs),
        "lr": float(lr),
        "lambda_group": float(lambda_group),
        "lambda_cycle": float(lambda_cycle),
        "edge_threshold": float(edge_threshold),
        "total_seconds": float(total_sec),
        "per_seed": per_seed,
    }


def summarize_seeds(dataset: str) -> Dict:
    ds_cfg = uconfig.DATASET_CONFIGS[dataset]
    loader = CausalDataLoader(str(ds_cfg["data_path"]), str(ds_cfg["metadata_path"]))
    var_structure = loader.get_variable_structure()
    evaluator = CausalGraphEvaluator(
        str(ds_cfg["ground_truth_path"]),
        var_structure,
        ground_truth_type=ds_cfg.get("ground_truth_type", "edge_list"),
    )

    base = NSR_DIR / "results" / "stability_runs" / dataset
    rows = []
    for seed_dir in sorted(base.glob("seed_*")):
        edges_path = seed_dir / "complete_edges.txt"
        if not edges_path.exists():
            continue
        edges = parse_edges_txt(edges_path)
        m = evaluator.evaluate(edges)
        rows.append(
            {
                "seed": seed_dir.name,
                "learned_edges": len(edges),
                "skeleton_shd": int(m.get("skeleton_shd", -1)),
                "full_shd": int(m.get("full_shd", -1)),
                "orientation_accuracy": float(m.get("orientation_accuracy", 0.0)),
                "seed_train_runtime_seconds": read_runtime_seconds(seed_dir),
            }
        )
    best = min(rows, key=lambda r: r["full_shd"]) if rows else None
    return {"dataset": dataset, "per_seed": rows, "best_by_full_shd": best}


def run_edge_kfold_internal(dataset: str) -> Dict:
    """
    Minimal in-process version of edge_kfold_internal.py, returning metrics and writing outputs.
    This only times the analysis (feature/KFold/scoring/assembly), not the seed training.
    """
    total_start = time.time()

    ds_cfg = uconfig.DATASET_CONFIGS[dataset]
    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "edge_list")

    seed_runs_dir = NSR_DIR / "results" / "stability_runs" / dataset
    adjacency_paths = list_seed_adjacencies(seed_runs_dir, ADJ_FILENAME)
    if len(adjacency_paths) < 2:
        raise RuntimeError(f"Need >=2 seed adjacencies for {dataset}, found {len(adjacency_paths)}")

    fci_csv = auto_detect_latest_fci_csv(dataset)

    # Var structure + evaluator
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()
    vt = var_structure["var_to_states"]
    evaluator = CausalGraphEvaluator(str(gt_path), var_structure, ground_truth_type=gt_type)

    # Symmetry-Unresolved uses the training-equivalent block structure.
    prior_builder = PriorBuilder(var_structure, dataset_name=dataset)
    skeleton_mask = prior_builder.build_skeleton_mask_from_fci(str(fci_csv))
    blocks = prior_builder.build_block_structure(skeleton_mask)
    sym_threshold = 0.08 if dataset == "andes" else 0.1

    # Skeleton pairs
    skel_pairs = load_fci_pairs(fci_csv)
    deg = skeleton_degree(skel_pairs)

    # GT edges
    from modules.ground_truth_loader import GroundTruthLoader

    gt_loader = GroundTruthLoader(str(gt_path), ground_truth_type=gt_type)
    gt_edges = gt_loader.get_edges()
    if not gt_edges:
        raise RuntimeError(f"Failed to load GT edges: {gt_path}")
    gt_undirected = {(a, b) if a < b else (b, a) for (a, b) in gt_edges}

    # FCI skeleton SHD (context)
    fci_add = len(skel_pairs - gt_undirected)
    fci_del = len(gt_undirected - skel_pairs)
    fci_skeleton_shd = fci_add + fci_del

    # Feature compute
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

    # KFold train
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    feature_cols = ["mean_diff", "agreement", "abs_mean_diff", "std_diff", "deg_skel_max", "k_a", "k_b"]
    X = gt_df[feature_cols].to_numpy(dtype=np.float64)
    y = gt_df["label"].to_numpy(dtype=np.int32)

    t1 = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
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

    # Final fit + score all
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

    # Score map
    score_map: Dict[Tuple[str, str], float] = {}
    for _, r in gt_df.iterrows():
        score_map[(r["var_a"], r["var_b"])] = float(r["oof_score"])
    for _, r in df.iterrows():
        key = (r["var_a"], r["var_b"])
        if key not in score_map:
            score_map[key] = float(r["score_fullfit"])

    # Assembly
    t3 = time.time()
    pairs_list = [(str(r["var_a"]), str(r["var_b"])) for _, r in df.iterrows()]
    oriented_edges, unoriented_pairs = greedy_dag_assembly(pairs_list, score_map)
    assembly_seconds = time.time() - t3

    eval_metrics = evaluator.evaluate(oriented_edges)

    total_pairs = len(pairs_list)
    unresolved_ratio_skeleton = len(unoriented_pairs) / max(1, total_pairs)
    gt_pairs = {(a, b) for (a, b) in skel_pairs if (a, b) in gt_undirected}
    gt_pairs_eval = {(a, b) for (a, b) in gt_pairs if a in vt and b in vt}
    unoriented_gt = {(a, b) for (a, b) in unoriented_pairs if (a, b) in gt_pairs_eval}
    unresolved_ratio_gt_true = len(unoriented_gt) / max(1, len(gt_pairs_eval))

    # Symmetry-Unresolved on each seed adjacency (training metric)
    per_seed_sym = []
    for ap in adjacency_paths:
        A = torch.load(ap, map_location="cpu")
        sym = compute_unresolved_ratio(A, blocks, threshold=float(sym_threshold))
        per_seed_sym.append(float(sym["unresolved_ratio"]))
    sym_mean = float(np.mean(per_seed_sym)) if per_seed_sym else None
    sym_min = float(np.min(per_seed_sym)) if per_seed_sym else None
    sym_max = float(np.max(per_seed_sym)) if per_seed_sym else None

    out_dir = REPO_ROOT / "results" / "edge_kfold_internal" / dataset / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "edge_table_all_pairs.csv", index=False)
    gt_df.to_csv(out_dir / "edge_table_gt_true_oof.csv", index=False)
    (out_dir / "assembled_edges.txt").write_text(
        "".join([f"{s} -> {t}\n" for s, t in sorted(oriented_edges)]), encoding="utf-8"
    )

    total_seconds = time.time() - total_start
    runtime = {
        "what_total": "edge_kfold_internal analysis only (excludes seed training)",
        "feature_compute_seconds": float(feature_compute_seconds),
        "edge_kfold_train_seconds": float(edge_kfold_train_seconds),
        "final_fit_and_score_seconds": float(final_fit_and_score_seconds),
        "assembly_seconds": float(assembly_seconds),
        "total_seconds": float(total_seconds),
    }

    metrics = {
        "dataset": dataset,
        "seed_runs": len(adjacency_paths),
        "fci_skeleton_pairs": int(total_pairs),
        "fci_skeleton_fp_additions": int(fci_add),
        "fci_skeleton_fn_deletions": int(fci_del),
        "fci_skeleton_shd": int(fci_skeleton_shd),
        "cv_acc_mean": float(np.mean([m["acc"] for m in fold_metrics])),
        "cv_auc_mean": float(np.nanmean([m["auc"] for m in fold_metrics])),
        "unresolved_ratio_skeleton": float(unresolved_ratio_skeleton),
        "unresolved_ratio_gt_true": float(unresolved_ratio_gt_true),
        "symmetry_unresolved_threshold": float(sym_threshold),
        "symmetry_unresolved_ratio_mean_seed": sym_mean,
        "symmetry_unresolved_ratio_min_seed": sym_min,
        "symmetry_unresolved_ratio_max_seed": sym_max,
        "evaluator_skeleton_shd": int(eval_metrics.get("skeleton_shd", -1)),
        "evaluator_full_shd": int(eval_metrics.get("full_shd", -1)),
        "evaluator_orientation_accuracy": float(eval_metrics.get("orientation_accuracy", 0.0)),
        "runtime": runtime,
        "paths": {"fci_csv": str(fci_csv), "seed_runs_dir": str(seed_runs_dir)},
        "outputs": {"out_dir": str(out_dir)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    PIPELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = PIPELINE_OUT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_summary = {
        "what": "run_kfold_pipeline.py (no CLI flags) end-to-end",
        "datasets": DEFAULT_DATASETS,
        "seeds": DEFAULT_SEEDS,
        "epochs": DEFAULT_EPOCHS,
        "lr": DEFAULT_LR,
        "timestamp": ts,
        "results": {},
    }

    for ds in DEFAULT_DATASETS:
        ds_entry = {}
        # 1) Multi-seed training
        ds_entry["multi_seed_training"] = run_multi_seed_training(ds, DEFAULT_SEEDS, DEFAULT_EPOCHS, DEFAULT_LR)
        # 2) Per-seed summary
        ds_entry["seed_summary"] = summarize_seeds(ds)
        # 3) Edge-KFold internal
        ds_entry["edge_kfold_internal"] = run_edge_kfold_internal(ds)
        pipeline_summary["results"][ds] = ds_entry

    (run_dir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"[PIPELINE] DONE. Summary: {run_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()

