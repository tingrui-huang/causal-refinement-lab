"""
Generate multiple no-LLM random-prior runs across seeds (for stability orientation).

This script runs train_complete() repeatedly with:
  - use_llm_prior = False
  - use_random_prior = True
  - fixed dataset/hyperparams
  - different random_seed per run

Outputs go under:
  Neuro-Symbolic-Reasoning/results/stability_runs/<dataset>/seed_<seed>/

Run (repo root):
  1) Edit the DEFAULTS section below (dataset / seeds / epochs / etc.)
  2) Run:
       python Neuro-Symbolic-Reasoning/asymmetry_analysis/run_multi_seed_random_prior.py

Note:
  This can take time. Start with small epochs for a quick sanity run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
NSR_DIR = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NSR_DIR) not in sys.path:
    sys.path.insert(0, str(NSR_DIR))

# --------------------------------------------------------------------------------------
# DEFAULTS (edit these like config.py style; CLI is optional)
# --------------------------------------------------------------------------------------
DEFAULT_DATASET = "andes"
DEFAULT_SEEDS = [0,1,2,3,4]
DEFAULT_EPOCHS = 1500
DEFAULT_LR = 0.01

# If None, dataset-specific defaults are used (see below)
DEFAULT_LAMBDA_GROUP = None
DEFAULT_LAMBDA_CYCLE = None
DEFAULT_EDGE_THRESHOLD = None


# Robustly import unified config from repo root (avoid accidentally importing Neuro-Symbolic-Reasoning/config.py)
import importlib.util

_spec = importlib.util.spec_from_file_location("unified_config", REPO_ROOT / "config.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to import unified config.py via importlib")
uconfig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uconfig)

from train_complete import train_complete


def main():
    ap = argparse.ArgumentParser()
    # CLI is optional: if you don't pass flags, DEFAULT_* above are used.
    ap.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--lambda_group", type=float, default=DEFAULT_LAMBDA_GROUP)
    ap.add_argument("--lambda_cycle", type=float, default=DEFAULT_LAMBDA_CYCLE)
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    args = ap.parse_args()

    if args.dataset not in uconfig.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{args.dataset}'")
    ds_cfg = uconfig.DATASET_CONFIGS[args.dataset]

    # Reasonable defaults (match experiment_llm_vs_random.py style)
    lambda_group = args.lambda_group
    lambda_cycle = args.lambda_cycle
    edge_threshold = args.edge_threshold

    if lambda_group is None or lambda_cycle is None or edge_threshold is None:
        # Dataset-specific fallbacks
        if args.dataset == "andes":
            lambda_group = 0.05 if lambda_group is None else lambda_group
            lambda_cycle = 0.05 if lambda_cycle is None else lambda_cycle
            edge_threshold = 0.08 if edge_threshold is None else edge_threshold
        else:
            lambda_group = 0.01 if lambda_group is None else lambda_group
            lambda_cycle = 0.001 if lambda_cycle is None else lambda_cycle
            edge_threshold = 0.1 if edge_threshold is None else edge_threshold

    # Track total runtime across seeds (this script only; excludes any downstream analysis)
    total_start = time.time()
    per_seed_rows = []

    for seed in args.seeds:
        out_dir = NSR_DIR / "results" / "stability_runs" / args.dataset / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "data_path": str(ds_cfg["data_path"]),
            "metadata_path": str(ds_cfg["metadata_path"]),
            "ground_truth_path": str(ds_cfg["ground_truth_path"]),
            "ground_truth_type": ds_cfg.get("ground_truth_type", "edge_list"),
            "n_epochs": int(args.epochs),
            "learning_rate": float(args.lr),
            "n_hops": 1,
            "lambda_group": float(lambda_group),
            "lambda_cycle": float(lambda_cycle),
            # Ensure we always log at least once even for short debug runs
            "monitor_interval": max(1, min(20, int(args.epochs))),
            "edge_threshold": float(edge_threshold),
            "use_llm_prior": False,
            "use_random_prior": True,
            "llm_direction_path": None,
            # Provide FCI skeleton if available (optional)
            "fci_skeleton_path": uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / args.dataset),
            "random_seed": int(seed),
            "output_dir": str(out_dir),
        }

        print("\n" + "=" * 80)
        print(f"STABILITY RUN: dataset={args.dataset}, seed={seed}")
        print("=" * 80)
        seed_start = time.time()
        train_complete(cfg)
        seed_sec = time.time() - seed_start
        per_seed_rows.append((int(seed), float(seed_sec)))

        # Write per-seed runtime (seconds) so we can audit/compare later
        (out_dir / "runtime_seconds.txt").write_text(
            f"dataset={args.dataset}\nseed={seed}\nwhat=train_complete(cfg)\nruntime_seconds={seed_sec:.4f}\n",
            encoding="utf-8",
        )

    total_sec = time.time() - total_start
    # Write dataset-level summary for this multi-seed invocation
    ds_dir = NSR_DIR / "results" / "stability_runs" / args.dataset
    summary_lines = [
        "MULTI-SEED STABILITY RUN SUMMARY",
        f"dataset={args.dataset}",
        f"what=run_multi_seed_random_prior.py (sum of train_complete over seeds)",
        f"seeds={list(map(int, args.seeds))}",
        f"epochs={int(args.epochs)}",
        f"lr={float(args.lr)}",
        f"lambda_group={float(lambda_group)}",
        f"lambda_cycle={float(lambda_cycle)}",
        f"edge_threshold={float(edge_threshold)}",
        f"total_runtime_seconds={total_sec:.4f}",
        "",
        "per_seed_runtime_seconds:",
    ]
    for s, sec in per_seed_rows:
        summary_lines.append(f"  seed_{s}: {sec:.4f}")
    (ds_dir / "multi_seed_runtime_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

