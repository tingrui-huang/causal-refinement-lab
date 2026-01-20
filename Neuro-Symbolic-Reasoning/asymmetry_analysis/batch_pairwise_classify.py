"""
Batch-run pairwise True-vs-Reverse on all configured datasets (excluding tuebingen)
and classify "A-type" datasets where frac(delta < 0) > THRESHOLD.

Definition used here:
  For each dataset and each metric (loss / nuclear norm / entropy), we compute:
    frac_neg = fraction of sampled GT edges where Delta = forward - reverse < 0
  A-type (per-metric) if frac_neg > THRESHOLD.

This script uses collider_filter="none" by default to keep sample size reasonable.

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/batch_pairwise_classify.py

Edit settings in the DEFAULTS section below (no CLI needed).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

from asymmetry_analysis.pairwise_true_vs_reverse import run_dataset


# --------------------------------------------------------------------------------------
# DEFAULTS (edit these)
# --------------------------------------------------------------------------------------
THRESHOLD = 0.75
COLLIDER_FILTER = "none"  # "none" or "strict"
N_EDGES = 100
EPOCHS = 150
LR = 0.05
WEIGHT_DECAY = 1e-4
SEED = 0
DEVICE = "cpu"

# Optionally restrict dataset list (None => auto from config, excluding tuebingen)
DATASETS_OVERRIDE = None  # e.g., ["alarm", "win95pts", "andes"]


@dataclass
class SummaryStats:
    n_edges: int
    frac_neg_loss: float
    frac_neg_rank: float
    frac_neg_entropy: float


def parse_frac_neg(summary_text: str) -> SummaryStats:
    # n_edges line: n_edges=100, ...
    m = re.search(r"n_edges=(\d+)", summary_text)
    n_edges = int(m.group(1)) if m else -1

    def grab_frac(block_name: str) -> float:
        # find block, then frac(delta<0)=X
        # block_name should be "Loss delta" / "Nuclear norm delta" / "Entropy delta"
        pat = rf"{re.escape(block_name)}[\s\S]*?frac\(delta<0\)=([0-9.]+)"
        mm = re.search(pat, summary_text)
        return float(mm.group(1)) if mm else float("nan")

    return SummaryStats(
        n_edges=n_edges,
        frac_neg_loss=grab_frac("Loss delta (CE) [forward - reverse]"),
        frac_neg_rank=grab_frac("Nuclear norm delta [forward - reverse]"),
        frac_neg_entropy=grab_frac("Entropy delta [forward - reverse]"),
    )


def dataset_keys() -> List[str]:
    if DATASETS_OVERRIDE:
        return list(DATASETS_OVERRIDE)
    keys = []
    for k in uconfig.DATASET_CONFIGS.keys():
        if k.startswith("tuebingen"):
            continue
        keys.append(k)
    # Keep a stable, human-friendly order: put the core benchmarks first if present
    preferred = ["alarm", "win95pts", "andes", "sachs", "child", "hailfinder", "insurance"]
    out = []
    for p in preferred:
        if p in keys:
            out.append(p)
    for k in sorted(keys):
        if k not in out:
            out.append(k)
    return out


def main():
    results_dir = REPO_ROOT / "results" / "pairwise_batch_classify"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\n" + "=" * 90)
    print("BATCH PAIRWISE CLASSIFY")
    print("=" * 90)
    print(f"threshold(frac_neg) > {THRESHOLD}")
    print(f"collider_filter={COLLIDER_FILTER}, n_edges={N_EDGES}, epochs={EPOCHS}, lr={LR}, wd={WEIGHT_DECAY}, seed={SEED}")

    for ds in dataset_keys():
        print("\n" + "-" * 90)
        print(f"Running dataset: {ds}")
        out = run_dataset(
            ds,
            n_edges=N_EDGES,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
            device=DEVICE,
            collider_filter=COLLIDER_FILTER,
        )
        summary_path = Path(out) / "summary.txt"
        txt = summary_path.read_text(encoding="utf-8", errors="ignore")
        st = parse_frac_neg(txt)

        row = {
            "dataset": ds,
            "n_edges_used": st.n_edges,
            "frac_neg_loss": st.frac_neg_loss,
            "frac_neg_rank": st.frac_neg_rank,
            "frac_neg_entropy": st.frac_neg_entropy,
            "A_loss": int(st.frac_neg_loss > THRESHOLD),
            "A_rank": int(st.frac_neg_rank > THRESHOLD),
            "A_entropy": int(st.frac_neg_entropy > THRESHOLD),
            "A_any": int((st.frac_neg_loss > THRESHOLD) or (st.frac_neg_rank > THRESHOLD) or (st.frac_neg_entropy > THRESHOLD)),
        }
        rows.append(row)
        print(
            f"frac_neg(loss/rank/ent)={row['frac_neg_loss']:.3f}/{row['frac_neg_rank']:.3f}/{row['frac_neg_entropy']:.3f}  "
            f"A_any={row['A_any']}"
        )

    # Print final table (console)
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (A-type if frac_neg > threshold)")
    print("=" * 90)
    header = "dataset  n  frac_loss  frac_rank  frac_ent  A_loss A_rank A_ent A_any"
    print(header)
    for r in rows:
        print(
            f"{r['dataset']:<10s} {r['n_edges_used']:>3d} "
            f"{r['frac_neg_loss']:.3f}     {r['frac_neg_rank']:.3f}     {r['frac_neg_entropy']:.3f}     "
            f"{r['A_loss']:>1d}      {r['A_rank']:>1d}      {r['A_entropy']:>1d}    {r['A_any']:>1d}"
        )

    # Save CSV for later
    try:
        import csv

        csv_path = results_dir / "batch_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {csv_path}")
    except Exception as e:
        print(f"[WARN] Could not save CSV: {e}")


if __name__ == "__main__":
    main()

