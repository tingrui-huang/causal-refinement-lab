"""
Summarize per-seed SHD metrics for a dataset (aligned with modules/evaluator.py),
and include per-seed runtime if available.

Runtime fields:
  - seed_train_runtime_seconds: measured by run_multi_seed_random_prior.py around train_complete(cfg)

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/seed_shd_summary.py --dataset hailfinder
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/seed_shd_summary.py --dataset win95pts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
NSR_DIR = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NSR_DIR) not in sys.path:
    sys.path.insert(0, str(NSR_DIR))

import importlib.util

_spec = importlib.util.spec_from_file_location("unified_config", REPO_ROOT / "config.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to import unified config.py via importlib")
uconfig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uconfig)

from modules.data_loader import CausalDataLoader
from modules.evaluator import CausalGraphEvaluator
from modules.metrics import compute_unresolved_ratio
from modules.prior_builder import PriorBuilder


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    args = ap.parse_args()

    ds = args.dataset
    ds_cfg = uconfig.DATASET_CONFIGS[ds]
    loader = CausalDataLoader(str(ds_cfg["data_path"]), str(ds_cfg["metadata_path"]))
    var_structure = loader.get_variable_structure()
    evaluator = CausalGraphEvaluator(
        str(ds_cfg["ground_truth_path"]),
        var_structure,
        ground_truth_type=ds_cfg.get("ground_truth_type", "edge_list"),
    )

    # Build the same block structure used by training, from the (latest) FCI skeleton.
    # This is needed to compute the Symmetry-Unresolved ratio (bidirectional/symmetric strong weights).
    fci_csv = uconfig._auto_detect_latest_file("edges_FCI_*.csv", uconfig.FCI_OUTPUT_DIR / ds)
    if not fci_csv:
        raise FileNotFoundError(f"No edges_FCI_*.csv found under {uconfig.FCI_OUTPUT_DIR / ds}")
    prior_builder = PriorBuilder(var_structure, dataset_name=ds)
    skeleton_mask = prior_builder.build_skeleton_mask_from_fci(str(fci_csv))
    blocks = prior_builder.build_block_structure(skeleton_mask)

    # Threshold used for "strong" direction in unresolved computation.
    # Match run_multi_seed_random_prior.py defaults: andes uses 0.08, others use 0.1.
    edge_threshold = 0.08 if ds == "andes" else 0.1

    base = NSR_DIR / "results" / "stability_runs" / ds
    rows = []
    for seed_dir in sorted(base.glob("seed_*")):
        edges_path = seed_dir / "complete_edges.txt"
        if not edges_path.exists():
            continue
        edges = parse_edges_txt(edges_path)
        m = evaluator.evaluate(edges)

        # Symmetry-Unresolved computed from adjacency (same definition as training monitors)
        adj_path = seed_dir / "complete_adjacency.pt"
        sym = None
        if adj_path.exists():
            A = __import__("torch").load(adj_path, map_location="cpu")
            sym = compute_unresolved_ratio(A, blocks, threshold=edge_threshold)

        rows.append(
            {
                "seed": seed_dir.name,
                "learned_edges": len(edges),
                "skeleton_shd": int(m.get("skeleton_shd", -1)),
                "full_shd": int(m.get("full_shd", -1)),
                "orientation_accuracy": float(m.get("orientation_accuracy", 0.0)),
                "seed_train_runtime_seconds": read_runtime_seconds(seed_dir),
                "symmetry_unresolved_ratio": None if sym is None else float(sym["unresolved_ratio"]),
                "symmetry_unresolved_unresolved": None if sym is None else int(sym["unresolved"]),
                "symmetry_unresolved_resolved": None if sym is None else int(sym["resolved"]),
                "symmetry_unresolved_no_direction": None if sym is None else int(sym["no_direction"]),
                "symmetry_unresolved_total_pairs": None if sym is None else int(sym["total_pairs"]),
            }
        )

    print("dataset:", ds)
    print(f"symmetry_unresolved_threshold={edge_threshold}  (what: compute_unresolved_ratio on complete_adjacency.pt)")
    print("seed\tlearned\tfull_shd\tskel_shd\torient_acc\tsym_unres_ratio\tseed_train_runtime_s")
    for r in rows:
        rt = "" if r["seed_train_runtime_seconds"] is None else f"{r['seed_train_runtime_seconds']:.2f}"
        su = "" if r["symmetry_unresolved_ratio"] is None else f"{r['symmetry_unresolved_ratio']:.4f}"
        print(
            f"{r['seed']}\t{r['learned_edges']}\t{r['full_shd']}\t{r['skeleton_shd']}\t"
            f"{r['orientation_accuracy']:.4f}\t{su}\t{rt}"
        )
    if rows:
        best = min(rows, key=lambda x: x["full_shd"])
        print(f"\nBEST (min full_shd): {best}")


if __name__ == "__main__":
    main()

