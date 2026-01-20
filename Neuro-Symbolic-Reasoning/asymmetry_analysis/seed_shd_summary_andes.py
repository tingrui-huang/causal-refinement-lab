"""
Compute aligned SHD metrics for each multi-seed run (Andes) using the project's evaluator.

This helps interpret the "5+random seeds" column:
  - best-of-5 (min full_shd)?
  - average?

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/seed_shd_summary_andes.py
"""

from __future__ import annotations

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


DATASET = "andes"
SEED_RUNS_DIR = NSR_DIR / "results" / "stability_runs" / DATASET


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


def main():
    ds_cfg = uconfig.DATASET_CONFIGS[DATASET]
    loader = CausalDataLoader(str(ds_cfg["data_path"]), str(ds_cfg["metadata_path"]))
    var_structure = loader.get_variable_structure()
    evaluator = CausalGraphEvaluator(
        str(ds_cfg["ground_truth_path"]),
        var_structure,
        ground_truth_type=ds_cfg.get("ground_truth_type", "edge_list"),
    )

    rows = []
    for seed_dir in sorted(SEED_RUNS_DIR.glob("seed_*")):
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
                "edge_f1": float(m.get("edge_f1", 0.0)),
                "directed_f1": float(m.get("directed_f1", 0.0)),
            }
        )

    print("seed\tlearned\tskel_shd\tfull_shd\torient_acc\tedge_f1\tdirected_f1")
    for r in rows:
        print(
            f"{r['seed']}\t{r['learned_edges']}\t{r['skeleton_shd']}\t{r['full_shd']}\t"
            f"{r['orientation_accuracy']:.4f}\t{r['edge_f1']:.4f}\t{r['directed_f1']:.4f}"
        )

    if rows:
        best = min(rows, key=lambda x: x["full_shd"])
        print(f"\nBEST (min full_shd): {best}")


if __name__ == "__main__":
    main()

