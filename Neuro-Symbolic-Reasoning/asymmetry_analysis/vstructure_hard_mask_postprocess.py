"""
V-structure hard mask postprocessing (fast validation)

Goal:
  Quickly test whether enforcing v-structure collider directions can improve
  direction credibility (without retraining).

Given:
  - a learned state-level adjacency (adjacency.pt / complete_adjacency.pt) in [0,1]
  - an FCI CSV with edge_type (directed / partial / undirected / tail-tail / bidirected)

We:
  1) Recover candidate collider triples X -> Z <- Y from the FCI PAG encoding.
     Since your CSV doesn't contain explicit collider lists, we infer arrowheads using:
       - edge_type in {'partial','directed','bidirected'} on row (source,target)
         => treat as an arrowhead INTO 'target'
     Then a v-structure at Z exists if there are two distinct incoming arrowheads
     X *-> Z and Y *-> Z, and X and Y are NOT adjacent in the FCI skeleton.

  2) For each inferred forced direction X -> Z, we hard-mask the reverse direction
     at the STATE level by setting the whole block (Z_states, X_states) to 0.

This is intentionally conservative: we do not boost X->Z; we only forbid Z->X.

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/vstructure_hard_mask_postprocess.py ^
    --dataset andes ^
    --adjacency_path "Neuro-Symbolic-Reasoning/results/experiment_llm_vs_random/andes/random_prior/complete_adjacency.pt" ^
    --fci_csv "Neuro-Symbolic-Reasoning/data/andes/edges_FCI_20260108_212351.csv"

Then evaluate with Phase1:
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/phase1_error_diagnosis.py --dataset andes --adjacency_path <output_pt> --aggs mean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
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


ARROW_IN_TYPES = {"partial", "directed", "bidirected"}


def load_fci_edges(fci_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(fci_csv)
    # Normalize expected columns
    cols = {c.lower(): c for c in df.columns}
    if "source" not in cols or "target" not in cols:
        raise ValueError(f"FCI CSV missing source/target columns: {list(df.columns)}")
    if "edge_type" not in cols:
        raise ValueError(f"FCI CSV missing edge_type column: {list(df.columns)}")
    df = df.rename(columns={cols["source"]: "source", cols["target"]: "target", cols["edge_type"]: "edge_type"})
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["edge_type"] = df["edge_type"].astype(str)
    return df[["source", "target", "edge_type"]]


def build_skeleton_pairs(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for s, t in zip(df["source"].tolist(), df["target"].tolist()):
        a, b = (s, t) if s < t else (t, s)
        pairs.add((a, b))
    return pairs


def build_arrow_in_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    target -> set(sources) where row encodes an arrowhead into target.
    """
    incoming: Dict[str, Set[str]] = {}
    for s, t, et in zip(df["source"].tolist(), df["target"].tolist(), df["edge_type"].tolist()):
        et = et.lower().strip()
        if et in ARROW_IN_TYPES:
            incoming.setdefault(t, set()).add(s)
    return incoming


def infer_vstructure_forced_edges(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Returns forced directed edges (parent, child) inferred from v-structures.
    """
    skeleton = build_skeleton_pairs(df)
    incoming = build_arrow_in_map(df)

    forced: Set[Tuple[str, str]] = set()
    for z, pars in incoming.items():
        if len(pars) < 2:
            continue
        pars_list = sorted(list(pars))
        for i in range(len(pars_list)):
            for j in range(i + 1, len(pars_list)):
                x = pars_list[i]
                y = pars_list[j]
                a, b = (x, y) if x < y else (y, x)
                # Unshielded triple requirement: x and y not adjacent in skeleton
                if (a, b) in skeleton:
                    continue
                forced.add((x, z))
                forced.add((y, z))
    return forced


def apply_hard_mask(adjacency: torch.Tensor, var_to_states: Dict[str, List[int]], forced_edges: Set[Tuple[str, str]]) -> torch.Tensor:
    """
    For each forced edge (x->z), set reverse block (z->x) to 0.
    """
    out = adjacency.clone()
    for x, z in forced_edges:
        if x not in var_to_states or z not in var_to_states:
            continue
        idx_x = var_to_states[x]
        idx_z = var_to_states[z]
        # IMPORTANT: avoid chained advanced indexing (it writes into a temporary).
        # Use a single advanced-index assignment.
        iz = torch.as_tensor(idx_z, dtype=torch.long)
        ix = torch.as_tensor(idx_x, dtype=torch.long)
        out[iz[:, None], ix[None, :]] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="andes")
    ap.add_argument("--adjacency_path", type=str, required=True)
    ap.add_argument("--fci_csv", type=str, required=True)
    ap.add_argument("--out_path", type=str, default=None, help="Output .pt path; default under results/vstructure_mask/")
    args = ap.parse_args()

    dataset = args.dataset
    if dataset not in uconfig.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'")
    ds_cfg = uconfig.DATASET_CONFIGS[dataset]
    metadata_path = Path(ds_cfg["metadata_path"])
    data_path = Path(ds_cfg["data_path"])

    adjacency_path = Path(args.adjacency_path)
    fci_csv = Path(args.fci_csv)
    if not adjacency_path.exists():
        raise FileNotFoundError(adjacency_path)
    if not fci_csv.exists():
        raise FileNotFoundError(fci_csv)

    # Load var_structure (metadata-only; don't call load_data)
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()
    var_to_states = var_structure["var_to_states"]

    # Load adjacency
    adjacency = torch.load(adjacency_path, map_location="cpu")
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.tensor(adjacency)

    df = load_fci_edges(fci_csv)
    forced = infer_vstructure_forced_edges(df)

    masked = apply_hard_mask(adjacency, var_to_states, forced)

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_dir = REPO_ROOT / "results" / "vstructure_mask" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{adjacency_path.stem}_vstructure_masked.pt"

    torch.save(masked, out_path)

    # Report stats
    n_forced = sum(1 for (x, z) in forced if x in var_to_states and z in var_to_states)
    print("\n" + "=" * 80)
    print("V-STRUCTURE HARD MASK POSTPROCESS")
    print("=" * 80)
    print(f"dataset:        {dataset}")
    print(f"adjacency_in:   {adjacency_path}")
    print(f"fci_csv:        {fci_csv}")
    print(f"forced_edges:   {n_forced} (variable-level)")
    print(f"adjacency_out:  {out_path}")
    print("\nNext (Phase1 before/after):")
    print(f"  python Neuro-Symbolic-Reasoning/asymmetry_analysis/phase1_error_diagnosis.py --dataset {dataset} --adjacency_path \"{adjacency_path}\" --aggs mean")
    print(f"  python Neuro-Symbolic-Reasoning/asymmetry_analysis/phase1_error_diagnosis.py --dataset {dataset} --adjacency_path \"{out_path}\" --aggs mean")


if __name__ == "__main__":
    main()

