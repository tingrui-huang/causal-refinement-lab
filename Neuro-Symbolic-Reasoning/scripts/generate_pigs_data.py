"""
Generate PIGS dataset from BIF file (discrete) in a way that matches the unified pipeline.

Outputs (default n_samples=50000):
  - Neuro-Symbolic-Reasoning/data/pigs/pigs_data_<n>.csv        (one-hot for training)
  - Neuro-Symbolic-Reasoning/data/pigs/metadata.json           (state mappings; preserves BIF variable order)
  - <project_root>/pigs_data_variable.csv                      (variable-level numeric for FCI)

Key design choices (IMPORTANT):
  - Variable order is taken from the BIF "variable ..." declaration order (NOT alphabetical).
  - One-hot columns are ordered by variable order, then by state order from the BIF declaration.
  - Metadata is generated directly (without re-reading the huge CSV) and uses the same variable order.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_bif_variables_and_states(bif_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Parse BIF variable declaration order and state lists.

    Expects blocks like:
      variable X {
        type discrete [ 3 ] { 0, 1, 2 };
      }
    """
    text = bif_path.read_text(encoding="utf-8", errors="ignore")

    var_order: List[str] = []
    var_to_states: Dict[str, List[str]] = {}

    current_var: str | None = None

    # Parse line-by-line to avoid confusion with nested braces in the state list.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Start of variable block
        m_var = re.match(r"^variable\s+([A-Za-z0-9_]+)\s*\{\s*$", line)
        if m_var:
            current_var = m_var.group(1)
            var_order.append(current_var)
            continue

        if current_var is None:
            continue

        # State declaration line inside the current variable block
        # Example: type discrete [ 3 ] { 0, 1, 2 };
        if line.startswith("type") and "discrete" in line:
            m_states = re.search(r"\{\s*([^}]*)\s*\}", line)
            if not m_states:
                raise ValueError(f"Failed to parse state list for variable '{current_var}' in {bif_path}")

            states_raw = m_states.group(1)
            states = [s.strip() for s in states_raw.split(",") if s.strip() != ""]
            if not states:
                raise ValueError(f"Empty state list for variable '{current_var}' in {bif_path}")

            # NOTE: metadata_generator assumes VARIABLE_STATE and splits by the LAST '_'.
            if any("_" in s for s in states):
                raise ValueError(
                    f"State values for '{current_var}' contain '_' which breaks VARIABLE_STATE parsing: {states}"
                )

            var_to_states[current_var] = states
            continue

        # End of variable block
        if line == "}":
            current_var = None
            continue

    if not var_order:
        raise ValueError(f"Failed to parse any variables from: {bif_path}")

    missing_states = [v for v in var_order if v not in var_to_states]
    if missing_states:
        raise ValueError(f"Missing state lists for variables: {missing_states[:10]} (total {len(missing_states)})")

    return var_order, var_to_states


def best_effort_set_seed(seed: int) -> None:
    np.random.seed(int(seed))


def sample_from_bif(bif_path: Path, n_samples: int, seed: int) -> pd.DataFrame:
    """
    Sample observational data from a Bayesian network in BIF using pgmpy.
    """
    try:
        from pgmpy.readwrite import BIFReader
        from pgmpy.sampling import BayesianModelSampling
    except ImportError as e:
        raise RuntimeError("pgmpy is required. Install with: pip install pgmpy") from e

    best_effort_set_seed(seed)

    reader = BIFReader(str(bif_path))
    model = reader.get_model()

    sampler = BayesianModelSampling(model)
    # pgmpy doesn't consistently expose a random_state parameter across versions;
    # seeding numpy is best-effort reproducibility.
    df = sampler.forward_sample(size=int(n_samples), show_progress=False)
    return df


def write_variable_level_csv(
    df_raw: pd.DataFrame,
    var_order: List[str],
    var_to_states: Dict[str, List[str]],
    output_path: Path,
) -> None:
    """
    Write variable-level numeric CSV for FCI, preserving var_order.
    Codes are 0..K-1 matching the BIF state order.
    """
    mappings: Dict[str, Dict[str, int]] = {
        v: {state: i for i, state in enumerate(var_to_states[v])} for v in var_order
    }

    df_num = pd.DataFrame(index=df_raw.index)
    for v in var_order:
        s = df_raw[v].astype(str)
        df_num[v] = s.map(mappings[v]).astype("int16")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_num.to_csv(output_path, index=False)


def write_onehot_csv_chunked(
    df_raw: pd.DataFrame,
    var_order: List[str],
    var_to_states: Dict[str, List[str]],
    output_path: Path,
    *,
    chunk_rows: int = 5000,
) -> List[str]:
    """
    Write one-hot CSV in a stable column order without holding the full one-hot matrix in memory.
    Returns the column header list.
    """
    # Build header in deterministic order: var order then state order
    header: List[str] = []
    for v in var_order:
        header.extend([f"{v}_{state}" for state in var_to_states[v]])

    # Precompute state -> offset mapping per variable
    state_to_offset: Dict[str, Dict[str, int]] = {
        v: {state: i for i, state in enumerate(var_to_states[v])} for v in var_order
    }
    n_vars = len(var_order)
    widths = [len(var_to_states[v]) for v in var_order]
    col_offsets = np.cumsum([0] + widths[:-1]).tolist()  # start offset in one-hot blocks (per var)
    total_cols = sum(widths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write header once
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")

    n = len(df_raw)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        chunk = df_raw.iloc[start:end]

        out = np.zeros((end - start, total_cols), dtype=np.uint8)

        # Fill block by block
        for j, v in enumerate(var_order):
            vals = chunk[v].astype(str).to_numpy()
            mapping = state_to_offset[v]
            base = col_offsets[j]
            # Map each value to its within-variable index
            idx = np.fromiter((mapping[x] for x in vals), dtype=np.int32, count=len(vals))
            out[np.arange(len(vals)), base + idx] = 1

        df_out = pd.DataFrame(out, columns=header)
        df_out.to_csv(output_path, mode="a", index=False, header=False)

    return header


def write_metadata_json(
    *,
    dataset_name: str,
    var_order: List[str],
    var_to_states: Dict[str, List[str]],
    onehot_columns: List[str],
    n_samples: int,
    output_path: Path,
    source_file: str,
) -> None:
    state_mappings: Dict[str, Dict[str, str]] = {}
    for v in var_order:
        state_mappings[v] = {str(i): f"{v}_{state}" for i, state in enumerate(var_to_states[v])}

    metadata = {
        "dataset_name": dataset_name,
        "n_variables": len(var_order),
        "n_states": len(onehot_columns),
        "variable_names": var_order,  # CRITICAL: preserve original order
        "state_mappings": state_mappings,
        "data_format": "one_hot_csv",
        "source_file": source_file,
        "n_samples": int(n_samples),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--chunk_rows", type=int, default=5000)
    args = ap.parse_args()

    project_root = Path(__file__).parent.parent.parent
    ns_root = project_root / "Neuro-Symbolic-Reasoning"
    pigs_dir = ns_root / "data" / "pigs"
    bif_path = pigs_dir / "pigs.bif"

    if not bif_path.exists():
        raise FileNotFoundError(f"Missing BIF file: {bif_path}")

    # Default seed: unified config if available, else 42
    if args.seed is None:
        try:
            import sys

            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            import config as unified_config

            seed = int(getattr(unified_config, "RANDOM_SEED", 42))
        except Exception:
            seed = 42
    else:
        seed = int(args.seed)

    n_samples = int(args.n_samples)

    print("=" * 80)
    print("GENERATING PIGS DATASET")
    print("=" * 80)
    print(f"BIF:       {bif_path}")
    print(f"Samples:   {n_samples}")
    print(f"Seed:      {seed}")
    print(f"ChunkRows: {int(args.chunk_rows)}")
    print("=" * 80)

    print("\n[1/5] Parsing BIF variable order + states...")
    var_order, var_to_states = parse_bif_variables_and_states(bif_path)
    print(f"  Variables: {len(var_order)}")
    print(f"  First 10 variables: {var_order[:10]}")
    # sanity: ensure uniform 3-state if expected
    uniq_state_counts = sorted({len(v) for v in var_to_states.values()})
    print(f"  Unique state counts: {uniq_state_counts}")

    print("\n[2/5] Sampling observational data from BIF (pgmpy)...")
    df_raw = sample_from_bif(bif_path, n_samples=n_samples, seed=seed)
    # Reorder columns strictly to BIF order
    df_raw = df_raw[var_order]
    print(f"  Sampled shape: {df_raw.shape}")

    print("\n[3/5] Writing FCI variable-level CSV (numeric codes, BIF order)...")
    fci_out = project_root / "pigs_data_variable.csv"
    write_variable_level_csv(df_raw, var_order, var_to_states, fci_out)
    print(f"  Saved: {fci_out}")

    print("\n[4/5] Writing one-hot CSV for training (chunked, deterministic column order)...")
    onehot_out = pigs_dir / f"pigs_data_{n_samples}.csv"
    header = write_onehot_csv_chunked(
        df_raw,
        var_order,
        var_to_states,
        onehot_out,
        chunk_rows=int(args.chunk_rows),
    )
    print(f"  Saved: {onehot_out}")
    print(f"  One-hot columns: {len(header)}")

    print("\n[5/5] Writing metadata.json (no alphabetical sorting)...")
    metadata_out = pigs_dir / "metadata.json"
    write_metadata_json(
        dataset_name="pigs",
        var_order=var_order,
        var_to_states=var_to_states,
        onehot_columns=header,
        n_samples=n_samples,
        output_path=metadata_out,
        source_file=onehot_out.name,
    )
    print(f"  Saved: {metadata_out}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Generated:")
    print(f"  - {onehot_out}")
    print(f"  - {metadata_out}")
    print(f"  - {fci_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

