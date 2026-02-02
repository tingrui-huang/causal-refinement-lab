"""
Baseline-dominance diagnostics (Normal/Normal-like flooding check)

Motivation
----------
For very large graphs, training can be dominated by the most frequent ("baseline") states.
If baseline states are extremely common AND mostly uninformative (low KL to the marginal),
then downweighting baseline-related contributions in the *reconstruction / likelihood term*
can amplify weak asymmetry signals from rarer states.

Important:
  - This script does NOT assume any state is literally named "Normal".
  - Baseline is defined empirically per variable as the most frequent state in the data.

Metrics
-------
1) baseline_freq per variable:
      p(A = a0) where a0 is the most frequent state of A
2) baseline-baseline joint for random variable pairs:
      p(A = a0, B = b0)
3) baseline informativeness proxy (pairwise):
      KL( P(B | A=a0) || P(B) )
   If this is ~0 while p(A=a0) is large, then A's baseline state is mostly uninformative for B.

Run (repo root)
--------------
python Neuro-Symbolic-Reasoning/asymmetry_analysis/baseline_dominance_diagnostics.py --dataset pigs
python Neuro-Symbolic-Reasoning/asymmetry_analysis/baseline_dominance_diagnostics.py --dataset link
python Neuro-Symbolic-Reasoning/asymmetry_analysis/baseline_dominance_diagnostics.py --dataset alarm
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
NSR_DIR = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NSR_DIR) not in sys.path:
    sys.path.insert(0, str(NSR_DIR))

import config  # unified config at repo root
from modules.data_loader import CausalDataLoader


EPS = 1e-12


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + EPS
    q = np.asarray(q, dtype=np.float64) + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _baseline_state_index(X: torch.Tensor, idxs: Sequence[int]) -> int:
    # X is one-hot; baseline is argmax of marginal state frequency.
    counts = X[:, idxs].sum(dim=0)  # (k,)
    j = int(torch.argmax(counts).item())
    return int(idxs[j])


@dataclass(frozen=True)
class Summary:
    dataset: str
    n_samples: int
    n_vars: int
    n_states: int
    baseline_freq_mean: float
    baseline_freq_median: float
    baseline_freq_frac_ge_090: float
    p00_mean: float
    p00_median: float
    p00_frac_ge_070: float
    kl_mean: float
    kl_median: float
    kl_frac_le_1e3: float


def run(dataset: str, *, n_pairs: int = 2000, seed: int = 0) -> Summary:
    if dataset not in config.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}' in config.DATASET_CONFIGS")
    ds = config.DATASET_CONFIGS[dataset]

    loader = CausalDataLoader(data_path=str(ds["data_path"]), metadata_path=str(ds["metadata_path"]))
    X = loader.load_data()  # (N, n_states) float/bool-ish
    vs = loader.get_variable_structure()
    vt: Dict[str, List[int]] = vs["var_to_states"]
    vars_ = list(vt.keys())

    N = int(X.shape[0])
    n_states = int(X.shape[1])
    n_vars = len(vars_)

    # Baseline index per variable
    base_idx: Dict[str, int] = {v: _baseline_state_index(X, vt[v]) for v in vars_}

    # baseline freq per variable
    base_freqs = np.asarray([float(X[:, base_idx[v]].float().mean().item()) for v in vars_], dtype=np.float64)

    # sample random pairs
    rng = np.random.default_rng(int(seed))
    pairs: List[Tuple[str, str]] = []
    if n_vars >= 2:
        for _ in range(int(n_pairs)):
            a, b = rng.choice(vars_, size=2, replace=False)
            pairs.append((str(a), str(b)))

    # compute P00 and KL
    p00_list: List[float] = []
    kl_list: List[float] = []
    for a, b in pairs:
        ia0 = base_idx[a]
        ib0 = base_idx[b]
        p00 = float((X[:, ia0] * X[:, ib0]).float().mean().item())
        p00_list.append(p00)

        idx_b = vt[b]
        mask = X[:, ia0] > 0.5
        if int(mask.sum().item()) < 25:
            continue
        pb = X[:, idx_b].float().mean(dim=0).cpu().numpy()
        pb_a0 = X[mask][:, idx_b].float().mean(dim=0).cpu().numpy()
        kl_list.append(_kl(pb_a0, pb))

    p00_arr = np.asarray(p00_list, dtype=np.float64) if p00_list else np.asarray([np.nan])
    kl_arr = np.asarray(kl_list, dtype=np.float64) if kl_list else np.asarray([np.nan])

    return Summary(
        dataset=dataset,
        n_samples=N,
        n_vars=n_vars,
        n_states=n_states,
        baseline_freq_mean=float(np.nanmean(base_freqs)),
        baseline_freq_median=float(np.nanmedian(base_freqs)),
        baseline_freq_frac_ge_090=float(np.nanmean(base_freqs >= 0.90)),
        p00_mean=float(np.nanmean(p00_arr)),
        p00_median=float(np.nanmedian(p00_arr)),
        p00_frac_ge_070=float(np.nanmean(p00_arr >= 0.70)),
        kl_mean=float(np.nanmean(kl_arr)),
        kl_median=float(np.nanmedian(kl_arr)),
        kl_frac_le_1e3=float(np.nanmean(kl_arr <= 1e-3)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s = run(args.dataset, n_pairs=int(args.n_pairs), seed=int(args.seed))
    print("\n" + "=" * 90)
    print("BASELINE DOMINANCE DIAGNOSTICS")
    print("=" * 90)
    print(f"dataset:    {s.dataset}")
    print(f"samples:    {s.n_samples}")
    print(f"vars:       {s.n_vars}")
    print(f"states:     {s.n_states}")
    print("-" * 90)
    print(f"baseline_freq: mean={s.baseline_freq_mean:.3f}  median={s.baseline_freq_median:.3f}  frac>=0.90={s.baseline_freq_frac_ge_090:.3f}")
    print(f"P00 (baseline-baseline): mean={s.p00_mean:.3f}  median={s.p00_median:.3f}  frac>=0.70={s.p00_frac_ge_070:.3f}")
    print(f"KL(P(B|A=base)||P(B)):   mean={s.kl_mean:.4g}  median={s.kl_median:.4g}  frac<=1e-3={s.kl_frac_le_1e3:.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()

