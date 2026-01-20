"""
Distribution diagnostics for pairwise deltas.

Computes, per dataset and per metric delta:
  - Shapiro-Wilk test (normality)
  - Skewness
  - Kurtosis (both excess and non-excess)
  - Breusch-Pagan and White heteroskedasticity tests on OLS:
        delta ~ 1 + k_u + k_v

Inputs:
  results/pairwise_asymmetry/<dataset>/deltas.csv
Columns expected:
  - loss_delta, nuc_delta, ent_delta
  - k_u, k_v

Run (repo root):
  python Neuro-Symbolic-Reasoning/asymmetry_analysis/distribution_diagnostics.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white


REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ["alarm", "win95pts", "andes", "sachs", "child", "hailfinder", "insurance"]
RESULTS_BASE = REPO_ROOT / "results" / "pairwise_asymmetry"
OUT_DIR = REPO_ROOT / "results" / "pairwise_distribution_tests"


METRICS = [
    ("loss_delta", "loss"),
    ("nuc_delta", "rank"),
    ("ent_delta", "entropy"),
]


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def run_tests_for_series(y: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """
    y: (n,)
    X: (n, p) exog WITHOUT constant; we'll add constant here.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)

    out: Dict[str, float] = {"n": n}
    if n < 3:
        # Too small for most tests
        out.update(
            {
                "shapiro_W": float("nan"),
                "shapiro_p": float("nan"),
                "skew": float("nan"),
                "kurtosis_excess": float("nan"),
                "kurtosis": float("nan"),
                "bp_lm": float("nan"),
                "bp_lm_p": float("nan"),
                "bp_f": float("nan"),
                "bp_f_p": float("nan"),
                "white_lm": float("nan"),
                "white_lm_p": float("nan"),
                "white_f": float("nan"),
                "white_f_p": float("nan"),
            }
        )
        return out

    # Shapiro-Wilk: scipy supports up to 5000, we are <=100 typically.
    try:
        W, p = stats.shapiro(y)
        out["shapiro_W"] = float(W)
        out["shapiro_p"] = float(p)
    except Exception:
        out["shapiro_W"] = float("nan")
        out["shapiro_p"] = float("nan")

    # Skew/Kurtosis
    out["skew"] = float(stats.skew(y, bias=False))
    out["kurtosis_excess"] = float(stats.kurtosis(y, fisher=True, bias=False))
    out["kurtosis"] = float(stats.kurtosis(y, fisher=False, bias=False))

    # Heteroskedasticity tests
    # Use complete cases for X as well
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # align to y length by taking first n rows (caller passes aligned arrays)
    # Filter finite rows
    mask = np.isfinite(y)
    # if y already filtered, mask is all True; keep for safety
    X2 = X[: n]
    mask2 = np.isfinite(X2).all(axis=1)
    y2 = y[mask2]
    X2 = X2[mask2]

    if y2.size < 5 or X2.shape[1] == 0:
        out.update(
            {
                "bp_lm": float("nan"),
                "bp_lm_p": float("nan"),
                "bp_f": float("nan"),
                "bp_f_p": float("nan"),
                "white_lm": float("nan"),
                "white_lm_p": float("nan"),
                "white_f": float("nan"),
                "white_f_p": float("nan"),
            }
        )
        return out

    exog = sm.add_constant(X2, has_constant="add")
    try:
        res = sm.OLS(y2, exog).fit()
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(res.resid, res.model.exog)
        out["bp_lm"] = float(bp_lm)
        out["bp_lm_p"] = float(bp_lm_p)
        out["bp_f"] = float(bp_f)
        out["bp_f_p"] = float(bp_f_p)

        w_lm, w_lm_p, w_f, w_f_p = het_white(res.resid, res.model.exog)
        out["white_lm"] = float(w_lm)
        out["white_lm_p"] = float(w_lm_p)
        out["white_f"] = float(w_f)
        out["white_f_p"] = float(w_f_p)
    except Exception:
        out.update(
            {
                "bp_lm": float("nan"),
                "bp_lm_p": float("nan"),
                "bp_f": float("nan"),
                "bp_f_p": float("nan"),
                "white_lm": float("nan"),
                "white_lm_p": float("nan"),
                "white_f": float("nan"),
                "white_f_p": float("nan"),
            }
        )

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for ds in DATASETS:
        csv_path = RESULTS_BASE / ds / "deltas.csv"
        if not csv_path.exists():
            print(f"[SKIP] missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "k_u" not in df.columns or "k_v" not in df.columns:
            print(f"[SKIP] missing k_u/k_v in {csv_path}")
            continue

        X = df[["k_u", "k_v"]].astype(float).to_numpy()

        for col, metric_name in METRICS:
            if col not in df.columns:
                continue
            y = df[col].astype(float).to_numpy()

            # Align + drop NaNs jointly
            mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
            y2 = y[mask]
            X2 = X[mask]

            stats_out = run_tests_for_series(y2, X2)
            rows.append(
                {
                    "dataset": ds,
                    "metric": metric_name,
                    **stats_out,
                }
            )

    # Save CSV
    out_csv = OUT_DIR / "diagnostics.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Print compact view
    print("\n" + "=" * 100)
    print("DISTRIBUTION DIAGNOSTICS (per dataset, per metric)")
    print("=" * 100)
    print("cols: n | shapiro_p | skew | kurtosis_excess | bp_lm_p | white_lm_p")
    for r in rows:
        print(
            f"{r['dataset']:<10s} {r['metric']:<8s} "
            f"n={int(r['n']):>4d} "
            f"shapiro_p={float(r['shapiro_p']):.3g} "
            f"skew={float(r['skew']):+.3f} "
            f"kurt_ex={float(r['kurtosis_excess']):+.3f} "
            f"bp_p={float(r['bp_lm_p']):.3g} "
            f"white_p={float(r['white_lm_p']):.3g}"
        )

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

