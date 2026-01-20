"""
Pairwise True-vs-Reverse Asymmetry Experiment (win95pts / andes)

Goal:
  Verify whether the TRUE causal direction (A->B from GT) differs from the REVERSE (B->A)
  along three "physics" dimensions:
    1) Prediction error (cross-entropy)
    2) Mechanism complexity (nuclear norm of W)
    3) Predictive determinism (mean predictive entropy)

Key design:
  - Do NOT train the full graph. Fit tiny pairwise softmax-linear models per edge.
  - Weight matrix is k_A x k_B (variable-specific number of states from metadata),
    NOT fixed 3x3.
  - Strict collider avoidance: skip edges that participate as an incoming edge of a
    v-structure u -> v <- w where u and w are not adjacent in GT skeleton.

Run:
  From repo root:
    python Neuro-Symbolic-Reasoning/asymmetry_analysis/pairwise_true_vs_reverse.py --dataset win95pts
    python Neuro-Symbolic-Reasoning/asymmetry_analysis/pairwise_true_vs_reverse.py --dataset andes

  Notes:
    - By default uses strict v-structure collider filtering, which can drastically reduce
      the number of eligible edges (you may end up with << n_edges).
    - To disable collider filtering: add --collider_filter none

Outputs:
  results/pairwise_asymmetry/<dataset>/
    - deltas.csv
    - summary.txt
    - money_plot_hist.png
    - money_plot_box.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# User-facing defaults (easy to edit without touching CLI)
# --------------------------------------------------------------------------------------
DEFAULT_DATASET = "win95pts"  # "win95pts" | "andes"
DEFAULT_RUN_BOTH = False
DEFAULT_COLLIDER_FILTER = "none"  # "none" | "strict"
DEFAULT_N_EDGES = 100
DEFAULT_EPOCHS = 150
DEFAULT_LR = 0.05
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cpu"  # "cpu" | "cuda"


# --------------------------------------------------------------------------------------
# Imports: make this runnable from repo root OR from Neuro-Symbolic-Reasoning/
# --------------------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
NSR_DIR = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NSR_DIR) not in sys.path:
    sys.path.insert(0, str(NSR_DIR))

import config  # unified config at repo root
from modules.data_loader import CausalDataLoader
from modules.ground_truth_loader import GroundTruthLoader


EPS = 1e-9


@dataclass(frozen=True)
class FitResult:
    loss: float
    nuclear_norm: float
    entropy: float
    k_in: int
    k_out: int


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Best-effort determinism (ok if CPU-only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_entropy_from_probs(probs: torch.Tensor) -> float:
    """
    probs: (N, K) row-stochastic probabilities
    returns mean entropy over samples
    """
    ent = -(probs * torch.log(probs + EPS)).sum(dim=1).mean()
    return float(ent.item())


def compute_nuclear_norm(W: torch.Tensor) -> float:
    """
    W: (K_in, K_out) weight matrix (rectangular allowed)
    """
    # torch.norm(..., p='nuc') supports matrices; keep it simple.
    return float(torch.norm(W, p="nuc").item())


def onehot_block(data: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """
    Extract one-hot block for a variable.
    data: (N, n_states)
    returns: (N, k_var)
    """
    return data[:, indices]


def onehot_to_class_index(onehot: torch.Tensor) -> torch.Tensor:
    """
    onehot: (N, K) (should be 0/1 with exactly one '1' per row)
    returns: (N,) long class index
    """
    return onehot.argmax(dim=1).long()


def fit_pairwise_softmax(
    X_in: torch.Tensor,
    y_out: torch.Tensor,
    *,
    epochs: int = 150,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Softmax linear model: logits = X_in @ W, with W shape (K_in, K_out)
    X_in is one-hot (N, K_in), y_out is class index (N,)
    Returns: (W, probs, final_loss)
    """
    set_seed(seed)

    X_in = X_in.to(device)
    y_out = y_out.to(device)

    k_in = int(X_in.shape[1])
    k_out = int(torch.max(y_out).item()) + 1

    W = torch.nn.Parameter(torch.randn(k_in, k_out, device=device) * 0.01)
    optimizer = torch.optim.Adam([W], lr=lr, weight_decay=float(weight_decay))
    criterion = torch.nn.CrossEntropyLoss()

    logits = None
    loss = None
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = X_in @ W  # (N, K_out)
        loss = criterion(logits, y_out)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = X_in @ W
        probs = torch.softmax(logits, dim=1)
        final_loss = float(criterion(logits, y_out).item())

    return W.detach().cpu(), probs.detach().cpu(), final_loss


def build_gt_maps(gt_edges: Set[Tuple[str, str]]) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]]]:
    """
    Returns:
      parents: child -> set(parents)
      skeleton_adj: undirected adjacency as a set of normalized pairs (min, max)
    """
    parents: Dict[str, Set[str]] = {}
    skeleton_adj: Set[Tuple[str, str]] = set()

    for u, v in gt_edges:
        parents.setdefault(v, set()).add(u)
        a, b = (u, v) if u < v else (v, u)
        skeleton_adj.add((a, b))

    return parents, skeleton_adj


def is_strict_collider_incoming_edge(
    u: str,
    v: str,
    *,
    parents: Dict[str, Set[str]],
    skeleton_adj: Set[Tuple[str, str]],
) -> bool:
    """
    True if edge u->v is part of a v-structure u->v<-w for some w,
    where u and w are NOT adjacent in the GT skeleton.
    """
    pa_v = parents.get(v, set())
    if len(pa_v) < 2:
        return False
    for w in pa_v:
        if w == u:
            continue
        a, b = (u, w) if u < w else (w, u)
        if (a, b) not in skeleton_adj:
            return True
    return False


def sample_edges(
    gt_edges: Set[Tuple[str, str]],
    var_structure: Dict,
    *,
    n_edges: int,
    rng: np.random.Generator,
    collider_filter: str = "strict",
) -> List[Tuple[str, str]]:
    """
    Sample up to n_edges GT edges (u->v) aligned to metadata.

    collider_filter:
      - "strict": drop incoming edges u->v that are part of a GT v-structure u->v<-w
                  where u and w are not adjacent in GT skeleton.
      - "none":   no collider filtering (keeps all aligned GT edges).
    """
    collider_filter = collider_filter.lower().strip()
    if collider_filter not in {"strict", "none"}:
        raise ValueError(f"Unsupported collider_filter='{collider_filter}'. Use 'strict' or 'none'.")

    parents, skeleton_adj = build_gt_maps(gt_edges)
    var_names = set(var_structure["var_to_states"].keys())

    candidates: List[Tuple[str, str]] = []
    for u, v in gt_edges:
        if u not in var_names or v not in var_names:
            continue
        if collider_filter == "strict":
            if is_strict_collider_incoming_edge(u, v, parents=parents, skeleton_adj=skeleton_adj):
                continue
        candidates.append((u, v))

    if not candidates:
        return []

    rng.shuffle(candidates)
    return candidates[: min(n_edges, len(candidates))]


def run_dataset(
    dataset: str,
    *,
    n_edges: int = 100,
    epochs: int = 150,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    collider_filter: str = "strict",
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Runs the pairwise experiment for a dataset.
    Returns output directory path.
    """
    if dataset not in config.DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Not in config.DATASET_CONFIGS.")
    ds_cfg = config.DATASET_CONFIGS[dataset]

    data_path = Path(ds_cfg["data_path"])
    metadata_path = Path(ds_cfg["metadata_path"])
    gt_path = Path(ds_cfg["ground_truth_path"])
    gt_type = ds_cfg.get("ground_truth_type", "bif")

    if out_dir is None:
        out_dir = REPO_ROOT / "results" / "pairwise_asymmetry" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PAIRWISE TRUE-vs-REVERSE: {dataset}")
    print("=" * 80)
    print(f"Data:      {data_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"GT:        {gt_path} (type={gt_type})")
    print(
        f"Params:    n_edges={n_edges}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, "
        f"seed={seed}, device={device}, collider_filter={collider_filter}"
    )
    print(f"Output:    {out_dir}")

    # Load observations (one-hot) and variable structure
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    observations = loader.load_data()  # (N, n_states)
    var_structure = loader.get_variable_structure()

    # Load GT edges
    gt_loader = GroundTruthLoader(str(gt_path), ground_truth_type=gt_type)
    gt_edges = gt_loader.get_edges()
    if not gt_edges:
        raise RuntimeError(f"Failed to load GT edges from {gt_path} (type={gt_type})")

    rng = np.random.default_rng(int(seed))
    sampled = sample_edges(
        gt_edges,
        var_structure,
        n_edges=n_edges,
        rng=rng,
        collider_filter=collider_filter,
    )
    if not sampled:
        raise RuntimeError("No valid GT edges left after filtering + var name alignment.")

    # Track how many got filtered (for transparency)
    parents, skeleton_adj = build_gt_maps(gt_edges)
    var_names = set(var_structure["var_to_states"].keys())
    total_align = sum(1 for (u, v) in gt_edges if u in var_names and v in var_names)
    total_collider_filtered = 0
    if collider_filter == "strict":
        total_collider_filtered = sum(
            1
            for (u, v) in gt_edges
            if u in var_names
            and v in var_names
            and is_strict_collider_incoming_edge(u, v, parents=parents, skeleton_adj=skeleton_adj)
        )

    print("\nEdge sampling:")
    print(f"  GT edges total:                 {len(gt_edges)}")
    print(f"  GT edges aligned to metadata:   {total_align}")
    if collider_filter == "strict":
        print(f"  Filtered (strict collider):     {total_collider_filtered}")
        print(f"  Candidates after filtering:     {total_align - total_collider_filtered}")
    else:
        print(f"  Collider filtering:             none")
        print(f"  Candidates:                     {total_align}")
    print(f"  Sampled edges:                  {len(sampled)}")

    # Run per-edge pairwise fits
    rows = []
    deltas_loss = []
    deltas_rank = []
    deltas_ent = []

    for i, (u, v) in enumerate(sampled):
        idx_u = var_structure["var_to_states"][u]
        idx_v = var_structure["var_to_states"][v]
        X_u = onehot_block(observations, idx_u)
        y_v = onehot_to_class_index(onehot_block(observations, idx_v))

        # Forward: u -> v
        W_f, P_f, L_f = fit_pairwise_softmax(
            X_u,
            y_v,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed * 100000 + i * 2 + 0,
            device=device,
        )
        R_f = compute_nuclear_norm(W_f)
        E_f = compute_entropy_from_probs(P_f)

        # Reverse: v -> u
        X_v = onehot_block(observations, idx_v)
        y_u = onehot_to_class_index(onehot_block(observations, idx_u))
        W_r, P_r, L_r = fit_pairwise_softmax(
            X_v,
            y_u,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed * 100000 + i * 2 + 1,
            device=device,
        )
        R_r = compute_nuclear_norm(W_r)
        E_r = compute_entropy_from_probs(P_r)

        d_loss = L_f - L_r
        d_rank = R_f - R_r
        d_ent = E_f - E_r

        deltas_loss.append(d_loss)
        deltas_rank.append(d_rank)
        deltas_ent.append(d_ent)

        rows.append(
            {
                "edge_u": u,
                "edge_v": v,
                "k_u": len(idx_u),
                "k_v": len(idx_v),
                "loss_fwd": L_f,
                "loss_rev": L_r,
                "loss_delta": d_loss,
                "nuc_fwd": R_f,
                "nuc_rev": R_r,
                "nuc_delta": d_rank,
                "ent_fwd": E_f,
                "ent_rev": E_r,
                "ent_delta": d_ent,
            }
        )

        if (i + 1) % max(1, math.floor(len(sampled) / 10)) == 0 or (i + 1) == len(sampled):
            print(f"  Progress: {i+1:4d}/{len(sampled)} edges")

    # Save CSV
    import csv

    csv_path = out_dir / "deltas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Simple summary stats (no scipy dependency)
    def summarize(name: str, xs: List[float]) -> str:
        arr = np.asarray(xs, dtype=np.float64)
        frac_neg = float((arr < 0).mean())
        return (
            f"{name}:\n"
            f"  n={arr.size}\n"
            f"  mean={arr.mean():+.6f}\n"
            f"  median={np.median(arr):+.6f}\n"
            f"  frac(delta<0)={frac_neg:.3f}\n"
        )

    summary_txt = (
        f"PAIRWISE TRUE-vs-REVERSE SUMMARY ({dataset})\n"
        f"n_edges={len(sampled)}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, seed={seed}\n"
        f"Collider filter: {collider_filter}\n\n"
        + summarize("Loss delta (CE) [forward - reverse]", deltas_loss)
        + "\n"
        + summarize("Nuclear norm delta [forward - reverse]", deltas_rank)
        + "\n"
        + summarize("Entropy delta [forward - reverse]", deltas_ent)
    )

    (out_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")

    # Money plots
    # 1) Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, xs, title in [
        (axes[0], deltas_loss, "Δ Loss (CE)\nforward - reverse"),
        (axes[1], deltas_rank, "Δ Nuclear Norm\nforward - reverse"),
        (axes[2], deltas_ent, "Δ Entropy\nforward - reverse"),
    ]:
        ax.hist(xs, bins=25, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(title)
    fig.suptitle(f"Pairwise True-vs-Reverse Asymmetry ({dataset})")
    fig.tight_layout()
    fig.savefig(out_dir / "money_plot_hist.png", dpi=200)
    plt.close(fig)

    # 2) Boxplots
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # Matplotlib 3.9+: "labels" renamed to "tick_labels"
    ax.boxplot(
        [deltas_loss, deltas_rank, deltas_ent],
        tick_labels=["DeltaLoss", "DeltaNuc", "DeltaEnt"],
        showmeans=True,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"Pairwise deltas (forward - reverse) ({dataset})")
    fig.tight_layout()
    fig.savefig(out_dir / "money_plot_box.png", dpi=200)
    plt.close(fig)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'summary.txt'}")
    print(f"  {out_dir / 'money_plot_hist.png'}")
    print(f"  {out_dir / 'money_plot_box.png'}")

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset key in config.DATASET_CONFIGS (ignored if --run_both)",
    )
    parser.add_argument("--run_both", action="store_true", default=DEFAULT_RUN_BOTH, help="Run win95pts then andes")
    parser.add_argument("--collider_filter", type=str, default=DEFAULT_COLLIDER_FILTER, choices=["strict", "none"])
    parser.add_argument("--n_edges", type=int, default=DEFAULT_N_EDGES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.run_both:
        for ds in ["win95pts", "andes"]:
            run_dataset(
                ds,
                n_edges=args.n_edges,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                device=args.device,
                collider_filter=args.collider_filter,
            )
    else:
        run_dataset(
            args.dataset,
            n_edges=args.n_edges,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            collider_filter=args.collider_filter,
        )


if __name__ == "__main__":
    main()

