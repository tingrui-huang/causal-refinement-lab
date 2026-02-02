"""
Trainer utilities.

This repo historically used script-driven training (`train.py`, `train_complete.py`).
To support more adaptive strategies, we provide a lightweight `CausalTrainer`
wrapper that can analyze dataset statistics at initialization time and expose
reliability-aware hyperparameters (e.g., baseline suppression).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DatasetStatistics:
    """
    Container for dataset-level statistics used by adaptive training strategies.
    """

    # (n_vars,) LOCAL baseline index per variable (0..n_states(var)-1)
    baseline_state_indices_local: torch.LongTensor
    # (n_vars,) GLOBAL baseline state index per variable (index into flattened one-hot)
    baseline_state_indices_global: torch.LongTensor
    # scalar in [0,1]: fraction of all variable-state activations that are baseline
    global_baseline_ratio: float
    # (n_vars,) number of states for each variable
    n_states_per_var: torch.LongTensor
    # (n_vars, max_states): per-variable frequency for each LOCAL state index
    # (padded with zeros for variables with fewer than max_states states)
    state_frequencies: torch.FloatTensor
    # (n_total_states,) global frequency of each state
    global_state_frequencies: torch.FloatTensor


class CausalTrainer:
    """
    Minimal trainer wrapper to host adaptive, dataset-driven configuration.

    NOTE: This does not replace the existing training scripts; it is meant as
    a shared place to compute dataset statistics and store derived hyperparams.
    """

    def __init__(
        self,
        var_structure: Dict,
        dataset_tensor: torch.Tensor,
        config: Optional[Dict] = None,
    ) -> None:
        self.var_structure = var_structure
        self.config = config or {}

        # Filled by setup/_analyze_dataset_statistics
        self.dataset_stats: Optional[DatasetStatistics] = None

        # Reliability-aware knobs (set in setup)
        self.gamma_base: float = 1.0
        self.gamma_noise: float = 0.3
        self.noise_threshold: float = 1e-4
        self.label_smoothing: float = float(self.config.get("label_smoothing", 0.1))

        self.setup(dataset_tensor)

    def setup(self, dataset_tensor: torch.Tensor) -> None:
        """
        Prepare trainer state given the dataset tensor (one-hot state matrix).
        """
        self.dataset_stats = self._analyze_dataset_statistics(dataset_tensor)

        # Spec-required hyperparameter rules
        if self.dataset_stats.global_baseline_ratio > 0.6:
            # Enable suppression, but use a safer default than 0.1 to avoid over-suppressing signal.
            # Can be overridden via config['gamma_base_suppressed'].
            self.gamma_base = float(self.config.get("gamma_base_suppressed", 0.5))
        else:
            self.gamma_base = 1.0  # balanced dataset, no suppression

        self.gamma_noise = float(self.config.get("gamma_noise", 0.3))
        self.noise_threshold = float(self.config.get("noise_threshold", 1e-4))
        print(
            "[OK] Adaptive Strategy Active: "
            f"gamma_base={self.gamma_base}, gamma_noise={self.gamma_noise}, "
            f"noise_threshold={self.noise_threshold}, label_smoothing={self.label_smoothing}"
        )

    def compute_adaptive_recon_loss(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Reliability-Aware reconstruction loss with three-level weighting.

        Inputs:
          - pred_logits: (B, N_vars, N_states_max)
          - targets: (B, N_vars) integer indices of the true state (LOCAL index per variable)

        Weighting priority (highest to lowest):
          A) Baseline: target == baseline_state_indices_local[var] -> weight = gamma_base
          B) Noise:    state_frequencies[var, target] < noise_threshold -> weight = gamma_noise
          C) Signal:   otherwise -> weight = 1.0
        """
        if self.dataset_stats is None:
            raise RuntimeError("dataset_stats is not initialized; call setup() or construct CausalTrainer with dataset_tensor.")

        if pred_logits.dim() != 3:
            raise ValueError(f"pred_logits must be (B, V, S); got {tuple(pred_logits.shape)}")
        if targets.dim() != 2:
            raise ValueError(f"targets must be (B, V); got {tuple(targets.shape)}")

        B, V, S = pred_logits.shape
        if targets.shape[0] != B or targets.shape[1] != V:
            raise ValueError(f"Shape mismatch: pred_logits is {(B, V, S)} but targets is {tuple(targets.shape)}")

        device = pred_logits.device
        targets = targets.to(device=device, dtype=torch.long)

        # ------------------------------------------------------------------
        # CE loss (reduction='none') over LOCAL state indices.
        # If variables have fewer than S states, mask invalid logits to -inf.
        # ------------------------------------------------------------------
        n_states_per_var = self.dataset_stats.n_states_per_var.to(device=device)
        if n_states_per_var.numel() != V:
            raise ValueError(
                f"dataset_stats.n_states_per_var has shape {tuple(n_states_per_var.shape)} but N_vars={V}. "
                "Make sure var_structure matches the logits/targets layout."
            )

        state_ids = torch.arange(S, device=device).view(1, 1, S)              # (1,1,S)
        valid = state_ids < n_states_per_var.view(1, V, 1)                    # (1,V,S) broadcast over B
        masked_logits = pred_logits.masked_fill(~valid, -1e9)

        # Per-item CE (optionally with label smoothing) over LOCAL indices.
        # We implement smoothing explicitly to keep reduction='none' and retain control over weighting.
        log_probs = F.log_softmax(masked_logits, dim=-1)  # (B,V,S)
        nll = -log_probs.gather(dim=-1, index=targets.view(B, V, 1)).squeeze(-1)  # (B,V)
        if self.label_smoothing and self.label_smoothing > 0.0:
            # IMPORTANT: only smooth over VALID states for each variable (ignore padded -inf slots).
            valid_f = valid.to(dtype=log_probs.dtype)  # (1,V,S) broadcast over B
            denom = valid_f.sum(dim=-1).clamp(min=1.0)  # (1,V)
            smooth = -(log_probs * valid_f).sum(dim=-1) / denom  # (B,V)
            eps = float(self.label_smoothing)
            ce = (1.0 - eps) * nll + eps * smooth
        else:
            ce = nll

        # ------------------------------------------------------------------
        # Build weight mask (start with 1.0; apply Noise; then Baseline override)
        # ------------------------------------------------------------------
        weight = torch.ones((B, V), device=device, dtype=ce.dtype)

        # Noise mask: lookup per-variable LOCAL frequency for each target index
        freqs = self.dataset_stats.state_frequencies.to(device=device, dtype=ce.dtype)  # (V, S_max)
        if freqs.shape[0] != V:
            raise ValueError(f"dataset_stats.state_frequencies has shape {tuple(freqs.shape)} but N_vars={V}")
        if freqs.shape[1] < S:
            raise ValueError(
                f"dataset_stats.state_frequencies has S_max={freqs.shape[1]} but pred_logits has S={S}. "
                "Ensure logits use the same max state dimension as dataset stats."
            )
        var_idx = torch.arange(V, device=device).view(1, V).expand(B, V)  # (B,V)
        target_freq = freqs[var_idx, targets]  # (B,V)
        noise_mask = target_freq < float(self.noise_threshold)
        if self.gamma_noise != 1.0:
            weight = torch.where(noise_mask, torch.full_like(weight, float(self.gamma_noise)), weight)

        # Baseline mask (override noise)
        baseline_local = self.dataset_stats.baseline_state_indices_local.to(device=device)  # (V,)
        baseline_mask = targets.eq(baseline_local.view(1, V))
        if self.gamma_base != 1.0:
            weight = torch.where(baseline_mask, torch.full_like(weight, float(self.gamma_base)), weight)

        return (ce * weight).mean()

    @torch.no_grad()
    def _analyze_dataset_statistics(self, dataset_tensor: torch.Tensor) -> DatasetStatistics:
        """
        Analyze baseline dominance and per-state frequencies.

        Args:
            dataset_tensor: (n_samples, n_total_states) one-hot state matrix.
                Each sample should have exactly one active state per variable
                (i.e., sum over states == n_vars).

        Returns:
            DatasetStatistics with:
              - baseline_state_indices: (n_vars,) global indices
              - global_baseline_ratio: scalar float
              - state_frequencies: (n_vars, n_total_states) float frequencies
              - global_state_frequencies: (n_total_states,) float frequencies
        """
        if dataset_tensor.dim() != 2:
            raise ValueError(f"dataset_tensor must be 2D (n_samples, n_states); got {tuple(dataset_tensor.shape)}")

        x = dataset_tensor
        if not torch.is_floating_point(x):
            x = x.float()

        n_samples, n_total_states = x.shape

        var_names = self.var_structure["variable_names"]
        var_to_states = self.var_structure["var_to_states"]
        n_vars = len(var_names)

        # Global frequency per state: P(state=1) across samples
        global_state_freq = x.mean(dim=0)  # (n_total_states,)

        # Per-variable state counts, baseline indices (local+global), and per-variable local frequencies
        n_states_per_var = torch.empty(n_vars, dtype=torch.long, device=x.device)
        baseline_local = torch.empty(n_vars, dtype=torch.long, device=x.device)
        baseline_global = torch.empty(n_vars, dtype=torch.long, device=x.device)

        max_states = 0
        per_var_freqs_list = []
        for i, var in enumerate(var_names):
            state_idxs = var_to_states[var]  # global indices in local order
            idx = torch.as_tensor(state_idxs, dtype=torch.long, device=x.device)
            n_states_per_var[i] = idx.numel()
            max_states = max(max_states, int(idx.numel()))

            local_freqs = global_state_freq.index_select(0, idx).to(torch.float32)  # (n_states(var),)
            per_var_freqs_list.append(local_freqs)

            best_local = torch.argmax(local_freqs)
            baseline_local[i] = best_local
            baseline_global[i] = idx[best_local]

        # Global baseline ratio: fraction of active variable-state slots that are baseline
        # Total active slots per sample is ~ n_vars for one-hot per variable.
        baseline_hits = x.index_select(1, baseline_global).sum()  # scalar
        denom = float(n_samples * n_vars) if n_samples > 0 else 1.0
        global_baseline_ratio = float(baseline_hits.item() / denom)

        # Per-variable LOCAL state frequencies in a padded (n_vars, max_states) table.
        state_frequencies = torch.zeros((n_vars, max_states), dtype=torch.float32, device=x.device)
        for i in range(n_vars):
            freqs_i = per_var_freqs_list[i]
            state_frequencies[i, : freqs_i.numel()] = freqs_i

        return DatasetStatistics(
            baseline_state_indices_local=baseline_local,
            baseline_state_indices_global=baseline_global,
            global_baseline_ratio=global_baseline_ratio,
            n_states_per_var=n_states_per_var,
            state_frequencies=state_frequencies,
            global_state_frequencies=global_state_freq.to(torch.float32),
        )

