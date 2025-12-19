"""
Step 5 (v3 - PRIOR LOSS): Neural LP with LLM Prior Loss

GEMINI'S IMPROVEMENT:
Add Prior Loss to penalize deviation from LLM suggestions:
    Loss = MSE + λ||W - W_LLM||²

This makes it harder to reverse LLM suggestions - requires stronger data evidence.

Key Changes from v2:
1. Added compute_prior_loss() function
2. Modified Phase 2 to include prior_lambda parameter
3. Prior loss penalizes: ||W - W_LLM||² (L2 distance from LLM prior)

Expected Result:
- Fewer false corrections (like BP -> CO)
- LLM suggestions only reversed with very strong data evidence
- Better balance between data-driven and knowledge-driven learning
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from neural_lp import NeuralLP


def load_real_alarm_data(csv_path):
    """Load and preprocess real ALARM data."""
    print("=" * 70)
    print("LOADING REAL ALARM DATA")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples from {csv_path}")
    print(f"Variables: {df.shape[1]}")
    
    var_file = Path('data/alarm_variables.txt')
    if var_file.exists():
        with open(var_file, 'r') as f:
            var_names = [line.strip().split('\t')[1] for line in f if line.strip()]
        print(f"\nUsing variable order from {var_file}")
    else:
        var_names = df.columns.tolist()
        print(f"\nUsing variable order from CSV columns")
        with open(var_file, 'w') as f:
            for i, var in enumerate(var_names):
                f.write(f"{i}\t{var}\n")
    
    csv_vars = set(df.columns)
    expected_vars = set(var_names)
    
    if csv_vars != expected_vars:
        print(f"\n[WARNING] Variable mismatch!")
        print(f"  In CSV but not expected: {csv_vars - expected_vars}")
        print(f"  Expected but not in CSV: {expected_vars - csv_vars}")
        var_names = [v for v in var_names if v in csv_vars]
        print(f"  Using {len(var_names)} common variables")
    
    df = df[var_names]
    
    print(f"\nData shape: {df.shape}")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    print(f"\n" + "-" * 70)
    print("PREPROCESSING: DISCRETE -> ORDINAL CONTINUOUS")
    print("-" * 70)
    print("Strategy: Treat categories (0, 1, 2) as continuous values")
    
    data = df.values.astype(float)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0) + 1e-8
    data_scaled = (data - data_mean) / data_std
    
    print(f"\nScaled data: Mean ~0, Std ~1")
    
    return torch.FloatTensor(data_scaled), var_names


def compute_dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """Compute NOTEARS DAG constraint: tr(e^{W o W}) - d"""
    n_vars = W.shape[0]
    W_squared = W * W
    M = W_squared
    expm = torch.eye(n_vars, device=W.device)
    M_power = torch.eye(n_vars, device=W.device)
    factorial = 1.0
    for i in range(1, 7):
        factorial *= i
        M_power = torch.matmul(M_power, M)
        expm = expm + M_power / factorial
    h = torch.trace(expm) - n_vars
    return h


def compute_prior_loss(W: torch.Tensor, W_prior: torch.Tensor) -> torch.Tensor:
    """
    Compute Prior Loss: penalize deviation from LLM suggestions.
    
    Prior Loss = ||W - W_LLM||²
    
    This makes it harder to reverse LLM suggestions.
    Model needs strong data evidence to overcome this penalty.
    
    Args:
        W: Current learned weights
        W_prior: LLM prior weights (initial suggestions)
        
    Returns:
        L2 distance between W and W_prior
    """
    return torch.sum((W - W_prior) ** 2)


def train_phase1_signal(
    model: NeuralLP,
    data: torch.Tensor,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    l1_lambda: float = 0.001,
    print_every: int = 20
):
    """Phase 1: Learn signal with minimal regularization (FULL GRAPH PREDICTION)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_vars = data.shape[1]
    
    history = {'total_loss': [], 'mse_loss': [], 'l1_loss': []}
    
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING SIGNAL - FULL GRAPH PREDICTION")
    print("=" * 70)
    print(f"Goal: Let data signals strengthen freely")
    print(f"  - Train on ALL {n_vars} variables (not just one target)")
    print(f"  - Both LLM and reverse directions can grow")
    print(f"  - No penalty for deviation yet")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"L1 lambda: {l1_lambda} (minimal)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        # Predict ALL variables
        all_predictions = model.forward(data)
        adjacency = model.adjacency()
        
        # Compute MSE loss for ALL variables
        mse_loss = torch.nn.functional.mse_loss(all_predictions, data)
        l1_loss = torch.sum(torch.abs(adjacency))
        total_loss = mse_loss + l1_lambda * l1_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        history['total_loss'].append(total_loss.item())
        history['mse_loss'].append(mse_loss.item())
        history['l1_loss'].append(l1_loss.item())
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | L1: {l1_loss.item():.4f}")
    
    print("-" * 70)
    print("Phase 1 complete! All variables trained.")
    return history


def train_phase2_prune_with_prior(
    model: NeuralLP,
    data: torch.Tensor,
    init_weights: torch.Tensor,
    n_epochs: int = 100,
    learning_rate: float = 0.005,
    l1_lambda: float = 0.08,
    dag_lambda: float = 2.0,
    prior_lambda: float = 0.5,
    print_every: int = 20
):
    """
    Phase 2: Prune noise with DAG constraint, L1, AND Prior Loss (FULL GRAPH).
    
    NEW: Prior Loss = λ||W - W_LLM||²
    
    This penalizes deviation from LLM suggestions, making reversals harder.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_vars = data.shape[1]
    
    history = {
        'total_loss': [], 
        'mse_loss': [], 
        'l1_loss': [], 
        'dag_loss': [],
        'prior_loss': []  # NEW!
    }
    
    print("\n" + "=" * 70)
    print("PHASE 2: PRUNING + DAG + PRIOR LOSS - FULL GRAPH PREDICTION")
    print("=" * 70)
    print(f"Goal: Resolve conflicts while respecting LLM knowledge")
    print(f"  - Train on ALL {n_vars} variables")
    print(f"  - DAG constraint eliminates cycles")
    print(f"  - Strong L1 suppresses weak edges")
    print(f"  - NEW: Prior loss penalizes deviation from LLM")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"L1 lambda: {l1_lambda}")
    print(f"DAG lambda: {dag_lambda}")
    print(f"Prior lambda: {prior_lambda} (NEW!)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        # Predict ALL variables
        all_predictions = model.forward(data)
        adjacency = model.adjacency()
        
        # MSE loss for ALL variables
        mse_loss = torch.nn.functional.mse_loss(all_predictions, data)
        l1_loss = torch.sum(torch.abs(adjacency))
        dag_loss = compute_dag_constraint(adjacency)
        prior_loss = compute_prior_loss(adjacency, init_weights)  # NEW!
        
        # Total loss with prior penalty
        total_loss = (mse_loss + 
                     l1_lambda * l1_loss + 
                     dag_lambda * dag_loss + 
                     prior_lambda * prior_loss)  # NEW!
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        history['total_loss'].append(total_loss.item())
        history['mse_loss'].append(mse_loss.item())
        history['l1_loss'].append(l1_loss.item())
        history['dag_loss'].append(dag_loss.item())
        history['prior_loss'].append(prior_loss.item())
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | L1: {l1_loss.item():.4f} | "
                  f"DAG: {dag_loss.item():.6f} | Prior: {prior_loss.item():.4f}")
    
    print("-" * 70)
    print("Phase 2 complete! All variables trained with prior loss.")
    return history


def evaluate_against_ground_truth(
    learned_adj: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.3
):
    """
    Complete evaluation: Precision, Recall, F1, Orientation Accuracy, SHD.
    
    Returns all metrics in one dictionary.
    """
    # Binarize
    pred = (np.abs(learned_adj) > threshold).astype(int)
    gt = ground_truth.astype(int)
    
    # Basic counts
    tp = ((gt == 1) & (pred == 1)).sum()
    fp = ((gt == 0) & (pred == 1)).sum()
    fn = ((gt == 1) & (pred == 0)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Orientation Accuracy
    n_vars = pred.shape[0]
    correct_orient = 0
    total_orient = 0
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            gt_has = (gt[i,j] == 1) or (gt[j,i] == 1)
            pred_has = (pred[i,j] == 1) or (pred[j,i] == 1)
            if gt_has and pred_has:
                total_orient += 1
                if gt[i,j] == pred[i,j] and gt[j,i] == pred[j,i]:
                    correct_orient += 1
    
    orientation_acc = correct_orient / total_orient if total_orient > 0 else 0
    
    # SHD = False Positives + False Negatives
    # Note: Reversals are already counted in FP+FN
    # (e.g., GT: A->B, Pred: B->A gives FP=1 for B->A, FN=1 for A->B)
    shd = fp + fn
    
    # Count reversals separately (for reporting, not for SHD)
    reversals = 0
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            gt_has = (gt[i,j] == 1) or (gt[j,i] == 1)
            pred_has = (pred[i,j] == 1) or (pred[j,i] == 1)
            if gt_has and pred_has:
                # Both have edge, check if direction matches
                if not (gt[i,j] == pred[i,j] and gt[j,i] == pred[j,i]):
                    reversals += 1
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'orientation_accuracy': orientation_acc,
        'shd': shd,
        'reversals': reversals,  # For info only
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'predicted_edges': pred.sum(),
        'ground_truth_edges': gt.sum()
    }


def analyze_llm_corrections(
    learned_adj: np.ndarray,
    init_weights: np.ndarray,
    var_names: list,
    threshold: float = 0.1
):
    """Analyze which LLM suggestions were kept vs corrected."""
    print("\n" + "=" * 70)
    print("ANALYZING LLM SUGGESTIONS vs LEARNED STRUCTURE")
    print("=" * 70)
    
    n_vars = len(var_names)
    
    llm_suggestions = []
    llm_kept = []
    llm_reversed = []
    llm_both_pruned = []
    
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            init_ij = init_weights[i, j]
            init_ji = init_weights[j, i]
            
            if init_ij > 0 and init_ji > 0 and abs(init_ij - init_ji) > 0.1:
                if init_ij > init_ji:
                    llm_dir = f"{var_names[i]} -> {var_names[j]}"
                    llm_forward = (i, j)
                    llm_reverse = (j, i)
                else:
                    llm_dir = f"{var_names[j]} -> {var_names[i]}"
                    llm_forward = (j, i)
                    llm_reverse = (i, j)
                
                llm_suggestions.append(llm_dir)
                
                learned_forward = abs(learned_adj[llm_forward[0], llm_forward[1]]) > threshold
                learned_reverse = abs(learned_adj[llm_reverse[0], llm_reverse[1]]) > threshold
                
                if learned_forward and not learned_reverse:
                    llm_kept.append(llm_dir)
                elif learned_reverse and not learned_forward:
                    reverse_dir = f"{var_names[llm_reverse[0]]} -> {var_names[llm_reverse[1]]}"
                    llm_reversed.append((llm_dir, reverse_dir))
                elif not learned_forward and not learned_reverse:
                    llm_both_pruned.append(llm_dir)
    
    print(f"\nTotal LLM directional suggestions: {len(llm_suggestions)}")
    print(f"  - Kept (LLM correct, data agrees): {len(llm_kept)}")
    print(f"  - REVERSED (LLM corrected by data): {len(llm_reversed)}")
    print(f"  - Both directions pruned: {len(llm_both_pruned)}")
    
    if llm_kept:
        print(f"\nLLM suggestions KEPT ({len(llm_kept)}):")
        for edge in llm_kept[:10]:
            print(f"  {edge}")
        if len(llm_kept) > 10:
            print(f"  ... and {len(llm_kept) - 10} more")
    
    if llm_reversed:
        print(f"\n*** LLM CORRECTIONS ({len(llm_reversed)}) ***")
        print("These reversals overcame the prior loss penalty!")
        for llm_dir, learned_dir in llm_reversed:
            print(f"  LLM said:  {llm_dir}")
            print(f"  Data says: {learned_dir}")
            print()
    
    if llm_both_pruned:
        print(f"\nEdges pruned ({len(llm_both_pruned)}):")
        for edge in llm_both_pruned[:10]:
            print(f"  {edge}")
        if len(llm_both_pruned) > 10:
            print(f"  ... and {len(llm_both_pruned) - 10} more")
    
    return {
        'total': len(llm_suggestions),
        'kept': len(llm_kept),
        'reversed': len(llm_reversed),
        'pruned': len(llm_both_pruned)
    }


def main():
    """Main training pipeline with Prior Loss - FULL GRAPH PREDICTION."""
    print("=" * 70)
    print("STEP 5 (v3): NEURAL LP + PRIOR LOSS - FULL GRAPH PREDICTION")
    print("=" * 70)
    print("\nKEY INNOVATIONS:")
    print("  1. FULL GRAPH PREDICTION: Train on ALL variables (not just one)")
    print("     - Can learn ALL edges in the graph")
    print("     - Expected: Much higher recall!")
    print("  2. Prior Loss: lambda * ||W - W_LLM||^2")
    print("     - Penalizes deviation from LLM suggestions")
    print("     - Requires stronger data evidence to reverse")
    print("     - Should prevent false corrections like BP -> CO")
    
    # Load REAL data
    data_path = Path('../alarm_data.csv')
    if not data_path.exists():
        data_path = Path('alarm_data.csv')
    if not data_path.exists():
        print(f"\n[ERROR] alarm_data.csv not found!")
        return
    
    data_tensor, var_names = load_real_alarm_data(data_path)
    n_vars = len(var_names)
    
    # Load v2 artifacts
    mask_file = Path('data/alarm_mask_skeleton.npy')
    init_weights_file = Path('data/alarm_init_weights.npy')
    
    if not mask_file.exists() or not init_weights_file.exists():
        print(f"\n[ERROR] v2 artifacts not found!")
        return
    
    mask = np.load(mask_file)
    init_weights = np.load(init_weights_file)
    
    print(f"\nLoaded Mask: {mask.shape}, Trainable: {mask.sum()}")
    print(f"Loaded LLM Prior: Max {init_weights.max():.2f}, Non-zero: {(init_weights > 0).sum()}")
    
    if mask.shape[0] != n_vars:
        print(f"\n[ERROR] Dimension mismatch!")
        return
    
    # Initialize Model
    print("\n" + "=" * 70)
    print("INITIALIZING NEURAL LP MODEL - FULL GRAPH MODE")
    print("=" * 70)
    print(f"Training strategy: Multi-target (all {n_vars} variables)")
    print(f"Expected improvement: Recall 71% -> 85%+ (can learn all edges)")
    
    model = NeuralLP(
        n_vars=n_vars,
        mask=mask,
        max_hops=2,
        init_weights=init_weights
    )
    
    # Convert init_weights to tensor for prior loss
    init_weights_tensor = torch.FloatTensor(init_weights)
    
    # Two-Phase Training (FULL GRAPH)
    print("\n" + "=" * 70)
    print("STARTING TWO-PHASE TRAINING - FULL GRAPH PREDICTION")
    print("=" * 70)
    
    history1 = train_phase1_signal(
        model, data_tensor,
        n_epochs=300, learning_rate=0.01, l1_lambda=0.001
    )
    
    history2 = train_phase2_prune_with_prior(
        model, data_tensor, init_weights_tensor,
        n_epochs=300, learning_rate=0.005,
        l1_lambda=0.08, dag_lambda=2.0, 
        prior_lambda=0.5  # NEW: Prior loss weight
    )
    
    # Get learned structure
    with torch.no_grad():
        learned_adj = model.adjacency().numpy()
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'alarm_learned_adjacency_v3_prior.npy', learned_adj)
    print(f"\nSaved to {output_dir / 'alarm_learned_adjacency_v3_prior.npy'}")
    
    # Analyze LLM corrections
    correction_stats = analyze_llm_corrections(
        learned_adj, init_weights, var_names, threshold=0.3
    )
    
    # Evaluate against ground truth if available
    gt_file = output_dir / 'alarm_ground_truth_aligned.npy'
    eval_metrics = None
    
    if gt_file.exists():
        print("\n" + "=" * 70)
        print("EVALUATION vs GROUND TRUTH")
        print("=" * 70)
        
        ground_truth = np.load(gt_file)
        eval_metrics = evaluate_against_ground_truth(learned_adj, ground_truth, threshold=0.3)
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision:            {eval_metrics['precision']:.1%}")
        print(f"  Recall:               {eval_metrics['recall']:.1%}")
        print(f"  F1 Score:             {eval_metrics['f1']:.1%}")
        print(f"  Orientation Accuracy: {eval_metrics['orientation_accuracy']:.1%}")
        
        print(f"\nEdge Counts:")
        print(f"  Ground Truth:   {eval_metrics['ground_truth_edges']} edges")
        print(f"  Predicted:      {eval_metrics['predicted_edges']} edges")
        print(f"  True Positive:  {eval_metrics['tp']}")
        print(f"  False Positive: {eval_metrics['fp']} (extra edges)")
        print(f"  False Negative: {eval_metrics['fn']} (missing edges)")
        print(f"  Reversals:      {eval_metrics['reversals']} (wrong direction)")
        
        print(f"\nStructural Hamming Distance (SHD): {eval_metrics['shd']}")
        print(f"  = FP({eval_metrics['fp']}) + FN({eval_metrics['fn']})")
        print(f"  Note: Reversals already counted in FP+FN")
        print(f"  (Lower is better, 0 = perfect match)")
    else:
        print("\n[INFO] Ground truth not found. Skipping evaluation.")
        print("       Run step6_evaluate_ground_truth.py first to generate ground truth.")
    
    # Show top edges
    print("\n" + "=" * 70)
    print("TOP-10 STRONGEST LEARNED EDGES")
    print("=" * 70)
    
    edges = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and abs(learned_adj[i, j]) > 0.01:
                edges.append((var_names[i], var_names[j], learned_adj[i, j]))
    
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for rank, (src, tgt, weight) in enumerate(edges[:10], 1):
        print(f"{rank:2d}. {src:15s} -> {tgt:15s}  weight: {weight:7.4f}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("STEP 5 (v3 - FULL GRAPH + PRIOR LOSS) COMPLETE!")
    print("=" * 70)
    
    print(f"\nLLM Correction Summary:")
    print(f"  Total suggestions: {correction_stats['total']}")
    print(f"  Kept:     {correction_stats['kept']}")
    print(f"  Reversed: {correction_stats['reversed']}")
    print(f"  Pruned:   {correction_stats['pruned']}")
    
    if eval_metrics:
        print(f"\nFinal Performance:")
        print(f"  Precision: {eval_metrics['precision']:.1%}")
        print(f"  Recall:    {eval_metrics['recall']:.1%}")
        print(f"  F1 Score:  {eval_metrics['f1']:.1%}")
        print(f"  SHD:       {eval_metrics['shd']}")
    
    if correction_stats['reversed'] == 0:
        print(f"\n*** SUCCESS! Prior loss prevented false corrections! ***")
    elif correction_stats['reversed'] == 1:
        print(f"\n*** 1 reversal (Prior loss made it harder but not impossible) ***")
    
    if eval_metrics:
        print("\n" + "=" * 70)
        print("VERSION COMPARISON")
        print("=" * 70)
        print("v2 (no prior loss):  Precision 33%, Recall 2%,  F1 4%,  SHD ~44")
        print("v3 (single-target):  Precision 80%, Recall 71%, F1 75%, SHD ~13")
        print(f"v3 FULL GRAPH:       Precision {eval_metrics['precision']:.0%}, Recall {eval_metrics['recall']:.0%}, F1 {eval_metrics['f1']:.0%}, SHD {eval_metrics['shd']}")
        print("\nKey improvement: Full graph can learn ALL edges (not just one target)")
    else:
        print("\n[Run step6_evaluate_ground_truth.py first to see performance metrics]")
    
    print("\n" + "=" * 70)
    if eval_metrics:
        print("TRAINING COMPLETE - All metrics calculated above!")
    else:
        print("TRAINING COMPLETE - Run step6_evaluate_ground_truth.py for metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
