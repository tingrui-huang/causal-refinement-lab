"""
Complete Training Script - Phase 4 & 5

Full integration of all components:
- Data loading
- Prior building (FCI + LLM)
- Dense matrix model
- Weighted Group Lasso + Cycle Consistency
- Comprehensive monitoring metrics

Focus: Direction learning and sparsity tracking
"""

import torch
import torch.optim as optim
from pathlib import Path
import json
import numpy as np

from modules.data_loader import CausalDataLoader
from modules.prior_builder import PriorBuilder
from modules.model import CausalDiscoveryModel
from modules.loss import LossComputer
from modules.evaluator import CausalGraphEvaluator


def compute_bidirectional_ratio(adjacency: torch.Tensor, 
                                block_structure: list,
                                threshold: float = 0.3) -> dict:
    """
    Compute bidirectional edge statistics
    
    Goal: Track how many block pairs have strong weights in BOTH directions
    We want this ratio to DECREASE during training (direction learning)
    
    FIXED: Now checks ALL variable pairs in FCI skeleton, not just those
    with both directions in block_structure. For pairs with only one direction
    in block_structure, we still check the reverse direction in the adjacency
    matrix (even though it's not explicitly in block_structure).
    
    Args:
        adjacency: (105, 105) adjacency matrix
        block_structure: List of blocks
        threshold: Threshold for "strong" weight
    
    Returns:
        Dictionary with bidirectional statistics
    """
    bidirectional_count = 0
    unidirectional_count = 0
    no_direction_count = 0
    processed_pairs = set()
    
    # Build block lookup
    block_lookup = {}
    for block in block_structure:
        var_a, var_b = block['var_pair']
        block_lookup[(var_a, var_b)] = block
    
    # Get all unique variable pairs from block_structure
    all_pairs = set()
    for block in block_structure:
        var_a, var_b = block['var_pair']
        pair_key = tuple(sorted([var_a, var_b]))
        all_pairs.add(pair_key)
    
    # Check each unique pair
    for pair_key in all_pairs:
        var_a, var_b = pair_key  # Already sorted
        
        # Try to get both directions
        forward_block = block_lookup.get((var_a, var_b))
        reverse_block = block_lookup.get((var_b, var_a))
        
        # Compute strengths for both directions
        # If a direction is not in block_structure, manually compute from adjacency
        if forward_block is not None:
            forward_weights = adjacency[forward_block['row_indices']][:, forward_block['col_indices']]
            forward_strength = forward_weights.mean().item()
        else:
            # Manually compute: need to get state indices from reverse_block
            if reverse_block is not None:
                # Swap indices: reverse_block is (var_b, var_a), we want (var_a, var_b)
                forward_weights = adjacency[reverse_block['col_indices']][:, reverse_block['row_indices']]
                forward_strength = forward_weights.mean().item()
            else:
                # Should not happen if block_structure is correct
                forward_strength = 0.0
        
        if reverse_block is not None:
            backward_weights = adjacency[reverse_block['row_indices']][:, reverse_block['col_indices']]
            backward_strength = backward_weights.mean().item()
        else:
            # Manually compute: need to get state indices from forward_block
            if forward_block is not None:
                # Swap indices: forward_block is (var_a, var_b), we want (var_b, var_a)
                backward_weights = adjacency[forward_block['col_indices']][:, forward_block['row_indices']]
                backward_strength = backward_weights.mean().item()
            else:
                # Should not happen if block_structure is correct
                backward_strength = 0.0
        
        # Classify
        forward_strong = forward_strength > threshold
        backward_strong = backward_strength > threshold
        
        if forward_strong and backward_strong:
            bidirectional_count += 1
        elif forward_strong or backward_strong:
            unidirectional_count += 1
        else:
            no_direction_count += 1
    
    total_pairs = bidirectional_count + unidirectional_count + no_direction_count
    
    return {
        'bidirectional': bidirectional_count,
        'unidirectional': unidirectional_count,
        'no_direction': no_direction_count,
        'total_pairs': total_pairs,
        'bidirectional_ratio': bidirectional_count / total_pairs if total_pairs > 0 else 0
    }


def compute_sparsity_metrics(adjacency: torch.Tensor,
                             skeleton_mask: torch.Tensor,
                             block_structure: list,
                             threshold: float = 0.1) -> dict:
    """
    Compute comprehensive sparsity metrics
    
    Args:
        adjacency: (105, 105) adjacency matrix
        skeleton_mask: (105, 105) skeleton constraint
        block_structure: List of blocks
        threshold: Threshold for "active" connection
    
    Returns:
        Dictionary with sparsity statistics
    """
    # Overall sparsity
    total_allowed = int(skeleton_mask.sum().item())
    adjacency_np = adjacency.detach().cpu().numpy()
    active_connections = (adjacency_np > threshold).sum()
    
    # Handle edge case: no allowed connections
    if total_allowed > 0:
        overall_sparsity = (total_allowed - active_connections) / total_allowed
    else:
        overall_sparsity = 0.0
    
    # Block-level sparsity
    active_blocks = 0
    for block in block_structure:
        block_weights = adjacency[block['row_indices']][:, block['col_indices']]
        if block_weights.mean().item() > threshold:
            active_blocks += 1
    
    # Handle edge case: no blocks
    if len(block_structure) > 0:
        block_sparsity = (len(block_structure) - active_blocks) / len(block_structure)
    else:
        block_sparsity = 0.0
    
    # Weight distribution
    active_weights = adjacency_np[adjacency_np > threshold]
    
    return {
        'total_allowed': total_allowed,
        'active_connections': int(active_connections),
        'overall_sparsity': overall_sparsity,
        'active_blocks': active_blocks,
        'total_blocks': len(block_structure),
        'block_sparsity': block_sparsity,
        'mean_active_weight': float(active_weights.mean()) if len(active_weights) > 0 else 0.0,
        'max_weight': float(adjacency_np.max()),
        'min_nonzero_weight': float(adjacency_np[adjacency_np > 0].min()) if (adjacency_np > 0).any() else 0.0
    }


def train_complete(config: dict):
    """
    Complete training with full monitoring
    
    Args:
        config: Training configuration
    """
    print("=" * 70)
    print("COMPLETE TRAINING: FULL INTEGRATION")
    print("=" * 70)
    print("Components:")
    print("  - Data: Direct CSV loading")
    print("  - Priors: FCI skeleton + LLM direction")
    print("  - Model: Dense 105x105 matrix")
    print("  - Loss: Reconstruction + Weighted Group Lasso + Cycle Consistency")
    print("  - Monitoring: Direction learning + Sparsity tracking")
    print("=" * 70)
    
    # === 1. LOAD DATA ===
    print("\n[1/6] Loading Data...")
    data_loader = CausalDataLoader(
        data_path=config['data_path'],
        metadata_path=config['metadata_path']
    )
    observations = data_loader.load_data()
    var_structure = data_loader.get_variable_structure()
    
    # === 2. BUILD PRIORS ===
    print("\n[2/6] Building Priors...")
    dataset_name = data_loader.metadata.get('dataset_name', 'Unknown')
    prior_builder = PriorBuilder(var_structure, dataset_name=dataset_name)
    
    # Check if using manual skeleton (for simple datasets like Tuebingen)
    manual_skeleton = config.get('manual_skeleton', None)
    
    priors = prior_builder.get_all_priors(
        fci_skeleton_path=config.get('fci_skeleton_path'),  # Pure FCI for hard mask (can be None if manual)
        llm_direction_path=config.get('llm_direction_path'),  # FCI+LLM for soft direction (optional)
        use_llm_prior=config.get('use_llm_prior', True),  # Whether to use LLM prior
        manual_skeleton=manual_skeleton  # Manual skeleton specification (for Tuebingen)
    )
    
    # Check if custom direction prior is provided (for experiments or LLM)
    if 'forward_bias' in config:
        # Create biased direction prior for experiment
        print(f"\n[EXPERIMENT MODE] Creating biased direction prior:")
        print(f"  Forward (Altitude->Temp): {config['forward_bias']:.1f}")
        print(f"  Backward (Temp->Alt): {1-config['forward_bias']:.1f}")
        
        direction_prior = torch.zeros_like(priors['direction_prior'])
        forward_bias = config['forward_bias']
        backward_bias = 1.0 - forward_bias
        
        # Apply bias to manual skeleton edges
        for var_a, var_b in manual_skeleton:
            states_a = var_structure['var_to_states'][var_a]
            states_b = var_structure['var_to_states'][var_b]
            
            # Forward direction
            for i in states_a:
                for j in states_b:
                    direction_prior[i, j] = forward_bias
            
            # Backward direction
            for i in states_b:
                for j in states_a:
                    direction_prior[i, j] = backward_bias
        
        priors['direction_prior'] = direction_prior
    
    # NEW: Check if LLM weights are provided (方案 A: 温和初始化)
    elif 'llm_forward_weight' in config and 'llm_backward_weight' in config:
        # Apply LLM-suggested weights to initialization
        # This gives model a gentle push (e.g., 0.6 vs 0.4) to help break symmetry
        print(f"\n[LLM PRIOR] Applying LLM-suggested direction weights:")
        print(f"  {config.get('llm_var_x', 'X')} -> {config.get('llm_var_y', 'Y')}: {config['llm_forward_weight']:.2f}")
        print(f"  {config.get('llm_var_y', 'Y')} -> {config.get('llm_var_x', 'X')}: {config['llm_backward_weight']:.2f}")
        print(f"  Advantage: {abs(config['llm_forward_weight'] - config['llm_backward_weight']):.2f}")
        
        direction_prior = torch.zeros_like(priors['direction_prior'])
        
        # Apply LLM weights to manual skeleton edges
        if manual_skeleton:
            for var_a, var_b in manual_skeleton:
                states_a = var_structure['var_to_states'][var_a]
                states_b = var_structure['var_to_states'][var_b]
                
                # Forward direction (X -> Y)
                for i in states_a:
                    for j in states_b:
                        direction_prior[i, j] = config['llm_forward_weight']
                
                # Backward direction (Y -> X)
                for i in states_b:
                    for j in states_a:
                        direction_prior[i, j] = config['llm_backward_weight']
        
        priors['direction_prior'] = direction_prior
        print(f"[LLM PRIOR] Direction prior updated with LLM weights")
    
    # === 3. INITIALIZE MODEL ===
    print("\n[3/6] Initializing Model...")
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    # === 4. INITIALIZE LOSS COMPUTER ===
    print("\n[4/6] Initializing Loss Computer...")
    loss_computer = LossComputer(
        block_structure=priors['blocks'],
        penalty_weights=priors['penalty_weights']
    )
    
    # === 5. SETUP TRAINING ===
    print("\n[5/6] Setting up Training...")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Initialize evaluator
    # For datasets with manual ground truth (like Tuebingen), pass ground_truth_edges directly
    ground_truth_edges = config.get('ground_truth_edges', None)
    evaluator = CausalGraphEvaluator(
        ground_truth_path=config.get('ground_truth_path'),
        var_structure=var_structure,
        ground_truth_edges=ground_truth_edges
    )
    
    # === 6. TRAINING LOOP WITH MONITORING ===
    print("\n[6/6] Starting Training Loop...")
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Epochs: {config['n_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Lambda Group Lasso: {config['lambda_group']}")
    print(f"Lambda Cycle: {config['lambda_cycle']}")
    print(f"N hops: {config['n_hops']}")
    print("-" * 70)
    
    # Tracking history
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_reconstruction': [],
        'loss_group_lasso': [],
        'loss_cycle': [],
        'bidirectional_ratio': [],
        'overall_sparsity': [],
        'block_sparsity': [],
        'active_connections': [],
        'active_blocks': []
    }
    
    for epoch in range(config['n_epochs']):
        # === FORWARD PASS ===
        predictions = model(observations, n_hops=config['n_hops'])
        adjacency = model.get_adjacency()
        
        # === COMPUTE LOSS ===
        losses = loss_computer.compute_total_loss(
            predictions=predictions,
            targets=observations,
            adjacency=adjacency,
            lambda_group=config['lambda_group'],
            lambda_cycle=config['lambda_cycle']
        )
        
        # === BACKWARD PASS ===
        optimizer.zero_grad()
        losses['total'].backward()
        
        # === GRADIENT MONITORING (for symmetry breaking analysis) ===
        # Compute gradient magnitudes for both directions
        if (epoch + 1) % config['monitor_interval'] == 0:
            with torch.no_grad():
                # Get gradients
                grad = model.raw_adj.grad
                
                # For Tuebingen: determine bin count from var_structure
                n_bins = var_structure['n_states'] // var_structure['n_variables']
                # Altitude (states 0:n_bins), Temperature (states n_bins:2*n_bins)
                grad_altitude_to_temp = grad[0:n_bins, n_bins:2*n_bins]  # Forward direction
                grad_temp_to_altitude = grad[n_bins:2*n_bins, 0:n_bins]  # Backward direction
                
                # Compute average gradient magnitude (L2 norm)
                grad_forward_mag = grad_altitude_to_temp.abs().mean().item()
                grad_backward_mag = grad_temp_to_altitude.abs().mean().item()
                grad_ratio = grad_forward_mag / grad_backward_mag if grad_backward_mag > 0 else float('inf')
        
        optimizer.step()
        
        # Clamp weights
        with torch.no_grad():
            model.raw_adj.data.clamp_(-5.0, 5.0)
        
        # === MONITORING METRICS ===
        if (epoch + 1) % config['monitor_interval'] == 0:
            with torch.no_grad():
                # Bidirectional ratio
                bidir_stats = compute_bidirectional_ratio(
                    adjacency, 
                    priors['blocks'],
                    threshold=config['edge_threshold']
                )
                
                # Sparsity metrics
                sparsity_stats = compute_sparsity_metrics(
                    adjacency,
                    priors['skeleton_mask'],
                    priors['blocks'],
                    threshold=config['edge_threshold']
                )
                
                # Record history
                history['epoch'].append(epoch + 1)
                history['loss_total'].append(losses['total'].item())
                history['loss_reconstruction'].append(losses['reconstruction'].item())
                history['loss_group_lasso'].append(losses['weighted_group_lasso'].item())
                history['loss_cycle'].append(losses['cycle_consistency'].item())
                history['bidirectional_ratio'].append(bidir_stats['bidirectional_ratio'])
                history['overall_sparsity'].append(sparsity_stats['overall_sparsity'])
                history['block_sparsity'].append(sparsity_stats['block_sparsity'])
                history['active_connections'].append(sparsity_stats['active_connections'])
                history['active_blocks'].append(sparsity_stats['active_blocks'])
                
                # Print progress
                print(f"\nEpoch {epoch + 1:3d}/{config['n_epochs']}")
                print(f"  Losses:")
                print(f"    Total:        {losses['total'].item():8.4f}")
                print(f"    Recon:        {losses['reconstruction'].item():8.4f}")
                print(f"    Group Lasso:  {losses['weighted_group_lasso'].item():8.4f}")
                print(f"    Cycle:        {losses['cycle_consistency'].item():8.4f}")
                print(f"  Gradient Analysis (Symmetry Breaking):")
                print(f"    Forward Grad (Alt->Temp):  {grad_forward_mag:8.6f}")
                print(f"    Backward Grad (Temp->Alt): {grad_backward_mag:8.6f}")
                print(f"    Ratio (Forward/Backward):  {grad_ratio:8.4f} [Want: >1 or <1, NOT ~1]")
                print(f"  Direction Learning:")
                print(f"    Bidirectional: {bidir_stats['bidirectional']:3d} / {bidir_stats['total_pairs']:3d} "
                      f"({bidir_stats['bidirectional_ratio']*100:5.1f}%) [Want: DOWN]")
                if bidir_stats['total_pairs'] > 0:
                    uni_pct = bidir_stats['unidirectional']/bidir_stats['total_pairs']*100
                else:
                    uni_pct = 0.0
                print(f"    Unidirectional: {bidir_stats['unidirectional']:3d} / {bidir_stats['total_pairs']:3d} "
                      f"({uni_pct:5.1f}%) [Want: UP]")
                print(f"  Sparsity:")
                print(f"    Overall:      {sparsity_stats['overall_sparsity']*100:5.1f}% "
                      f"({sparsity_stats['active_connections']}/{sparsity_stats['total_allowed']})")
                print(f"    Block-level:  {sparsity_stats['block_sparsity']*100:5.1f}% "
                      f"({sparsity_stats['active_blocks']}/{sparsity_stats['total_blocks']})")
    
    # === FINAL EVALUATION ===
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    adjacency = model.get_adjacency()
    learned_edges = evaluator.extract_learned_edges(adjacency, threshold=config['edge_threshold'])
    metrics = evaluator.evaluate(learned_edges)
    evaluator.print_metrics(metrics)
    
    # === SAVE RESULTS ===
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'history': history
    }, output_dir / 'complete_model.pt')
    
    # Save adjacency
    torch.save(adjacency, output_dir / 'complete_adjacency.pt')
    
    # Save learned edges
    with open(output_dir / 'complete_edges.txt', 'w') as f:
        f.write("Learned Causal Edges (Complete Training)\n")
        f.write("=" * 70 + "\n\n")
        for var_a, var_b in sorted(learned_edges):
            # Get block strength
            block = None
            for b in priors['blocks']:
                if b['var_pair'] == (var_a, var_b):
                    block = b
                    break
            if block:
                block_weights = adjacency[block['row_indices']][:, block['col_indices']]
                strength = block_weights.mean().item()
                f.write(f"{var_a} -> {var_b} (strength: {strength:.4f})\n")
    
    # Save metrics
    with open(output_dir / 'complete_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training history
    with open(output_dir / 'complete_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save summary report
    with open(output_dir / 'complete_summary.txt', 'w') as f:
        f.write("Complete Training Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Epochs: {config['n_epochs']}\n")
        f.write(f"  Learning rate: {config['learning_rate']}\n")
        f.write(f"  Lambda Group Lasso: {config['lambda_group']}\n")
        f.write(f"  Lambda Cycle: {config['lambda_cycle']}\n\n")
        
        f.write("Final Losses:\n")
        f.write(f"  Total: {history['loss_total'][-1]:.4f}\n")
        f.write(f"  Reconstruction: {history['loss_reconstruction'][-1]:.4f}\n")
        f.write(f"  Group Lasso: {history['loss_group_lasso'][-1]:.4f}\n")
        f.write(f"  Cycle Consistency: {history['loss_cycle'][-1]:.4f}\n\n")
        
        f.write("Direction Learning:\n")
        f.write(f"  Initial Bidirectional Ratio: {history['bidirectional_ratio'][0]*100:.1f}%\n")
        f.write(f"  Final Bidirectional Ratio: {history['bidirectional_ratio'][-1]*100:.1f}%\n")
        f.write(f"  Change: {(history['bidirectional_ratio'][-1] - history['bidirectional_ratio'][0])*100:+.1f}%\n\n")
        
        f.write("Sparsity:\n")
        f.write(f"  Initial Overall: {history['overall_sparsity'][0]*100:.1f}%\n")
        f.write(f"  Final Overall: {history['overall_sparsity'][-1]*100:.1f}%\n")
        f.write(f"  Initial Block: {history['block_sparsity'][0]*100:.1f}%\n")
        f.write(f"  Final Block: {history['block_sparsity'][-1]*100:.1f}%\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  Edge Precision: {metrics['edge_precision']*100:.1f}%\n")
        f.write(f"  Edge Recall: {metrics['edge_recall']*100:.1f}%\n")
        f.write(f"  Orientation Accuracy: {metrics['orientation_accuracy']*100:.1f}%\n")
        f.write(f"  Learned Edges: {metrics['learned_edges']}\n")
        f.write(f"  Ground Truth Edges: {metrics['ground_truth_edges']}\n")
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, metrics, history


if __name__ == "__main__":
    # Configuration
    config = {
        # Data paths (updated to new structure)
        'data_path': 'data/alarm/alarm_data_10000.csv',
        'metadata_path': 'data/alarm/metadata.json',
        'fci_skeleton_path': 'data/alarm/edges_FCI_20251207_230824.csv',  # Pure FCI for HARD skeleton mask
        'llm_direction_path': 'data/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv',  # FCI+LLM for SOFT direction prior
        'ground_truth_path': 'data/alarm/alarm.bif',
        
        # Training parameters
        'n_epochs': 200,  # Pilot training: 200 epochs
        'learning_rate': 0.01,
        'n_hops': 1,  # Single-hop reasoning
        
        # Loss hyperparameters (Conservative start)
        'lambda_group': 0.01,    # Group Lasso weight
        'lambda_cycle': 0.001,   # Cycle Consistency weight
        
        # Monitoring
        'monitor_interval': 20,  # Monitor every 20 epochs
        'edge_threshold': 0.1,   # Threshold for active edge
        
        # Output
        'output_dir': 'results/complete'
    }
    
    # Train
    print("\n" + "=" * 70)
    print("PILOT TRAINING: 200 EPOCHS")
    print("=" * 70)
    print("Goal: Verify loss convergence and direction learning")
    print()
    
    model, metrics, history = train_complete(config)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\nLoss Convergence:")
    print(f"  Reconstruction: {history['loss_reconstruction'][0]:.4f} -> {history['loss_reconstruction'][-1]:.4f}")
    print(f"  Group Lasso:    {history['loss_group_lasso'][0]:.4f} -> {history['loss_group_lasso'][-1]:.4f}")
    print(f"  Cycle:          {history['loss_cycle'][0]:.4f} -> {history['loss_cycle'][-1]:.4f}")
    
    print("\nDirection Learning:")
    print(f"  Bidirectional Ratio: {history['bidirectional_ratio'][0]*100:.1f}% -> {history['bidirectional_ratio'][-1]*100:.1f}%")
    change = (history['bidirectional_ratio'][-1] - history['bidirectional_ratio'][0]) * 100
    print(f"  Change: {change:+.1f}% {'[GOOD]' if change < 0 else '[NEEDS TUNING]'}")
    
    print("\nSparsity:")
    print(f"  Overall: {history['overall_sparsity'][-1]*100:.1f}% ({history['active_connections'][-1]}/{416})")
    print(f"  Block:   {history['block_sparsity'][-1]*100:.1f}% ({history['active_blocks'][-1]}/{1332})")
    
    print("\nEvaluation:")
    print(f"  Precision: {metrics['edge_precision']*100:.1f}%")
    print(f"  Recall: {metrics['edge_recall']*100:.1f}%")
    print(f"  Orientation: {metrics['orientation_accuracy']*100:.1f}%")
    print(f"  Edges: {metrics['learned_edges']} / {metrics['ground_truth_edges']}")

