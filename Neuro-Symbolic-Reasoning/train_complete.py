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
    
    for block in block_structure:
        var_a, var_b = block['var_pair']
        
        # Avoid double counting
        pair_key = tuple(sorted([var_a, var_b]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        # Find reverse block
        reverse_block = block_lookup.get((var_b, var_a))
        if reverse_block is None:
            continue
        
        # Compute block strengths
        forward_weights = adjacency[block['row_indices']][:, block['col_indices']]
        backward_weights = adjacency[reverse_block['row_indices']][:, reverse_block['col_indices']]
        
        forward_strength = forward_weights.mean().item()
        backward_strength = backward_weights.mean().item()
        
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
    overall_sparsity = (total_allowed - active_connections) / total_allowed
    
    # Block-level sparsity
    active_blocks = 0
    for block in block_structure:
        block_weights = adjacency[block['row_indices']][:, block['col_indices']]
        if block_weights.mean().item() > threshold:
            active_blocks += 1
    
    block_sparsity = (len(block_structure) - active_blocks) / len(block_structure)
    
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
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_skeleton_path=config['fci_skeleton_path'],  # Pure FCI for hard mask
        llm_direction_path=config.get('llm_direction_path'),  # FCI+LLM for soft direction (optional)
        use_llm_prior=config.get('use_llm_prior', True)  # Whether to use LLM prior
    )
    
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
    evaluator = CausalGraphEvaluator(
        ground_truth_path=config['ground_truth_path'],
        var_structure=var_structure
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
                print(f"  Direction Learning:")
                print(f"    Bidirectional: {bidir_stats['bidirectional']:3d} / {bidir_stats['total_pairs']:3d} "
                      f"({bidir_stats['bidirectional_ratio']*100:5.1f}%) [Want: DOWN]")
                print(f"    Unidirectional: {bidir_stats['unidirectional']:3d} / {bidir_stats['total_pairs']:3d} "
                      f"({bidir_stats['unidirectional']/bidir_stats['total_pairs']*100:5.1f}%) [Want: UP]")
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
        # Data paths
        'data_path': 'data/alarm_data_10000.csv',
        'metadata_path': 'output/knowledge_graph_metadata.json',
        'fci_skeleton_path': 'data/edges_FCI_20251207_230824.csv',  # Pure FCI for HARD skeleton mask
        'llm_direction_path': 'data/edges_Hybrid_FCI_LLM_20251207_230956.csv',  # FCI+LLM for SOFT direction prior
        'ground_truth_path': 'data/alarm.bif',
        
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

