"""
Phase 2 Training Script

Enhanced training with:
- Weighted Group Lasso: Block-level sparsity with Normal protection
- Cycle Consistency: Direction learning
- Detailed loss monitoring
"""

import torch
import torch.optim as optim
from pathlib import Path
import json

from modules.data_loader import CausalDataLoader
from modules.prior_builder import PriorBuilder
from modules.model import CausalDiscoveryModel
from modules.loss import LossComputer
from modules.evaluator import CausalGraphEvaluator


def train_phase2(config: dict):
    """
    Phase 2 training: Dense matrix + Weighted Group Lasso + Cycle Consistency
    
    Args:
        config: Training configuration dictionary
    """
    print("=" * 70)
    print("PHASE 2 TRAINING: WEIGHTED GROUP LASSO + CYCLE CONSISTENCY")
    print("=" * 70)
    
    # === 1. LOAD DATA ===
    data_loader = CausalDataLoader(
        data_path=config['data_path'],
        metadata_path=config['metadata_path']
    )
    observations = data_loader.load_data()
    var_structure = data_loader.get_variable_structure()
    
    # === 2. BUILD PRIORS ===
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_csv_path=config['fci_edges_path'],
        llm_rules_path=config['llm_rules_path']
    )
    
    # === 3. INITIALIZE MODEL ===
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    # === 4. SETUP LOSS COMPUTER (PHASE 2) ===
    loss_computer = LossComputer(
        block_structure=priors['blocks'],
        penalty_weights=priors['penalty_weights']
    )
    
    # === 5. SETUP TRAINING ===
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # === 6. INITIALIZE EVALUATOR ===
    evaluator = CausalGraphEvaluator(
        ground_truth_path=config['ground_truth_path'],
        var_structure=var_structure
    )
    
    # === 7. TRAINING LOOP ===
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Epochs: {config['n_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Lambda Group Lasso: {config['lambda_group']}")
    print(f"Lambda Cycle: {config['lambda_cycle']}")
    print(f"Batch size: Full dataset ({observations.shape[0]} samples)")
    print("-" * 70)
    
    # Track loss history
    loss_history = {
        'total': [],
        'reconstruction': [],
        'weighted_group_lasso': [],
        'cycle_consistency': []
    }
    
    for epoch in range(config['n_epochs']):
        # Forward pass
        predictions = model(observations, n_hops=config['n_hops'])
        adjacency = model.get_adjacency()
        
        # Compute loss (Phase 2: all components)
        losses = loss_computer.compute_total_loss(
            predictions=predictions,
            targets=observations,
            adjacency=adjacency,
            lambda_group=config['lambda_group'],
            lambda_cycle=config['lambda_cycle']
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Clamp weights to reasonable range
        with torch.no_grad():
            model.raw_adj.data.clamp_(-5.0, 5.0)
        
        # Record loss history
        for key in loss_history:
            loss_history[key].append(losses[key].item())
        
        # Log progress
        if (epoch + 1) % config['log_interval'] == 0:
            print(f"Epoch {epoch + 1:3d} | "
                  f"Total: {losses['total'].item():.4f} | "
                  f"Recon: {losses['reconstruction'].item():.4f} | "
                  f"Group: {losses['weighted_group_lasso'].item():.4f} | "
                  f"Cycle: {losses['cycle_consistency'].item():.4f}")
    
    # === 8. EVALUATION ===
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    adjacency = model.get_adjacency()
    learned_edges = evaluator.extract_learned_edges(adjacency, threshold=config['edge_threshold'])
    metrics = evaluator.evaluate(learned_edges)
    evaluator.print_metrics(metrics)
    
    # === 9. ANALYZE SPARSITY ===
    print("\n" + "=" * 70)
    print("SPARSITY ANALYSIS")
    print("=" * 70)
    
    adjacency_np = adjacency.detach().cpu().numpy()
    
    # Overall sparsity
    total_allowed = int(priors['skeleton_mask'].sum().item())
    active_connections = (adjacency_np > config['edge_threshold']).sum()
    
    print(f"\nOverall Sparsity:")
    print(f"  Allowed by skeleton: {total_allowed}")
    print(f"  Active (>{config['edge_threshold']}): {active_connections}")
    print(f"  Sparsity: {(total_allowed - active_connections) / total_allowed * 100:.1f}%")
    
    # Block-level sparsity
    active_blocks = 0
    for block in priors['blocks']:
        block_weights = adjacency[block['row_indices']][:, block['col_indices']]
        if block_weights.mean().item() > config['edge_threshold']:
            active_blocks += 1
    
    print(f"\nBlock-level Sparsity:")
    print(f"  Total blocks: {len(priors['blocks'])}")
    print(f"  Active blocks: {active_blocks}")
    print(f"  Sparsity: {(len(priors['blocks']) - active_blocks) / len(priors['blocks']) * 100:.1f}%")
    
    # === 10. SAVE RESULTS ===
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'loss_history': loss_history
    }, output_dir / 'phase2_model.pt')
    
    # Save adjacency matrix
    torch.save(adjacency, output_dir / 'phase2_adjacency.pt')
    
    # Save learned edges
    with open(output_dir / 'phase2_edges.txt', 'w') as f:
        f.write("Learned Causal Edges (Phase 2)\n")
        f.write("=" * 70 + "\n\n")
        for var_a, var_b in sorted(learned_edges):
            f.write(f"{var_a} -> {var_b}\n")
    
    # Save metrics
    with open(output_dir / 'phase2_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save loss history
    with open(output_dir / 'phase2_loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, metrics, loss_history


if __name__ == "__main__":
    # Configuration
    config = {
        # Data paths
        'data_path': 'data/alarm_data_10000.csv',
        'metadata_path': 'output/knowledge_graph_metadata.json',
        'fci_edges_path': 'data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
        'llm_rules_path': 'llm_prior_rules',
        'ground_truth_path': '../alarm.bif',
        
        # Training parameters
        'n_epochs': 300,
        'learning_rate': 0.01,
        'n_hops': 1,  # Single-hop reasoning
        'log_interval': 50,
        
        # Loss hyperparameters (PHASE 2)
        'lambda_group': 0.01,    # Weighted Group Lasso weight
        'lambda_cycle': 0.001,   # Cycle Consistency weight
        
        # Evaluation
        'edge_threshold': 0.1,  # Lower threshold for Phase 2 (sparsity from Group Lasso)
        
        # Output
        'output_dir': 'results/phase2'
    }
    
    # Train
    model, metrics, loss_history = train_phase2(config)
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Edge Recall: {metrics['edge_recall']:.1%}")
    print(f"Directed Recall: {metrics['directed_recall']:.1%}")
    print(f"Orientation Accuracy: {metrics['orientation_accuracy']:.1%}")
    print(f"\nFinal Losses:")
    print(f"  Reconstruction: {loss_history['reconstruction'][-1]:.4f}")
    print(f"  Group Lasso: {loss_history['weighted_group_lasso'][-1]:.4f}")
    print(f"  Cycle Consistency: {loss_history['cycle_consistency'][-1]:.4f}")

