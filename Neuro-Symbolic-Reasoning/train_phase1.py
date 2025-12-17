"""
Phase 1 Training Script

Basic training with:
- Dense 105x105 matrix (not sparse rules)
- Reconstruction loss only
- FCI skeleton mask constraint
- LLM direction prior initialization
"""

import torch
import torch.optim as optim
from pathlib import Path
import json

from modules.data_loader import CausalDataLoader
from modules.prior_builder import PriorBuilder
from modules.model import CausalDiscoveryModel, LossComputer
from modules.evaluator import CausalGraphEvaluator


def train_phase1(config: dict):
    """
    Phase 1 training: Basic reconstruction with dense matrix
    
    Args:
        config: Training configuration dictionary
    """
    print("=" * 70)
    print("PHASE 1 TRAINING: DENSE MATRIX + RECONSTRUCTION LOSS")
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
    
    # === 4. SETUP TRAINING ===
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_computer = LossComputer()
    
    # === 5. INITIALIZE EVALUATOR ===
    evaluator = CausalGraphEvaluator(
        ground_truth_path=config['ground_truth_path'],
        var_structure=var_structure
    )
    
    # === 6. TRAINING LOOP ===
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Epochs: {config['n_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: Full dataset ({observations.shape[0]} samples)")
    print("-" * 70)
    
    for epoch in range(config['n_epochs']):
        # Forward pass
        predictions = model(observations, n_hops=config['n_hops'])
        
        # Compute loss
        losses = loss_computer.compute_total_loss(predictions, observations)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Clamp weights to reasonable range
        with torch.no_grad():
            model.raw_adj.data.clamp_(-5.0, 5.0)
        
        # Log progress
        if (epoch + 1) % config['log_interval'] == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {losses['total'].item():.4f} | "
                  f"Reconstruction: {losses['reconstruction'].item():.4f}")
    
    # === 7. EVALUATION ===
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    adjacency = model.get_adjacency()
    learned_edges = evaluator.extract_learned_edges(adjacency, threshold=config['edge_threshold'])
    metrics = evaluator.evaluate(learned_edges)
    evaluator.print_metrics(metrics)
    
    # === 8. SAVE RESULTS ===
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics
    }, output_dir / 'phase1_model.pt')
    
    # Save adjacency matrix
    torch.save(adjacency, output_dir / 'phase1_adjacency.pt')
    
    # Save learned edges
    with open(output_dir / 'phase1_edges.txt', 'w') as f:
        f.write("Learned Causal Edges (Phase 1)\n")
        f.write("=" * 70 + "\n\n")
        for var_a, var_b in sorted(learned_edges):
            f.write(f"{var_a} -> {var_b}\n")
    
    # Save metrics
    with open(output_dir / 'phase1_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, metrics


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
        'n_hops': 1,  # Single-hop reasoning for Phase 1
        'log_interval': 50,
        
        # Evaluation
        'edge_threshold': 0.3,
        
        # Output
        'output_dir': 'results/phase1'
    }
    
    # Train
    model, metrics = train_phase1(config)
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Edge Recall: {metrics['edge_recall']:.1%}")
    print(f"Directed Recall: {metrics['directed_recall']:.1%}")
    print(f"Orientation Accuracy: {metrics['orientation_accuracy']:.1%}")

