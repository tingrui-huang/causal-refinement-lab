"""
Training Template with Timing and Result Management

This template shows how to use the new result management system with
comprehensive timing statistics and metadata tracking.
"""

import time
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# Import modules
from modules.data_loader import CausalDataLoader
from modules.prior_builder import PriorBuilder
from modules.model import CausalDiscoveryModel
from modules.loss import CausalDiscoveryLoss
from modules.evaluator import CausalGraphEvaluator
from modules.result_manager import ResultManager


def train_causal_discovery(config: dict):
    """
    Train causal discovery model with comprehensive timing and result tracking
    
    Args:
        config: Configuration dictionary with all parameters
    """
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print("=" * 80)
    print("CAUSAL DISCOVERY TRAINING")
    print("=" * 80)
    print(f"Dataset: {config['dataset_name']}")
    print(f"LLM Model: {config.get('llm_model', 'None')}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize timing
    timing = {
        'total_start': time.time(),
        'data_loading': 0,
        'prior_building': 0,
        'model_init': 0,
        'training': 0,
        'evaluation': 0,
        'saving': 0
    }
    
    # Initialize result manager
    result_manager = ResultManager(base_dir=config.get('results_dir', 'results'))
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("Loading data...")
    t_start = time.time()
    
    data_loader = CausalDataLoader(
        data_path=config['data_path'],
        metadata_path=config['metadata_path']
    )
    
    data = data_loader.load_data()
    var_structure = data_loader.get_variable_structure()
    
    timing['data_loading'] = time.time() - t_start
    print(f"✓ Data loaded in {timing['data_loading']:.2f}s")
    print()
    
    # ========================================================================
    # PRIOR BUILDING
    # ========================================================================
    print("Building priors...")
    t_start = time.time()
    
    prior_builder = PriorBuilder(
        var_structure=var_structure,
        dataset_name=config['dataset_name']
    )
    
    priors = prior_builder.get_all_priors(
        fci_skeleton_path=config['fci_skeleton_path'],
        llm_direction_path=config.get('llm_direction_path'),
        use_llm_prior=config.get('use_llm_prior', False)
    )
    
    timing['prior_building'] = time.time() - t_start
    print(f"✓ Priors built in {timing['prior_building']:.2f}s")
    print()
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    print("Initializing model...")
    t_start = time.time()
    
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    loss_fn = CausalDiscoveryLoss(
        blocks=priors['blocks'],
        penalty_weights=priors['penalty_weights'],
        lambda_group_lasso=config.get('lambda_group_lasso', 0.1),
        lambda_cycle=config.get('lambda_cycle', 0.01)
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.01)
    )
    
    timing['model_init'] = time.time() - t_start
    print(f"✓ Model initialized in {timing['model_init']:.2f}s")
    print()
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    t_start = time.time()
    n_epochs = config.get('n_epochs', 1000)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data)
        
        # Compute loss
        adjacency = model.get_adjacency()
        loss, loss_components = loss_fn(logits, data, adjacency)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Project to ensure constraints
        model.project_to_constraints()
        
        # Log progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Recon: {loss_components['reconstruction']:.4f} | "
                  f"Lasso: {loss_components['group_lasso']:.4f} | "
                  f"Cycle: {loss_components['cycle']:.4f}")
    
    timing['training'] = time.time() - t_start
    print(f"\n✓ Training completed in {timing['training']:.2f}s")
    print(f"  Average: {timing['training']/n_epochs:.3f}s per epoch")
    print()
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    t_start = time.time()
    
    evaluator = CausalGraphEvaluator(
        ground_truth_path=config['ground_truth_path'],
        var_structure=var_structure
    )
    
    # Extract learned edges
    adjacency = model.get_adjacency()
    learned_edges = evaluator.extract_learned_edges(
        adjacency,
        threshold=config.get('threshold', 0.3)
    )
    
    # Evaluate
    metrics = evaluator.evaluate(learned_edges)
    evaluator.print_metrics(metrics)
    
    timing['evaluation'] = time.time() - t_start
    print(f"\n✓ Evaluation completed in {timing['evaluation']:.2f}s")
    print()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    t_start = time.time()
    
    # Create run directory
    run_dir = result_manager.create_run_directory(
        dataset_name=config['dataset_name'],
        llm_model=config.get('llm_model'),
        config=config
    )
    
    print(f"Run directory: {run_dir}")
    print()
    
    # Save model and adjacency
    result_manager.save_model(model, run_dir)
    result_manager.save_adjacency(adjacency, run_dir)
    result_manager.save_config(config, run_dir)
    
    # Calculate total time
    timing['saving'] = time.time() - t_start
    timing['total'] = time.time() - timing['total_start']
    
    # Save evaluation results with timing
    evaluator.save_results(
        metrics=metrics,
        learned_edges=learned_edges,
        output_dir=str(run_dir),
        config=config,
        timing_info=timing
    )
    
    print(f"\n✓ Results saved in {timing['saving']:.2f}s")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time:       {timing['total']:.2f}s ({timing['total']/60:.1f} min)")
    print(f"  Data loading:   {timing['data_loading']:.2f}s ({timing['data_loading']/timing['total']*100:.1f}%)")
    print(f"  Prior building: {timing['prior_building']:.2f}s ({timing['prior_building']/timing['total']*100:.1f}%)")
    print(f"  Model init:     {timing['model_init']:.2f}s ({timing['model_init']/timing['total']*100:.1f}%)")
    print(f"  Training:       {timing['training']:.2f}s ({timing['training']/timing['total']*100:.1f}%)")
    print(f"  Evaluation:     {timing['evaluation']:.2f}s ({timing['evaluation']/timing['total']*100:.1f}%)")
    print(f"  Saving:         {timing['saving']:.2f}s ({timing['saving']/timing['total']*100:.1f}%)")
    print()
    print(f"Edge F1:          {metrics['edge_f1']:.1%}")
    print(f"Orient. Accuracy: {metrics['orientation_accuracy']:.1%}")
    print(f"SHD:              {metrics['shd']}")
    print()
    print(f"Results saved to: {run_dir}")
    print("=" * 80)
    
    return {
        'metrics': metrics,
        'timing': timing,
        'run_dir': run_dir
    }


if __name__ == "__main__":
    # Example configuration for ALARM dataset with GPT-3.5
    config = {
        # Dataset
        'dataset_name': 'alarm',
        'data_path': 'data/alarm/alarm_data_10000.csv',
        'metadata_path': 'data/alarm/metadata.json',
        'ground_truth_path': 'data/alarm/alarm.bif',
        
        # FCI and LLM
        'fci_skeleton_path': 'data/alarm/edges_FCI_20251207_230824.csv',
        'llm_direction_path': 'data/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv',
        'use_llm_prior': True,
        'llm_model': 'gpt-3.5-turbo',
        
        # Hyperparameters
        'learning_rate': 0.01,
        'n_epochs': 1000,
        'lambda_group_lasso': 0.1,
        'lambda_cycle': 0.01,
        'threshold': 0.3,
        
        # Output
        'results_dir': 'results'
    }
    
    # Train
    results = train_causal_discovery(config)
    
    print("\nTraining complete!")
    print(f"Check results in: {results['run_dir']}")
