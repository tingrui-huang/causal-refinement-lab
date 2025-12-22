"""
Main Training Script

Simply run this script to train with the settings in config.py
To change dataset or LLM, edit config.py
"""

import time
import torch
import torch.optim as optim
from datetime import datetime

# Import configuration
import config

# Import modules
from modules.data_loader import CausalDataLoader
from modules.prior_builder import PriorBuilder
from modules.model import CausalDiscoveryModel
from modules.loss import LossComputer
from modules.evaluator import CausalGraphEvaluator
from modules.result_manager import ResultManager
from modules.metrics import compute_unresolved_ratio, compute_sparsity_metrics
from modules.ground_truth_loader import GroundTruthLoader, TuebingenEvaluator


def main():
    """Main training function"""
    
    # ========================================================================
    # PRINT CONFIGURATION
    # ========================================================================
    config.print_config()
    
    # Validate configuration
    try:
        config.validate_config()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] Configuration Error: {e}")
        print("\nPlease fix the configuration in config.py and try again.")
        return
    
    print()
    
    # Get config dict
    cfg = config.get_config()
    
    # ========================================================================
    # INITIALIZE TIMING
    # ========================================================================
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
    result_manager = ResultManager(base_dir=cfg['results_dir'])
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    t_start = time.time()
    
    data_loader = CausalDataLoader(
        data_path=cfg['data_path'],
        metadata_path=cfg['metadata_path']
    )
    
    data = data_loader.load_data()
    var_structure = data_loader.get_variable_structure()
    
    timing['data_loading'] = time.time() - t_start
    print(f"\n[OK] Data loaded in {timing['data_loading']:.2f}s")
    print()
    
    # ========================================================================
    # PRIOR BUILDING
    # ========================================================================
    print("=" * 80)
    print("STEP 2: BUILDING PRIORS")
    print("=" * 80)
    t_start = time.time()
    
    prior_builder = PriorBuilder(
        var_structure=var_structure,
        dataset_name=cfg['dataset_name']
    )
    
    priors = prior_builder.get_all_priors(
        fci_skeleton_path=cfg['fci_skeleton_path'],
        llm_direction_path=cfg['llm_direction_path'],
        use_llm_prior=cfg['use_llm_prior']
    )
    
    timing['prior_building'] = time.time() - t_start
    print(f"\n[OK] Priors built in {timing['prior_building']:.2f}s")
    print()
    
    # ========================================================================
    # COMPUTE FCI BASELINE
    # ========================================================================
    fci_baseline_unresolved_ratio = None
    if cfg.get('fci_skeleton_path'):
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'refactored'))
            from evaluate_fci import parse_fci_csv
            
            # Parse FCI CSV to get edge type breakdown
            fci_directed, fci_undirected, edge_counts = parse_fci_csv(cfg['fci_skeleton_path'])
            
            # Calculate unresolved ratio (all non-directed edges)
            total_edges = sum(edge_counts.values())
            directed_edges = edge_counts.get('directed', 0)
            unresolved_edges = total_edges - directed_edges
            fci_baseline_unresolved_ratio = unresolved_edges / total_edges if total_edges > 0 else 0
            
            print("=" * 80)
            print("FCI BASELINE (No LLM, No Training)")
            print("=" * 80)
            print(f"Total FCI edges: {total_edges}")
            print(f"  Directed (->):       {directed_edges:3d}  ({directed_edges/total_edges*100:.1f}%) [direction resolved]")
            print(f"  Unresolved:          {unresolved_edges:3d}  ({fci_baseline_unresolved_ratio*100:.1f}%) [direction NOT resolved]")
            print(f"    - Bidirected (<->): {edge_counts.get('bidirected', 0):3d}")
            print(f"    - Partial (o->):    {edge_counts.get('partial', 0):3d}")
            print(f"    - Undirected (o-o): {edge_counts.get('undirected', 0):3d}")
            print(f"    - Tail-tail (--):   {edge_counts.get('tail-tail', 0):3d}")
            print(f"\nFCI Unresolved Ratio (Baseline): {fci_baseline_unresolved_ratio*100:.1f}%")
            print("=" * 80)
            print()
        except Exception as e:
            print(f"[WARN] Could not compute FCI baseline: {e}")
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    print("=" * 80)
    print("STEP 3: INITIALIZING MODEL")
    print("=" * 80)
    t_start = time.time()
    
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    loss_computer = LossComputer(
        block_structure=priors['blocks'],
        penalty_weights=priors['penalty_weights']
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['learning_rate']
    )
    
    timing['model_init'] = time.time() - t_start
    print(f"[OK] Model initialized in {timing['model_init']:.2f}s")
    print()
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    print("=" * 80)
    print("STEP 4: TRAINING")
    print("=" * 80)
    print()
    
    t_start = time.time()
    n_epochs = cfg['n_epochs']
    log_interval = cfg['log_interval']
    
    # Initialize history tracking
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_reconstruction': [],
        'loss_group_lasso': [],
        'loss_cycle': [],
        'unresolved_ratio': [],
        'unresolved_count': [],
        'resolved_count': [],
        'overall_sparsity': [],
        'active_connections': [],
        'active_blocks': [],
        'block_sparsity': []
    }
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data, n_hops=cfg['n_hops'])
        
        # Compute loss
        adjacency = model.get_adjacency()
        loss_dict = loss_computer.compute_total_loss(
            predictions=logits,
            targets=data,
            adjacency=adjacency,
            lambda_group=cfg['lambda_group_lasso'],
            lambda_cycle=cfg['lambda_cycle']
        )
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Clamp weights to prevent explosion
        with torch.no_grad():
            model.raw_adj.data.clamp_(-5.0, 5.0)
        
        # Compute monitoring metrics
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            with torch.no_grad():
                # Unresolved ratio (key metric for direction learning / symmetry breaking)
                bidir_stats = compute_unresolved_ratio(
                    adjacency, 
                    priors['blocks'],
                    threshold=cfg['threshold']
                )
                
                # Sparsity metrics
                sparsity_stats = compute_sparsity_metrics(
                    adjacency,
                    priors['skeleton_mask'],
                    priors['blocks'],
                    threshold=cfg['threshold']
                )
                
                # Record history
                history['epoch'].append(epoch + 1)
                history['loss_total'].append(loss.item())
                history['loss_reconstruction'].append(loss_dict['reconstruction'].item())
                history['loss_group_lasso'].append(loss_dict['weighted_group_lasso'].item())
                history['loss_cycle'].append(loss_dict['cycle_consistency'].item())
                history['unresolved_ratio'].append(bidir_stats['unresolved_ratio'])
                history['unresolved_count'].append(bidir_stats['unresolved'])
                history['resolved_count'].append(bidir_stats['resolved'])
                history['overall_sparsity'].append(sparsity_stats['overall_sparsity'])
                history['active_connections'].append(sparsity_stats['active_connections'])
                history['active_blocks'].append(sparsity_stats['active_blocks'])
                history['block_sparsity'].append(sparsity_stats['block_sparsity'])
                
                # Print comprehensive log
                print(f"\nEpoch {epoch+1:3d}/{n_epochs}")
                print(f"  Loss: {loss.item():.4f} "
                      f"(Recon: {loss_dict['reconstruction'].item():.4f}, "
                      f"Lasso: {loss_dict['weighted_group_lasso'].item():.4f}, "
                      f"Cycle: {loss_dict['cycle_consistency'].item():.4f})")
                print(f"  Direction: Unresolved {bidir_stats['unresolved_ratio']*100:.1f}% "
                      f"({bidir_stats['unresolved']}/{bidir_stats['total_pairs']} pairs)")
                print(f"  Sparsity: Overall {sparsity_stats['overall_sparsity']*100:.1f}%, "
                      f"Block {sparsity_stats['block_sparsity']*100:.1f}% "
                      f"({sparsity_stats['active_blocks']}/{sparsity_stats['total_blocks']} active)")
    
    timing['training'] = time.time() - t_start
    print(f"\n[OK] Training completed in {timing['training']:.2f}s")
    print(f"  Average: {timing['training']/n_epochs:.3f}s per epoch")
    
    # Print training summary
    if len(history['unresolved_ratio']) > 0:
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"\nDirection Learning (Symmetry Breaking):")
        
        if fci_baseline_unresolved_ratio is not None:
            print(f"  FCI Baseline (edges without direction):     {fci_baseline_unresolved_ratio*100:.1f}%")
            # print(f"  After LLM Prior (Epoch 0, symmetric pairs): {history['unresolved_ratio'][0]*100:.1f}%")
            print(f"  After Training (Final, symmetric pairs):    {history['unresolved_ratio'][-1]*100:.1f}%")
            total_improvement = (fci_baseline_unresolved_ratio - history['unresolved_ratio'][-1]) * 100
            print(f"  Total Improvement (FCI → Final): {total_improvement:+.1f}%")
        else:
            print(f"  Unresolved Ratio: {history['unresolved_ratio'][0]*100:.1f}% -> {history['unresolved_ratio'][-1]*100:.1f}%")
            change = (history['unresolved_ratio'][-1] - history['unresolved_ratio'][0]) * 100
            status = '[GOOD]' if change < 0 else '[NEEDS TUNING]'
            print(f"  Change: {change:+.1f}% {status}")
        
        print(f"\nSparsity Evolution:")
        print(f"  Overall: {history['overall_sparsity'][0]*100:.1f}% -> {history['overall_sparsity'][-1]*100:.1f}%")
        print(f"  Active Connections: {history['active_connections'][0]} -> {history['active_connections'][-1]}")
        print(f"  Active Blocks: {history['active_blocks'][0]}/{history['active_blocks'][0]} -> {history['active_blocks'][-1]}/{len(priors['blocks'])}")
        print("=" * 80)
    print()
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("=" * 80)
    print("STEP 5: EVALUATION")
    print("=" * 80)
    print()
    
    t_start = time.time()
    
    # Get adjacency matrix
    adjacency = model.get_adjacency()
    
    # Determine evaluation type based on dataset
    if cfg['ground_truth_path']:
        gt_type = cfg.get('ground_truth_type', 'bif')
        
        if gt_type == 'json':
            # Tuebingen-style pairwise evaluation
            print("[INFO] Using pairwise evaluation (Tuebingen format)")
            
            gt_loader = GroundTruthLoader(cfg['ground_truth_path'], 'json')
            tuebingen_eval = TuebingenEvaluator(gt_loader, var_structure)
            
            # Extract pair_id from dataset name if available
            pair_id = cfg['dataset_name'] if 'pair' in cfg['dataset_name'] else None
            
            results = tuebingen_eval.evaluate_pairwise(adjacency, pair_id)
            tuebingen_eval.print_results(results)
            
            # Convert to metrics dict for consistency
            metrics = {
                'predicted_direction': results['predicted_direction'],
                'confidence': results['confidence'],
                'correct': results['correct'] if results['correct'] is not None else 'N/A'
            }
            learned_edges = set()  # Not applicable for pairwise
            
        else:
            # Graph-based evaluation (ALARM, Insurance, Sachs, etc.)
            gt_type = cfg.get('ground_truth_type', 'bif')
            print(f"[INFO] Using graph-based evaluation ({gt_type} format)")
            
            evaluator = CausalGraphEvaluator(
                ground_truth_path=cfg['ground_truth_path'],
                var_structure=var_structure,
                ground_truth_type=gt_type
            )
            
            # Extract learned edges
            learned_edges = evaluator.extract_learned_edges(
                adjacency,
                threshold=cfg['threshold']
            )
            
            # Evaluate
            metrics = evaluator.evaluate(learned_edges)
            evaluator.print_metrics(metrics)
    else:
        print("[WARN] No ground truth available for this dataset")
        print("  Skipping evaluation...")
        metrics = {}
        learned_edges = set()
    
    timing['evaluation'] = time.time() - t_start
    print(f"\n[OK] Evaluation completed in {timing['evaluation']:.2f}s")
    print()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("=" * 80)
    print("STEP 6: SAVING RESULTS")
    print("=" * 80)
    print()
    
    t_start = time.time()
    
    # Create run directory
    run_dir = result_manager.create_run_directory(
        dataset_name=cfg['dataset_name'],
        llm_model=config.get_llm_short_name(),
        config=cfg
    )
    
    print(f"Run directory: {run_dir}")
    print()
    
    # Save model and adjacency
    result_manager.save_model(model, run_dir)
    result_manager.save_adjacency(adjacency, run_dir)
    result_manager.save_config(cfg, run_dir)
    result_manager.save_history(history, run_dir)
    
    # Calculate total time
    timing['saving'] = time.time() - t_start
    timing['total'] = time.time() - timing['total_start']
    
    # Save evaluation results with timing
    if cfg['ground_truth_path']:
        evaluator.save_results(
            metrics=metrics,
            learned_edges=learned_edges,
            output_dir=str(run_dir),
            config=cfg,
            timing_info=timing
        )
    
    print(f"\n[OK] Results saved in {timing['saving']:.2f}s")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Total time:       {timing['total']:.2f}s ({timing['total']/60:.1f} min)")
    print(f"  Data loading:   {timing['data_loading']:.2f}s ({timing['data_loading']/timing['total']*100:.1f}%)")
    print(f"  Prior building: {timing['prior_building']:.2f}s ({timing['prior_building']/timing['total']*100:.1f}%)")
    print(f"  Model init:     {timing['model_init']:.2f}s ({timing['model_init']/timing['total']*100:.1f}%)")
    print(f"  Training:       {timing['training']:.2f}s ({timing['training']/timing['total']*100:.1f}%)")
    print(f"  Evaluation:     {timing['evaluation']:.2f}s ({timing['evaluation']/timing['total']*100:.1f}%)")
    print(f"  Saving:         {timing['saving']:.2f}s ({timing['saving']/timing['total']*100:.1f}%)")
    print()
    
    if metrics:
        print(f"Edge F1:          {metrics['edge_f1']:.1%}")
        print(f"Orient. Accuracy: {metrics['orientation_accuracy']:.1%}")
        print(f"Skeleton SHD:     {metrics['skeleton_shd']} (undirected)")
        print(f"Full SHD:         {metrics['full_shd']} (directed, standard metric)")
        print()
    
    print(f"Results saved to: {run_dir}")
    print("=" * 80)
    
    return {
        'metrics': metrics,
        'timing': timing,
        'run_dir': run_dir
    }


if __name__ == "__main__":
    results = main()
