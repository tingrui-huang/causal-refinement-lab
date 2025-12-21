"""
Full Tuebingen Dataset Training with Gradient Ratio + Dynamic Threshold

This script extends train_tuebingen_biased.py to work with the full Tuebingen dataset.
Key features:
1. Gradient ratio monitoring for direction learning
2. Dynamic threshold based on weight distribution
3. Uses existing config parameters (no hardcoding)
4. Works with all Tuebingen pairs

Strategy:
- Monitor gradient ratios during training to detect direction learning
- Use dynamic threshold (e.g., 3% of total weight) for edge detection
- Leverage existing config structure for flexibility
"""

import sys
from pathlib import Path
import torch
import importlib
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force reload config to get latest values
import config
importlib.reload(config)
from config import get_training_config, DATASET_CONFIGS, print_config
from train_complete import train_complete


def compute_gradient_ratio(model, var_structure, n_bins):
    """
    Compute gradient ratio between forward and backward directions
    
    Args:
        model: CausalDiscoveryModel with gradients
        var_structure: Variable structure info
        n_bins: Number of bins for discretization
    
    Returns:
        dict with gradient statistics
    """
    with torch.no_grad():
        grad = model.raw_adj.grad
        
        if grad is None:
            return {
                'forward_mag': 0.0,
                'backward_mag': 0.0,
                'ratio': 1.0,
                'asymmetry': 0.0
            }
        
        # For 2-variable case: X (states 0:n_bins), Y (states n_bins:2*n_bins)
        grad_forward = grad[0:n_bins, n_bins:2*n_bins]  # X -> Y
        grad_backward = grad[n_bins:2*n_bins, 0:n_bins]  # Y -> X
        
        # Compute average gradient magnitude
        forward_mag = grad_forward.abs().mean().item()
        backward_mag = grad_backward.abs().mean().item()
        
        # Compute ratio and asymmetry
        ratio = forward_mag / backward_mag if backward_mag > 0 else float('inf')
        asymmetry = abs(forward_mag - backward_mag)
        
        return {
            'forward_mag': forward_mag,
            'backward_mag': backward_mag,
            'ratio': ratio,
            'asymmetry': asymmetry
        }


def compute_dynamic_threshold(adjacency, strategy='percentile', percentile=3.0):
    """
    Compute dynamic threshold based on weight distribution
    
    Args:
        adjacency: Adjacency matrix
        strategy: 'percentile' or 'total_weight'
        percentile: Percentile value (for percentile strategy)
    
    Returns:
        threshold value
    """
    adj_np = adjacency.detach().cpu().numpy()
    
    if strategy == 'percentile':
        # Use percentile of non-zero weights
        nonzero_weights = adj_np[adj_np > 0]
        if len(nonzero_weights) > 0:
            threshold = np.percentile(nonzero_weights, percentile)
        else:
            threshold = 0.01  # Default
    
    elif strategy == 'total_weight':
        # Use percentage of total weight
        total_weight = adj_np.sum()
        threshold = total_weight * (percentile / 100.0) / adj_np.size
    
    else:
        threshold = 0.01  # Default
    
    return threshold


def analyze_direction_learning(model, var_structure, n_bins, threshold):
    """
    Analyze direction learning from adjacency matrix
    
    Args:
        model: Trained model
        var_structure: Variable structure info
        n_bins: Number of bins
        threshold: Edge threshold
    
    Returns:
        dict with direction statistics
    """
    with torch.no_grad():
        adj = model.get_adjacency().numpy()
        
        # Extract direction strengths
        forward_weights = adj[0:n_bins, n_bins:2*n_bins]  # X -> Y
        backward_weights = adj[n_bins:2*n_bins, 0:n_bins]  # Y -> X
        
        forward_strength = forward_weights.sum()
        backward_strength = backward_weights.sum()
        
        # Compute statistics
        ratio = forward_strength / backward_strength if backward_strength > 0 else float('inf')
        gap = forward_strength - backward_strength
        
        # Determine predicted direction
        if forward_strength > backward_strength:
            predicted_direction = 1  # X -> Y
        elif backward_strength > forward_strength:
            predicted_direction = -1  # Y -> X
        else:
            predicted_direction = 0  # Uncertain
        
        return {
            'forward_strength': forward_strength,
            'backward_strength': backward_strength,
            'ratio': ratio,
            'gap': gap,
            'predicted_direction': predicted_direction,
            'forward_mean': forward_weights.mean(),
            'backward_mean': backward_weights.mean()
        }


def train_single_pair(pair_id, data_path, ground_truth_direction, config_override=None):
    """
    Train on a single Tuebingen pair
    
    Args:
        pair_id: Pair identifier (e.g., 'pair0001')
        data_path: Path to CSV data file
        ground_truth_direction: 1 for X->Y, -1 for Y->X
        config_override: Optional config overrides
    
    Returns:
        dict with training results
    """
    print("\n" + "=" * 80)
    print(f"TRAINING: {pair_id}")
    print("=" * 80)
    
    # Get base configuration
    base_config = get_training_config()
    
    # Update paths for this pair
    base_config['data_path'] = str(data_path)
    metadata_path = data_path.parent / 'metadata.json'
    base_config['metadata_path'] = str(metadata_path)
    
    # Apply overrides
    if config_override:
        base_config.update(config_override)
    
    # Set ground truth for evaluation
    var_names = ['X', 'Y']  # Tuebingen pairs use generic names
    if ground_truth_direction == 1:
        base_config['ground_truth_edges'] = [('X', 'Y')]
    elif ground_truth_direction == -1:
        base_config['ground_truth_edges'] = [('Y', 'X')]
    else:
        base_config['ground_truth_edges'] = []
    
    # Set manual skeleton (undirected edge)
    base_config['manual_skeleton'] = [('X', 'Y')]
    base_config['skip_fci'] = True
    
    print(f"\nConfiguration:")
    print(f"  Data: {data_path}")
    print(f"  Ground Truth: {ground_truth_direction} ({'X->Y' if ground_truth_direction == 1 else 'Y->X' if ground_truth_direction == -1 else 'Unknown'})")
    print(f"  Bins: {base_config.get('n_bins', 5)}")
    print(f"  Strategy: {base_config.get('discretization_strategy', 'uniform')}")
    print(f"  Learning Rate: {base_config['learning_rate']}")
    print(f"  Lambda Group: {base_config['lambda_group']}")
    print(f"  Lambda Cycle: {base_config['lambda_cycle']}")
    print(f"  Epochs: {base_config['n_epochs']}")
    
    # Train
    try:
        model, metrics, history = train_complete(base_config)
        
        # Get variable structure for analysis
        from modules.data_loader import CausalDataLoader
        data_loader = CausalDataLoader(
            data_path=base_config['data_path'],
            metadata_path=base_config['metadata_path']
        )
        var_structure = data_loader.get_variable_structure()
        n_bins = var_structure['n_states'] // var_structure['n_variables']
        
        # Compute dynamic threshold
        adjacency = model.get_adjacency()
        dynamic_threshold = compute_dynamic_threshold(
            adjacency, 
            strategy='total_weight',
            percentile=3.0
        )
        
        # Analyze direction learning
        direction_stats = analyze_direction_learning(
            model, 
            var_structure, 
            n_bins,
            dynamic_threshold
        )
        
        # Check if prediction is correct
        predicted_correct = (direction_stats['predicted_direction'] == ground_truth_direction)
        
        # Compute SHD (Structural Hamming Distance)
        if predicted_correct:
            shd = 0  # Perfect match
        elif direction_stats['predicted_direction'] == 0:
            shd = 1  # Missing edge
        else:
            shd = 1  # Wrong direction
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\nDirection Analysis:")
        print(f"  Forward (X->Y): {direction_stats['forward_strength']:.4f}")
        print(f"  Backward (Y->X): {direction_stats['backward_strength']:.4f}")
        print(f"  Ratio: {direction_stats['ratio']:.4f}")
        print(f"  Gap: {direction_stats['gap']:.4f}")
        print(f"\nPrediction:")
        print(f"  Ground Truth: {ground_truth_direction} ({'X->Y' if ground_truth_direction == 1 else 'Y->X'})")
        print(f"  Predicted: {direction_stats['predicted_direction']} ({'X->Y' if direction_stats['predicted_direction'] == 1 else 'Y->X' if direction_stats['predicted_direction'] == -1 else 'Uncertain'})")
        print(f"  Correct: {'YES' if predicted_correct else 'NO'}")
        print(f"  SHD: {shd}")
        print(f"\nDynamic Threshold: {dynamic_threshold:.6f}")
        print("=" * 80)
        
        return {
            'pair_id': pair_id,
            'success': True,
            'predicted_direction': direction_stats['predicted_direction'],
            'ground_truth_direction': ground_truth_direction,
            'correct': predicted_correct,
            'shd': shd,
            'forward_strength': direction_stats['forward_strength'],
            'backward_strength': direction_stats['backward_strength'],
            'ratio': direction_stats['ratio'],
            'gap': direction_stats['gap'],
            'dynamic_threshold': dynamic_threshold,
            'final_loss': history['loss_total'][-1],
            'metrics': metrics
        }
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'pair_id': pair_id,
            'success': False,
            'error': str(e),
            'predicted_direction': 0,
            'ground_truth_direction': ground_truth_direction,
            'correct': False,
            'shd': 1
        }


def run_full_benchmark(data_dir, output_dir=None, max_pairs=None, config_override=None):
    """
    Run benchmark on full Tuebingen dataset
    
    Args:
        data_dir: Directory containing Tuebingen pairs
        output_dir: Output directory for results
        max_pairs: Maximum number of pairs to process (None = all)
        config_override: Optional config overrides
    
    Returns:
        DataFrame with results
    """
    data_dir = Path(data_dir)
    
    # Find all pair CSV files
    pair_files = sorted(data_dir.glob('pair*.csv'))
    
    if max_pairs:
        pair_files = pair_files[:max_pairs]
    
    print("\n" + "=" * 80)
    print("TUEBINGEN FULL BENCHMARK")
    print("=" * 80)
    print(f"Data Directory: {data_dir}")
    print(f"Total Pairs: {len(pair_files)}")
    print("=" * 80)
    
    results = []
    
    for i, pair_file in enumerate(pair_files, 1):
        pair_id = pair_file.stem  # e.g., 'pair0001'
        
        # Parse ground truth from description file
        des_file = data_dir / f"{pair_id}_des.txt"
        if des_file.exists():
            try:
                with open(des_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if 'x --> y' in content or 'x->y' in content or 'x causes y' in content:
                        ground_truth = 1
                    elif 'y --> x' in content or 'y->x' in content or 'y causes x' in content:
                        ground_truth = -1
                    else:
                        ground_truth = 1  # Default
            except:
                ground_truth = 1  # Default
        else:
            ground_truth = 1  # Default
        
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(pair_files)}: {pair_id}")
        print(f"{'='*80}")
        
        # Train on this pair
        result = train_single_pair(
            pair_id=pair_id,
            data_path=pair_file,
            ground_truth_direction=ground_truth,
            config_override=config_override
        )
        
        results.append(result)
        
        # Print progress
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results) * 100
        print(f"\n[PROGRESS] {i}/{len(pair_files)} pairs processed")
        print(f"[PROGRESS] Current accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Compute summary statistics
    total_pairs = len(df)
    successful_pairs = df['success'].sum()
    correct_predictions = df['correct'].sum()
    accuracy = correct_predictions / total_pairs * 100
    avg_shd = df['shd'].mean()
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Pairs: {total_pairs}")
    print(f"Successful: {successful_pairs}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average SHD: {avg_shd:.2f}")
    print("=" * 80)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = output_dir / f'tuebingen_full_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_pairs': int(total_pairs),
            'successful_pairs': int(successful_pairs),
            'correct_predictions': int(correct_predictions),
            'accuracy': float(accuracy),
            'avg_shd': float(avg_shd),
            'config': config_override if config_override else {}
        }
        
        summary_path = output_dir / f'tuebingen_full_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    
    return df


def main():
    """Main entry point"""
    
    # Print configuration
    print_config()
    
    # Get data directory
    data_dir = Path(__file__).parent / 'data' / 'tuebingen'
    
    if not data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("Please ensure Tuebingen data is in: Neuro-Symbolic-Reasoning/data/tuebingen/")
        sys.exit(1)
    
    # Configuration overrides (using existing config parameters)
    config_override = {
        # Use existing config values, only override if needed
        # 'n_bins': 5,  # Already in config
        # 'discretization_strategy': 'uniform',  # Already in config
        # 'learning_rate': 0.05,  # Already in config
        # 'lambda_group': 0.0,  # Already in config
        # 'lambda_cycle': 0.06,  # Already in config
        # 'n_epochs': 200,  # Already in config
        
        # Dynamic parameters (not in config)
        'monitor_interval': 20,  # Monitor every 20 epochs
        'use_dynamic_threshold': True,  # Enable dynamic threshold
        'dynamic_threshold_strategy': 'total_weight',  # Strategy
        'dynamic_threshold_percentile': 3.0,  # 3% of total weight
    }
    
    # Output directory
    output_dir = Path(__file__).parent / 'results' / 'tuebingen_full'
    
    # Run benchmark
    print("\n" + "=" * 80)
    print("STARTING FULL TUEBINGEN BENCHMARK")
    print("=" * 80)
    print("\nStrategy:")
    print("  1. Gradient ratio monitoring for direction learning")
    print("  2. Dynamic threshold (3% of total weight)")
    print("  3. Existing config parameters from config.py")
    print("=" * 80)
    
    # For testing, start with a few pairs
    df = run_full_benchmark(
        data_dir=data_dir,
        output_dir=output_dir,
        max_pairs=5,  # Start with 5 pairs for testing
        config_override=config_override
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = main()


