"""
Reorganize Results Script

Moves existing results from flat structure to dataset-organized structure.

Old structure:
results/
├── no_llm/
├── gpt35/
└── ...

New structure:
results/
├── alarm/
│   ├── no_llm_TIMESTAMP/
│   ├── gpt35_TIMESTAMP/
│   └── ...
└── ...
"""

import shutil
from pathlib import Path
from datetime import datetime


def reorganize_results(results_dir: str = 'results', dataset_name: str = 'alarm'):
    """
    Reorganize results into dataset-specific folders
    
    Args:
        results_dir: Base results directory
        dataset_name: Name of dataset (default: 'alarm')
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return
    
    # Create dataset directory
    dataset_dir = results_path / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    print(f"Reorganizing results for dataset: {dataset_name}")
    print(f"Target directory: {dataset_dir}")
    print()
    
    # Get all subdirectories
    subdirs = [d for d in results_path.iterdir() if d.is_dir() and d.name != dataset_name]
    
    moved_count = 0
    skipped_count = 0
    
    for subdir in subdirs:
        # Skip if already in dataset folder
        if subdir.parent.name == dataset_name:
            continue
        
        # Check if it's a result directory (has .pt files)
        has_results = any(subdir.glob('*.pt'))
        
        if has_results:
            # Move to dataset directory
            target_dir = dataset_dir / subdir.name
            
            if target_dir.exists():
                print(f"[SKIP] {subdir.name} (already exists in {dataset_name}/)")
                skipped_count += 1
            else:
                print(f"[MOVE] {subdir.name} -> {dataset_name}/{subdir.name}")
                shutil.move(str(subdir), str(target_dir))
                moved_count += 1
        else:
            print(f"[SKIP] {subdir.name} (not a result directory)")
            skipped_count += 1
    
    print()
    print("=" * 70)
    print("REORGANIZATION COMPLETE")
    print("=" * 70)
    print(f"Moved:   {moved_count} directories")
    print(f"Skipped: {skipped_count} directories")
    print()
    print(f"Results are now organized in: {dataset_dir}")


def create_readme(results_dir: str = 'results'):
    """Create a README explaining the new structure"""
    readme_path = Path(results_dir) / 'README.md'
    
    content = """# Results Directory

This directory contains experimental results organized by dataset.

## Structure

```
results/
├── alarm/                          # ALARM dataset results
│   ├── no_llm_20251219_140530/    # Run without LLM prior
│   │   ├── model.pt               # Trained model
│   │   ├── adjacency.pt           # Learned adjacency matrix
│   │   ├── config.json            # Configuration
│   │   ├── evaluation_results.json # Metrics (JSON)
│   │   └── evaluation_results.txt  # Metrics (human-readable)
│   ├── gpt35_20251219_141020/     # Run with GPT-3.5 prior
│   └── ...
├── tuebingen/                      # Tuebingen dataset results
└── ...
```

## Naming Convention

Run directories follow this pattern:
```
{llm_model}_{timestamp}[_cycle{lambda_cycle}][_lasso{lambda_lasso}]
```

Examples:
- `no_llm_20251219_140530` - No LLM, default hyperparameters
- `gpt35_20251219_141020` - GPT-3.5, default hyperparameters
- `gpt35_20251219_141030_cycle0.01_lasso0.1` - GPT-3.5 with custom hyperparameters

## Files in Each Run

### model.pt
PyTorch model state dict

### adjacency.pt
Learned adjacency matrix (n_states × n_states)

### config.json
Complete configuration including:
- Dataset information
- LLM model (if used)
- Hyperparameters
- File paths
- Timestamp

### evaluation_results.json
Evaluation metrics in JSON format:
- Metadata (dataset, LLM, timestamp)
- Configuration
- Timing information
- Metrics (precision, recall, F1, SHD, etc.)
- Learned edges
- Ground truth edges

### evaluation_results.txt
Human-readable version of evaluation results

## Usage

### Using ResultManager

```python
from modules.result_manager import ResultManager

# Initialize manager
manager = ResultManager()

# Create new run directory
run_dir = manager.create_run_directory('alarm', 'gpt35')

# Save results
manager.save_model(model, run_dir)
manager.save_adjacency(adjacency, run_dir)
manager.save_config(config, run_dir)

# List all runs
runs = manager.list_runs('alarm')

# Get run information
info = manager.get_run_info('alarm', 'gpt35_20251219_141020')
```

### Using Evaluator

```python
from modules.evaluator import CausalGraphEvaluator

# Initialize evaluator
evaluator = CausalGraphEvaluator(
    ground_truth_path='data/alarm/alarm.bif',
    var_structure=var_structure
)

# Evaluate and save results
metrics = evaluator.evaluate(learned_edges)
evaluator.save_results(
    metrics=metrics,
    learned_edges=learned_edges,
    output_dir=run_dir,
    config=config,
    timing_info=timing_info
)
```

## Migration

Old results (flat structure) have been moved to `alarm/` directory.
If you have results for other datasets, organize them similarly.
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"README created: {readme_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize results into dataset-specific folders')
    parser.add_argument('--results-dir', type=str, default='../results', 
                       help='Base results directory (default: ../results)')
    parser.add_argument('--dataset', type=str, default='alarm',
                       help='Dataset name (default: alarm)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually moving files')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be moved")
        print()
    
    # Reorganize results
    if not args.dry_run:
        reorganize_results(args.results_dir, args.dataset)
        create_readme(args.results_dir)
    else:
        print("Would reorganize results (use without --dry-run to actually move files)")
