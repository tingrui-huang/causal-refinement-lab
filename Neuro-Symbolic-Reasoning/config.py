"""
Neuro-Symbolic Training Configuration

This file now imports from the unified config at project root.
All settings are managed in ../config.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from unified config (use absolute import to avoid circular import)
import importlib.util
spec = importlib.util.spec_from_file_location("unified_config", PROJECT_ROOT / "config.py")
unified_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified_config)

# Extract needed variables
DATASET = unified_config.DATASET
DATASET_CONFIGS = unified_config.DATASET_CONFIGS
get_current_dataset_config = unified_config.get_current_dataset_config
LLM_MODEL = unified_config.LLM_MODEL
USE_LLM_PRIOR = unified_config.USE_LLM_PRIOR
LEARNING_RATE = unified_config.LEARNING_RATE
N_EPOCHS = unified_config.N_EPOCHS
N_HOPS = unified_config.N_HOPS
BATCH_SIZE = unified_config.BATCH_SIZE
LAMBDA_GROUP_LASSO = unified_config.LAMBDA_GROUP_LASSO
LAMBDA_CYCLE = unified_config.LAMBDA_CYCLE
THRESHOLD = unified_config.THRESHOLD
LOG_INTERVAL = unified_config.LOG_INTERVAL
VERBOSE = unified_config.VERBOSE
RANDOM_SEED = unified_config.RANDOM_SEED
DEVICE = unified_config.DEVICE
EARLY_STOPPING = unified_config.EARLY_STOPPING
PATIENCE = unified_config.PATIENCE
TRAINING_RESULTS_DIR = unified_config.TRAINING_RESULTS_DIR
get_training_config = unified_config.get_training_config
print_unified_config = unified_config.print_config
validate_config = unified_config.validate_config

# Legacy imports (kept for backward compatibility)
from_config = (
)

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================
# These are kept for backward compatibility with train.py

# Base directory
BASE_DIR = Path(__file__).parent

# Current dataset config
CURRENT_DATASET_CONFIG = get_current_dataset_config()

# Results directory
RESULTS_DIR = TRAINING_RESULTS_DIR

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_config():
    """
    Get complete configuration for training
    
    This is the main function used by train.py
    """
    return get_training_config()


def print_config():
    """Print training configuration"""
    cfg = get_config()
    
    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    
    print("\nDataset Settings:")
    print(f"  Dataset:           {cfg['dataset_name']}")
    print(f"  Data path:         {cfg['data_path']}")
    print(f"  Metadata path:     {cfg['metadata_path']}")
    print(f"  Ground truth:      {cfg['ground_truth_path']}")
    
    print("\nLLM Settings:")
    print(f"  LLM Model:         {cfg['llm_model'] if cfg['llm_model'] else 'None (FCI only)'}")
    print(f"  Use LLM Prior:     {cfg['use_llm_prior']}")
    if cfg['use_llm_prior']:
        print(f"  FCI Skeleton:      {cfg['fci_skeleton_path']}")
        print(f"  LLM Direction:     {cfg['llm_direction_path']}")
    else:
        print(f"  FCI Skeleton:      {cfg['fci_skeleton_path']}")
    
    print("\nTraining Hyperparameters:")
    print(f"  Learning Rate:     {cfg['learning_rate']}")
    print(f"  Epochs:            {cfg['n_epochs']}")
    print(f"  N Hops:            {cfg['n_hops']}")
    print(f"  Lambda Lasso:      {cfg['lambda_group_lasso']}")
    print(f"  Lambda Cycle:      {cfg['lambda_cycle']}")
    print(f"  Threshold:         {cfg['threshold']}")
    
    print("\nOutput Settings:")
    print(f"  Results Dir:       {cfg['results_dir']}")
    print(f"  Verbose:           {cfg['verbose']}")
    print(f"  Log Interval:      {cfg['log_interval']}")
    
    print("\n" + "=" * 80)


def get_llm_short_name():
    """Get short name for LLM model"""
    if not LLM_MODEL:
        return None
    
    llm_mapping = {
        'gpt-3.5-turbo': 'gpt35',
        'gpt-4': 'gpt4',
        'zephyr-7b': 'zephyr',
        'mistral-7b': 'mistral',
    }
    
    return llm_mapping.get(LLM_MODEL, LLM_MODEL.replace('-', '').replace('.', ''))


# ============================================================================
# VALIDATION
# ============================================================================
def validate_training_config():
    """Validate training configuration"""
    cfg = get_config()
    
    # Check if data files exist
    if not Path(cfg['data_path']).exists():
        raise FileNotFoundError(f"Data file not found: {cfg['data_path']}")
    
    if not Path(cfg['metadata_path']).exists():
        raise FileNotFoundError(f"Metadata file not found: {cfg['metadata_path']}")
    
    # Check FCI skeleton
    if not cfg['fci_skeleton_path']:
        print("[WARN] No FCI skeleton found. Will use ground truth skeleton if available.")
    elif not Path(cfg['fci_skeleton_path']).exists():
        raise FileNotFoundError(f"FCI skeleton not found: {cfg['fci_skeleton_path']}")
    
    # Check LLM prior requirement
    if cfg['use_llm_prior'] and not cfg['llm_direction_path']:
        raise ValueError(f"USE_LLM_PRIOR is True but no LLM direction path found for {cfg['dataset_name']}")
    
    if cfg['use_llm_prior'] and cfg['llm_direction_path']:
        if not Path(cfg['llm_direction_path']).exists():
            raise FileNotFoundError(f"LLM direction file not found: {cfg['llm_direction_path']}")
    
    print("[OK] Training configuration validated")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print_config()
    print("\n[INFO] All settings are managed in ../config.py")
    print("[INFO] Edit ../config.py to change any settings")
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    try:
        validate_training_config()
    except Exception as e:
        print(f"[ERROR] {e}")
