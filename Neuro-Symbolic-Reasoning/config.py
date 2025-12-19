"""
Central Configuration File

This file contains all configuration settings for the causal discovery pipeline.
Simply modify the values here to change dataset, LLM model, hyperparameters, etc.
"""

from pathlib import Path

# ============================================================================
# DATASET SELECTION
# ============================================================================
# Options: 'alarm', 'tuebingen_pair1', 'tuebingen_pair2', etc.
DATASET = 'alarm'

# ============================================================================
# LLM MODEL SELECTION
# ============================================================================
# Options: 
#   - None (no LLM prior)
#   - 'gpt-3.5-turbo' (GPT-3.5)
#   - 'gpt-4' (GPT-4)
#   - 'zephyr-7b' (Zephyr)
LLM_MODEL = 'gpt-3.5-turbo'  # Set to None for no LLM

# Use LLM prior for direction initialization?
USE_LLM_PRIOR = True if LLM_MODEL else False

# ============================================================================
# DATASET-SPECIFIC PATHS
# ============================================================================
# Base directory
BASE_DIR = Path(__file__).parent

# Dataset configurations
DATASET_CONFIGS = {
    'alarm': {
        'data_path': BASE_DIR / 'data' / 'alarm' / 'alarm_data_10000.csv',
        'metadata_path': BASE_DIR / 'data' / 'alarm' / 'metadata.json',
        'ground_truth_path': BASE_DIR / 'data' / 'alarm' / 'alarm.bif',
        'fci_skeleton_path': BASE_DIR / 'data' / 'alarm' / 'edges_FCI_20251207_230824.csv',
        'llm_direction_path': BASE_DIR / 'data' / 'alarm' / 'edges_Hybrid_FCI_LLM_20251207_230956.csv',
    },
    'tuebingen_pair1': {
        'data_path': BASE_DIR / 'data' / 'tuebingen' / 'pair0001.csv',
        'metadata_path': BASE_DIR / 'data' / 'tuebingen' / 'metadata_pair1.json',
        'ground_truth_path': None,  # Tuebingen doesn't have ground truth
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    # Add more datasets here...
}

# Get current dataset config
CURRENT_DATASET_CONFIG = DATASET_CONFIGS.get(DATASET, DATASET_CONFIGS['alarm'])

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
LEARNING_RATE = 0.01
N_EPOCHS = 200  # Pilot training: 200 epochs (same as train_complete.py)
N_HOPS = 1  # Single-hop reasoning (same as train_complete.py)
BATCH_SIZE = None  # None = full batch

# Regularization
LAMBDA_GROUP_LASSO = 0.01   # Group Lasso penalty weight (same as train_complete.py)
LAMBDA_CYCLE = 0.001        # Cycle penalty weight (same as train_complete.py)

# Evaluation
THRESHOLD = 0.1  # Threshold for edge detection (same as train_complete.py)

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
RESULTS_DIR = BASE_DIR / 'results'

# Logging
VERBOSE = True
LOG_INTERVAL = 20  # Print every N epochs

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
# Random seed
RANDOM_SEED = 42

# Device
DEVICE = 'cuda' if False else 'cpu'  # Change first False to torch.cuda.is_available()

# Early stopping
EARLY_STOPPING = False
PATIENCE = 50

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_config():
    """
    Get complete configuration as a dictionary
    
    Returns:
        Dictionary with all configuration settings
    """
    config = {
        # Dataset
        'dataset_name': DATASET,
        'data_path': str(CURRENT_DATASET_CONFIG['data_path']),
        'metadata_path': str(CURRENT_DATASET_CONFIG['metadata_path']),
        'ground_truth_path': str(CURRENT_DATASET_CONFIG['ground_truth_path']) if CURRENT_DATASET_CONFIG['ground_truth_path'] else None,
        'fci_skeleton_path': str(CURRENT_DATASET_CONFIG['fci_skeleton_path']) if CURRENT_DATASET_CONFIG['fci_skeleton_path'] else None,
        'llm_direction_path': str(CURRENT_DATASET_CONFIG['llm_direction_path']) if CURRENT_DATASET_CONFIG['llm_direction_path'] else None,
        
        # LLM
        'llm_model': LLM_MODEL,
        'use_llm_prior': USE_LLM_PRIOR,
        
        # Hyperparameters
        'learning_rate': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'n_hops': N_HOPS,
        'batch_size': BATCH_SIZE,
        'lambda_group_lasso': LAMBDA_GROUP_LASSO,
        'lambda_cycle': LAMBDA_CYCLE,
        'threshold': THRESHOLD,
        
        # Output
        'results_dir': str(RESULTS_DIR),
        'verbose': VERBOSE,
        'log_interval': LOG_INTERVAL,
        
        # Advanced
        'random_seed': RANDOM_SEED,
        'device': DEVICE,
        'early_stopping': EARLY_STOPPING,
        'patience': PATIENCE,
    }
    
    return config


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    print()
    
    print("Dataset Settings:")
    print(f"  Dataset:           {DATASET}")
    print(f"  Data path:         {CURRENT_DATASET_CONFIG['data_path']}")
    print(f"  Metadata path:     {CURRENT_DATASET_CONFIG['metadata_path']}")
    print(f"  Ground truth:      {CURRENT_DATASET_CONFIG['ground_truth_path']}")
    print()
    
    print("LLM Settings:")
    print(f"  LLM Model:         {LLM_MODEL if LLM_MODEL else 'None (No LLM)'}")
    print(f"  Use LLM Prior:     {USE_LLM_PRIOR}")
    if USE_LLM_PRIOR:
        print(f"  LLM Direction:     {CURRENT_DATASET_CONFIG['llm_direction_path']}")
    print()
    
    print("Training Hyperparameters:")
    print(f"  Learning Rate:     {LEARNING_RATE}")
    print(f"  Epochs:            {N_EPOCHS}")
    print(f"  N Hops:            {N_HOPS}")
    print(f"  Lambda Lasso:      {LAMBDA_GROUP_LASSO}")
    print(f"  Lambda Cycle:      {LAMBDA_CYCLE}")
    print(f"  Threshold:         {THRESHOLD}")
    print()
    
    print("Output Settings:")
    print(f"  Results Dir:       {RESULTS_DIR}")
    print(f"  Verbose:           {VERBOSE}")
    print(f"  Log Interval:      {LOG_INTERVAL}")
    print()
    
    print("=" * 80)


def get_llm_short_name():
    """Get short name for LLM model (for directory naming)"""
    if not LLM_MODEL:
        return None
    
    llm_mapping = {
        'gpt-3.5-turbo': 'gpt35',
        'gpt-4': 'gpt4',
        'gpt-4-turbo': 'gpt4turbo',
        'zephyr-7b': 'zephyr',
        'claude-3': 'claude3',
    }
    
    return llm_mapping.get(LLM_MODEL, LLM_MODEL.replace('-', '').replace('.', ''))


def validate_config():
    """
    Validate configuration settings
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check if dataset exists
    if DATASET not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {DATASET}. Available: {list(DATASET_CONFIGS.keys())}")
    
    # Check if required files exist
    if not CURRENT_DATASET_CONFIG['data_path'].exists():
        raise FileNotFoundError(f"Data file not found: {CURRENT_DATASET_CONFIG['data_path']}")
    
    if not CURRENT_DATASET_CONFIG['metadata_path'].exists():
        raise FileNotFoundError(f"Metadata file not found: {CURRENT_DATASET_CONFIG['metadata_path']}\n"
                              f"Run: python scripts/generate_{DATASET}_metadata.py")
    
    # Check LLM settings
    if USE_LLM_PRIOR and not CURRENT_DATASET_CONFIG['llm_direction_path']:
        raise ValueError(f"USE_LLM_PRIOR is True but no LLM direction path specified for {DATASET}")
    
    if USE_LLM_PRIOR and CURRENT_DATASET_CONFIG['llm_direction_path']:
        if not CURRENT_DATASET_CONFIG['llm_direction_path'].exists():
            raise FileNotFoundError(f"LLM direction file not found: {CURRENT_DATASET_CONFIG['llm_direction_path']}")
    
    print("[OK] Configuration validated successfully!")


# ============================================================================
# QUICK CONFIGURATION PRESETS
# ============================================================================
def set_alarm_no_llm():
    """Quick preset: ALARM dataset without LLM"""
    global DATASET, LLM_MODEL, USE_LLM_PRIOR, CURRENT_DATASET_CONFIG
    DATASET = 'alarm'
    LLM_MODEL = None
    USE_LLM_PRIOR = False
    CURRENT_DATASET_CONFIG = DATASET_CONFIGS['alarm']


def set_alarm_gpt35():
    """Quick preset: ALARM dataset with GPT-3.5"""
    global DATASET, LLM_MODEL, USE_LLM_PRIOR, CURRENT_DATASET_CONFIG
    DATASET = 'alarm'
    LLM_MODEL = 'gpt-3.5-turbo'
    USE_LLM_PRIOR = True
    CURRENT_DATASET_CONFIG = DATASET_CONFIGS['alarm']


def set_alarm_zephyr():
    """Quick preset: ALARM dataset with Zephyr"""
    global DATASET, LLM_MODEL, USE_LLM_PRIOR, CURRENT_DATASET_CONFIG
    DATASET = 'alarm'
    LLM_MODEL = 'zephyr-7b'
    USE_LLM_PRIOR = True
    CURRENT_DATASET_CONFIG = DATASET_CONFIGS['alarm']


# ============================================================================
# MAIN (for testing)
# ============================================================================
if __name__ == "__main__":
    print("Testing configuration module\n")
    
    # Print current config
    print_config()
    
    # Validate
    try:
        validate_config()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] Validation Error: {e}")
    
    # Get config dict
    config = get_config()
    print("\nConfiguration dictionary:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test presets
    print("\n" + "=" * 80)
    print("Testing presets:")
    print("=" * 80)
    
    print("\n1. ALARM without LLM:")
    set_alarm_no_llm()
    print(f"   Dataset: {DATASET}, LLM: {LLM_MODEL}")
    
    print("\n2. ALARM with GPT-3.5:")
    set_alarm_gpt35()
    print(f"   Dataset: {DATASET}, LLM: {LLM_MODEL}")
    
    print("\n3. ALARM with Zephyr:")
    set_alarm_zephyr()
    print(f"   Dataset: {DATASET}, LLM: {LLM_MODEL}")
