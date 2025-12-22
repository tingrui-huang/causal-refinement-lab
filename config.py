"""
Unified Configuration for Causal Discovery Pipeline

This single config file controls BOTH:
1. FCI/LLM algorithms (refactored/)
2. Neuro-Symbolic training (Neuro-Symbolic-Reasoning/)

Usage:
    - Edit this file to change any settings
    - Both modules will automatically use these settings
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
REFACTORED_DIR = PROJECT_ROOT / 'refactored'
NEURO_SYMBOLIC_DIR = PROJECT_ROOT / 'Neuro-Symbolic-Reasoning'

# ============================================================================
# DATASET SELECTION
# ============================================================================
# Options: 'alarm', 'insurance', 'sachs', 'child', 'tuebingen_pair1', etc.
DATASET = 'child'

# ============================================================================
# STEP 1: FCI ALGORITHM SETTINGS
# ============================================================================
# Independence test for FCI
# Options: 'chisq' (discrete data), 'fisherz' (continuous Gaussian), 'gsq'
FCI_INDEPENDENCE_TEST = 'chisq'

# Significance level for FCI
FCI_ALPHA = 0.05

# Validation alpha (for LLM-based direction resolution)
VALIDATION_ALPHA = 0.01

# ============================================================================
# STEP 2: LLM SETTINGS (Optional)
# ============================================================================
# LLM Model Selection
# Options:
#   - None (no LLM, use FCI skeleton only)
#   - 'gpt-3.5-turbo' (GPT-3.5)
#   - 'gpt-4' (GPT-4)
#   - 'zephyr-7b' (Zephyr)
LLM_MODEL = 'gpt-3.5-turbo'  # Set to None for FCI-only pipeline (testing GSB framework)

# Use LLM prior for direction initialization in neural training?
USE_LLM_PRIOR = True if LLM_MODEL else False

# LLM API settings
LLM_TEMPERATURE = 0.0  # 0.0 for deterministic results
LLM_MAX_TOKENS = 500  # Prevent overly long responses

# ============================================================================
# STEP 3: NEURAL TRAINING SETTINGS
# ============================================================================
# Training hyperparameters
LEARNING_RATE = 0.01
N_EPOCHS = 300  # Number of training epochs
N_HOPS = 1  # Number of reasoning hops (1 = single-hop)
BATCH_SIZE = None  # None = full batch

# Regularization
LAMBDA_GROUP_LASSO = 0.005  # Group lasso penalty weight
LAMBDA_CYCLE = 0.005    # Cycle consistency penalty weight
# alarm is 0.01, 0.005, insurance is 0.001,0.05, different datasets have different configurations.
# Sachs for these two should be as much lower as possible, like 0 and 0.001
# child is 0.005, 0,.005


# Threshold for edge detection
THRESHOLD = 0.05
# Lower = more edges, Higher = fewer edges Sachs dataset should be lower, like 0.008
# child is 0.05

# Logging
LOG_INTERVAL = 20  # Print training stats every N epochs
VERBOSE = True

# ============================================================================
# DATASET-SPECIFIC PATHS
# ============================================================================
DATASET_CONFIGS = {
    'alarm': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (37 columns), neural training needs one-hot data (105 columns)
        'fci_data_path': PROJECT_ROOT / 'alarm_data.csv',  # Variable-level data for FCI (37 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'alarm' / 'alarm_data_10000.csv',  # One-hot data for neural training (105 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'alarm' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': PROJECT_ROOT / 'alarm.bif',
        'ground_truth_type': 'bif',  # Type: 'bif', 'json', or None
        
        # Data type
        'data_type': 'discrete',  # 'discrete' or 'continuous'
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'insurance': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (27 columns), neural training needs one-hot data (88 columns)
        'fci_data_path': PROJECT_ROOT / 'insurance_data.csv',  # Variable-level data for FCI (27 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'insurance' / 'insurance_data_10000.csv',  # One-hot data for neural training (88 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'insurance' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'insurance' / 'insurance.bif',
        'ground_truth_type': 'bif',  # Type: 'bif', 'json', or None
        
        # Data type
        'data_type': 'discrete',  # 'discrete' or 'continuous'
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'tuebingen_pair1': {
        # Data files
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001.csv',
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'metadata_pair1.json',
        
        # Ground truth
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'ground_truth.json',
        'ground_truth_type': 'json',  # Pairwise evaluation
        
        # Data type
        'data_type': 'continuous',  # Needs discretization
        
        # FCI/LLM outputs
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    
    'sachs': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (11 columns), neural training needs one-hot data (~33 columns)
        'fci_data_path': PROJECT_ROOT / 'sachs_data_variable.csv',  # Variable-level data for FCI (11 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'sachs' / 'sachs_data.csv',  # One-hot data for neural training (~33 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'sachs' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'sachs' / 'sachs_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Pre-discretized from bnlearn (3 states per variable)
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'child': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (20 columns), neural training needs one-hot data
        'fci_data_path': PROJECT_ROOT / 'child_data_variable.csv',  # Variable-level data for FCI (20 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'child' / 'child_data.csv',  # One-hot data for neural training
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'child' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'child' / 'child_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Discrete medical diagnosis variables
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    # Add more datasets here...
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
# FCI/LLM outputs
FCI_OUTPUT_DIR = REFACTORED_DIR / 'output'

# Neural training results
TRAINING_RESULTS_DIR = NEURO_SYMBOLIC_DIR / 'results'

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
# Random seed
RANDOM_SEED = 42

# Device
DEVICE = 'cpu'  # 'cuda' or 'cpu'

# Early stopping
EARLY_STOPPING = False
PATIENCE = 50

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_current_dataset_config():
    """Get configuration for current dataset"""
    return DATASET_CONFIGS.get(DATASET, DATASET_CONFIGS['alarm'])


def get_fci_config():
    """Get FCI-specific configuration"""
    dataset_cfg = get_current_dataset_config()
    
    # Auto-select independence test based on data type
    default_test = 'fisherz' if dataset_cfg['data_type'] == 'continuous' else 'chisq'
    
    return {
        'dataset': DATASET,
        'independence_test': default_test,  # Auto-selected based on data_type
        'alpha': FCI_ALPHA,
        'validation_alpha': VALIDATION_ALPHA,
        'output_dir': str(FCI_OUTPUT_DIR),
        'ground_truth_path': str(dataset_cfg['ground_truth_path']),
        'ground_truth_type': dataset_cfg.get('ground_truth_type', 'bif'),
        'llm_temperature': LLM_TEMPERATURE,
        'llm_max_tokens': LLM_MAX_TOKENS,
    }


def get_training_config():
    """Get training-specific configuration"""
    dataset_cfg = get_current_dataset_config()
    
    return {
        # Dataset
        'dataset_name': DATASET,
        'data_path': str(dataset_cfg['data_path']),
        'metadata_path': str(dataset_cfg['metadata_path']),
        'ground_truth_path': str(dataset_cfg['ground_truth_path']) if dataset_cfg['ground_truth_path'] else None,
        'ground_truth_type': dataset_cfg.get('ground_truth_type', 'bif'),
        'data_type': dataset_cfg.get('data_type', 'discrete'),
        
        # FCI/LLM paths (auto-detected from FCI output directory, not data directory!)
        'fci_skeleton_path': _auto_detect_latest_file('edges_FCI_*.csv', FCI_OUTPUT_DIR / DATASET),
        'llm_direction_path': _auto_detect_latest_file('edges_FCI_LLM_*.csv', FCI_OUTPUT_DIR / DATASET) if USE_LLM_PRIOR else None,
        
        # LLM settings
        'llm_model': LLM_MODEL,
        'use_llm_prior': USE_LLM_PRIOR,
        'llm_temperature': LLM_TEMPERATURE,
        'llm_max_tokens': LLM_MAX_TOKENS,
        
        # Hyperparameters
        'learning_rate': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'n_hops': N_HOPS,
        'batch_size': BATCH_SIZE,
        'lambda_group_lasso': LAMBDA_GROUP_LASSO,
        'lambda_cycle': LAMBDA_CYCLE,
        'threshold': THRESHOLD,
        
        # Output
        'results_dir': str(TRAINING_RESULTS_DIR),
        'verbose': VERBOSE,
        'log_interval': LOG_INTERVAL,
        
        # Advanced
        'random_seed': RANDOM_SEED,
        'device': DEVICE,
        'early_stopping': EARLY_STOPPING,
        'patience': PATIENCE,
    }


def _auto_detect_latest_file(pattern, directory):
    """Auto-detect the latest file matching pattern"""
    from pathlib import Path
    
    data_dir = Path(directory)
    if not data_dir.exists():
        return None
    
    files = list(data_dir.glob(pattern))
    if not files:
        return None
    
    # Return the most recent file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def print_config():
    """Print current configuration"""
    print("\n" + "=" * 80)
    print("UNIFIED CONFIGURATION")
    print("=" * 80)
    
    print(f"\nDataset: {DATASET}")
    print(f"Ground Truth: {get_current_dataset_config()['ground_truth_path']}")
    
    print(f"\n--- STEP 1: FCI Algorithm ---")
    print(f"  Independence Test: {FCI_INDEPENDENCE_TEST}")
    print(f"  Alpha: {FCI_ALPHA}")
    print(f"  Output: {FCI_OUTPUT_DIR}")
    
    print(f"\n--- STEP 2: LLM (Optional) ---")
    print(f"  Model: {LLM_MODEL if LLM_MODEL else 'None (FCI only)'}")
    print(f"  Use LLM Prior: {USE_LLM_PRIOR}")
    
    print(f"\n--- STEP 3: Neural Training ---")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Lambda Lasso: {LAMBDA_GROUP_LASSO}")
    print(f"  Lambda Cycle: {LAMBDA_CYCLE}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Results: {TRAINING_RESULTS_DIR}")
    
    print("\n" + "=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration"""
    dataset_cfg = get_current_dataset_config()
    
    # Check if data files exist
    if not dataset_cfg['data_path'].exists():
        raise FileNotFoundError(f"Data file not found: {dataset_cfg['data_path']}")
    
    if not dataset_cfg['metadata_path'].exists():
        raise FileNotFoundError(f"Metadata file not found: {dataset_cfg['metadata_path']}")
    
    # Check ground truth (optional)
    if dataset_cfg['ground_truth_path'] and not Path(dataset_cfg['ground_truth_path']).exists():
        print(f"[WARN] Ground truth file not found: {dataset_cfg['ground_truth_path']}")
    
    # Check LLM prior requirement
    if USE_LLM_PRIOR and not LLM_MODEL:
        raise ValueError("USE_LLM_PRIOR is True but LLM_MODEL is None")
    
    print("[OK] Configuration validated")


# ============================================================================
# QUICK PRESETS
# ============================================================================
def use_fci_only():
    """Preset: FCI only (no LLM)"""
    global LLM_MODEL, USE_LLM_PRIOR
    LLM_MODEL = None
    USE_LLM_PRIOR = False
    print("[CONFIG] Using FCI only (no LLM)")


def use_fci_gpt35():
    """Preset: FCI + GPT-3.5"""
    global LLM_MODEL, USE_LLM_PRIOR
    LLM_MODEL = 'gpt-3.5-turbo'
    USE_LLM_PRIOR = True
    print("[CONFIG] Using FCI + GPT-3.5")


def use_fci_zephyr():
    """Preset: FCI + Zephyr"""
    global LLM_MODEL, USE_LLM_PRIOR
    LLM_MODEL = 'zephyr-7b'
    USE_LLM_PRIOR = True
    print("[CONFIG] Using FCI + Zephyr")


# ============================================================================
# MAIN (for testing)
# ============================================================================
if __name__ == "__main__":
    print_config()
    
    print("\n" + "=" * 80)
    print("TESTING PRESETS")
    print("=" * 80)
    
    print("\n1. FCI Only:")
    use_fci_only()
    print(f"   LLM Model: {LLM_MODEL}, Use LLM Prior: {USE_LLM_PRIOR}")
    
    print("\n2. FCI + GPT-3.5:")
    use_fci_gpt35()
    print(f"   LLM Model: {LLM_MODEL}, Use LLM Prior: {USE_LLM_PRIOR}")
    
    print("\n3. FCI + Zephyr:")
    use_fci_zephyr()
    print(f"   LLM Model: {LLM_MODEL}, Use LLM Prior: {USE_LLM_PRIOR}")
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    try:
        validate_config()
    except Exception as e:
        print(f"[ERROR] {e}")

