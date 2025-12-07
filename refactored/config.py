"""
Configuration file for causal discovery experiments

Edit this file to switch datasets without changing main code
"""

# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Available datasets
DATASETS = {
    "lucas": {
        "path": "../lucas0_train.csv",
        "loader": "LUCASDataLoader",
        "description": "LUCAS lung cancer dataset (12 variables, 2000 samples)"
    },
    "alarm": {
        "path": "../alarm_data.csv",  # Change this to your ALARM data path
        "loader": "ALARMDataLoader",
        "description": "ALARM medical monitoring dataset (37 variables)"
    },
    "custom": {
        "path": "../your_data.csv",  # Change this to your custom data path
        "loader": "DataLoader",  # Generic loader
        "description": "Custom dataset"
    }
}

# ============================================================
# SELECT DATASET HERE (change this to switch datasets)
# ============================================================
ACTIVE_DATASET = "lucas"  # Options: "lucas", "alarm", "custom"

# ============================================================
# ALGORITHM PARAMETERS
# ============================================================

# For LLM-based methods
MAX_ITERATIONS = 50
MI_THRESHOLD = 0.05
VALIDATION_ALPHA = 0.05

# For FCI/GES baseline
FCI_ALPHA = 0.05
GES_SCORE_TYPE = 'bic'

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
OUTPUT_DIR = "../outputs"

# Auto-generate dataset-specific output directory
def get_output_dir():
    """Get output directory for the active dataset"""
    import os
    dataset_dir = os.path.join(OUTPUT_DIR, ACTIVE_DATASET)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

