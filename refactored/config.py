"""
FCI/LLM Configuration (refactored/)

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
FCI_INDEPENDENCE_TEST = unified_config.FCI_INDEPENDENCE_TEST
FCI_ALPHA = unified_config.FCI_ALPHA
VALIDATION_ALPHA = unified_config.VALIDATION_ALPHA
LLM_MODEL = unified_config.LLM_MODEL
FCI_OUTPUT_DIR = unified_config.FCI_OUTPUT_DIR
get_fci_config = unified_config.get_fci_config
print_unified_config = unified_config.print_config

# Legacy imports (kept for backward compatibility)
from_config = (
)

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================
# These are kept for backward compatibility with existing scripts

# Active dataset
ACTIVE_DATASET = DATASET

# Output directory
OUTPUT_DIR = str(FCI_OUTPUT_DIR / DATASET)

# Ground truth path
GROUND_TRUTH_PATH = str(get_current_dataset_config()['ground_truth_path'])

# Neuro-Symbolic data directory
NEURO_SYMBOLIC_DATA_DIR = str(get_current_dataset_config()['data_path'].parent)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_output_dir():
    """Get output directory for current dataset"""
    return OUTPUT_DIR


def print_config():
    """Print FCI configuration"""
    print("\n" + "=" * 80)
    print("FCI/LLM CONFIGURATION")
    print("=" * 80)
    print(f"\nDataset: {DATASET}")
    print(f"Independence Test: {FCI_INDEPENDENCE_TEST}")
    print(f"Alpha: {FCI_ALPHA}")
    print(f"Validation Alpha: {VALIDATION_ALPHA}")
    print(f"LLM Model: {LLM_MODEL if LLM_MODEL else 'None (FCI only)'}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Ground Truth: {GROUND_TRUTH_PATH}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print_config()
    print("\n[INFO] All settings are managed in ../config.py")
    print("[INFO] Edit ../config.py to change any settings")
