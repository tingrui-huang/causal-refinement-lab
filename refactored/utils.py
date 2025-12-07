"""
Utility functions for causal discovery experiments
"""

from config import DATASETS, ACTIVE_DATASET
from modules.data_loader import DataLoader, LUCASDataLoader, ALARMDataLoader


def get_active_data_loader():
    """
    Get data loader for the active dataset specified in config.py
    
    Returns:
    --------
    data_loader : DataLoader instance
        Configured data loader for the active dataset
    """
    if ACTIVE_DATASET not in DATASETS:
        raise ValueError(f"Dataset '{ACTIVE_DATASET}' not found in config.py")
    
    dataset_config = DATASETS[ACTIVE_DATASET]
    data_path = dataset_config["path"]
    loader_name = dataset_config["loader"]
    
    # Instantiate the appropriate loader
    if loader_name == "LUCASDataLoader":
        return LUCASDataLoader(data_path)
    elif loader_name == "ALARMDataLoader":
        return ALARMDataLoader(data_path)
    else:
        return DataLoader(data_path, dataset_name=ACTIVE_DATASET.upper())


def print_dataset_info():
    """Print information about the active dataset"""
    print("\n" + "=" * 60)
    print(f"Active Dataset: {ACTIVE_DATASET.upper()}")
    print(f"Description: {DATASETS[ACTIVE_DATASET]['description']}")
    print(f"Path: {DATASETS[ACTIVE_DATASET]['path']}")
    print(f"To change dataset, edit config.py (line 33)")
    print("=" * 60 + "\n")

