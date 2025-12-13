"""
Symbolic causal refinement module.

This module implements Neural LP-inspired methods for refining 
causal discovery results from FCI/LLM outputs.
"""

from .data_generator import CausalChainGenerator
from .fci_simulator import FCISimulator, SimpleThreeVarFCISimulator
from .neural_lp import (
    NeuralLP,
    MaskedAdjacencyMatrix,
    MultiHopReasoning,
    train_neural_lp,
    compute_loss,
    print_adjacency_comparison
)

__all__ = [
    'CausalChainGenerator',
    'FCISimulator',
    'SimpleThreeVarFCISimulator',
    'NeuralLP',
    'MaskedAdjacencyMatrix',
    'MultiHopReasoning',
    'train_neural_lp',
    'compute_loss',
    'print_adjacency_comparison',
]

