"""
Scanner Utilities for Causal Discovery

Provides functions to identify candidate variable pairs for causal analysis.
Uses Mutual Information (MI) for discrete/binary data.
"""

from sklearn.metrics import mutual_info_score


def scan_with_mutual_information(df, graph, nodes, threshold=0.05):
    """
    Scan for candidate variable pairs using Mutual Information.
    
    Why Mutual Information (MI)?
    - Better than Pearson correlation for discrete/binary data
    - Captures both linear and nonlinear dependencies
    - MI = 0 means independent, higher MI means stronger dependency
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    graph : networkx.DiGraph
        Current causal graph (to avoid re-testing existing edges)
    nodes : list
        List of variable names
    threshold : float
        MI threshold (typically 0.05-0.1 for binary data)
        Note: MI thresholds are much smaller than correlation thresholds
    
    Returns:
    --------
    list of tuples : (node_a, node_b, mi_score)
        Sorted by MI score (highest first)
    """
    candidates = []
    
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i >= j:
                continue
            
            # Skip if edge already exists in either direction
            if graph.has_edge(node_a, node_b) or graph.has_edge(node_b, node_a):
                continue
            
            # Calculate Mutual Information
            # MI >= 0, higher means stronger dependency
            mi_score = mutual_info_score(df[node_a], df[node_b])
            
            if mi_score > threshold:
                candidates.append((node_a, node_b, mi_score))
    
    # RankedExpand: prioritize highest MI scores
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    return candidates

