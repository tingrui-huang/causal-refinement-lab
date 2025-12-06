"""
Report Generator for Causal Discovery Experiments

This module provides utilities to generate human-readable reports
from causal discovery results.
"""

from datetime import datetime
import networkx as nx


def save_text_report(graph, model_name="model", cot_log=None, output_dir="."):
    """
    Generate a human-readable text report from a causal graph.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The causal graph with edges (may have 'type'='rejected' attribute)
    model_name : str
        Name of the model/method used (e.g., "GPT-3.5", "GES", "FCI")
    cot_log : list, optional
        List of CoT reasoning logs (if available)
    output_dir : str
        Directory to save the report
    
    Returns:
    --------
    str : Path to the generated report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/report_{model_name.replace(' ', '_')}_{timestamp}.txt"
    
    accepted_edges = []
    rejected_edges = []
    
    # Classify edges into accepted and rejected
    for u, v, data in graph.edges(data=True):
        edge_str = f"{u} -> {v}"
        if data.get('type') == 'rejected':
            rejected_edges.append(edge_str)
        else:
            accepted_edges.append(edge_str)
    
    # Write report
    with open(filename, "w", encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Causal Discovery Report: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Nodes: {graph.number_of_nodes()}\n")
        f.write(f"Total Iterations: {len(cot_log) if cot_log else 'N/A'}\n")
        f.write("=" * 60 + "\n\n")
        
        # Accepted edges section
        f.write(f"ACCEPTED CAUSAL EDGES: {len(accepted_edges)}\n")
        f.write("-" * 60 + "\n")
        f.write("These edges represent causal relationships affirmed by the method.\n\n")
        if accepted_edges:
            for i, edge in enumerate(sorted(accepted_edges), 1):
                f.write(f"  {i:2d}. [YES] {edge}\n")
        else:
            f.write("  (No edges accepted)\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        # Rejected edges section
        f.write(f"REJECTED EDGES: {len(rejected_edges)}\n")
        f.write("-" * 60 + "\n")
        f.write("High correlation in data, but rejected by the method.\n")
        f.write("(Likely spurious correlations or confounded relationships)\n\n")
        if rejected_edges:
            for i, edge in enumerate(sorted(rejected_edges), 1):
                f.write(f"  {i:2d}. [NO]  {edge}\n")
        else:
            f.write("  (No edges rejected)\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Nodes:          {graph.number_of_nodes()}\n")
        f.write(f"Accepted Edges:       {len(accepted_edges)}\n")
        f.write(f"Rejected Edges:       {len(rejected_edges)}\n")
        f.write(f"Total Edges Tested:   {len(accepted_edges) + len(rejected_edges)}\n")
        if len(accepted_edges) + len(rejected_edges) > 0:
            acceptance_rate = len(accepted_edges) / (len(accepted_edges) + len(rejected_edges)) * 100
            f.write(f"Acceptance Rate:      {acceptance_rate:.1f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n[REPORT] Text report saved to: {filename}")
    return filename


def save_edge_list(graph, model_name="model", output_dir="."):
    """
    Save a simple edge list (CSV format) for further analysis.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The causal graph
    model_name : str
        Name of the model/method
    output_dir : str
        Directory to save the file
    
    Returns:
    --------
    str : Path to the generated CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/edges_{model_name.replace(' ', '_')}_{timestamp}.csv"
    
    with open(filename, "w", encoding='utf-8') as f:
        f.write("source,target,status\n")
        for u, v, data in graph.edges(data=True):
            status = "rejected" if data.get('type') == 'rejected' else "accepted"
            f.write(f"{u},{v},{status}\n")
    
    print(f"[REPORT] Edge list saved to: {filename}")
    return filename

