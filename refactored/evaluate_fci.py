"""
Evaluate FCI Results Against Ground Truth

Automatically evaluates the latest FCI output and generates a report.
This script is called after FCI runs to assess skeleton quality.
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime


def parse_ground_truth(bif_path):
    """Parse ground truth edges from BIF file"""
    ground_truth_edges = set()
    
    with open(bif_path, 'r') as f:
        content = f.read()
    
    # Extract probability declarations
    # Format: probability ( CHILD | PARENT1, PARENT2, ... )
    prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'
    
    for match in re.finditer(prob_pattern, content):
        child = match.group(1)
        parents_str = match.group(2)
        parents = [p.strip() for p in parents_str.split(',')]
        
        for parent in parents:
            if parent:
                ground_truth_edges.add((parent, child))
    
    return ground_truth_edges


def parse_fci_csv(fci_csv_path):
    """Parse FCI edges from CSV file"""
    df = pd.read_csv(fci_csv_path)
    
    # Determine column names (case-insensitive compatibility)
    if 'Source' in df.columns and 'Target' in df.columns:
        source_col, target_col = 'Source', 'Target'
    elif 'source' in df.columns and 'target' in df.columns:
        source_col, target_col = 'source', 'target'
    else:
        # Use first two columns as fallback
        source_col, target_col = df.columns[0], df.columns[1]
        print(f"[WARN] Standard columns not found, using: {source_col} -> {target_col}")
    
    # Determine edge type column
    if 'edge_type' in df.columns:
        edge_type_col = 'edge_type'
    elif 'Edge_Type' in df.columns:
        edge_type_col = 'Edge_Type'
    elif 'type' in df.columns:
        edge_type_col = 'type'
    else:
        edge_type_col = None
        print("[WARN] No edge_type column found, assuming all edges are directed")
    
    fci_directed = set()
    fci_undirected = set()
    
    edge_counts = {
        'directed': 0,
        'undirected': 0,
        'partial': 0,
        'tail-tail': 0,
        'bidirected': 0
    }
    
    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        edge_type = row[edge_type_col] if edge_type_col else 'directed'
        
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        if edge_type == 'directed':
            fci_directed.add((source, target))
        elif edge_type in ['undirected', 'partial', 'tail-tail', 'bidirected']:
            # For undirected/partial edges, consider both directions
            fci_undirected.add(tuple(sorted([source, target])))
    
    return fci_directed, fci_undirected, edge_counts


def evaluate_fci(fci_csv_path, ground_truth_path, output_dir=None):
    """
    Evaluate FCI skeleton against ground truth
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("FCI EVALUATION AGAINST GROUND TRUTH")
    print("=" * 80)
    
    # Load data
    gt_edges = parse_ground_truth(ground_truth_path)
    fci_directed, fci_undirected, edge_counts = parse_fci_csv(fci_csv_path)
    
    print(f"\nGround Truth: {len(gt_edges)} directed edges")
    print(f"FCI Edges: {len(fci_directed)} directed + {len(fci_undirected)} undirected/partial")
    print(f"  - Directed: {edge_counts['directed']}")
    print(f"  - Undirected: {edge_counts.get('undirected', 0)}")
    print(f"  - Partial: {edge_counts.get('partial', 0)}")
    print(f"  - Tail-tail: {edge_counts.get('tail-tail', 0)}")
    print(f"  - Bidirected: {edge_counts.get('bidirected', 0)}")
    
    # === 1. UNDIRECTED SKELETON METRICS ===
    # Convert GT to undirected
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    
    # All FCI edges (both directed and undirected)
    fci_all_undirected = set()
    for edge in fci_directed:
        fci_all_undirected.add(tuple(sorted([edge[0], edge[1]])))
    fci_all_undirected.update(fci_undirected)
    
    # Calculate undirected metrics
    undirected_tp = len(fci_all_undirected & gt_undirected)
    undirected_fp = len(fci_all_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - fci_all_undirected)
    
    edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
    edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
    
    print("\n" + "=" * 80)
    print("EDGE DISCOVERY (Undirected Skeleton)")
    print("=" * 80)
    print(f"True Positives (TP):  {undirected_tp}")
    print(f"False Positives (FP): {undirected_fp}")
    print(f"False Negatives (FN): {undirected_fn}")
    print(f"\nPrecision: {edge_precision*100:.1f}%")
    print(f"Recall:    {edge_recall*100:.1f}%")
    print(f"F1 Score:  {edge_f1*100:.1f}%")
    
    # === 2. ORIENTATION ACCURACY ===
    correctly_oriented = 0
    incorrectly_oriented = 0
    
    for fci_edge in fci_directed:
        undirected_edge = tuple(sorted([fci_edge[0], fci_edge[1]]))
        
        # Check if this edge exists in GT (undirected)
        if undirected_edge in gt_undirected:
            # Check if direction is correct
            if fci_edge in gt_edges:
                correctly_oriented += 1
            else:
                reversed_edge = (fci_edge[1], fci_edge[0])
                if reversed_edge in gt_edges:
                    incorrectly_oriented += 1
    
    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
    
    print("\n" + "=" * 80)
    print("ORIENTATION ACCURACY (Directed Edges Only)")
    print("=" * 80)
    print(f"FCI Directed Edges: {len(fci_directed)}")
    print(f"Correctly Oriented: {correctly_oriented}")
    print(f"Incorrectly Oriented: {incorrectly_oriented}")
    print(f"\nOrientation Accuracy: {orientation_accuracy*100:.1f}%")
    
    # === 3. UNDIRECTED RATIO ===
    undirected_ratio = len(fci_undirected) / len(fci_all_undirected) if len(fci_all_undirected) > 0 else 0
    
    print("\n" + "=" * 80)
    print("UNDIRECTED / PARTIAL EDGES")
    print("=" * 80)
    print(f"Total FCI edges: {len(fci_all_undirected)}")
    print(f"Directed: {len(fci_directed)}")
    print(f"Undirected/Partial: {len(fci_undirected)}")
    print(f"Undirected Ratio: {undirected_ratio*100:.1f}%")
    
    # === 4. SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Edge F1:          {edge_f1*100:.1f}%")
    print(f"Precision:        {edge_precision*100:.1f}%")
    print(f"Recall:           {edge_recall*100:.1f}%")
    print(f"Orient. Accuracy: {orientation_accuracy*100:.1f}%")
    print(f"Undirected Ratio: {undirected_ratio*100:.1f}%")
    print("=" * 80)
    
    # Save evaluation report
    if output_dir:
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_FCI_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FCI EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"FCI CSV: {fci_csv_path.name}\n")
            f.write(f"Ground Truth: {ground_truth_path.name}\n\n")
            
            f.write("METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Edge F1:              {edge_f1*100:.1f}%\n")
            f.write(f"Edge Precision:       {edge_precision*100:.1f}%\n")
            f.write(f"Edge Recall:          {edge_recall*100:.1f}%\n")
            f.write(f"Orientation Accuracy: {orientation_accuracy*100:.1f}%\n")
            f.write(f"Undirected Ratio:     {undirected_ratio*100:.1f}%\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 80 + "\n")
            f.write(f"True Positives:  {undirected_tp}\n")
            f.write(f"False Positives: {undirected_fp}\n")
            f.write(f"False Negatives: {undirected_fn}\n\n")
            
            f.write("ORIENTATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Correctly Oriented:   {correctly_oriented}\n")
            f.write(f"Incorrectly Oriented: {incorrectly_oriented}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Evaluation report saved to: {report_path}")
    
    return {
        'edge_f1': edge_f1,
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'orientation_accuracy': orientation_accuracy,
        'undirected_ratio': undirected_ratio,
        'undirected_tp': undirected_tp,
        'undirected_fp': undirected_fp,
        'undirected_fn': undirected_fn,
        'correctly_oriented': correctly_oriented,
        'incorrectly_oriented': incorrectly_oriented
    }


def find_latest_fci_csv(output_dir='output'):
    """Find the most recent FCI CSV file"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    fci_csvs = list(output_path.glob('edges_FCI_*.csv'))
    
    if not fci_csvs:
        return None
    
    return max(fci_csvs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    from config import GROUND_TRUTH_PATH, OUTPUT_DIR
    
    # Find latest FCI output
    latest_fci = find_latest_fci_csv(OUTPUT_DIR)
    
    if not latest_fci:
        print(f"[ERROR] No FCI CSV files found in {OUTPUT_DIR}/")
        print("Run main_fci.py first to generate FCI results.")
        exit(1)
    
    print(f"Found latest FCI output: {latest_fci.name}")
    
    # Ground truth path
    gt_path = Path(GROUND_TRUTH_PATH)
    
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        print("Please update GROUND_TRUTH_PATH in config.py")
        exit(1)
    
    # Evaluate
    metrics = evaluate_fci(latest_fci, gt_path, output_dir=OUTPUT_DIR)
