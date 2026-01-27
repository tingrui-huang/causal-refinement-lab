"""
Evaluate FCI Skeleton Performance

Calculate Precision, Recall, F1, and Orientation Accuracy for the FCI skeleton
compared to Ground Truth from alarm.bif
"""

import re
import pandas as pd
from pathlib import Path


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


def parse_fci_edges(fci_csv_path):
    """Parse FCI edges from CSV file"""
    df = pd.read_csv(fci_csv_path)
    
    fci_directed = set()
    fci_undirected = set()
    
    edge_counts = {
        'directed': 0,
        'undirected': 0,
        'partial': 0,
        'tail-tail': 0
    }
    
    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row['edge_type']
        
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        if edge_type == 'directed':
            fci_directed.add((source, target))
        elif edge_type in ['undirected', 'partial', 'tail-tail']:
            # For undirected/partial edges, consider both directions
            fci_undirected.add(tuple(sorted([source, target])))
    
    return fci_directed, fci_undirected, edge_counts


def evaluate_fci(fci_csv_path, ground_truth_path):
    """
    Evaluate FCI skeleton against ground truth
    """
    print("=" * 80)
    print("FCI SKELETON EVALUATION")
    print("=" * 80)
    
    # Load data
    gt_edges = parse_ground_truth(ground_truth_path)
    fci_directed, fci_undirected, edge_counts = parse_fci_edges(fci_csv_path)
    
    print(f"\nGround Truth: {len(gt_edges)} directed edges")
    print(f"FCI Edges: {len(fci_directed)} directed + {len(fci_undirected)} undirected/partial")
    print(f"  - Directed: {edge_counts['directed']}")
    print(f"  - Undirected: {edge_counts['undirected']}")
    print(f"  - Partial: {edge_counts['partial']}")
    print(f"  - Tail-tail: {edge_counts['tail-tail']}")
    
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
    print("UNDIRECTED SKELETON METRICS (Edge Discovery)")
    print("=" * 80)
    print(f"True Positives (TP):  {undirected_tp}")
    print(f"False Positives (FP): {undirected_fp}")
    print(f"False Negatives (FN): {undirected_fn}")
    print(f"\nPrecision: {edge_precision*100:.1f}%")
    print(f"Recall:    {edge_recall*100:.1f}%")
    print(f"F1 Score:  {edge_f1*100:.1f}%")
    
    # Show missing edges
    if undirected_fn > 0:
        print(f"\nMissing edges (FN = {undirected_fn}):")
        missing = gt_undirected - fci_all_undirected
        for edge in sorted(missing):
            # Find the directed version in GT
            for gt_edge in gt_edges:
                if tuple(sorted([gt_edge[0], gt_edge[1]])) == edge:
                    print(f"  {gt_edge[0]} → {gt_edge[1]}")
                    break
    
    # Show extra edges
    if undirected_fp > 0:
        print(f"\nExtra edges (FP = {undirected_fp}):")
        extra = fci_all_undirected - gt_undirected
        for edge in sorted(extra):
            print(f"  {edge[0]} - {edge[1]}")
    
    # === 2. ORIENTATION ACCURACY (for directed edges only) ===
    correctly_oriented = 0
    incorrectly_oriented = 0
    cannot_evaluate = 0
    
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
                else:
                    cannot_evaluate += 1
    
    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
    
    print("\n" + "=" * 80)
    print("ORIENTATION ACCURACY (Directed Edges Only)")
    print("=" * 80)
    print(f"FCI Directed Edges: {len(fci_directed)}")
    print(f"Correctly Oriented: {correctly_oriented}")
    print(f"Incorrectly Oriented: {incorrectly_oriented}")
    print(f"Cannot Evaluate: {cannot_evaluate}")
    print(f"\nOrientation Accuracy: {orientation_accuracy*100:.1f}%")
    
    # Show reversed edges
    if incorrectly_oriented > 0:
        print(f"\nReversed edges ({incorrectly_oriented}):")
        for fci_edge in fci_directed:
            reversed_edge = (fci_edge[1], fci_edge[0])
            undirected_edge = tuple(sorted([fci_edge[0], fci_edge[1]]))
            if undirected_edge in gt_undirected and reversed_edge in gt_edges and fci_edge not in gt_edges:
                print(f"  FCI: {fci_edge[0]} → {fci_edge[1]}")
                print(f"  GT:  {reversed_edge[0]} → {reversed_edge[1]}")
    
    # === 3. BIDIRECTIONAL RATIO ===
    # Count undirected/partial edges
    bidirectional_ratio = len(fci_undirected) / len(fci_all_undirected) if len(fci_all_undirected) > 0 else 0
    
    print("\n" + "=" * 80)
    print("BIDIRECTIONAL / UNDIRECTED EDGES")
    print("=" * 80)
    print(f"Total FCI edges: {len(fci_all_undirected)}")
    print(f"Directed: {len(fci_directed)}")
    print(f"Undirected/Partial: {len(fci_undirected)}")
    print(f"Bidirectional Ratio: {bidirectional_ratio*100:.1f}%")
    
    # === 4. SUMMARY TABLE ===
    print("\n" + "=" * 80)
    print("SUMMARY (Comparable to Model Results)")
    print("=" * 80)
    print(f"\n{'Metric':<25} | {'FCI Skeleton':<15}")
    print("-" * 45)
    print(f"{'Edge F1':<25} | {edge_f1*100:>13.1f}%")
    print(f"{'Precision':<25} | {edge_precision*100:>13.1f}%")
    print(f"{'Recall':<25} | {edge_recall*100:>13.1f}%")
    print(f"{'Orient. Acc.':<25} | {orientation_accuracy*100:>13.1f}%")
    print(f"{'Bidir. Ratio':<25} | {bidirectional_ratio*100:>13.1f}%")
    
    return {
        'edge_f1': edge_f1,
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'orientation_accuracy': orientation_accuracy,
        'bidirectional_ratio': bidirectional_ratio,
        'undirected_tp': undirected_tp,
        'undirected_fp': undirected_fp,
        'undirected_fn': undirected_fn,
        'correctly_oriented': correctly_oriented,
        'incorrectly_oriented': incorrectly_oriented
    }


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    fci_path = base_dir / 'data' / 'alarm' / 'edges_FCI_20251207_230824.csv'
    gt_path = base_dir / 'data' / 'alarm' / 'alarm.bif'
    
    metrics = evaluate_fci(fci_path, gt_path)
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH MODEL RESULTS")
    print("=" * 80)
    print("\nFrom your table:")
    print(f"{'Configuration':<25} | {'Edge F1':<10} | {'Precision':<10} | {'Recall':<10} | {'Orient. Acc.':<15} | {'Bidir. Ratio':<15}")
    print("-" * 110)
    print(f"{'FCI Skeleton':<25} | {metrics['edge_f1']*100:>8.1f}% | {metrics['edge_precision']*100:>8.1f}% | {metrics['edge_recall']*100:>8.1f}% | {metrics['orientation_accuracy']*100:>13.1f}% | {metrics['bidirectional_ratio']*100:>13.1f}%")
    print(f"{'No LLM (Ours)':<25} |     94.4% |     97.7% |     91.3% |          78.3% |           100%")
    print(f"{'LLM (GPT-3.5)':<25} |     92.0% |     97.6% |     87.0% |          82.5% |             0%")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"\n1. FCI provides a {'good' if metrics['edge_recall'] > 0.85 else 'moderate'} skeleton:")
    print(f"   - Recall: {metrics['edge_recall']*100:.1f}% (finds {metrics['undirected_tp']}/{metrics['undirected_tp']+metrics['undirected_fn']} true edges)")
    print(f"   - Precision: {metrics['edge_precision']*100:.1f}% ({metrics['undirected_fp']} false positives)")
    
    print(f"\n2. FCI's orientation:")
    print(f"   - {metrics['correctly_oriented']}/{metrics['correctly_oriented']+metrics['incorrectly_oriented']} directed edges are correct ({metrics['orientation_accuracy']*100:.1f}%)")
    print(f"   - {metrics['incorrectly_oriented']} edges are reversed")
    
    print(f"\n3. Models improve upon FCI:")
    print(f"   - No-LLM improves orientation: {metrics['orientation_accuracy']*100:.1f}% → 78.3%")
    print(f"   - GPT-3.5 further improves: 78.3% → 82.5%")
    print(f"   - Both models resolve undirected edges!")

