"""
Evaluator Module

Evaluate learned causal graph against ground truth with comprehensive metrics
and metadata tracking.
"""

import re
import torch
import json
import time
from datetime import datetime
from typing import Dict, Set, Tuple, Optional
from pathlib import Path


class CausalGraphEvaluator:
    """
    Evaluate learned causal structure against ground truth
    
    Metrics:
    1. Edge-level (undirected): Precision, Recall, F1
    2. Directed edge-level: Precision, Recall, F1
    3. Orientation accuracy: Among correct edges, how many have correct direction
    4. Structural Hamming Distance (SHD)
    """
    
    def __init__(self, ground_truth_path: str, var_structure: Dict, ground_truth_type: str = 'bif'):
        """
        Args:
            ground_truth_path: Path to ground truth file
            var_structure: Variable structure from DataLoader
            ground_truth_type: Type of ground truth file ('bif', 'edge_list', etc.)
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.var_structure = var_structure
        self.ground_truth_type = ground_truth_type
        
        # Parse ground truth based on type
        if ground_truth_type == 'bif':
            self.ground_truth_edges, self.all_variables = self._parse_bif()
        elif ground_truth_type == 'edge_list':
            self.ground_truth_edges, self.all_variables = self._parse_edge_list()
        else:
            raise ValueError(f"Unsupported ground truth type: {ground_truth_type}")
        
        print("=" * 70)
        print("EVALUATOR INITIALIZED")
        print("=" * 70)
        print(f"Ground truth type: {ground_truth_type}")
        print(f"Ground truth edges: {len(self.ground_truth_edges)}")
        print(f"Variables: {len(self.all_variables)}")
    
    def _parse_bif(self) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """
        Parse ground truth from BIF file
        
        Returns:
            (ground_truth_edges, all_variables)
            ground_truth_edges: Set of (parent, child) tuples
            all_variables: Set of variable names
        """
        ground_truth_edges = set()
        all_variables = set()
        
        with open(self.ground_truth_path, 'r') as f:
            content = f.read()
        
        # Extract variable declarations
        var_pattern = r'variable\s+(\w+)\s*\{'
        variables = re.findall(var_pattern, content)
        all_variables = set(variables)
        
        # Extract probability declarations (define causal structure)
        # Format: probability ( CHILD | PARENT1, PARENT2, ... )
        prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'
        
        for match in re.finditer(prob_pattern, content):
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(',')]
            
            for parent in parents:
                if parent:  # Skip empty strings
                    ground_truth_edges.add((parent, child))
        
        return ground_truth_edges, all_variables
    
    def _parse_edge_list(self) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """
        Parse ground truth from edge list file
        
        Format:
            # Comments start with #
            source1 -> target1
            source2 -> target2
            ...
        
        Returns:
            (ground_truth_edges, all_variables)
            ground_truth_edges: Set of (source, target) tuples
            all_variables: Set of variable names (extracted from edges)
        """
        ground_truth_edges = set()
        all_variables = set()
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse edge: "A -> B"
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        source = parts[0].strip()
                        target = parts[1].strip()
                        if source and target:
                            ground_truth_edges.add((source, target))
                            all_variables.add(source)
                            all_variables.add(target)
        
        return ground_truth_edges, all_variables
    
    def extract_learned_edges(self, adjacency: torch.Tensor, 
                              threshold: float = 0.3) -> Set[Tuple[str, str]]:
        """
        Extract variable-level edges from learned adjacency matrix
        
        Args:
            adjacency: Learned adjacency matrix (105, 105)
            threshold: Threshold for considering a block as having an edge
        
        Returns:
            Set of (var_a, var_b) directed edges
        """
        learned_edges = set()
        
        for var_a in self.var_structure['variable_names']:
            for var_b in self.var_structure['variable_names']:
                if var_a == var_b:
                    continue
                
                # Get block for this variable pair
                states_a = self.var_structure['var_to_states'][var_a]
                states_b = self.var_structure['var_to_states'][var_b]
                
                # Extract block
                block = adjacency[states_a][:, states_b]
                
                # Compute block strength (mean)
                block_strength = block.mean().item()
                
                # If block strength exceeds threshold, add edge
                if block_strength > threshold:
                    learned_edges.add((var_a, var_b))
        
        return learned_edges
    
    def evaluate(self, learned_edges: Set[Tuple[str, str]]) -> Dict:
        """
        Compute all evaluation metrics
        
        Args:
            learned_edges: Set of learned directed edges
        
        Returns:
            Dictionary with all metrics
        """
        # Convert to undirected edges
        learned_undirected = {tuple(sorted([e[0], e[1]])) for e in learned_edges}
        gt_undirected = {tuple(sorted([e[0], e[1]])) for e in self.ground_truth_edges}
        
        # === 1. UNDIRECTED EDGE METRICS ===
        undirected_tp = len(learned_undirected & gt_undirected)
        undirected_fp = len(learned_undirected - gt_undirected)
        undirected_fn = len(gt_undirected - learned_undirected)
        
        edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
        edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
        
        # === 2. DIRECTED EDGE METRICS ===
        directed_tp = len(learned_edges & self.ground_truth_edges)
        directed_fp = len(learned_edges - self.ground_truth_edges)
        directed_fn = len(self.ground_truth_edges - learned_edges)
        
        directed_precision = directed_tp / (directed_tp + directed_fp) if (directed_tp + directed_fp) > 0 else 0
        directed_recall = directed_tp / (directed_tp + directed_fn) if (directed_tp + directed_fn) > 0 else 0
        directed_f1 = 2 * directed_precision * directed_recall / (directed_precision + directed_recall) if (directed_precision + directed_recall) > 0 else 0
        
        # === 3. ORIENTATION ACCURACY ===
        correctly_oriented = 0
        incorrectly_oriented = 0
        
        for learned_edge in learned_edges:
            undirected_edge = tuple(sorted([learned_edge[0], learned_edge[1]]))
            if undirected_edge in gt_undirected:
                # We found this edge, check if direction is correct
                if learned_edge in self.ground_truth_edges:
                    correctly_oriented += 1
                else:
                    # Check if it's reversed
                    reversed_edge = (learned_edge[1], learned_edge[0])
                    if reversed_edge in self.ground_truth_edges:
                        incorrectly_oriented += 1
        
        orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
        
        # === 4. STRUCTURAL HAMMING DISTANCE (SHD) ===
        # According to standard definition (Gemini's clarification):
        # 
        # Skeleton SHD: Only considers edge existence (undirected)
        #   SHD_skeleton = E_add + E_del
        #   E_add (False Positive): edges in learned but not in GT (undirected)
        #   E_del (False Negative): edges in GT but not in learned (undirected)
        #
        # Full SHD: Considers both edge existence and direction
        #   SHD_full = E_add + E_del + E_rev
        #   E_add: edges added (not in GT at all)
        #   E_del: edges missing (in GT but not learned at all)
        #   E_rev: edges with correct skeleton but wrong direction
        
        # Count reversals (edges with correct skeleton but wrong direction)
        reversals = 0
        for learned_edge in learned_edges:
            reversed_edge = (learned_edge[1], learned_edge[0])
            if reversed_edge in self.ground_truth_edges and learned_edge not in self.ground_truth_edges:
                reversals += 1
        
        # Skeleton SHD: undirected edge errors
        skeleton_shd = undirected_fp + undirected_fn
        
        # Full SHD: directed edge errors (standard NeurIPS/ICLR metric)
        # E_add = edges added (not in GT undirected graph)
        # E_del = edges deleted (in GT undirected graph but not learned)
        # E_rev = edges with correct skeleton but wrong direction
        full_shd = undirected_fp + undirected_fn + reversals
        
        # Compile metrics
        metrics = {
            # Undirected (skeleton) metrics
            'edge_precision': edge_precision,
            'edge_recall': edge_recall,
            'edge_f1': edge_f1,
            'undirected_tp': undirected_tp,
            'undirected_fp': undirected_fp,
            'undirected_fn': undirected_fn,
            
            # Directed metrics
            'directed_precision': directed_precision,
            'directed_recall': directed_recall,
            'directed_f1': directed_f1,
            'directed_tp': directed_tp,
            'directed_fp': directed_fp,
            'directed_fn': directed_fn,
            
            # Orientation
            'orientation_accuracy': orientation_accuracy,
            'correctly_oriented': correctly_oriented,
            'incorrectly_oriented': incorrectly_oriented,
            
            # SHD (Structural Hamming Distance)
            'skeleton_shd': skeleton_shd,  # Only edge existence (undirected)
            'full_shd': full_shd,          # Edge existence + direction (standard metric)
            'shd': full_shd,               # Default to full_shd for backward compatibility
            'reversals': reversals,        # Number of edges with wrong direction
            
            # Counts
            'learned_edges': len(learned_edges),
            'ground_truth_edges': len(self.ground_truth_edges)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way"""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        
        print("\n--- EDGE-LEVEL (Ignoring Direction) ---")
        print(f"Edge Precision:     {metrics['edge_precision']:.1%}")
        print(f"Edge Recall:        {metrics['edge_recall']:.1%}")
        print(f"Edge F1 Score:      {metrics['edge_f1']:.1%}")
        print(f"True Positives:     {metrics['undirected_tp']}")
        print(f"False Positives:    {metrics['undirected_fp']}")
        print(f"False Negatives:    {metrics['undirected_fn']}")
        
        print("\n--- DIRECTED EDGE-LEVEL (With Direction) ---")
        print(f"Directed Precision: {metrics['directed_precision']:.1%}")
        print(f"Directed Recall:    {metrics['directed_recall']:.1%}")
        print(f"Directed F1 Score:  {metrics['directed_f1']:.1%}")
        print(f"True Positives:     {metrics['directed_tp']}")
        print(f"False Positives:    {metrics['directed_fp']}")
        print(f"False Negatives:    {metrics['directed_fn']}")
        
        print("\n--- ORIENTATION ACCURACY ---")
        print(f"Orientation Accuracy: {metrics['orientation_accuracy']:.1%}")
        print(f"Correctly Oriented:   {metrics['correctly_oriented']}")
        print(f"Incorrectly Oriented: {metrics['incorrectly_oriented']}")
        
        print("\n--- STRUCTURAL HAMMING DISTANCE (SHD) ---")
        print(f"Skeleton SHD:  {metrics['skeleton_shd']}  (E_add + E_del, undirected)")
        print(f"  E_add (FP):  {metrics['undirected_fp']}  (edges added)")
        print(f"  E_del (FN):  {metrics['undirected_fn']}  (edges missing)")
        print(f"\nFull SHD:      {metrics['full_shd']}  (E_add + E_del + E_rev, directed)")
        print(f"  E_add (FP):  {metrics['undirected_fp']}  (edges added)")
        print(f"  E_del (FN):  {metrics['undirected_fn']}  (edges missing)")
        print(f"  E_rev:       {metrics['reversals']}  (edges reversed)")
        
        print("\n--- SUMMARY ---")
        print(f"Learned Edges:      {metrics['learned_edges']}")
        print(f"Ground Truth Edges: {metrics['ground_truth_edges']}")
    
    def save_results(self, 
                     metrics: Dict, 
                     learned_edges: Set[Tuple[str, str]],
                     output_dir: str,
                     config: Dict,
                     timing_info: Optional[Dict] = None):
        """
        Save evaluation results with comprehensive metadata
        
        Args:
            metrics: Evaluation metrics dictionary
            learned_edges: Set of learned edges
            output_dir: Directory to save results
            config: Training configuration with metadata
            timing_info: Dictionary with timing information (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare comprehensive results
        results = {
            # Metadata
            'metadata': {
                'dataset': config.get('dataset_name', 'Unknown'),
                'llm_model': config.get('llm_model', 'None'),
                'use_llm_prior': config.get('use_llm_prior', False),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'run_id': config.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
            },
            
            # Configuration
            'config': {
                'learning_rate': config.get('learning_rate', None),
                'lambda_group_lasso': config.get('lambda_group_lasso', None),
                'lambda_cycle': config.get('lambda_cycle', None),
                'n_epochs': config.get('n_epochs', None),
                'threshold': config.get('threshold', 0.3),
                'fci_skeleton_path': config.get('fci_skeleton_path', None),
                'llm_direction_path': config.get('llm_direction_path', None)
            },
            
            # Timing information
            'timing': timing_info or {},
            
            # Metrics
            'metrics': metrics,
            
            # Learned edges
            'learned_edges': [{'source': e[0], 'target': e[1]} for e in sorted(learned_edges)],
            
            # Ground truth edges
            'ground_truth_edges': [{'source': e[0], 'target': e[1]} for e in sorted(self.ground_truth_edges)]
        }
        
        # Save as JSON
        json_path = output_path / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as human-readable text
        txt_path = output_path / 'evaluation_results.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CAUSAL DISCOVERY EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset:        {results['metadata']['dataset']}\n")
            f.write(f"LLM Model:      {results['metadata']['llm_model']}\n")
            f.write(f"Use LLM Prior:  {results['metadata']['use_llm_prior']}\n")
            f.write(f"Timestamp:      {results['metadata']['timestamp']}\n")
            f.write(f"Run ID:         {results['metadata']['run_id']}\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            for key, value in results['config'].items():
                if value is not None:
                    f.write(f"{key:25s}: {value}\n")
            f.write("\n")
            
            # Timing
            if timing_info:
                f.write("TIMING INFORMATION\n")
                f.write("-" * 80 + "\n")
                for key, value in timing_info.items():
                    if isinstance(value, float):
                        f.write(f"{key:25s}: {value:.2f} seconds\n")
                    else:
                        f.write(f"{key:25s}: {value}\n")
                f.write("\n")
            
            # Metrics
            f.write("EVALUATION METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Edge-Level (Undirected Skeleton)\n")
            f.write(f"  Precision:      {metrics['edge_precision']:.1%}\n")
            f.write(f"  Recall:         {metrics['edge_recall']:.1%}\n")
            f.write(f"  F1 Score:       {metrics['edge_f1']:.1%}\n")
            f.write(f"  True Positives: {metrics['undirected_tp']}\n")
            f.write(f"  False Positives:{metrics['undirected_fp']}\n")
            f.write(f"  False Negatives:{metrics['undirected_fn']}\n\n")
            
            f.write("Directed Edge-Level\n")
            f.write(f"  Precision:      {metrics['directed_precision']:.1%}\n")
            f.write(f"  Recall:         {metrics['directed_recall']:.1%}\n")
            f.write(f"  F1 Score:       {metrics['directed_f1']:.1%}\n")
            f.write(f"  True Positives: {metrics['directed_tp']}\n")
            f.write(f"  False Positives:{metrics['directed_fp']}\n")
            f.write(f"  False Negatives:{metrics['directed_fn']}\n\n")
            
            f.write("Orientation Accuracy\n")
            f.write(f"  Accuracy:       {metrics['orientation_accuracy']:.1%}\n")
            f.write(f"  Correct:        {metrics['correctly_oriented']}\n")
            f.write(f"  Incorrect:      {metrics['incorrectly_oriented']}\n\n")
            
            f.write("Structural Hamming Distance (SHD)\n")
            f.write(f"  Skeleton SHD:   {metrics['skeleton_shd']} (E_add + E_del, undirected)\n")
            f.write(f"    E_add (FP):   {metrics['undirected_fp']}\n")
            f.write(f"    E_del (FN):   {metrics['undirected_fn']}\n")
            f.write(f"  Full SHD:       {metrics['full_shd']} (E_add + E_del + E_rev, directed)\n")
            f.write(f"    E_add (FP):   {metrics['undirected_fp']}\n")
            f.write(f"    E_del (FN):   {metrics['undirected_fn']}\n")
            f.write(f"    E_rev:        {metrics['reversals']}\n\n")
            
            f.write("Summary\n")
            f.write(f"  Learned Edges:  {metrics['learned_edges']}\n")
            f.write(f"  GT Edges:       {metrics['ground_truth_edges']}\n\n")
            
            # Learned edges
            f.write("=" * 80 + "\n")
            f.write("LEARNED EDGES\n")
            f.write("=" * 80 + "\n")
            for edge in sorted(learned_edges):
                status = "✓" if edge in self.ground_truth_edges else ("↔" if (edge[1], edge[0]) in self.ground_truth_edges else "✗")
                f.write(f"{status} {edge[0]:20s} → {edge[1]:20s}\n")
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
    # Test the evaluator
    import sys
    sys.path.append('..')
    from modules.data_loader import CausalDataLoader
    from modules.prior_builder import PriorBuilder
    from modules.model import CausalDiscoveryModel
    
    # Load data
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='output/knowledge_graph_metadata.json'
    )
    var_structure = loader.get_variable_structure()
    
    # Build priors
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_csv_path='data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
        llm_rules_path='llm_prior_rules'
    )
    
    # Initialize model
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    # Initialize evaluator
    evaluator = CausalGraphEvaluator(
        ground_truth_path='../alarm.bif',
        var_structure=var_structure
    )
    
    # Extract learned edges (before training, should match LLM prior)
    adjacency = model.get_adjacency()
    learned_edges = evaluator.extract_learned_edges(adjacency, threshold=0.3)
    
    # Evaluate
    metrics = evaluator.evaluate(learned_edges)
    evaluator.print_metrics(metrics)

