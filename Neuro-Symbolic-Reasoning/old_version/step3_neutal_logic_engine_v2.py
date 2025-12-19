"""
Neural Logic Programming (Neural LP) - Multi-Hop Reasoning

Implementation based on Yang et al. 2017 "Differentiable Learning of Logical Rules for Knowledge Base Reasoning"

Key Features:
1. Multi-hop reasoning (configurable depth, default 2 hops)
2. Attention-based rule composition across reasoning paths
3. Differentiable rule learning with gradient descent
4. Sparse logic network initialized with LLM priors

Architecture:
- Input: Patient state vectors (observed facts)
- Rules: Weighted adjacency matrix (learnable)
- Reasoning: Iterative matrix multiplication for multi-hop inference
- Aggregation: Attention mechanism combines reasoning at different depths
- Output: Inferred state beliefs after multi-hop reasoning

Training Objective:
- Reconstruct observed facts through multi-hop reasoning
- L1 regularization for rule sparsity
- Attention entropy regularization for diverse hop usage
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import re


class NeuralLP(nn.Module):
    """
    Neural Logic Programming Model with Multi-Hop Reasoning

    Following Yang et al. 2017 approach:
    - Learns weighted logic rules
    - Performs multi-hop inference through rule composition
    - Uses attention mechanism to combine reasoning paths
    """

    def __init__(self, n_entities, rule_indices, init_weights, max_hops=2):
        """
        Args:
            n_entities: Number of discrete states/entities
            rule_indices: (2, n_rules) tensor of [body_idx, head_idx]
            init_weights: Initial rule confidences from LLM
            max_hops: Maximum reasoning depth (default: 2)
        """
        super().__init__()
        self.n_entities = n_entities
        self.max_hops = max_hops
        self.rule_indices = rule_indices

        # Learnable rule weights (initialized from LLM priors)
        self.rule_weights = nn.Parameter(torch.FloatTensor(init_weights))

        # Attention weights for combining multi-hop paths
        self.hop_attention = nn.Parameter(torch.ones(max_hops) / max_hops)

        print(f"Neural LP initialized:")
        print(f"  - Entities: {n_entities}")
        print(f"  - Rules: {len(init_weights)}")
        print(f"  - Max hops: {max_hops}")

    def build_adjacency_matrix(self):
        """
        Build weighted adjacency matrix from rules.
        A[i,j] = weight of rule (i -> j)
        """
        # Apply sigmoid to weights for soft logic
        weights = torch.sigmoid(self.rule_weights)

        # Build sparse adjacency matrix
        adj = torch.sparse_coo_tensor(
            self.rule_indices,
            weights,
            (self.n_entities, self.n_entities)
        )

        # Convert to dense for matrix operations
        # (In production, use sparse operations for efficiency)
        return adj.to_dense()

    def single_hop(self, state, adjacency):
        """
        Single-hop reasoning: apply rules once

        Args:
            state: (batch, n_entities) current state beliefs
            adjacency: (n_entities, n_entities) rule matrix

        Returns:
            (batch, n_entities) inferred state after one hop
        """
        # Matrix multiplication: state @ adjacency
        # Interpretation: aggregate evidence from all connected states
        next_state = torch.matmul(state, adjacency)

        # Normalize to prevent explosion
        next_state = torch.clamp(next_state, 0.0, 1.0)

        return next_state

    def forward(self, initial_state):
        """
        Multi-hop reasoning with attention-based path aggregation

        Args:
            initial_state: (batch, n_entities) observed facts

        Returns:
            (batch, n_entities) inferred beliefs after multi-hop reasoning
        """
        batch_size = initial_state.size(0)

        # Build adjacency matrix from current rule weights
        adjacency = self.build_adjacency_matrix()

        # Store reasoning at each hop
        hop_results = []
        current_state = initial_state

        # Multi-hop reasoning with residual connections at each step
        for hop in range(self.max_hops):
            # Apply rules for one hop
            next_state = self.single_hop(current_state, adjacency)

            # CRITICAL FIX: Preserve initial observations at each hop (residual connection)
            # Without this, observed facts can disappear after the first hop when inference
            # values are low (e.g., < 0.5). The residual connection ensures that:
            # 1. Observed facts (initial_state) are never lost
            # 2. Multi-hop reasoning can build upon previous inferences
            # 3. The model learns true multi-hop paths rather than just 1-hop rules
            next_state = torch.max(next_state, initial_state)

            hop_results.append(next_state)
            current_state = next_state

        # Attention-based aggregation of multi-hop paths
        # Normalize attention weights
        attention = torch.softmax(self.hop_attention, dim=0)

        # Weighted sum of hop results
        final_state = torch.zeros_like(initial_state)
        for hop_idx, hop_result in enumerate(hop_results):
            final_state += attention[hop_idx] * hop_result

        # Final residual connection to guarantee observed facts are preserved
        final_state = torch.max(final_state, initial_state)

        # Clamp to valid probability range [0, 1]
        final_state = torch.clamp(final_state, 0.0, 1.0)

        return final_state

    def get_reasoning_path_weights(self):
        """Return attention weights for each reasoning hop"""
        return torch.softmax(self.hop_attention, dim=0).detach().cpu().numpy()


def parse_bif_ground_truth(bif_path='../alarm.bif'):
    """
    Parse ground truth causal structure from BIF file

    Args:
        bif_path: Path to the BIF file

    Returns:
        ground_truth_edges: Set of (parent, child) tuples representing true causal edges
        all_variables: Set of all variable names
    """
    print("=" * 70)
    print("PARSING GROUND TRUTH FROM BIF FILE")
    print("=" * 70)

    ground_truth_edges = set()
    all_variables = set()

    with open(bif_path, 'r') as f:
        content = f.read()

    # Extract variable declarations
    var_pattern = r'variable\s+(\w+)\s*\{'
    variables = re.findall(var_pattern, content)
    all_variables = set(variables)

    # Extract probability declarations (these define the causal structure)
    # Format: probability ( CHILD | PARENT1, PARENT2, ... )
    prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'

    for match in re.finditer(prob_pattern, content):
        child = match.group(1)
        parents_str = match.group(2)
        parents = [p.strip() for p in parents_str.split(',')]

        for parent in parents:
            if parent:  # Skip empty strings
                ground_truth_edges.add((parent, child))

    print(f"\nGround Truth Statistics:")
    print(f"  Variables: {len(all_variables)}")
    print(f"  Causal edges: {len(ground_truth_edges)}")

    # Show sample edges
    print(f"\nSample ground truth edges:")
    for edge in list(ground_truth_edges)[:10]:
        print(f"  {edge[0]} -> {edge[1]}")

    return ground_truth_edges, all_variables


def evaluate_learned_rules(learned_rules, ground_truth_edges, all_variables, threshold=0.3):
    """
    Evaluate learned rules against ground truth

    Metrics:
    1. Edge Precision/Recall/F1 (ignoring direction)
    2. Directed Edge Precision/Recall/F1 (with direction)
    3. Orientation Accuracy (among correctly identified edges)
    4. Structural Hamming Distance (SHD)

    Args:
        learned_rules: List of rule dicts with 'body', 'head', 'w_final'
        ground_truth_edges: Set of (parent, child) tuples
        all_variables: Set of all variable names
        threshold: Confidence threshold for keeping learned rules

    Returns:
        Dictionary with all evaluation metrics
    """
    print("\n" + "=" * 70)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("=" * 70)

    def extract_variable_name(state_name):
        """Extract variable name from state name like 'HYPOVOLEMIA_True' -> 'HYPOVOLEMIA'"""
        # State names are in format: VARIABLE_StateValue
        # Split by underscore and take all but last part
        parts = state_name.split('_')
        if len(parts) >= 2:
            # Handle cases like LVEDVOLUME_Low, FIO2_False, etc.
            # The variable name is everything before the last underscore
            return '_'.join(parts[:-1])
        return state_name

    # Extract learned edges from rules (with direction)
    learned_directed_edges = set()
    for rule in learned_rules:
        if rule['w_final'] > threshold:
            # Rule format: head :- body means body -> head
            # Extract variable names from state names
            body_var = extract_variable_name(rule['body'])
            head_var = extract_variable_name(rule['head'])
            learned_directed_edges.add((body_var, head_var))

    # Convert to undirected edges for edge-level evaluation
    learned_undirected = {tuple(sorted([e[0], e[1]])) for e in learned_directed_edges}
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in ground_truth_edges}

    # === 1. UNDIRECTED EDGE METRICS (Edge Recall) ===
    undirected_tp = len(learned_undirected & gt_undirected)
    undirected_fp = len(learned_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - learned_undirected)

    edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
    edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (
                                                                                               edge_precision + edge_recall) > 0 else 0

    # === 2. DIRECTED EDGE METRICS ===
    directed_tp = len(learned_directed_edges & ground_truth_edges)
    directed_fp = len(learned_directed_edges - ground_truth_edges)
    directed_fn = len(ground_truth_edges - learned_directed_edges)

    directed_precision = directed_tp / (directed_tp + directed_fp) if (directed_tp + directed_fp) > 0 else 0
    directed_recall = directed_tp / (directed_tp + directed_fn) if (directed_tp + directed_fn) > 0 else 0
    directed_f1 = 2 * directed_precision * directed_recall / (directed_precision + directed_recall) if (
                                                                                                                   directed_precision + directed_recall) > 0 else 0

    # === 3. ORIENTATION ACCURACY ===
    # Among edges where we got the skeleton right, how many have correct direction?
    correctly_oriented = 0
    incorrectly_oriented = 0

    for learned_edge in learned_directed_edges:
        undirected_edge = tuple(sorted([learned_edge[0], learned_edge[1]]))
        if undirected_edge in gt_undirected:
            # We found this edge, check if direction is correct
            if learned_edge in ground_truth_edges:
                correctly_oriented += 1
            else:
                # Check if it's reversed
                reversed_edge = (learned_edge[1], learned_edge[0])
                if reversed_edge in ground_truth_edges:
                    incorrectly_oriented += 1

    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (
                                                                                                           correctly_oriented + incorrectly_oriented) > 0 else 0

    # === 4. STRUCTURAL HAMMING DISTANCE (SHD) ===
    # SHD = number of edge additions + deletions + reversals needed
    # Reversals count as 1 operation (not 2)

    # Count reversals
    reversals = 0
    for learned_edge in learned_directed_edges:
        reversed_edge = (learned_edge[1], learned_edge[0])
        if reversed_edge in ground_truth_edges and learned_edge not in ground_truth_edges:
            reversals += 1

    # SHD = FP + FN, but reversals are already in both FP and FN
    # So we need to subtract reversals once
    shd = directed_fp + directed_fn - reversals

    # === SUMMARY ===
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

        # SHD
        'shd': shd,
        'reversals': reversals,

        # Counts
        'learned_edges': len(learned_directed_edges),
        'ground_truth_edges': len(ground_truth_edges)
    }

    # Print results
    print(f"\n{'METRIC':<30} | {'VALUE':<15}")
    print("-" * 50)

    print("\n--- EDGE-LEVEL (Ignoring Direction) ---")
    print(f"{'Edge Precision':<30} | {metrics['edge_precision']:.1%}")
    print(f"{'Edge Recall':<30} | {metrics['edge_recall']:.1%}")
    print(f"{'Edge F1 Score':<30} | {metrics['edge_f1']:.1%}")
    print(f"{'True Positives (edges)':<30} | {metrics['undirected_tp']}")
    print(f"{'False Positives (edges)':<30} | {metrics['undirected_fp']}")
    print(f"{'False Negatives (edges)':<30} | {metrics['undirected_fn']}")

    print("\n--- DIRECTED EDGE-LEVEL (With Direction) ---")
    print(f"{'Directed Precision':<30} | {metrics['directed_precision']:.1%}")
    print(f"{'Directed Recall':<30} | {metrics['directed_recall']:.1%}")
    print(f"{'Directed F1 Score':<30} | {metrics['directed_f1']:.1%}")
    print(f"{'True Positives (directed)':<30} | {metrics['directed_tp']}")
    print(f"{'False Positives (directed)':<30} | {metrics['directed_fp']}")
    print(f"{'False Negatives (directed)':<30} | {metrics['directed_fn']}")

    print("\n--- ORIENTATION ACCURACY ---")
    print(f"{'Orientation Accuracy':<30} | {metrics['orientation_accuracy']:.1%}")
    print(f"{'Correctly Oriented':<30} | {metrics['correctly_oriented']}")
    print(f"{'Incorrectly Oriented':<30} | {metrics['incorrectly_oriented']}")

    print("\n--- STRUCTURAL HAMMING DISTANCE ---")
    print(f"{'SHD':<30} | {metrics['shd']}")
    print(f"{'Reversals':<30} | {metrics['reversals']}")

    print("\n--- SUMMARY ---")
    print(f"{'Learned Edges':<30} | {metrics['learned_edges']}")
    print(f"{'Ground Truth Edges':<30} | {metrics['ground_truth_edges']}")

    return metrics


def load_knowledge_graph_and_rules():
    """
    Load knowledge graph data and LLM-provided rules

    Returns:
        data_matrix: (n_patients, n_states) observed facts
        rule_indices: (2, n_rules) rule structure
        init_weights: (n_rules,) initial rule confidences
        state_to_idx: dict mapping state names to indices
        idx_to_state: dict mapping indices to state names
    """
    print("=" * 70)
    print("LOADING KNOWLEDGE GRAPH & RULES")
    print("=" * 70)

    # Load metadata
    with open('output/knowledge_graph_metadata.json', 'r') as f:
        meta = json.load(f)

    # Build state index mapping
    state_to_idx = {}
    idx_to_state = {}
    idx_counter = 0

    for var, mapping in meta['state_mappings'].items():
        sorted_vals = sorted(mapping.keys(), key=lambda x: int(x))
        for val_code in sorted_vals:
            state_name = mapping[val_code]
            state_to_idx[state_name] = idx_counter
            idx_to_state[idx_counter] = state_name
            idx_counter += 1

    print(f"\nEntities (States): {len(state_to_idx)}")

    # Load patient data (facts)
    with open('output/knowledge_graph_triples.json', 'r') as f:
        triples = json.load(f)

    # Build data matrix
    patient_ids = set()
    for t in triples:
        if isinstance(t, dict):
            patient_ids.add(t['subject'])
        else:
            patient_ids.add(t[0])

    n_patients = len(patient_ids)
    patient_to_row = {pid: i for i, pid in enumerate(sorted(list(patient_ids)))}

    data_matrix = torch.zeros(n_patients, len(state_to_idx))

    for t in triples:
        if isinstance(t, dict):
            sub, pred, obj = t['subject'], t['predicate'], t['object']
        else:
            sub, pred, obj = t[0], t[1], t[2]

        if obj in state_to_idx:
            row = patient_to_row[sub]
            col = state_to_idx[obj]
            data_matrix[row, col] = 1.0

    print(f"Patients: {n_patients}")
    print(f"Facts (triples): {int(data_matrix.sum().item())}")

    # Load LLM rules
    print("\nLoading LLM Prior Rules...")

    with open('llm_prior_rules', 'r') as f:
        rule_lines = f.readlines()

    bodies = []
    heads = []
    initial_confidences = []
    valid_rules = []

    for line in rule_lines:
        try:
            # Format: "0.9 :: Head :- Body"
            parts = line.strip().split('::')
            conf = float(parts[0].strip())
            logic = parts[1].strip().split(':-')
            head_state = logic[0].strip()
            body_state = logic[1].strip()

            if head_state in state_to_idx and body_state in state_to_idx:
                heads.append(state_to_idx[head_state])
                bodies.append(state_to_idx[body_state])
                initial_confidences.append(conf)
                valid_rules.append(line.strip())
        except:
            continue

    print(f"Valid rules: {len(valid_rules)} / {len(rule_lines)}")

    # Build rule indices tensor
    rule_indices = torch.tensor([bodies, heads], dtype=torch.long)

    return data_matrix, rule_indices, initial_confidences, state_to_idx, idx_to_state


def train_neural_lp(max_hops=2, n_epochs=300, learning_rate=0.01):
    """
    Train Neural LP model with multi-hop reasoning

    Args:
        max_hops: Maximum reasoning depth
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Load data
    data, rule_indices, init_weights, state_map, idx_map = load_knowledge_graph_and_rules()

    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING NEURAL LP MODEL")
    print("=" * 70)

    model = NeuralLP(
        n_entities=len(state_map),
        rule_indices=rule_indices,
        init_weights=init_weights,
        max_hops=max_hops
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING NEURAL LP - MULTI-HOP REASONING")
    print("=" * 70)
    print(f"Objective: Learn rule weights through {max_hops}-hop reasoning")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 70)

    for epoch in range(n_epochs):
        # Forward pass: multi-hop reasoning
        predictions = model(data)

        # Loss: how well can we reconstruct observed facts
        reconstruction_loss = criterion(predictions, data)

        # L1 regularization on rule weights (encourage sparsity)
        l1_reg = 0.001 * torch.sum(torch.abs(model.rule_weights))

        # Attention regularization (encourage diverse hop usage)
        attention = torch.softmax(model.hop_attention, dim=0)
        attention_entropy = -torch.sum(attention * torch.log(attention + 1e-10))
        attention_reg = -0.01 * attention_entropy  # Negative to encourage diversity

        total_loss = reconstruction_loss + l1_reg + attention_reg

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Clamp rule weights to reasonable range
        with torch.no_grad():
            model.rule_weights.data.clamp_(-5.0, 5.0)

        if (epoch + 1) % 50 == 0:
            hop_weights = model.get_reasoning_path_weights()
            hop_str = ", ".join([f"Hop{i + 1}:{w:.2f}" for i, w in enumerate(hop_weights)])
            print(f"Epoch {epoch + 1:3d} | Loss: {total_loss.item():.4f} | "
                  f"Recon: {reconstruction_loss.item():.4f} | {hop_str}")

    # Analysis
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - ANALYZING LEARNED RULES")
    print("=" * 70)

    # Get final rule weights
    final_weights = torch.sigmoid(model.rule_weights).detach().numpy()
    init_weights_array = np.array(init_weights)

    # Get hop attention weights
    hop_weights = model.get_reasoning_path_weights()
    print("\nReasoning Path Importance:")
    for hop_idx, weight in enumerate(hop_weights):
        print(f"  Hop {hop_idx + 1}: {weight:.3f} ({weight * 100:.1f}%)")

    # Analyze rules
    print("\n" + "-" * 70)
    print("LEARNED RULES (sorted by final confidence)")
    print("-" * 70)

    results = []
    for i in range(len(final_weights)):
        body_idx = rule_indices[0, i].item()
        head_idx = rule_indices[1, i].item()
        w_init = init_weights_array[i]
        w_final = final_weights[i]

        results.append({
            'rule': f"{idx_map[head_idx]} :- {idx_map[body_idx]}",
            'body': idx_map[body_idx],
            'head': idx_map[head_idx],
            'w_init': w_init,
            'w_final': w_final,
            'delta': w_final - w_init
        })

    # Sort by final weight
    results.sort(key=lambda x: x['w_final'], reverse=True)

    print(f"\n{'RULE':<50} | {'INIT':<6} | {'FINAL':<6} | {'CHANGE':<7} | {'STATUS'}")
    print("-" * 100)

    kept_rules = []
    for r in results:
        if r['w_final'] > 0.1:  # Threshold for keeping rules
            delta = r['delta']

            if delta > 0.1:
                status = "STRENGTHENED"
            elif delta < -0.1:
                status = "WEAKENED"
            else:
                status = "STABLE"

            print(f"{r['rule']:<50} | {r['w_init']:.2f}   | {r['w_final']:.2f}   | "
                  f"{delta:+.2f}    | {status}")
            kept_rules.append(r)

    print("-" * 100)
    print(f"Rules kept (confidence > 0.3): {len(kept_rules)} / {len(results)}")

    # Save results
    output_path = Path('../results')
    output_path.mkdir(exist_ok=True)

    # Save learned rules
    with open(output_path / 'neural_lp_rules.txt', 'w') as f:
        f.write(f"# Neural LP Learned Rules (Max Hops: {max_hops})\n")
        f.write(f"# Hop Weights: {', '.join([f'Hop{i + 1}={w:.3f}' for i, w in enumerate(hop_weights)])}\n\n")
        for r in kept_rules:
            f.write(f"{r['w_final']:.4f} :: {r['rule']}\n")

    print(f"\nResults saved to {output_path / 'neural_lp_rules.txt'}")

    # Save reasoning paths analysis
    with open(output_path / 'reasoning_paths.txt', 'w') as f:
        f.write("Multi-Hop Reasoning Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Max Hops: {max_hops}\n\n")
        f.write("Hop Importance:\n")
        for hop_idx, weight in enumerate(hop_weights):
            f.write(f"  Hop {hop_idx + 1}: {weight:.4f} ({weight * 100:.1f}%)\n")
        f.write("\n")
        f.write("Interpretation:\n")
        if max_hops >= 2:
            if hop_weights[0] > 0.7:
                f.write("  - Direct rules (1-hop) dominate reasoning\n")
            elif hop_weights[1] > 0.5:
                f.write("  - Transitive reasoning (2-hop) is important\n")
            else:
                f.write("  - Balanced use of direct and transitive reasoning\n")

    print(f"Reasoning analysis saved to {output_path / 'reasoning_paths.txt'}")

    # Evaluate against ground truth
    try:
        ground_truth_edges, all_vars = parse_bif_ground_truth('../../alarm.bif')
        eval_metrics = evaluate_learned_rules(results, ground_truth_edges, all_vars, threshold=0.3)

        # Save evaluation metrics
        with open(output_path / 'evaluation_metrics.txt', 'w') as f:
            f.write("Neural LP Evaluation Metrics\n")
            f.write("=" * 70 + "\n\n")

            f.write("EDGE-LEVEL METRICS (Ignoring Direction)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Edge Precision:     {eval_metrics['edge_precision']:.1%}\n")
            f.write(f"Edge Recall:        {eval_metrics['edge_recall']:.1%}\n")
            f.write(f"Edge F1 Score:      {eval_metrics['edge_f1']:.1%}\n")
            f.write(f"True Positives:     {eval_metrics['undirected_tp']}\n")
            f.write(f"False Positives:    {eval_metrics['undirected_fp']}\n")
            f.write(f"False Negatives:    {eval_metrics['undirected_fn']}\n\n")

            f.write("DIRECTED EDGE METRICS (With Direction)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Directed Precision: {eval_metrics['directed_precision']:.1%}\n")
            f.write(f"Directed Recall:    {eval_metrics['directed_recall']:.1%}\n")
            f.write(f"Directed F1 Score:  {eval_metrics['directed_f1']:.1%}\n")
            f.write(f"True Positives:     {eval_metrics['directed_tp']}\n")
            f.write(f"False Positives:    {eval_metrics['directed_fp']}\n")
            f.write(f"False Negatives:    {eval_metrics['directed_fn']}\n\n")

            f.write("ORIENTATION METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Orientation Accuracy: {eval_metrics['orientation_accuracy']:.1%}\n")
            f.write(f"Correctly Oriented:   {eval_metrics['correctly_oriented']}\n")
            f.write(f"Incorrectly Oriented: {eval_metrics['incorrectly_oriented']}\n\n")

            f.write("STRUCTURAL HAMMING DISTANCE\n")
            f.write("-" * 70 + "\n")
            f.write(f"SHD:       {eval_metrics['shd']}\n")
            f.write(f"Reversals: {eval_metrics['reversals']}\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Learned Edges:      {eval_metrics['learned_edges']}\n")
            f.write(f"Ground Truth Edges: {eval_metrics['ground_truth_edges']}\n")

        print(f"\nEvaluation metrics saved to {output_path / 'evaluation_metrics.txt'}")

    except FileNotFoundError:
        print("\n[WARNING] Ground truth BIF file not found. Skipping evaluation.")
        print("          Place alarm.bif in parent directory for evaluation.")
        eval_metrics = None

    # Demonstrate multi-hop inference on sample
    print("\n" + "=" * 70)
    print("SAMPLE MULTI-HOP INFERENCE (WITH RESIDUAL CONNECTIONS)")
    print("=" * 70)

    sample_idx = 0
    sample_input = data[sample_idx:sample_idx + 1]

    with torch.no_grad():
        # Get predictions at each hop WITH residual connections
        adjacency = model.build_adjacency_matrix()
        current = sample_input

        print(f"\nPatient sample (index {sample_idx}):")
        print(f"Initial facts: {int(sample_input.sum().item())} states observed")

        for hop in range(max_hops):
            # Apply one hop of reasoning
            next_state = model.single_hop(current, adjacency)

            # Apply residual connection (preserve initial facts)
            next_state = torch.max(next_state, sample_input)

            # Count inferred states
            inferred = (next_state > 0.5).float()
            new_facts = int((inferred - sample_input).clamp(min=0).sum().item())
            total_facts = int(inferred.sum().item())

            print(f"After hop {hop + 1}: {new_facts} new states inferred (total: {total_facts})")

            # Update current state for next hop
            current = next_state

    print("\n" + "=" * 70)
    print("NEURAL LP TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Train with 2-hop reasoning (can be increased to 3 or more)
    train_neural_lp(max_hops=3, n_epochs=300, learning_rate=0.01)
