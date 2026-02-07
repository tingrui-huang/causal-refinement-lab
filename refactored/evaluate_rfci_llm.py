"""
Evaluate RFCI+LLM Results Against Ground Truth (refactored/)

Goal:
  Match the evaluation metrics/format of refactored/evaluate_fci.py, but for
  RFCI+LLM edge lists (edges_RFCI_LLM_*.csv) produced by the hybrid pipelines.

Key differences vs evaluate_fci.py:
  - Treat edge_type == "llm_resolved" as DIRECTED (it has an oriented direction).
  - Optionally ignore rejected edges if a "status" column is present.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
import re
from typing import Dict, Optional, Set, Tuple


def parse_ground_truth(gt_path: str | Path) -> Set[Tuple[str, str]]:
    """
    Parse ground truth edges from file.

    Supports:
    - BIF format (Bayesian Network)
    - Edge list format (simple text: source -> target)
    """
    ground_truth_edges: Set[Tuple[str, str]] = set()
    gt_path = Path(gt_path)

    with open(gt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try BIF format first
    prob_pattern = r"probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)"
    matches = list(re.finditer(prob_pattern, content))

    if matches:
        for match in matches:
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(",")]
            for parent in parents:
                if parent:
                    ground_truth_edges.add((parent, child))
    else:
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    if source and target:
                        ground_truth_edges.add((source, target))

    return ground_truth_edges


def parse_rfci_llm_csv(rfci_llm_csv_path: str | Path) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Dict[str, int]]:
    """
    Parse RFCI+LLM edges from CSV file.

    Returns:
      - directed_edges: set[(source, target)] where direction is explicitly chosen
      - undirected_edges: set[sorted(source, target)] for edges that remain unresolved
      - edge_counts: counts by edge_type

    Conventions:
      - edge_type in {"directed", "llm_resolved"} is treated as DIRECTED
      - edge_type in {"undirected", "partial", "tail-tail", "bidirected"} is treated as UNRESOLVED (skeleton-only)
      - if a "status" column exists, rows with status == "rejected" are ignored
    """
    df = pd.read_csv(rfci_llm_csv_path)

    # Determine column names (case-insensitive compatibility)
    if "Source" in df.columns and "Target" in df.columns:
        source_col, target_col = "Source", "Target"
    elif "source" in df.columns and "target" in df.columns:
        source_col, target_col = "source", "target"
    else:
        source_col, target_col = df.columns[0], df.columns[1]
        print(f"[WARN] Standard columns not found, using: {source_col} -> {target_col}")

    # Determine edge type column
    if "edge_type" in df.columns:
        edge_type_col = "edge_type"
    elif "Edge_Type" in df.columns:
        edge_type_col = "Edge_Type"
    elif "type" in df.columns:
        edge_type_col = "type"
    else:
        edge_type_col = None
        print("[WARN] No edge_type column found, assuming all edges are directed")

    # Optional status column (to ignore rejected)
    status_col = None
    for c in ["status", "Status"]:
        if c in df.columns:
            status_col = c
            break

    directed_types = {"directed", "llm_resolved"}
    unresolved_types = {"undirected", "partial", "tail-tail", "bidirected"}

    directed_edges: Set[Tuple[str, str]] = set()
    undirected_edges: Set[Tuple[str, str]] = set()
    edge_counts: Dict[str, int] = {
        "directed": 0,
        "llm_resolved": 0,
        "undirected": 0,
        "partial": 0,
        "tail-tail": 0,
        "bidirected": 0,
    }

    for _, row in df.iterrows():
        if status_col is not None:
            st = str(row.get(status_col, "accepted")).strip().lower()
            if st == "rejected":
                continue

        source = row[source_col]
        target = row[target_col]
        edge_type = str(row[edge_type_col]).strip() if edge_type_col else "directed"

        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

        if edge_type in directed_types:
            directed_edges.add((source, target))
        elif edge_type in unresolved_types:
            undirected_edges.add(tuple(sorted([source, target])))
        else:
            # Unknown types: be conservative and treat as directed if it is not explicitly unresolved
            directed_edges.add((source, target))

    return directed_edges, undirected_edges, edge_counts


def compute_unresolved_ratio(rfci_llm_csv_path: str | Path) -> Dict:
    """
    Unresolved ratio = (# of non-directed edges) / (# of total edges).
    Mirrors compute_fci_unresolved_ratio() semantics in evaluate_fci.py.
    """
    directed_edges, undirected_edges, edge_counts = parse_rfci_llm_csv(rfci_llm_csv_path)

    total_edges = sum(edge_counts.values())
    resolved_edges = edge_counts.get("directed", 0) + edge_counts.get("llm_resolved", 0)
    unresolved_edges = total_edges - resolved_edges

    unresolved_ratio = unresolved_edges / total_edges if total_edges > 0 else 0

    return {
        "unresolved_count": unresolved_edges,
        "resolved_count": resolved_edges,
        "total_edges": total_edges,
        "unresolved_ratio": unresolved_ratio,
        "edge_type_breakdown": edge_counts,
    }


def compute_shd(rfci_llm_csv_path: str | Path, ground_truth_path: str | Path) -> Dict:
    """
    Compute Structural Hamming Distance (SHD), mirroring evaluate_fci.py:
      - skeleton_shd = additions + deletions (undirected)
      - full_shd = additions + deletions + reversals + unresolved_as_errors
    """
    gt_edges = parse_ground_truth(ground_truth_path)
    directed_edges, undirected_edges, _ = parse_rfci_llm_csv(rfci_llm_csv_path)

    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    all_undirected = {tuple(sorted([e[0], e[1]])) for e in directed_edges}
    all_undirected.update(undirected_edges)

    additions = len(all_undirected - gt_undirected)
    deletions = len(gt_undirected - all_undirected)

    reversals = 0
    unresolved_as_errors = 0

    for edge in directed_edges:
        und = tuple(sorted([edge[0], edge[1]]))
        if und in gt_undirected:
            rev = (edge[1], edge[0])
            if rev in gt_edges and edge not in gt_edges:
                reversals += 1

    for und_edge in undirected_edges:
        if und_edge in gt_undirected:
            unresolved_as_errors += 1

    skeleton_shd = additions + deletions
    full_shd = additions + deletions + reversals + unresolved_as_errors

    return {
        "skeleton_shd": skeleton_shd,
        "full_shd": full_shd,
        "shd": full_shd,
        "additions": additions,
        "deletions": deletions,
        "reversals": reversals,
        "unresolved_as_errors": unresolved_as_errors,
        "total_direction_errors": reversals + unresolved_as_errors,
    }


def evaluate_rfci_llm(rfci_llm_csv_path: str | Path, ground_truth_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict:
    """
    Evaluate RFCI+LLM edge list against ground truth.
    Mirrors evaluate_fci.py output/metrics.
    """
    rfci_llm_csv_path = Path(rfci_llm_csv_path)
    ground_truth_path = Path(ground_truth_path)

    print("\n" + "=" * 80)
    print("RFCI+LLM EVALUATION AGAINST GROUND TRUTH")
    print("=" * 80)

    gt_edges = parse_ground_truth(ground_truth_path)
    directed_edges, undirected_edges, edge_counts = parse_rfci_llm_csv(rfci_llm_csv_path)

    print(f"\nGround Truth: {len(gt_edges)} directed edges")
    print(f"RFCI+LLM Edges: {len(directed_edges)} directed + {len(undirected_edges)} undirected/partial")
    print(f"  - Directed: {edge_counts.get('directed', 0)}")
    print(f"  - LLM-resolved: {edge_counts.get('llm_resolved', 0)}")
    print(f"  - Undirected: {edge_counts.get('undirected', 0)}")
    print(f"  - Partial: {edge_counts.get('partial', 0)}")
    print(f"  - Tail-tail: {edge_counts.get('tail-tail', 0)}")
    print(f"  - Bidirected: {edge_counts.get('bidirected', 0)}")

    # === 1. UNDIRECTED SKELETON METRICS ===
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}

    all_undirected = {tuple(sorted([e[0], e[1]])) for e in directed_edges}
    all_undirected.update(undirected_edges)

    undirected_tp = len(all_undirected & gt_undirected)
    undirected_fp = len(all_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - all_undirected)

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

    for edge in directed_edges:
        und = tuple(sorted([edge[0], edge[1]]))
        if und in gt_undirected:
            if edge in gt_edges:
                correctly_oriented += 1
            else:
                rev = (edge[1], edge[0])
                if rev in gt_edges:
                    incorrectly_oriented += 1

    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0

    print("\n" + "=" * 80)
    print("ORIENTATION ACCURACY (Directed Edges Only)")
    print("=" * 80)
    print(f"RFCI+LLM Directed Edges: {len(directed_edges)}")
    print(f"Correctly Oriented: {correctly_oriented}")
    print(f"Incorrectly Oriented: {incorrectly_oriented}")
    print(f"\nOrientation Accuracy: {orientation_accuracy*100:.1f}%")

    # === 3. UNRESOLVED RATIO ===
    unresolved_stats = compute_unresolved_ratio(rfci_llm_csv_path)

    print("\n" + "=" * 80)
    print("UNRESOLVED RATIO (RFCI+LLM)")
    print("=" * 80)
    print(f"Total edges: {unresolved_stats['total_edges']}")
    print(f"  Directed (->):       {unresolved_stats['resolved_count']:3d}  ({unresolved_stats['resolved_count']/unresolved_stats['total_edges']*100:.1f}%) [direction resolved]")
    print(f"  Unresolved:          {unresolved_stats['unresolved_count']:3d}  ({unresolved_stats['unresolved_ratio']*100:.1f}%) [direction NOT resolved]")

    breakdown = unresolved_stats["edge_type_breakdown"]
    print(f"    - Bidirected (<->): {breakdown.get('bidirected', 0):3d}")
    print(f"    - Partial (o->):    {breakdown.get('partial', 0):3d}")
    print(f"    - Undirected (o-o): {breakdown.get('undirected', 0):3d}")
    print(f"    - Tail-tail (--):   {breakdown.get('tail-tail', 0):3d}")

    print(f"\nUnresolved Ratio: {unresolved_stats['unresolved_ratio']*100:.1f}%")

    # === 4. SHD ===
    shd_stats = compute_shd(rfci_llm_csv_path, ground_truth_path)

    print("\n" + "=" * 80)
    print("STRUCTURAL HAMMING DISTANCE (SHD)")
    print("=" * 80)
    print(f"Skeleton SHD: {shd_stats['skeleton_shd']}  (E_add + E_del, undirected)")
    print(f"  E_add (FP):   {shd_stats['additions']}  (edges added)")
    print(f"  E_del (FN):   {shd_stats['deletions']}  (edges missing)")
    print(f"\nFull SHD:     {shd_stats['full_shd']}  (E_add + E_del + E_rev, directed)")
    print(f"  E_add (FP):   {shd_stats['additions']}  (edges added)")
    print(f"  E_del (FN):   {shd_stats['deletions']}  (edges missing)")
    print(f"  E_rev:        {shd_stats['total_direction_errors']}  (direction errors)")
    print(f"    - Reversed: {shd_stats['reversals']}  (wrong direction)")
    print(f"    - Unresolved: {shd_stats['unresolved_as_errors']}  (undirected/partial in RFCI+LLM)")
    print("\n[NOTE] Unresolved edges count as direction errors for fair comparison with Neural Network")

    # === 5. UNDIRECTED RATIO ===
    undirected_ratio = len(undirected_edges) / len(all_undirected) if len(all_undirected) > 0 else 0

    print("\n" + "=" * 80)
    print("UNDIRECTED / PARTIAL EDGES")
    print("=" * 80)
    print(f"Total edges: {len(all_undirected)}")
    print(f"Directed: {len(directed_edges)}")
    print(f"Undirected/Partial: {len(undirected_edges)}")
    print(f"Undirected Ratio: {undirected_ratio*100:.1f}%")

    # === 6. SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skeleton SHD:         {shd_stats['skeleton_shd']}  (undirected)")
    print(f"Full SHD:             {shd_stats['full_shd']}  (directed, standard metric)")
    print(f"Edge F1:              {edge_f1*100:.1f}%")
    print(f"Precision:            {edge_precision*100:.1f}%")
    print(f"Recall:               {edge_recall*100:.1f}%")
    print(f"Orient. Accuracy:     {orientation_accuracy*100:.1f}%")
    print(f"Unresolved Ratio:     {unresolved_stats['unresolved_ratio']*100:.1f}%")
    print(f"Undirected Ratio:     {undirected_ratio*100:.1f}%")
    print("=" * 80)

    if output_dir:
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_RFCI_LLM_{timestamp}.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RFCI+LLM EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV: {rfci_llm_csv_path.name}\n")
            f.write(f"Ground Truth: {ground_truth_path.name}\n\n")

            f.write("METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Edge F1:              {edge_f1*100:.1f}%\n")
            f.write(f"Edge Precision:       {edge_precision*100:.1f}%\n")
            f.write(f"Edge Recall:          {edge_recall*100:.1f}%\n")
            f.write(f"Orientation Accuracy: {orientation_accuracy*100:.1f}%\n")
            f.write(f"Undirected Ratio:     {undirected_ratio*100:.1f}%\n")
            f.write(f"Unresolved Ratio:     {unresolved_stats['unresolved_ratio']*100:.1f}%\n\n")

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

        print(f"\n[OK] Evaluation report saved to: {report_path}")

    return {
        "skeleton_shd": shd_stats["skeleton_shd"],
        "full_shd": shd_stats["full_shd"],
        "shd": shd_stats["shd"],
        "shd_additions": shd_stats["additions"],
        "shd_deletions": shd_stats["deletions"],
        "shd_reversals": shd_stats["reversals"],
        "edge_f1": edge_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "orientation_accuracy": orientation_accuracy,
        "unresolved_ratio": unresolved_stats["unresolved_ratio"],
        "unresolved_count": unresolved_stats["unresolved_count"],
        "resolved_count": unresolved_stats["resolved_count"],
        "undirected_ratio": undirected_ratio,
        "undirected_tp": undirected_tp,
        "undirected_fp": undirected_fp,
        "undirected_fn": undirected_fn,
        "correctly_oriented": correctly_oriented,
        "incorrectly_oriented": incorrectly_oriented,
    }


def find_latest_rfci_llm_csv(output_dir: str | Path = "output") -> Optional[Path]:
    """
    Find the most recent RFCI+LLM edge list CSV for the current dataset.
    Prefers edges_RFCI_LLM_*.csv, but falls back to edges_FCI_LLM_*.csv if needed.
    """
    from config import DATASET

    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    dataset_dir = output_path / DATASET
    patterns = ["edges_RFCI_LLM_*.csv", "edges_FCI_LLM_*.csv"]

    def _pick_latest(paths):
        return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

    if dataset_dir.exists():
        for pat in patterns:
            hits = list(dataset_dir.glob(pat))
            if hits:
                return _pick_latest(hits)

    for pat in patterns:
        hits = list(output_path.glob(pat))
        if hits:
            return _pick_latest(hits)

    return None


if __name__ == "__main__":
    from config import GROUND_TRUTH_PATH, OUTPUT_DIR

    latest = find_latest_rfci_llm_csv(OUTPUT_DIR)
    if not latest:
        print(f"[ERROR] No RFCI_LLM / FCI_LLM CSV files found in {OUTPUT_DIR}/")
        print("Run refactored/main_hybrid_fci_llm.py first to generate hybrid results.")
        raise SystemExit(1)

    gt_path = Path(GROUND_TRUTH_PATH)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        raise SystemExit(1)

    print(f"Found latest RFCI+LLM output: {latest.name}")
    evaluate_rfci_llm(latest, gt_path, output_dir=Path(OUTPUT_DIR))

