"""
Evaluate GES Results Against Ground Truth

Mirrors the FCI/PC evaluation style and metrics, but targets GES outputs
saved as edges_GES_*.csv.
"""

from pathlib import Path
from datetime import datetime

# Reuse robust GT parser + CSV parser + SHD implementation.
from evaluate_fci import parse_ground_truth, parse_fci_csv, compute_shd


def compute_ges_unresolved_ratio(ges_csv_path):
    """
    For GES (typically CPDAG-like output), treat non-directed edge pairs as unresolved.
    """
    ges_directed, ges_undirected, _edge_counts = parse_fci_csv(ges_csv_path)

    resolved = len(ges_directed)
    unresolved = len(ges_undirected)
    total = resolved + unresolved

    return {
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "total_pairs": total,
        "unresolved_ratio": (unresolved / total) if total > 0 else 0.0,
    }


def evaluate_ges(ges_csv_path, ground_truth_path, output_dir=None):
    """
    Evaluate GES graph against directed ground truth.

    Returns:
        dict of metrics, aligned with evaluate_fci / evaluate_pc keys where possible.
    """
    print("\n" + "=" * 80)
    print("GES EVALUATION AGAINST GROUND TRUTH")
    print("=" * 80)

    gt_edges = parse_ground_truth(ground_truth_path)
    ges_directed, ges_undirected, edge_counts = parse_fci_csv(ges_csv_path)

    print(f"\nGround Truth: {len(gt_edges)} directed edges")
    print(f"GES Edges: {len(ges_directed)} directed + {len(ges_undirected)} undirected (pairs)")
    print(f"  - Directed rows:   {edge_counts.get('directed', 0)}")
    print(f"  - Undirected rows: {edge_counts.get('undirected', 0)}")

    # 1) Skeleton metrics
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    ges_all_undirected = {tuple(sorted([e[0], e[1]])) for e in ges_directed}
    ges_all_undirected.update(ges_undirected)

    undirected_tp = len(ges_all_undirected & gt_undirected)
    undirected_fp = len(ges_all_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - ges_all_undirected)

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

    # 2) Orientation accuracy (directed edges only)
    correctly_oriented = 0
    incorrectly_oriented = 0

    for ges_edge in ges_directed:
        undirected_edge = tuple(sorted([ges_edge[0], ges_edge[1]]))
        if undirected_edge in gt_undirected:
            if ges_edge in gt_edges:
                correctly_oriented += 1
            else:
                reversed_edge = (ges_edge[1], ges_edge[0])
                if reversed_edge in gt_edges:
                    incorrectly_oriented += 1

    orientation_accuracy = (
        correctly_oriented / (correctly_oriented + incorrectly_oriented)
        if (correctly_oriented + incorrectly_oriented) > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("ORIENTATION ACCURACY (Directed Edges Only)")
    print("=" * 80)
    print(f"GES Directed Edges: {len(ges_directed)}")
    print(f"Correctly Oriented: {correctly_oriented}")
    print(f"Incorrectly Oriented: {incorrectly_oriented}")
    print(f"\nOrientation Accuracy: {orientation_accuracy*100:.1f}%")

    # 3) Unresolved ratio
    unresolved_stats = compute_ges_unresolved_ratio(ges_csv_path)

    print("\n" + "=" * 80)
    print("UNRESOLVED RATIO (GES)")
    print("=" * 80)
    print(f"Total edge pairs: {unresolved_stats['total_pairs']}")
    print(f"  Directed (resolved):     {unresolved_stats['resolved_count']:3d}")
    print(f"  Undirected (unresolved): {unresolved_stats['unresolved_count']:3d}")
    print(f"\nGES Unresolved Ratio: {unresolved_stats['unresolved_ratio']*100:.1f}%")

    # 4) SHD
    shd_stats = compute_shd(ges_csv_path, ground_truth_path)

    print("\n" + "=" * 80)
    print("STRUCTURAL HAMMING DISTANCE (SHD)")
    print("=" * 80)
    print(f"Skeleton SHD: {shd_stats['skeleton_shd']}  (E_add + E_del, undirected)")
    print(f"Full SHD:     {shd_stats['full_shd']}  (E_add + E_del + E_rev, directed)")
    print(f"  E_add (FP):   {shd_stats['additions']}")
    print(f"  E_del (FN):   {shd_stats['deletions']}")
    print(f"  E_rev:        {shd_stats['total_direction_errors']}  (direction errors)")
    print(f"    - Reversed:   {shd_stats['reversals']}")
    print(f"    - Unresolved: {shd_stats['unresolved_as_errors']}  (undirected in GES)")

    # 5) Undirected ratio
    undirected_ratio = len(ges_undirected) / len(ges_all_undirected) if len(ges_all_undirected) > 0 else 0

    print("\n" + "=" * 80)
    print("UNDIRECTED EDGES (GES)")
    print("=" * 80)
    print(f"Total GES edge pairs: {len(ges_all_undirected)}")
    print(f"Directed pairs: {len(ges_directed)}")
    print(f"Undirected pairs: {len(ges_undirected)}")
    print(f"Undirected Ratio: {undirected_ratio*100:.1f}%")

    # 6) Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skeleton SHD:         {shd_stats['skeleton_shd']}")
    print(f"Full SHD:             {shd_stats['full_shd']}")
    print(f"Edge F1:              {edge_f1*100:.1f}%")
    print(f"Precision:            {edge_precision*100:.1f}%")
    print(f"Recall:               {edge_recall*100:.1f}%")
    print(f"Orient. Accuracy:     {orientation_accuracy*100:.1f}%")
    print(f"Unresolved Ratio:     {unresolved_stats['unresolved_ratio']*100:.1f}%  <- GES")
    print(f"Undirected Ratio:     {undirected_ratio*100:.1f}%")
    print("=" * 80)

    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_GES_{timestamp}.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("GES EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GES CSV: {Path(ges_csv_path).name}\n")
            f.write(f"Ground Truth: {Path(ground_truth_path).name}\n\n")

            f.write("METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Edge F1:              {edge_f1*100:.1f}%\n")
            f.write(f"Edge Precision:       {edge_precision*100:.1f}%\n")
            f.write(f"Edge Recall:          {edge_recall*100:.1f}%\n")
            f.write(f"Orientation Accuracy: {orientation_accuracy*100:.1f}%\n")
            f.write(f"Undirected Ratio:     {undirected_ratio*100:.1f}%\n")
            f.write(f"Unresolved Ratio:     {unresolved_stats['unresolved_ratio']*100:.1f}%\n\n")

            f.write("SHD\n")
            f.write("-" * 80 + "\n")
            f.write(f"Skeleton SHD: {shd_stats['skeleton_shd']}\n")
            f.write(f"Full SHD:     {shd_stats['full_shd']}\n")
            f.write(f"E_add (FP):   {shd_stats['additions']}\n")
            f.write(f"E_del (FN):   {shd_stats['deletions']}\n")
            f.write(f"E_rev:        {shd_stats['total_direction_errors']}\n")
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


def find_latest_ges_csv(output_dir="output"):
    """Find the most recent GES CSV file (edges_GES_*.csv) under output_dir/DATASET."""
    from config import DATASET

    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Caller may already pass dataset-specific dir (.../output/<dataset>)
    if output_path.name == DATASET and output_path.parent.name == "output":
        ges_csvs = list(output_path.glob("edges_GES_*.csv"))
        return max(ges_csvs, key=lambda p: p.stat().st_mtime) if ges_csvs else None

    dataset_dir = output_path / DATASET
    if dataset_dir.exists():
        ges_csvs = list(dataset_dir.glob("edges_GES_*.csv"))
        if ges_csvs:
            return max(ges_csvs, key=lambda p: p.stat().st_mtime)

    ges_csvs = list(output_path.glob("edges_GES_*.csv"))
    if not ges_csvs:
        return None

    return max(ges_csvs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    from config import GROUND_TRUTH_PATH, OUTPUT_DIR

    latest_ges = find_latest_ges_csv(OUTPUT_DIR)
    if not latest_ges:
        print(f"[ERROR] No GES CSV files found in {OUTPUT_DIR}/")
        print("Run main_ges.py first to generate GES results.")
        raise SystemExit(1)

    gt_path = Path(GROUND_TRUTH_PATH)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        print("Please update GROUND_TRUTH_PATH in config.py")
        raise SystemExit(1)

    evaluate_ges(latest_ges, gt_path, output_dir=Path(OUTPUT_DIR))
