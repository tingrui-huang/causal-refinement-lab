"""
Evaluate PC Results Against Ground Truth

Mirrors the FCI evaluation metrics (SHD / Edge F1 / Orientation Accuracy),
but targets PC outputs (CPDAG) saved as edges_PC_*.csv.
"""

from pathlib import Path
from datetime import datetime

# Reuse the robust parsers + SHD computation from FCI evaluator
from evaluate_fci import parse_ground_truth, parse_fci_csv, compute_shd


def compute_pc_unresolved_ratio(pc_csv_path):
    """
    For PC (CPDAG), "unresolved" means edges that remain undirected in the CPDAG.
    We compute this in *pair space* (undirected pairs), not per-row in the CSV.
    """
    pc_directed, pc_undirected, _edge_counts = parse_fci_csv(pc_csv_path)

    resolved = len(pc_directed)
    unresolved = len(pc_undirected)
    total = resolved + unresolved

    return {
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "total_pairs": total,
        "unresolved_ratio": (unresolved / total) if total > 0 else 0.0,
    }


def evaluate_pc(pc_csv_path, ground_truth_path, output_dir=None):
    """
    Evaluate PC CPDAG against directed ground truth.

    Returns:
        dict of metrics, matching the keys used by evaluate_fci where possible.
    """
    print("\n" + "=" * 80)
    print("PC EVALUATION AGAINST GROUND TRUTH")
    print("=" * 80)

    gt_edges = parse_ground_truth(ground_truth_path)  # directed
    pc_directed, pc_undirected, edge_counts = parse_fci_csv(pc_csv_path)

    print(f"\nGround Truth: {len(gt_edges)} directed edges")
    print(f"PC Edges: {len(pc_directed)} directed + {len(pc_undirected)} undirected (pairs)")
    print(f"  - Directed rows:   {edge_counts.get('directed', 0)}")
    print(f"  - Undirected rows: {edge_counts.get('undirected', 0)}")

    # === 1. UNDIRECTED SKELETON METRICS ===
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}

    pc_all_undirected = {tuple(sorted([e[0], e[1]])) for e in pc_directed}
    pc_all_undirected.update(pc_undirected)

    undirected_tp = len(pc_all_undirected & gt_undirected)
    undirected_fp = len(pc_all_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - pc_all_undirected)

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

    # === 2. ORIENTATION ACCURACY (Directed edges only) ===
    correctly_oriented = 0
    incorrectly_oriented = 0

    for pc_edge in pc_directed:
        undirected_edge = tuple(sorted([pc_edge[0], pc_edge[1]]))
        if undirected_edge in gt_undirected:
            if pc_edge in gt_edges:
                correctly_oriented += 1
            else:
                reversed_edge = (pc_edge[1], pc_edge[0])
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
    print(f"PC Directed Edges: {len(pc_directed)}")
    print(f"Correctly Oriented: {correctly_oriented}")
    print(f"Incorrectly Oriented: {incorrectly_oriented}")
    print(f"\nOrientation Accuracy: {orientation_accuracy*100:.1f}%")

    # === 3. UNRESOLVED RATIO (PC) ===
    unresolved_stats = compute_pc_unresolved_ratio(pc_csv_path)

    print("\n" + "=" * 80)
    print("UNRESOLVED RATIO (PC CPDAG)")
    print("=" * 80)
    print(f"Total edge pairs: {unresolved_stats['total_pairs']}")
    print(f"  Directed (resolved):   {unresolved_stats['resolved_count']:3d}")
    print(f"  Undirected (unresolved): {unresolved_stats['unresolved_count']:3d}")
    print(f"\nPC Unresolved Ratio: {unresolved_stats['unresolved_ratio']*100:.1f}%")

    # === 4. SHD ===
    shd_stats = compute_shd(pc_csv_path, ground_truth_path)

    print("\n" + "=" * 80)
    print("STRUCTURAL HAMMING DISTANCE (SHD)")
    print("=" * 80)
    print(f"Skeleton SHD: {shd_stats['skeleton_shd']}  (E_add + E_del, undirected)")
    print(f"Full SHD:     {shd_stats['full_shd']}  (E_add + E_del + E_rev, directed)")
    print(f"  E_add (FP):   {shd_stats['additions']}")
    print(f"  E_del (FN):   {shd_stats['deletions']}")
    print(f"  E_rev:        {shd_stats['total_direction_errors']}  (direction errors)")
    print(f"    - Reversed:   {shd_stats['reversals']}")
    print(f"    - Unresolved: {shd_stats['unresolved_as_errors']}  (undirected in PC)")

    # === 5. UNDIRECTED RATIO (pairs) ===
    undirected_ratio = len(pc_undirected) / len(pc_all_undirected) if len(pc_all_undirected) > 0 else 0

    print("\n" + "=" * 80)
    print("UNDIRECTED EDGES (CPDAG)")
    print("=" * 80)
    print(f"Total PC edge pairs: {len(pc_all_undirected)}")
    print(f"Directed pairs: {len(pc_directed)}")
    print(f"Undirected pairs: {len(pc_undirected)}")
    print(f"Undirected Ratio: {undirected_ratio*100:.1f}%")

    # === 6. SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skeleton SHD:         {shd_stats['skeleton_shd']}")
    print(f"Full SHD:             {shd_stats['full_shd']}")
    print(f"Edge F1:              {edge_f1*100:.1f}%")
    print(f"Precision:            {edge_precision*100:.1f}%")
    print(f"Recall:               {edge_recall*100:.1f}%")
    print(f"Orient. Accuracy:     {orientation_accuracy*100:.1f}%")
    print(f"Unresolved Ratio:     {unresolved_stats['unresolved_ratio']*100:.1f}%  <- PC")
    print(f"Undirected Ratio:     {undirected_ratio*100:.1f}%")
    print("=" * 80)

    # Save evaluation report
    if output_dir:
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_PC_{timestamp}.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PC EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PC CSV: {Path(pc_csv_path).name}\n")
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


def find_latest_pc_csv(output_dir="output"):
    """Find the most recent PC CSV file (edges_PC_*.csv) under output_dir/DATASET."""
    from config import DATASET

    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # If caller already passed the dataset-specific output dir (e.g., .../output/alarm),
    # don't append DATASET again.
    if output_path.name == DATASET and output_path.parent.name == "output":
        pc_csvs = list(output_path.glob("edges_PC_*.csv"))
        return max(pc_csvs, key=lambda p: p.stat().st_mtime) if pc_csvs else None

    dataset_dir = output_path / DATASET
    if dataset_dir.exists():
        pc_csvs = list(dataset_dir.glob("edges_PC_*.csv"))
        if pc_csvs:
            return max(pc_csvs, key=lambda p: p.stat().st_mtime)

    pc_csvs = list(output_path.glob("edges_PC_*.csv"))
    if not pc_csvs:
        return None

    return max(pc_csvs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    from config import GROUND_TRUTH_PATH, OUTPUT_DIR

    latest_pc = find_latest_pc_csv(OUTPUT_DIR)
    if not latest_pc:
        print(f"[ERROR] No PC CSV files found in {OUTPUT_DIR}/")
        print("Run main_pc.py first to generate PC results.")
        raise SystemExit(1)

    gt_path = Path(GROUND_TRUTH_PATH)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        print("Please update GROUND_TRUTH_PATH in config.py")
        raise SystemExit(1)

    evaluate_pc(latest_pc, gt_path, output_dir=Path(OUTPUT_DIR))

