"""
Visualize comparison of all Step 4 refinement approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results():
    """Load all results."""
    results = {}
    results['step3'] = np.load('results/learned_adjacency.npy')
    results['v1'] = np.load('results/learned_adjacency_refined.npy')
    results['v2'] = np.load('results/learned_adjacency_refined_v2.npy')
    results['final'] = np.load('results/learned_adjacency_final.npy')
    results['optimized'] = np.load('results/learned_adjacency_optimized.npy')
    results['selective'] = np.load('results/learned_adjacency_selective.npy')
    return results


def evaluate_success(adj):
    """Calculate success rate."""
    checks = [
        adj[0, 1] > 1.0,      # X->Y strong
        adj[1, 2] < -1.0,     # Y->Z strong
        abs(adj[1, 0]) < 0.2, # Y->X weak
        abs(adj[2, 1]) < 0.2, # Z->Y weak
        abs(adj[0, 2]) < 0.2, # X->Z weak
        abs(adj[2, 0]) < 0.2  # Z->X weak
    ]
    return sum(checks) / len(checks)


def visualize_comparison():
    """Create comprehensive visualization."""
    results = load_results()
    gt_weights = np.load('data/ground_truth_weights.npy')
    
    fig = plt.figure(figsize=(16, 10))
    
    # Method names and labels
    methods = {
        'step3': 'Step 3\n(Baseline)',
        'v1': 'Step 4 v1\n(DAG+L1)',
        'v2': 'Step 4 v2\n(Stronger)',
        'final': 'Step 4 Final\n(Two-Phase)',
        'optimized': 'Step 4 Opt\n(Multi-Target)',
        'selective': 'Step 4 Sel\n(Weighted)'
    }
    
    edges = ['X→Y', 'Y→Z', 'Y→X', 'Z→Y', 'X→Z', 'Z→X']
    edge_indices = [(0,1), (1,2), (1,0), (2,1), (0,2), (2,0)]
    gt_values = [2.0, -3.0, 0.0, 0.0, 0.0, 0.0]
    
    # 1. Success Rate Comparison (top)
    ax1 = plt.subplot(2, 3, 1)
    success_rates = [evaluate_success(results[k]) * 100 for k in methods.keys()]
    method_labels = list(methods.values())
    colors = ['#ff6b6b', '#ff8787', '#ffa94d', '#4dabf7', '#74c0fc', '#c5c5c5']
    
    bars = ax1.bar(range(len(methods)), success_rates, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(method_labels, fontsize=9)
    ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.axhline(66.7, color='red', linestyle='--', alpha=0.5, label='Baseline (66.7%)')
    ax1.axhline(83.3, color='green', linestyle='--', alpha=0.5, label='Best (83.3%)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. True Edges (X->Y, Y->Z)
    ax2 = plt.subplot(2, 3, 2)
    method_keys = list(methods.keys())
    x_y_vals = [results[k][0, 1] for k in method_keys]
    y_z_vals = [results[k][1, 2] for k in method_keys]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    ax2.bar(x_pos - width/2, x_y_vals, width, label='X→Y (truth: 2.0)', 
            color='#4dabf7', edgecolor='black', linewidth=1)
    ax2.bar(x_pos + width/2, y_z_vals, width, label='Y→Z (truth: -3.0)', 
            color='#ff6b6b', edgecolor='black', linewidth=1)
    ax2.axhline(2.0, color='blue', linestyle='--', alpha=0.3)
    ax2.axhline(-3.0, color='red', linestyle='--', alpha=0.3)
    ax2.axhline(1.0, color='blue', linestyle=':', alpha=0.3, label='Threshold (1.0)')
    ax2.axhline(-1.0, color='red', linestyle=':', alpha=0.3, label='Threshold (-1.0)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_labels, fontsize=9)
    ax2.set_ylabel('Weight', fontsize=11, fontweight='bold')
    ax2.set_title('True Causal Edges (should be strong)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Reverse Edges (Y->X, Z->Y)
    ax3 = plt.subplot(2, 3, 3)
    y_x_vals = [abs(results[k][1, 0]) for k in method_keys]
    z_y_vals = [abs(results[k][2, 1]) for k in method_keys]
    
    ax3.bar(x_pos - width/2, y_x_vals, width, label='|Y→X| (should be ~0)', 
            color='#ffa94d', edgecolor='black', linewidth=1)
    ax3.bar(x_pos + width/2, z_y_vals, width, label='|Z→Y| (should be ~0)', 
            color='#ffd43b', edgecolor='black', linewidth=1)
    ax3.axhline(0.2, color='green', linestyle='--', alpha=0.5, label='Threshold (0.2)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_labels, fontsize=9)
    ax3.set_ylabel('Absolute Weight', fontsize=11, fontweight='bold')
    ax3.set_title('Reverse Edges (should be suppressed)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(max(y_x_vals), max(z_y_vals)) * 1.2)
    
    # 4. Spurious Edges (X->Z, Z->X)
    ax4 = plt.subplot(2, 3, 4)
    x_z_vals = [abs(results[k][0, 2]) for k in method_keys]
    z_x_vals = [abs(results[k][2, 0]) for k in method_keys]
    
    ax4.bar(x_pos - width/2, x_z_vals, width, label='|X→Z| (should be ~0)', 
            color='#a9e34b', edgecolor='black', linewidth=1)
    ax4.bar(x_pos + width/2, z_x_vals, width, label='|Z→X| (should be ~0)', 
            color='#94d82d', edgecolor='black', linewidth=1)
    ax4.axhline(0.2, color='green', linestyle='--', alpha=0.5, label='Threshold (0.2)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(method_labels, fontsize=9)
    ax4.set_ylabel('Absolute Weight', fontsize=11, fontweight='bold')
    ax4.set_title('Spurious Edges (should be removed)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, max(max(x_z_vals), max(z_x_vals)) * 1.2)
    
    # 5. Heatmap of Best Result (Step 4 Final)
    ax5 = plt.subplot(2, 3, 5)
    best_adj = results['final']
    im = ax5.imshow(best_adj, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    ax5.set_xticks([0, 1, 2])
    ax5.set_yticks([0, 1, 2])
    ax5.set_xticklabels(['X', 'Y', 'Z'], fontsize=11)
    ax5.set_yticklabels(['X', 'Y', 'Z'], fontsize=11)
    ax5.set_title('Best Result: Step 4 Final\n(Two-Phase Training)', 
                 fontsize=12, fontweight='bold')
    
    # Add values on heatmap
    for i in range(3):
        for j in range(3):
            text = ax5.text(j, i, f'{best_adj[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax5, label='Weight')
    
    # 6. Ground Truth Heatmap
    ax6 = plt.subplot(2, 3, 6)
    im2 = ax6.imshow(gt_weights, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(['X', 'Y', 'Z'], fontsize=11)
    ax6.set_yticklabels(['X', 'Y', 'Z'], fontsize=11)
    ax6.set_title('Ground Truth\n(X → Y → Z)', fontsize=12, fontweight='bold')
    
    # Add values on heatmap
    for i in range(3):
        for j in range(3):
            text = ax6.text(j, i, f'{gt_weights[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.colorbar(im2, ax=ax6, label='Weight')
    
    plt.suptitle('Step 4 Refinement: Comprehensive Comparison\n' +
                 'DAG Constraints + Stronger Sparsity', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/step4_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comprehensive comparison to results/step4_comprehensive_comparison.png")
    plt.close()
    
    # Create summary table figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Method', 'Strategy', 'X→Y', 'Y→Z', 'Y→X', 'Z→Y', 'X→Z', 'Z→X', 'Success'])
    table_data.append(['Ground Truth', '-', '2.0', '-3.0', '0.0', '0.0', '0.0', '0.0', '-'])
    
    for key, label in methods.items():
        adj = results[key]
        success = evaluate_success(adj) * 100
        
        strategies = {
            'step3': 'No constraints',
            'v1': 'DAG=1.0, L1=0.05',
            'v2': 'DAG=3.0, L1=0.1',
            'final': 'Two-phase',
            'optimized': 'Multi-target',
            'selective': 'Weighted reg'
        }
        
        row = [
            label.replace('\n', ' '),
            strategies[key],
            f'{adj[0,1]:.3f}',
            f'{adj[1,2]:.3f}',
            f'{adj[1,0]:.3f}',
            f'{adj[2,1]:.3f}',
            f'{adj[0,2]:.3f}',
            f'{adj[2,0]:.3f}',
            f'{success:.1f}%'
        ]
        table_data.append(row)
    
    # Color coding for cells
    cell_colors = []
    for i, row in enumerate(table_data):
        if i == 0:  # Header
            cell_colors.append(['#e0e0e0'] * len(row))
        elif i == 1:  # Ground truth
            cell_colors.append(['#f0f0f0'] * len(row))
        else:
            colors = ['white'] * len(row)
            # Color code success rate
            success_val = float(row[-1].rstrip('%'))
            if success_val >= 80:
                colors[-1] = '#c3fae8'  # Light green
            elif success_val >= 60:
                colors[-1] = '#ffe066'  # Light yellow
            else:
                colors[-1] = '#ffc9c9'  # Light red
            cell_colors.append(colors)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    cellColours=cell_colors, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Bold header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=11)
    
    plt.title('Step 4 Refinement: Detailed Results Table', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('results/step4_results_table.png', dpi=150, bbox_inches='tight')
    print("Saved results table to results/step4_results_table.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING STEP 4 VISUALIZATIONS")
    print("=" * 70)
    visualize_comparison()
    print("\n" + "=" * 70)
    print("VISUALIZATIONS COMPLETE!")
    print("=" * 70)
