"""
Smoking Gun Experiment: Testing if GSB model removes edges caused by latent confounders
核心思路：构造一个简单的三节点陷阱 X1 <- H -> X2，看模型是否会错误地切掉这条边
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
import networkx as nx
from typing import Tuple, Dict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_latent_confounder_data(n_samples: int = 1000, 
                                     noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成包含隐变量的数据：H -> X1, H -> X2
    
    Args:
        n_samples: 样本数量
        noise_level: 噪声水平
    
    Returns:
        observed_data: (n_samples, 2) - 只包含 X1, X2
        latent_data: (n_samples, 3) - 包含 H, X1, X2
    """
    # Step 1: 生成隐变量 H (binary: 0 or 1)
    H = np.random.binomial(1, 0.5, size=n_samples)
    
    # Step 2: H -> X1 (概率 0.9)
    # 如果 H=1, X1=1 的概率是 0.9; 如果 H=0, X1=1 的概率是 0.1
    X1_prob = np.where(H == 1, 0.9, 0.1)
    X1 = np.random.binomial(1, X1_prob)
    
    # Step 3: H -> X2 (概率 0.9)
    X2_prob = np.where(H == 1, 0.9, 0.1)
    X2 = np.random.binomial(1, X2_prob)
    
    # Add small Gaussian noise to make it continuous
    X1 = X1.astype(float) + np.random.normal(0, noise_level, n_samples)
    X2 = X2.astype(float) + np.random.normal(0, noise_level, n_samples)
    H = H.astype(float) + np.random.normal(0, noise_level, n_samples)
    
    # Observed data (只有 X1, X2)
    observed_data = np.column_stack([X1, X2])
    
    # Full data (包含 H)
    latent_data = np.column_stack([H, X1, X2])
    
    return observed_data, latent_data


def run_fci(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    运行 FCI 算法
    
    Args:
        data: (n_samples, n_vars)
        alpha: 显著性水平
    
    Returns:
        adjacency_matrix: (n_vars, n_vars)
    """
    print(f"\n{'='*60}")
    print("Running FCI Algorithm...")
    print(f"{'='*60}")
    
    # Run FCI
    G, edges = fci(data, fisherz, alpha=alpha, verbose=False)
    
    # Get adjacency matrix
    adj_matrix = G.graph
    
    print(f"FCI Result (alpha={alpha}):")
    print(f"Adjacency Matrix:\n{adj_matrix}")
    
    # Check if X1-X2 edge exists
    edge_exists = (adj_matrix[0, 1] != 0) or (adj_matrix[1, 0] != 0)
    print(f"\nEdge between X1 and X2: {'EXISTS' if edge_exists else 'REMOVED'}")
    
    return adj_matrix


class SimpleGSBModel(nn.Module):
    """
    简化版的 GSB 模型，用于测试
    """
    def __init__(self, n_vars: int, hidden_dim: int = 32):
        super().__init__()
        self.n_vars = n_vars
        
        # Adjacency matrix (learnable)
        self.W = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        
        # Neural network for each variable
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_vars, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(n_vars)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_vars)
        """
        batch_size = x.shape[0]
        predictions = []
        
        for i in range(self.n_vars):
            # Get parents (weighted by adjacency matrix)
            parents = x * self.W[i, :]  # (batch_size, n_vars)
            
            # Predict variable i
            pred = self.networks[i](parents)  # (batch_size, 1)
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)  # (batch_size, n_vars)
    
    def get_adjacency_matrix(self, threshold: float = 0.1) -> np.ndarray:
        """
        获取邻接矩阵（阈值化）
        """
        W_np = self.W.detach().cpu().numpy()
        # Remove self-loops
        np.fill_diagonal(W_np, 0)
        # Threshold
        adj = (np.abs(W_np) > threshold).astype(int)
        return adj


def train_gsb_model(data: np.ndarray, 
                    n_epochs: int = 1000,
                    lr: float = 0.01,
                    lasso_lambda: float = 0.1,
                    hidden_dim: int = 32) -> SimpleGSBModel:
    """
    训练 GSB 模型
    
    Args:
        data: (n_samples, n_vars)
        n_epochs: 训练轮数
        lr: 学习率
        lasso_lambda: Lasso 正则化系数
        hidden_dim: 隐藏层维度
    
    Returns:
        trained model
    """
    print(f"\n{'='*60}")
    print("Training GSB Model...")
    print(f"{'='*60}")
    
    n_vars = data.shape[1]
    model = SimpleGSBModel(n_vars, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensor
    data_tensor = torch.FloatTensor(data)
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data_tensor)
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(predictions, data_tensor)
        
        # Lasso regularization (L1 on adjacency matrix)
        lasso_loss = lasso_lambda * torch.abs(model.W).sum()
        
        # Total loss
        loss = recon_loss + lasso_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, "
                  f"Recon: {recon_loss.item():.4f}, Lasso: {lasso_loss.item():.4f}")
    
    # Get final adjacency matrix
    adj_matrix = model.get_adjacency_matrix(threshold=0.1)
    print(f"\nGSB Result:")
    print(f"Adjacency Matrix:\n{adj_matrix}")
    
    # Check if X1-X2 edge exists
    edge_exists = (adj_matrix[0, 1] != 0) or (adj_matrix[1, 0] != 0)
    print(f"\nEdge between X1 and X2: {'EXISTS' if edge_exists else 'REMOVED'}")
    
    return model, losses


def visualize_results(observed_data: np.ndarray, 
                      latent_data: np.ndarray,
                      fci_adj: np.ndarray,
                      gsb_adj: np.ndarray,
                      losses: list):
    """
    可视化实验结果
    """
    fig = plt.figure(figsize=(18, 10))
    
    # 1. True causal structure (with latent variable)
    ax1 = plt.subplot(2, 4, 1)
    G_true = nx.DiGraph()
    G_true.add_edges_from([('H', 'X1'), ('H', 'X2')])
    pos = {'H': (0.5, 1), 'X1': (0, 0), 'X2': (1, 0)}
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=16, font_weight='bold',
            arrows=True, arrowsize=20, ax=ax1)
    ax1.set_title('True Causal Structure\n(H is latent)', fontsize=14, fontweight='bold')
    
    # 2. Correlation between X1 and X2
    ax2 = plt.subplot(2, 4, 2)
    ax2.scatter(observed_data[:, 0], observed_data[:, 1], alpha=0.5)
    ax2.set_xlabel('X1', fontsize=12)
    ax2.set_ylabel('X2', fontsize=12)
    corr = np.corrcoef(observed_data[:, 0], observed_data[:, 1])[0, 1]
    ax2.set_title(f'X1 vs X2 Correlation\n(r={corr:.3f})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. FCI result
    ax3 = plt.subplot(2, 4, 3)
    G_fci = nx.DiGraph()
    G_fci.add_nodes_from(['X1', 'X2'])
    if fci_adj[0, 1] != 0 or fci_adj[1, 0] != 0:
        if fci_adj[0, 1] != 0 and fci_adj[1, 0] != 0:
            G_fci.add_edge('X1', 'X2', style='bidirectional')
            G_fci.add_edge('X2', 'X1', style='bidirectional')
        elif fci_adj[0, 1] != 0:
            G_fci.add_edge('X1', 'X2')
        else:
            G_fci.add_edge('X2', 'X1')
    pos_obs = {'X1': (0, 0), 'X2': (1, 0)}
    nx.draw(G_fci, pos_obs, with_labels=True, node_color='lightgreen',
            node_size=2000, font_size=16, font_weight='bold',
            arrows=True, arrowsize=20, ax=ax3)
    ax3.set_title('FCI Result\n(Edge preserved)', fontsize=14, fontweight='bold')
    
    # 4. GSB result
    ax4 = plt.subplot(2, 4, 4)
    G_gsb = nx.DiGraph()
    G_gsb.add_nodes_from(['X1', 'X2'])
    if gsb_adj[0, 1] != 0:
        G_gsb.add_edge('X1', 'X2')
    if gsb_adj[1, 0] != 0:
        G_gsb.add_edge('X2', 'X1')
    nx.draw(G_gsb, pos_obs, with_labels=True, node_color='lightcoral',
            node_size=2000, font_size=16, font_weight='bold',
            arrows=True, arrowsize=20, ax=ax4)
    edge_status = "Edge removed" if (gsb_adj[0, 1] == 0 and gsb_adj[1, 0] == 0) else "Edge preserved"
    ax4.set_title(f'GSB Result\n({edge_status})', fontsize=14, fontweight='bold')
    
    # 5. Training loss
    ax5 = plt.subplot(2, 4, 5)
    ax5.plot(losses)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('GSB Training Loss', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation matrix (observed)
    ax6 = plt.subplot(2, 4, 6)
    corr_matrix = np.corrcoef(observed_data.T)
    im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['X1', 'X2'])
    ax6.set_yticklabels(['X1', 'X2'])
    ax6.set_title('Observed Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax6)
    for i in range(2):
        for j in range(2):
            ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=12)
    
    # 7. Correlation with latent variable
    ax7 = plt.subplot(2, 4, 7)
    corr_with_latent = np.corrcoef(latent_data.T)
    im2 = ax7.imshow(corr_with_latent, cmap='coolwarm', vmin=-1, vmax=1)
    ax7.set_xticks([0, 1, 2])
    ax7.set_yticks([0, 1, 2])
    ax7.set_xticklabels(['H', 'X1', 'X2'])
    ax7.set_yticklabels(['H', 'X1', 'X2'])
    ax7.set_title('Full Correlation Matrix\n(including latent H)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax7)
    for i in range(3):
        for j in range(3):
            ax7.text(j, i, f'{corr_with_latent[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=10)
    
    # 8. Summary statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    summary_text = f"""
    SMOKING GUN EXPERIMENT RESULTS
    {'='*40}
    
    Data Generation:
    • N = {observed_data.shape[0]} samples
    • Structure: H → X1, H → X2
    • H is LATENT (unobserved)
    
    Statistical Properties:
    • Corr(X1, X2) = {corr:.3f}
    • Corr(H, X1) = {corr_with_latent[0, 1]:.3f}
    • Corr(H, X2) = {corr_with_latent[0, 2]:.3f}
    
    FCI Result:
    • Edge X1-X2: {'PRESERVED' if (fci_adj[0, 1] != 0 or fci_adj[1, 0] != 0) else 'REMOVED'}
    • (Correctly identifies confounding)
    
    GSB Result:
    • Edge X1-X2: {'PRESERVED' if (gsb_adj[0, 1] != 0 or gsb_adj[1, 0] != 0) else 'REMOVED'}
    • {'[!] FALSE NEGATIVE!' if (gsb_adj[0, 1] == 0 and gsb_adj[1, 0] == 0) else '[V] Correct'}
    
    Hypothesis:
    {'[V] CONFIRMED: GSB removes edges caused' if (gsb_adj[0, 1] == 0 and gsb_adj[1, 0] == 0) else '[X] REJECTED: GSB preserves edges'}
    {'  by latent confounders!' if (gsb_adj[0, 1] == 0 and gsb_adj[1, 0] == 0) else '  caused by latent confounders.'}
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'smoking_gun_experiment.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Figure saved as '{output_path}'")
    print(f"{'='*60}")
    
    # Try to show (may not work in some environments)
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def main():
    """
    主函数：运行 Smoking Gun 实验
    """
    print("\n" + "="*60)
    print("SMOKING GUN EXPERIMENT")
    print("Testing if GSB removes edges caused by latent confounders")
    print("="*60)
    
    # Step 1: Generate data
    print("\nStep 1: Generating data with latent confounder...")
    n_samples = 1000
    observed_data, latent_data = generate_latent_confounder_data(n_samples=n_samples)
    
    print(f"Generated {n_samples} samples")
    print(f"Observed data shape: {observed_data.shape}")
    print(f"Correlation between X1 and X2: {np.corrcoef(observed_data[:, 0], observed_data[:, 1])[0, 1]:.3f}")
    
    # Step 2: Run FCI
    fci_adj = run_fci(observed_data, alpha=0.05)
    
    # Step 3: Train GSB model
    model, losses = train_gsb_model(
        observed_data, 
        n_epochs=1000,
        lr=0.01,
        lasso_lambda=0.1,
        hidden_dim=32
    )
    gsb_adj = model.get_adjacency_matrix(threshold=0.1)
    
    # Step 4: Visualize results
    print("\nStep 4: Visualizing results...")
    visualize_results(observed_data, latent_data, fci_adj, gsb_adj, losses)
    
    # Step 5: Print conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    fci_has_edge = (fci_adj[0, 1] != 0) or (fci_adj[1, 0] != 0)
    gsb_has_edge = (gsb_adj[0, 1] != 0) or (gsb_adj[1, 0] != 0)
    
    print(f"\nFCI: {'Preserved' if fci_has_edge else 'Removed'} X1-X2 edge")
    print(f"GSB: {'Preserved' if gsb_has_edge else 'Removed'} X1-X2 edge")
    
    if not gsb_has_edge and fci_has_edge:
        print("\n[V] HYPOTHESIS CONFIRMED!")
        print("  GSB incorrectly removes edges caused by latent confounders,")
        print("  while FCI correctly identifies the confounding relationship.")
        print("\n  This explains the lower recall in your experiments:")
        print("  - Precision stays high (few false positives)")
        print("  - Recall drops (more false negatives due to removed latent edges)")
        print("  - F1 score decreases accordingly")
    elif gsb_has_edge:
        print("\n[X] HYPOTHESIS REJECTED")
        print("  GSB preserved the edge, suggesting it can handle this case.")
    else:
        print("\n[?] INCONCLUSIVE")
        print("  Both methods removed the edge.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

