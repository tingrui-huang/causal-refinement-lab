import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================
# 1. 核心模型: Neural Logic Module
# ==========================================
class NeuralLogicReasoning(nn.Module):
    def __init__(self, n_states, rule_indices, init_weights):
        """
        n_states: 离散状态的总数 (如 LVEDVOLUME_Low, HR_High...)
        rule_indices: (2, n_rules) 的张量，表示 [Body_Idx, Head_Idx]
        init_weights: LLM 给出的初始置信度
        """
        super().__init__()
        self.n_states = n_states

        # 我们只学习 LLM 提供的规则的权重 (Sparse Logic Network)
        # 使用 Parameter 让他可训练
        self.rule_weights = nn.Parameter(torch.FloatTensor(init_weights))
        self.rule_indices = rule_indices

        # 稀疏矩阵用于前向传播: Body -> Head
        # 形状: (n_states, n_states)
        self.sparse_mask = torch.sparse_coo_tensor(
            rule_indices,
            torch.ones(len(init_weights)),
            (n_states, n_states)
        )

    def forward(self, x):
        """
        x: (Batch, n_states) - 病人的状态向量 (0或1)
        """
        # 构造当前的权重矩阵 (利用 rule_weights 更新稀疏矩阵的值)
        # Adjacency: Body (Rows) -> Head (Cols)
        adj = torch.sparse_coo_tensor(
            self.rule_indices,
            self.rule_weights,  # 可学习的权重
            (self.n_states, self.n_states)
        ).to(x.device)

        # 逻辑推理: Head = Body * Weight
        # 相当于矩阵乘法: (Batch, States) @ (States, States)
        # 这里的加法相当于逻辑 OR，乘法相当于逻辑 AND (带权重)
        out = torch.sparse.mm(adj.t(), x.t()).t()

        # Sigmoid 激活，因为状态是概率 (0-1)
        return torch.sigmoid(out)


# ==========================================
# 2. 数据处理与加载
# ==========================================
def load_data_and_rules():
    print("-" * 50)
    print("LOADING KNOWLEDGE GRAPH & RULES")

    # 1. Load Metadata to build State Map
    with open('knowledge_graph_metadata.json', 'r') as f:
        meta = json.load(f)

    # 建立状态索引: "LVEDVOLUME_Low" -> 0, "LVEDVOLUME_High" -> 1
    state_to_idx = {}
    idx_to_state = {}
    idx_counter = 0

    for var, mapping in meta['value_map'].items():
        # 这里需要兼容 step 1 生成的后缀格式
        # 假设 Step 1 用的映射是: 0->Low, 1->High, 0->False 等
        # 我们重新遍历一遍 value_map 确保顺序一致
        sorted_vals = sorted(mapping.keys(), key=lambda x: int(x))
        for val_code in sorted_vals:
            val_name = mapping[val_code]  # e.g., 'Low', 'True'
            state_name = f"{var}_{val_name}"
            state_to_idx[state_name] = idx_counter
            idx_to_state[idx_counter] = state_name
            idx_counter += 1

    print(f"Total Discrete States (Nodes): {len(state_to_idx)}")

    # 2. Load Patient Data (Facts)
    # 我们直接读 CSV 或者 triples JSON 都可以，读 triples 更符合 KG 逻辑
    with open('knowledge_graph_triples.json', 'r') as f:
        triples = json.load(f)

    # 构建数据矩阵 (Patients x States)
    # 找出有多少病人
    patient_ids = set()
    for t in triples:
        patient_ids.add(t[0])

    n_patients = len(patient_ids)
    patient_to_row = {pid: i for i, pid in enumerate(sorted(list(patient_ids)))}

    data_matrix = torch.zeros(n_patients, len(state_to_idx))

    print("Building Data Matrix...")
    for sub, pred, obj in triples:
        # triple: ["Patient_0", "HasState", "LVEDVOLUME_Low"]
        if obj in state_to_idx:
            row = patient_to_row[sub]
            col = state_to_idx[obj]
            data_matrix[row, col] = 1.0

    print(f"Data Matrix Shape: {data_matrix.shape}")

    # 3. Load LLM Rules
    print("\nParsing LLM Rules...")
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
            else:
                # 可能是 LLM 生成的变量名不匹配，跳过
                pass
        except:
            continue

    print(f"Loaded {len(valid_rules)} valid rules out of {len(rule_lines)}")

    # Build Tensor indices for Sparse Matrix
    # Indices shape: (2, n_rules) -> [Body_Indices, Head_Indices]
    # Because we do MatMul: Body * W = Head
    rule_indices = torch.tensor([bodies, heads], dtype=torch.long)

    return data_matrix, rule_indices, initial_confidences, state_to_idx, idx_to_state


# ==========================================
# 3. 训练循环
# ==========================================
def train_logic_engine():
    # Load Data
    data, rule_indices, init_weights, state_map, idx_map = load_data_and_rules()

    # Hyperparameters
    n_epochs = 500
    learning_rate = 0.05

    # Init Model
    model = NeuralLogicReasoning(
        n_states=len(state_map),
        rule_indices=rule_indices,
        init_weights=init_weights
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Binary Cross Entropy

    print("\n" + "=" * 50)
    print("STARTING NEURAL LOGIC TRAINING")
    print("=" * 50)
    print("Task: Link Prediction (Reasoning)")
    print("Logic: Adjusting rule weights to fit patient data distributions")

    for epoch in range(n_epochs):
        # Forward pass
        # 我们试图用数据推导自身，看规则是否自洽 (Self-Consistency)
        # 或者可以做 Masked Training (更高级)，这里用简单的自回归拟合
        predictions = model(data)

        # 只计算有规则涉及到的那些状态的 Loss，避免稀疏矩阵全0梯度的干扰
        # 但为了简单，我们计算全图 Loss，因为没连接的地方导数也是0
        loss = criterion(predictions, data)

        # L1 Regularization to prune weak rules
        l1_loss = 0.001 * torch.sum(torch.abs(model.rule_weights))
        total_loss = loss + l1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Clamp weights to be positive (Logic rules shouldn't be negative here)
        with torch.no_grad():
            model.rule_weights.data.clamp_(0.0, 5.0)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {total_loss.item():.4f}")

    # ==========================================
    # 4. 结果分析与导出
    # ==========================================
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - ANALYZING RULES")
    print("=" * 50)

    final_weights = model.rule_weights.detach().numpy()

    # Combine results
    results = []
    for i in range(len(final_weights)):
        body_idx = rule_indices[0, i].item()
        head_idx = rule_indices[1, i].item()
        w_old = init_weights[i]
        w_new = final_weights[i]

        results.append({
            'rule': f"{idx_map[head_idx]} :- {idx_map[body_idx]}",
            'w_init': w_old,
            'w_learned': w_new,
            'delta': w_new - w_old
        })

    # Sort by final learned weight (Importance)
    results.sort(key=lambda x: x['w_learned'], reverse=True)

    print(f"{'RULE':<50} | {'INIT':<6} | {'FINAL':<6} | {'STATUS'}")
    print("-" * 80)

    valid_count = 0
    for r in results:
        # Threshold to keep rule
        if r['w_learned'] > 0.1:
            status = "CONFIRMED" if r['w_learned'] > 0.5 else "WEAK"
            if r['w_learned'] > r['w_init'] + 0.1: status = "STRENGTHENED"
            if r['w_learned'] < r['w_init'] - 0.2: status = "WEAKENED"

            print(f"{r['rule']:<50} | {r['w_init']:.2f}   | {r['w_learned']:.2f}   | {status}")
            valid_count += 1

    print("-" * 80)
    print(f"Total Rules Kept: {valid_count} / {len(results)}")

    # Save final Causal Graph
    output_path = Path('results')
    output_path.mkdir(exist_ok=True)

    with open(output_path / 'final_causal_rules.txt', 'w') as f:
        for r in results:
            if r['w_learned'] > 0.1:
                f.write(f"{r['w_learned']:.4f} :: {r['rule']}\n")

    print(f"\nFinal Causal Rules saved to {output_path / 'final_causal_rules.txt'}")


if __name__ == "__main__":
    train_logic_engine()