# Critical Bug Fix: FCI Skeleton Constraint

## 问题发现 (Problem Identified)

用户发现了一个严重的架构问题：

**原始错误**:
- `build_block_structure()` 为所有 37×36 = **1332 个变量对**都创建了 block
- 但 FCI skeleton 只允许 **~45 条边**
- 这导致模型在 1287 个根本不应该存在的变量对上浪费计算

**用户的关键观察**:
> "你的报告里还写了报告数据里有1330对变量都在'互殴'，请问有把FCI的skeleton弄进来吗？FCI给的只有四十多条边，哪来那么多变量互相博弈方向？"

---

## 修正方案 (Solution)

### 1. 数据源分离

**用户要求**:
- **硬 mask (skeleton)**: 使用纯 FCI 结果 (`edges_FCI_20251207_230824.csv`)
- **软 mask (direction prior)**: 使用 FCI+LLM 混合结果 (`edges_Hybrid_FCI_LLM_20251207_230956.csv`)

### 2. 代码修改

#### A. `prior_builder.py` - `build_block_structure()`

**修改前**:
```python
def build_block_structure(self) -> List[Dict]:
    blocks = []
    for var_a in var_names:
        for var_b in var_names:
            if var_a == var_b:
                continue
            # 无条件创建所有 block！
            blocks.append({...})
    return blocks  # 返回 1332 个 blocks
```

**修改后**:
```python
def build_block_structure(self, skeleton_mask: torch.Tensor) -> List[Dict]:
    blocks = []
    for var_a in var_names:
        for var_b in var_names:
            if var_a == var_b:
                continue
            
            # 检查 FCI skeleton 是否允许这个 block
            block_mask = skeleton_mask[row_indices][:, col_indices]
            
            # 只有 FCI 允许的才创建 block
            if block_mask.sum().item() > 0:
                blocks.append({...})
    
    return blocks  # 返回 ~47 个 blocks (减少 96.5%!)
```

#### B. `prior_builder.py` - `build_direction_prior_from_llm()`

**修改前**:
- 读取旧的 `llm_prior_rules` 文本文件
- 格式: `0.92 :: CVP_Low :- HYPOVOLEMIA_True`

**修改后**:
- 读取新的 CSV 文件 (`edges_Hybrid_FCI_LLM_*.csv`)
- 根据 `edge_type` 列设置不同的 confidence:
  - `directed` / `llm_resolved`: 0.7 (high confidence)
  - `undirected` / `partial`: 0.3 (low confidence)

#### C. `prior_builder.py` - `get_all_priors()`

**修改前**:
```python
def get_all_priors(self, fci_csv_path: str, llm_rules_path: str):
    skeleton_mask = self.build_skeleton_mask_from_fci(fci_csv_path)
    direction_prior = self.build_direction_prior_from_llm(llm_rules_path)
    blocks = self.build_block_structure()  # 不依赖 skeleton!
    ...
```

**修改后**:
```python
def get_all_priors(self, fci_skeleton_path: str, llm_direction_path: str):
    # 1. 从纯 FCI 构建硬 skeleton
    skeleton_mask = self.build_skeleton_mask_from_fci(fci_skeleton_path)
    
    # 2. 从 FCI+LLM 混合构建软 direction prior
    direction_prior = self.build_direction_prior_from_llm(llm_direction_path)
    
    # 3. 只为 FCI 允许的边创建 blocks
    blocks = self.build_block_structure(skeleton_mask)
    ...
```

#### D. `train_complete.py` - Configuration

**修改前**:
```python
config = {
    'fci_edges_path': 'data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
    'llm_rules_path': 'llm_prior_rules',
    'ground_truth_path': '../alarm.bif',
    ...
}
```

**修改后**:
```python
config = {
    'fci_skeleton_path': 'data/edges_FCI_20251207_230824.csv',  # 纯 FCI
    'llm_direction_path': 'data/edges_Hybrid_FCI_LLM_20251207_230956.csv',  # FCI+LLM
    'ground_truth_path': 'data/alarm.bif',  # 修正路径
    ...
}
```

---

## 结果对比 (Results Comparison)

### 修正前 (Before Fix)

| Metric | Value | Issue |
|--------|-------|-------|
| Total Blocks | 1332 | ❌ 包含大量不应存在的变量对 |
| FCI-Allowed Blocks | 4 | ❌ 实际只有 4 个 block 有连接 |
| Bidirectional Pairs | 0 / 666 | ❌ 分母错误 (应该是 ~20) |
| **Edge Precision** | 100.0% | ⚠️ 过于保守 |
| **Edge Recall** | 26.1% | ❌ 太低 |
| **Orientation Accuracy** | 91.7% | ⚠️ 基数太小 (12 edges) |
| Learned Edges | 12 | ❌ 远低于 ground truth (46) |

### 修正后 (After Fix)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Total Blocks | 47 | ✅ 减少 96.5% |
| FCI-Allowed Blocks | 47 | ✅ 全部来自 FCI |
| Bidirectional Pairs | 0 / 4 | ✅ 分母正确 |
| **Edge Precision** | 97.6% | ✅ 保持高精度 |
| **Edge Recall** | 87.0% | ✅ 提升 60.9% |
| **Edge F1 Score** | 92.0% | ✅ 从 41.4% 提升到 92.0% |
| **Orientation Accuracy** | 82.5% | ✅ 基数合理 (41 edges) |
| Learned Edges | 41 | ✅ 接近 ground truth (46) |

### 方向学习 (Direction Learning)

**修正前**:
- Bidirectional Ratio: 0.0% → 0.0% (无变化)
- 原因: 大部分 block 根本没有连接

**修正后**:
- Bidirectional Ratio: 50.0% → 0.0% ✅
- 变化: **-50.0%** [GOOD]
- 说明: Cycle Consistency Loss 成功解决了方向冲突

### 稀疏性 (Sparsity)

**修正前**:
- Block Sparsity: 99.7% (4/1332 active)
- 问题: 分母包含大量不应存在的 block

**修正后**:
- Block Sparsity: 12.8% (41/47 active)
- 说明: 在 FCI 允许的 47 个 block 中，41 个被激活，合理

---

## 关键洞察 (Key Insights)

### 1. FCI Skeleton 的重要性

FCI skeleton 不仅仅是一个"软约束"，而是**硬约束**：
- 如果 FCI 说 A 和 B 没有连接，那么模型就**不应该**考虑 A→B 或 B→A
- 原始实现违反了这个原则，导致模型在 1287 个不可能的边上浪费计算

### 2. 方向学习的前提

方向学习（Cycle Consistency Loss）只有在**存在方向冲突**时才有意义：
- 修正前: 大部分变量对根本没有连接 → 无冲突可解决
- 修正后: FCI 允许的变量对中，有 50% 存在双向连接 → Cycle Loss 成功将其降至 0%

### 3. 评估指标的可靠性

修正前的高 Orientation Accuracy (91.7%) 是**误导性的**：
- 基数太小 (12 edges)
- 大部分 ground truth edges 被遗漏

修正后的 Orientation Accuracy (82.5%) 更**可靠**：
- 基数合理 (41 edges)
- 覆盖了 87% 的 ground truth edges

---

## 性能提升总结 (Performance Improvement Summary)

### 计算效率

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Blocks Computed | 1332 | 47 | **96.5% reduction** |
| Cycle Loss Pairs | 666 | 4 | **99.4% reduction** |

### 模型质量

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Edge Recall | 26.1% | 87.0% | **+60.9%** |
| Edge F1 Score | 41.4% | 92.0% | **+50.6%** |
| Learned Edges | 12 | 41 | **+242%** |
| Direction Learning | 0% → 0% | 50% → 0% | **Now working!** |

---

## 下一步优化 (Next Steps)

### 1. 提升 Orientation Accuracy (82.5% → 90%+)

**当前问题**: 7 条边方向错误

**可能方案**:
- 增加 `lambda_cycle` 从 0.001 到 0.01 (更强的方向惩罚)
- 检查这 7 条边的条件概率不对称性
- 分析 LLM prior 对这些边的 confidence

### 2. 提升 Recall (87.0% → 95%+)

**当前问题**: 6 条 ground truth edges 未发现

**可能方案**:
- 减少 `lambda_group` 从 0.01 到 0.005
- 检查这 6 条边是否在数据中很弱
- 分析 FCI 是否遗漏了这些边

### 3. 降低 False Positives (1 edge)

**当前问题**: 1 条学到的边不在 ground truth 中

**可能方案**:
- 检查这条边是否是 FCI 的误报
- 分析数据中的条件概率支持
- 可能是 ground truth 的遗漏？

---

## 结论 (Conclusion)

这次修正解决了一个**架构级别的根本性问题**：

1. ✅ **正确实现了 FCI skeleton 约束**: 从 1332 个 blocks 减少到 47 个
2. ✅ **方向学习开始工作**: Bidirectional ratio 从 50% 降至 0%
3. ✅ **大幅提升 recall**: 从 26.1% 提升到 87.0%
4. ✅ **保持高 precision**: 97.6%
5. ✅ **F1 score 翻倍**: 从 41.4% 提升到 92.0%

**用户的观察是完全正确的** - 原始实现确实没有正确使用 FCI skeleton，导致模型在大量不应存在的变量对上"互殴"。

现在模型的性能已经达到了**生产级别** (F1 = 92.0%, Orientation = 82.5%)，可以进行下一阶段的超参数优化。

---

**Date**: 2025-12-16
**Bug Severity**: Critical (Architecture-level)
**Fix Impact**: Transformative
**Status**: ✅ RESOLVED



