import torch
import json
from pathlib import Path
from modules.data_loader import CausalDataLoader
from modules.prior_builder import PriorBuilder


# === FIXED VERSION: 检查所有 43 对，而不是只检查 4 对 ===
def compute_bidirectional_ratio_real(adjacency, block_structure, threshold=0.3):
    """
    计算双向边比例
    
    FIXED: 现在检查所有 FCI 变量对（43 对），而不是只检查
    两个方向都在 block_structure 中的对（4 对）
    """
    bidirectional_count = 0
    unidirectional_count = 0
    no_direction_count = 0

    # 建立块查找表
    block_lookup = {}
    for block in block_structure:
        var_a, var_b = block['var_pair']
        block_lookup[(var_a, var_b)] = block

    # 获取所有唯一变量对
    all_pairs = set()
    for block in block_structure:
        var_a, var_b = block['var_pair']
        pair_key = tuple(sorted([var_a, var_b]))
        all_pairs.add(pair_key)

    # 检查每个唯一对
    for pair_key in all_pairs:
        var_a, var_b = pair_key  # 已经排序过

        # 尝试获取两个方向
        forward_block = block_lookup.get((var_a, var_b))
        reverse_block = block_lookup.get((var_b, var_a))

        # 计算两个方向的强度
        # 如果某个方向不在 block_structure 中，手动从 adjacency 计算
        if forward_block is not None:
            forward_weights = adjacency[forward_block['row_indices']][:, forward_block['col_indices']]
            forward_strength = forward_weights.mean().item()
        else:
            # 手动计算：从 reverse_block 的索引推导
            if reverse_block is not None:
                # reverse_block 是 (var_b, var_a)，我们要 (var_a, var_b)
                forward_weights = adjacency[reverse_block['col_indices']][:, reverse_block['row_indices']]
                forward_strength = forward_weights.mean().item()
            else:
                forward_strength = 0.0

        if reverse_block is not None:
            backward_weights = adjacency[reverse_block['row_indices']][:, reverse_block['col_indices']]
            backward_strength = backward_weights.mean().item()
        else:
            # 手动计算：从 forward_block 的索引推导
            if forward_block is not None:
                # forward_block 是 (var_a, var_b)，我们要 (var_b, var_a)
                backward_weights = adjacency[forward_block['col_indices']][:, forward_block['row_indices']]
                backward_strength = backward_weights.mean().item()
            else:
                backward_strength = 0.0

        # 分类
        forward_strong = forward_strength > threshold
        backward_strong = backward_strength > threshold

        if forward_strong and backward_strong:
            bidirectional_count += 1
        elif forward_strong or backward_strong:
            unidirectional_count += 1
        else:
            no_direction_count += 1

    total_pairs = bidirectional_count + unidirectional_count + no_direction_count

    return bidirectional_count, total_pairs


def main():
    print("=" * 50)
    print("正在读取真实训练结果...")
    print("=" * 50)

    # 1. 设置路径 (根据你的 train_complete.py 配置)
    # 请确保这个路径下有 complete_adjacency.pt
    result_path = Path('../results/no_llm/complete_adjacency.pt')

    if not result_path.exists():
        print(f"Error: File not found {result_path}")
        print("Please make sure you have run train_complete.py first")
        return

    # 2. 加载邻接矩阵
    adjacency = torch.load(result_path, map_location='cpu')
    print("Success: Loaded adjacency matrix")

    # 3. 重新构建 Block 结构 (需要加载数据来获取结构)
    print("正在重建变量结构...")
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='output/knowledge_graph_metadata.json'
    )
    var_structure = loader.get_variable_structure()

    # 获取 priors (主要是为了拿到 blocks)
    # 这里不需要 LLM 路径，只需要 FCI 骨架路径来确定哪些边是存在的
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_skeleton_path='data/edges_FCI_20251207_230824.csv',  # 确保这个文件存在
        use_llm_prior=False
    )
    blocks = priors['blocks']

    # 4. 计算真实的 Bidirectional Ratio
    # 注意：阈值要和你 evaluator 里用的一样，通常是 0.3 或 0.1
    # 我们可以把两个都算一下

    print("\n" + "=" * 50)
    print("真实统计结果 (NO-LLM Baseline)")
    print("=" * 50)

    for th in [0.1, 0.3]:
        count, total = compute_bidirectional_ratio_real(adjacency, blocks, threshold=th)
        ratio = count / total * 100 if total > 0 else 0
        print(f"阈值 (Threshold) = {th}:")
        print(f"  - 双向边数量: {count} / {total}")
        print(f"  - 真实双向率: {ratio:.1f}%")
        print("-" * 30)


if __name__ == "__main__":
    main()