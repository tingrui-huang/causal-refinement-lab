import pandas as pd
import numpy as np


def check_state_asymmetry(df, var_A, var_B):
    """
    检查具体的 High -> High 关系的不对称性
    """
    # 假设 2 代表 High (如果不确定，可以打印 unique 看看)
    state_val = 2

    # 1. P(B=High | A=High)
    mask_A = df[var_A] == state_val
    mask_B = df[var_B] == state_val

    if mask_A.sum() == 0 or mask_B.sum() == 0:
        return 0, 0

    p_B_given_A = (mask_A & mask_B).sum() / mask_A.sum()

    # 2. P(A=High | B=High)
    p_A_given_B = (mask_A & mask_B).sum() / mask_B.sum()

    return p_B_given_A, p_A_given_B


# 加载数据
df = pd.read_csv('alarm_data.csv')

# 测试一对已知的强因果边
# LVEDVOLUME -> PCWP (容积 -> 压力)
var_a, var_b = 'LVEDVOLUME', 'PCWP'

p_fw, p_bw = check_state_asymmetry(df, var_a, var_b)

print(f"Checking Asymmetry for {var_a} <-> {var_b} (High State)")
print(f"P({var_b}=High | {var_a}=High) = {p_fw:.4f}")
print(f"P({var_a}=High | {var_b}=High) = {p_bw:.4f}")
print(f"Difference: {abs(p_fw - p_bw):.4f}")

if abs(p_fw - p_bw) > 0.1:
    print("✅ 结论：数据存在显著的不对称性，Cycle Loss 有效！")
else:
    print("⚠️ 结论：数据确实对称，需依赖 LLM Prior。")