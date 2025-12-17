import pandas as pd
import numpy as np


def check_state_asymmetry(df, var_A, var_B):
    """
    检查具体的 High -> High 关系的不对称性
    数据是 one-hot 编码的，所以直接使用 var_A_High 和 var_B_High 列
    """
    # #region agent log
    import json
    from pathlib import Path
    log_path = Path(r'd:\Users\trhua\Research\causal-refinement-lab\.cursor\debug.log')
    col_A = f"{var_A}_High"
    col_B = f"{var_B}_High"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'location':'analysis_asymmetry.py:5','message':'Function entry','data':{'var_A':var_A,'var_B':var_B,'col_A':col_A,'col_B':col_B,'col_A_exists':col_A in df.columns,'col_B_exists':col_B in df.columns},'timestamp':pd.Timestamp.now().timestamp()*1000,'sessionId':'debug-session','runId':'post-fix','hypothesisId':'H1'}) + '\n')
    # #endregion
    
    # 使用 one-hot 编码的 High 列
    col_A_high = f"{var_A}_High"
    col_B_high = f"{var_B}_High"

    # #region agent log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'location':'analysis_asymmetry.py:18','message':'Before masking','data':{'col_A_high':col_A_high,'col_B_high':col_B_high,'A_sum':int(df[col_A_high].sum()),'B_sum':int(df[col_B_high].sum())},'timestamp':pd.Timestamp.now().timestamp()*1000,'sessionId':'debug-session','runId':'post-fix','hypothesisId':'H1'}) + '\n')
    # #endregion

    # 1. P(B=High | A=High)
    mask_A = df[col_A_high] == 1
    mask_B = df[col_B_high] == 1

    if mask_A.sum() == 0 or mask_B.sum() == 0:
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'location':'analysis_asymmetry.py:31','message':'Empty mask detected','data':{'mask_A_sum':int(mask_A.sum()),'mask_B_sum':int(mask_B.sum())},'timestamp':pd.Timestamp.now().timestamp()*1000,'sessionId':'debug-session','runId':'post-fix','hypothesisId':'H1'}) + '\n')
        # #endregion
        return 0, 0

    p_B_given_A = (mask_A & mask_B).sum() / mask_A.sum()

    # 2. P(A=High | B=High)
    p_A_given_B = (mask_A & mask_B).sum() / mask_B.sum()

    # #region agent log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'location':'analysis_asymmetry.py:42','message':'Function exit','data':{'p_B_given_A':float(p_B_given_A),'p_A_given_B':float(p_A_given_B),'mask_A_sum':int(mask_A.sum()),'mask_B_sum':int(mask_B.sum()),'both_high':int((mask_A & mask_B).sum())},'timestamp':pd.Timestamp.now().timestamp()*1000,'sessionId':'debug-session','runId':'post-fix','hypothesisId':'H1'}) + '\n')
    # #endregion

    return p_B_given_A, p_A_given_B


# 加载数据
df = pd.read_csv('data/alarm_data_10000.csv')

# #region agent log
import json
from pathlib import Path
log_path = Path(r'd:\Users\trhua\Research\causal-refinement-lab\.cursor\debug.log')
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps({'location':'analysis_asymmetry.py:28','message':'CSV loaded','data':{'shape':df.shape,'columns':list(df.columns),'first_5_cols':list(df.columns[:5]),'dtypes':str(df.dtypes.to_dict())},'timestamp':pd.Timestamp.now().timestamp()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1'}) + '\n')
# #endregion

# 测试一对已知的强因果边
# LVEDVOLUME -> PCWP (容积 -> 压力)
var_a, var_b = 'LVEDVOLUME', 'PCWP'

# #region agent log
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps({'location':'analysis_asymmetry.py:32','message':'Checking if columns exist','data':{'var_a':var_a,'var_b':var_b,'var_a_exists':var_a in df.columns,'var_b_exists':var_b in df.columns,'available_cols':list(df.columns)},'timestamp':pd.Timestamp.now().timestamp()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1,H2,H3'}) + '\n')
# #endregion

p_fw, p_bw = check_state_asymmetry(df, var_a, var_b)

print(f"Checking Asymmetry for {var_a} <-> {var_b} (High State)")
print(f"P({var_b}=High | {var_a}=High) = {p_fw:.4f}")
print(f"P({var_a}=High | {var_b}=High) = {p_bw:.4f}")
print(f"Difference: {abs(p_fw - p_bw):.4f}")

if abs(p_fw - p_bw) > 0.1:
    print("结论：数据存在显著的不对称性，Cycle Loss 有效！")
else:
    print("结论：数据确实对称，需依赖 LLM Prior。")