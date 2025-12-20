"""
Fine-grained Asymmetry Analysis

Gemini's Hypothesis:
- FCI/GES works at VARIABLE level (A, B)
- Neural LP works at STATE level (A_High, B_High)
- Data may be symmetric at variable level, but ASYMMETRIC at state level
- Especially: High->High strong, Normal->Normal weak
- Neural LP can capture this fine-grained asymmetry that GES cannot see

This script tests this hypothesis systematically.
"""

# #region agent log
import json
import sys
from pathlib import Path
log_path = Path(r'/.cursor/debug.log')
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:1','message':'Script started','data':{'cwd':str(Path.cwd()),'script_path':str(Path(__file__).absolute())},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1,H2'}) + '\n')
# #endregion

import pandas as pd
import numpy as np


def analyze_variable_level_asymmetry(df, var_a, var_b):
    """
    Variable-level asymmetry (what FCI/GES sees)
    
    Treats all states of A and B equally, computes overall P(B|A) vs P(A|B)
    """
    # Get all columns for var_a and var_b
    cols_a = [col for col in df.columns if col.startswith(f"{var_a}_")]
    cols_b = [col for col in df.columns if col.startswith(f"{var_b}_")]
    
    if not cols_a or not cols_b:
        return None
    
    # For each sample, check if A is active (any state) and B is active (any state)
    a_active = df[cols_a].sum(axis=1) > 0
    b_active = df[cols_b].sum(axis=1) > 0
    
    n_a = a_active.sum()
    n_b = b_active.sum()
    n_ab = (a_active & b_active).sum()
    
    if n_a == 0 or n_b == 0:
        return None
    
    p_b_given_a = n_ab / n_a
    p_a_given_b = n_ab / n_b
    
    return {
        'P(B|A)': p_b_given_a,
        'P(A|B)': p_a_given_b,
        'asymmetry': p_b_given_a - p_a_given_b,
        'n_a': n_a,
        'n_b': n_b,
        'n_ab': n_ab
    }


def analyze_state_level_asymmetry(df, var_a, var_b):
    """
    State-level asymmetry (what Neural LP sees)
    
    Computes P(B_state | A_state) for each state pair
    Returns a matrix of asymmetries
    """
    # Get all columns for var_a and var_b
    cols_a = [col for col in df.columns if col.startswith(f"{var_a}_")]
    cols_b = [col for col in df.columns if col.startswith(f"{var_b}_")]
    
    if not cols_a or not cols_b:
        return None
    
    results = []
    
    for col_a in cols_a:
        state_a = col_a.replace(f"{var_a}_", "")
        
        for col_b in cols_b:
            state_b = col_b.replace(f"{var_b}_", "")
            
            mask_a = df[col_a] == 1
            mask_b = df[col_b] == 1
            
            n_a = mask_a.sum()
            n_b = mask_b.sum()
            n_ab = (mask_a & mask_b).sum()
            
            if n_a == 0 or n_b == 0:
                continue
            
            p_b_given_a = n_ab / n_a
            p_a_given_b = n_ab / n_b
            asymmetry = p_b_given_a - p_a_given_b
            
            results.append({
                'state_a': state_a,
                'state_b': state_b,
                'P(B|A)': p_b_given_a,
                'P(A|B)': p_a_given_b,
                'asymmetry': asymmetry,
                'n_a': n_a,
                'n_b': n_b,
                'n_ab': n_ab
            })
    
    return results


def analyze_edge_asymmetry(df, var_a, var_b):
    """
    Test Gemini's hypothesis:
    1. Variable-level might be symmetric
    2. But state-level (especially High->High) might be asymmetric
    """
    print("=" * 70)
    print(f"Testing: {var_a} <-> {var_b}")
    print("=" * 70)
    
    # 1. Variable-level analysis
    print("\n[1] VARIABLE-LEVEL ASYMMETRY (What FCI/GES sees)")
    print("-" * 70)
    var_result = analyze_variable_level_asymmetry(df, var_a, var_b)
    
    if var_result is None:
        print("ERROR: Variables not found in data")
        return
    
    print(f"P({var_b} | {var_a}) = {var_result['P(B|A)']:.4f}")
    print(f"P({var_a} | {var_b}) = {var_result['P(A|B)']:.4f}")
    print(f"Asymmetry = {var_result['asymmetry']:+.4f}")
    print(f"Sample counts: n({var_a})={var_result['n_a']}, n({var_b})={var_result['n_b']}, n(both)={var_result['n_ab']}")
    
    if abs(var_result['asymmetry']) < 0.05:
        print("=> SYMMETRIC at variable level")
    else:
        print(f"=> ASYMMETRIC at variable level ({'A->B' if var_result['asymmetry'] > 0 else 'B->A'})")
    
    # 2. State-level analysis
    print("\n[2] STATE-LEVEL ASYMMETRY (What Neural LP sees)")
    print("-" * 70)
    state_results = analyze_state_level_asymmetry(df, var_a, var_b)
    
    if not state_results:
        print("ERROR: No valid state pairs found")
        return
    
    # Sort by absolute asymmetry
    state_results_sorted = sorted(state_results, key=lambda x: abs(x['asymmetry']), reverse=True)
    
    print(f"Found {len(state_results)} state pairs\n")
    
    # Show top asymmetric pairs
    print("Top 5 Most Asymmetric State Pairs:")
    for i, result in enumerate(state_results_sorted[:5], 1):
        print(f"\n  {i}. {var_a}_{result['state_a']} -> {var_b}_{result['state_b']}")
        print(f"     P(B|A) = {result['P(B|A)']:.4f}, P(A|B) = {result['P(A|B)']:.4f}")
        print(f"     Asymmetry = {result['asymmetry']:+.4f}")
        print(f"     Samples: n(A)={result['n_a']}, n(B)={result['n_b']}, n(both)={result['n_ab']}")
    
    # Check Gemini's specific hypothesis: High->High vs Normal->Normal
    print("\n" + "-" * 70)
    print("Gemini's Hypothesis: High->High strong, Normal->Normal weak")
    print("-" * 70)
    
    high_high = None
    normal_normal = None
    
    for result in state_results:
        if 'High' in result['state_a'] and 'High' in result['state_b']:
            high_high = result
        if 'Normal' in result['state_a'] and 'Normal' in result['state_b']:
            normal_normal = result
    
    if high_high:
        print(f"\nHigh -> High:")
        print(f"  P({var_b}_High | {var_a}_High) = {high_high['P(B|A)']:.4f}")
        print(f"  P({var_a}_High | {var_b}_High) = {high_high['P(A|B)']:.4f}")
        print(f"  Asymmetry = {high_high['asymmetry']:+.4f}")
        print(f"  Samples: n={high_high['n_ab']}/{high_high['n_a']}")
    else:
        print(f"\nHigh -> High: NOT FOUND")
    
    if normal_normal:
        print(f"\nNormal -> Normal:")
        print(f"  P({var_b}_Normal | {var_a}_Normal) = {normal_normal['P(B|A)']:.4f}")
        print(f"  P({var_a}_Normal | {var_b}_Normal) = {normal_normal['P(A|B)']:.4f}")
        print(f"  Asymmetry = {normal_normal['asymmetry']:+.4f}")
        print(f"  Samples: n={normal_normal['n_ab']}/{normal_normal['n_a']}")
    else:
        print(f"\nNormal -> Normal: NOT FOUND")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    max_state_asymmetry = max([abs(r['asymmetry']) for r in state_results])
    
    if abs(var_result['asymmetry']) < 0.05 and max_state_asymmetry > 0.1:
        print("[CORRECT] Gemini's hypothesis is CORRECT!")
        print(f"  - Variable-level asymmetry: {abs(var_result['asymmetry']):.4f} (symmetric)")
        print(f"  - Max state-level asymmetry: {max_state_asymmetry:.4f} (asymmetric)")
        print("  => Neural LP's fine-grained modeling DOES help!")
    elif abs(var_result['asymmetry']) > 0.1:
        print("[WRONG] Variable-level is already asymmetric")
        print(f"  - Variable-level asymmetry: {abs(var_result['asymmetry']):.4f}")
        print("  => FCI/GES should already capture this")
    elif max_state_asymmetry < 0.05:
        print("[WRONG] Data is symmetric at BOTH levels")
        print(f"  - Variable-level asymmetry: {abs(var_result['asymmetry']):.4f}")
        print(f"  - Max state-level asymmetry: {max_state_asymmetry:.4f}")
        print("  => Fine-grained modeling does NOT help for this edge")
    else:
        print("[MIXED] Mixed results")
        print(f"  - Variable-level asymmetry: {abs(var_result['asymmetry']):.4f}")
        print(f"  - Max state-level asymmetry: {max_state_asymmetry:.4f}")
    
    return {
        'variable_asymmetry': var_result['asymmetry'],
        'max_state_asymmetry': max_state_asymmetry,
        'state_results': state_results
    }


def main():
    # #region agent log
    import json
    from pathlib import Path
    import sys
    log_path = Path(r'/.cursor/debug.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:main','message':'Main function entry','data':{'cwd':str(Path.cwd())},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1,H2,H3'}) + '\n')
    # #endregion
    
    print("\n" + "=" * 70, flush=True)
    print("FINE-GRAINED ASYMMETRY ANALYSIS", flush=True)
    print("Testing Gemini's Hypothesis", flush=True)
    print("=" * 70, flush=True)
    
    # #region agent log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:main','message':'Before loading CSV','data':{'csv_path':'data/alarm_data_10000.csv','exists':Path('data/alarm_data_10000.csv').exists()},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1'}) + '\n')
    # #endregion
    
    # Load data
    try:
        df = pd.read_csv('data/alarm_data_10000.csv')
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:main','message':'CSV loaded successfully','data':{'shape':list(df.shape),'columns_count':len(df.columns)},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1,H2'}) + '\n')
        # #endregion
        print(f"\nLoaded data: {df.shape[0]} samples, {df.shape[1]} states", flush=True)
    except Exception as e:
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:main','message':'CSV load FAILED','data':{'error':str(e),'error_type':type(e).__name__},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H1,H3'}) + '\n')
        # #endregion
        print(f"ERROR loading CSV: {e}", flush=True)
        sys.exit(1)
    
    # Test on several edges
    test_cases = [
        ('LVEDVOLUME', 'HYPOVOLEMIA'),  # One of the 7 reversals
        ('CATECHOL', 'TPR'),             # One of the 7 reversals
        ('LVEDVOLUME', 'PCWP'),          # A correctly oriented edge
        ('HR', 'CO'),                    # A correctly oriented edge
        ('STROKEVOLUME', 'CO'),          # A correctly oriented edge
    ]
    
    results_summary = []
    
    for var_a, var_b in test_cases:
        result = analyze_edge_asymmetry(df, var_a, var_b)
        if result:
            results_summary.append({
                'edge': f"{var_a} <-> {var_b}",
                'var_asymmetry': result['variable_asymmetry'],
                'max_state_asymmetry': result['max_state_asymmetry']
            })
        print("\n")
    
    # Overall summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    print("\n| Edge | Variable Asymmetry | Max State Asymmetry | Gemini Hypothesis |")
    print("|------|-------------------|---------------------|-------------------|")
    
    for result in results_summary:
        var_asym = abs(result['var_asymmetry'])
        state_asym = result['max_state_asymmetry']
        
        if var_asym < 0.05 and state_asym > 0.1:
            verdict = "[CORRECT]"
        elif var_asym < 0.05 and state_asym < 0.05:
            verdict = "[WRONG] Both symmetric"
        else:
            verdict = "[MIXED]"
        
        print(f"| {result['edge']} | {var_asym:.4f} | {state_asym:.4f} | {verdict} |")
    
    # Final conclusion
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)
    
    correct_count = sum(1 for r in results_summary if abs(r['var_asymmetry']) < 0.05 and r['max_state_asymmetry'] > 0.1)
    both_symmetric_count = sum(1 for r in results_summary if abs(r['var_asymmetry']) < 0.05 and r['max_state_asymmetry'] < 0.05)
    
    print(f"\nTested {len(results_summary)} edges:")
    print(f"  - Gemini's hypothesis holds: {correct_count}/{len(results_summary)}")
    print(f"  - Both levels symmetric: {both_symmetric_count}/{len(results_summary)}")
    print(f"  - Other cases: {len(results_summary) - correct_count - both_symmetric_count}/{len(results_summary)}")
    
    if correct_count > len(results_summary) / 2:
        print("\n[CORRECT] Fine-grained modeling DOES help!")
        print("  Neural LP can capture state-level asymmetries that FCI/GES cannot see.")
    elif both_symmetric_count > len(results_summary) / 2:
        print("\n[WRONG] Fine-grained modeling does NOT help for this dataset")
        print("  Data is highly symmetric at both variable and state levels.")
        print("  Direction learning must rely on LLM priors or other mechanisms.")
    else:
        print("\n[MIXED] Mixed results")
        print("  Fine-grained modeling helps for some edges but not others.")


if __name__ == "__main__":
    # #region agent log
    import json
    from pathlib import Path
    import sys
    log_path = Path(r'/.cursor/debug.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:__main__','message':'Script __main__ block executing','data':{'python_version':sys.version,'executable':sys.executable},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H2,H3'}) + '\n')
    # #endregion
    
    try:
        main()
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:__main__','message':'Main completed successfully','data':{},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H3'}) + '\n')
        # #endregion
    except Exception as e:
        # #region agent log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'location':'fine_grained_asymmetry_analysis.py:__main__','message':'Main FAILED with exception','data':{'error':str(e),'error_type':type(e).__name__,'traceback':__import__('traceback').format_exc()},'timestamp':__import__('time').time()*1000,'sessionId':'debug-session','runId':'run1','hypothesisId':'H3'}) + '\n')
        # #endregion
        print(f"\nFATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

