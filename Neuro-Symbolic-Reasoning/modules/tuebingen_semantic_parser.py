# -*- coding: utf-8 -*-
"""
Tuebingen Semantic Parser

Parses _des.txt files to extract semantic information about variables.
This enables LLM to make informed causal judgments based on domain knowledge.
"""

import re
from pathlib import Path
from typing import Dict, Optional


def parse_tuebingen_description(des_path: Path) -> Dict:
    """
    Parse Tuebingen description file to extract variable semantics
    
    Args:
        des_path: Path to _des.txt file
    
    Returns:
        {
            'var_x_description': 'altitude',
            'var_y_description': 'temperature (average over 1961-1990)',
            'ground_truth_direction': 1,  # 1=x->y, -1=y->x, 0=unknown
            'has_semantic_info': True,
            'context': 'DWD data (Deutscher Wetterdienst)...'
        }
    """
    result = {
        'var_x_description': None,
        'var_y_description': None,
        'ground_truth_direction': 0,
        'has_semantic_info': False,
        'context': None
    }
    
    try:
        with open(des_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract variable descriptions
        # Pattern: "x: altitude" or "x = Organic C content"
        x_match = re.search(r'^x[:\s=]\s*(.+?)$', content, re.MULTILINE | re.IGNORECASE)
        y_match = re.search(r'^y[:\s=]\s*(.+?)$', content, re.MULTILINE | re.IGNORECASE)
        
        if x_match:
            result['var_x_description'] = x_match.group(1).strip()
        
        if y_match:
            result['var_y_description'] = y_match.group(1).strip()
        
        # Extract ground truth direction
        # Pattern: "x --> y" or "y --> x" or "ground truth: x --> y"
        gt_match = re.search(
            r'(?:ground truth[:\s]*)?([xy])\s*--?>?\s*([xy])', 
            content, 
            re.IGNORECASE
        )
        
        if gt_match:
            from_var = gt_match.group(1).lower()
            to_var = gt_match.group(2).lower()
            
            if from_var == 'x' and to_var == 'y':
                result['ground_truth_direction'] = 1
            elif from_var == 'y' and to_var == 'x':
                result['ground_truth_direction'] = -1
        
        # Extract context (first few lines, usually contains dataset info)
        lines = content.split('\n')
        context_lines = []
        for line in lines[:5]:  # First 5 lines usually have context
            line = line.strip()
            if line and not line.startswith('http') and not line.startswith('x') and not line.startswith('y'):
                context_lines.append(line)
        
        if context_lines:
            result['context'] = ' '.join(context_lines)
        
        # Check if we have semantic info
        result['has_semantic_info'] = (
            result['var_x_description'] is not None and 
            result['var_y_description'] is not None
        )
        
        return result
        
    except Exception as e:
        print(f"[WARN] Failed to parse {des_path}: {e}")
        return result


def format_variable_for_llm(var_name: str, var_description: Optional[str]) -> str:
    """
    Format variable for LLM prompt
    
    Args:
        var_name: Original variable name (e.g., "x")
        var_description: Semantic description (e.g., "altitude")
    
    Returns:
        Formatted string for LLM (e.g., "altitude" or "x (no description)")
    """
    if var_description and var_description.strip():
        return var_description.strip()
    else:
        return f"{var_name} (no description available)"


# Test the parser
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Test on a few pairs
    test_pairs = ['pair0001', 'pair0073', 'pair0077', 'pair0092']
    data_dir = Path(__file__).parent.parent / 'data' / 'tuebingen'
    
    print("=" * 80)
    print("TESTING TUEBINGEN SEMANTIC PARSER")
    print("=" * 80)
    
    for pair_id in test_pairs:
        des_path = data_dir / f"{pair_id}_des.txt"
        
        if not des_path.exists():
            print(f"\n[SKIP] {pair_id}: File not found")
            continue
        
        print(f"\n{pair_id}:")
        print("-" * 80)
        
        result = parse_tuebingen_description(des_path)
        
        print(f"  X: {result['var_x_description']}")
        print(f"  Y: {result['var_y_description']}")
        print(f"  Ground Truth: {'x->y' if result['ground_truth_direction'] == 1 else 'y->x' if result['ground_truth_direction'] == -1 else 'unknown'}")
        print(f"  Has Semantic Info: {result['has_semantic_info']}")
        
        if result['context']:
            print(f"  Context: {result['context'][:100]}...")
        
        # Test formatting for LLM
        x_formatted = format_variable_for_llm('x', result['var_x_description'])
        y_formatted = format_variable_for_llm('y', result['var_y_description'])
        print(f"\n  LLM Format:")
        print(f"    Variable X: {x_formatted}")
        print(f"    Variable Y: {y_formatted}")
    
    print("\n" + "=" * 80)



