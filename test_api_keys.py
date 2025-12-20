"""
Test API Key Configuration

This script helps you verify that your API keys are properly configured.
"""

import os
import sys

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] python-dotenv is installed")
except ImportError:
    print("[WARN] python-dotenv not installed (optional)")
    print("  Install with: pip install python-dotenv")

print("\n" + "=" * 80)
print("API KEY CONFIGURATION TEST")
print("=" * 80)

# Check for API keys
keys_to_check = {
    'OPENAI_API_KEY': 'OpenAI (GPT-3.5, GPT-4)',
    'HUGGINGFACE_TOKEN': 'Hugging Face (Zephyr, Mistral)',
    'GOOGLE_API_KEY': 'Google AI (Gemini)'
}

found_keys = {}
missing_keys = []

for key_name, description in keys_to_check.items():
    key_value = os.getenv(key_name)
    
    if key_value:
        # Mask the key for security
        masked_value = key_value[:10] + "..." + key_value[-4:] if len(key_value) > 14 else key_value[:5] + "..."
        found_keys[key_name] = masked_value
        print(f"\n[OK] {key_name}")
        print(f"  Description: {description}")
        print(f"  Value: {masked_value}")
    else:
        missing_keys.append(key_name)
        print(f"\n[MISSING] {key_name}")
        print(f"  Description: {description}")
        print(f"  Status: Not set")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nFound: {len(found_keys)} API key(s)")
for key_name in found_keys:
    print(f"  [OK] {key_name}")

if missing_keys:
    print(f"\nMissing: {len(missing_keys)} API key(s)")
    for key_name in missing_keys:
        print(f"  [X] {key_name}")
    
    print("\n" + "-" * 80)
    print("HOW TO FIX")
    print("-" * 80)
    print("\nOption 1: Create .env file (recommended)")
    print("  1. Copy .env.example to .env")
    print("  2. Edit .env and add your API keys")
    print("  3. Run this script again")
    
    print("\nOption 2: Set environment variables")
    print("  Windows PowerShell:")
    for key_name in missing_keys:
        print(f"    $env:{key_name}=\"your_key_here\"")
    
    print("\n  Linux/Mac:")
    for key_name in missing_keys:
        print(f"    export {key_name}=\"your_key_here\"")
else:
    print("\n[OK] All API keys configured!")
    print("  You can now run the pipeline without manual input")

# Check .env file
print("\n" + "=" * 80)
print("FILE CHECK")
print("=" * 80)

env_file = ".env"
env_example = ".env.example"

if os.path.exists(env_file):
    print(f"\n[OK] {env_file} exists")
    
    # Check if it's in .gitignore
    gitignore_file = ".gitignore"
    if os.path.exists(gitignore_file):
        with open(gitignore_file, 'r') as f:
            gitignore_content = f.read()
        
        if '.env' in gitignore_content:
            print(f"[OK] {env_file} is in .gitignore (safe)")
        else:
            print(f"[WARN] {env_file} is NOT in .gitignore!")
            print("  Add '.env' to .gitignore to prevent accidental commits")
    else:
        print(f"[WARN] .gitignore not found")
else:
    print(f"\n[MISSING] {env_file} not found")
    
    if os.path.exists(env_example):
        print(f"[INFO] {env_example} exists")
        print(f"  Run: cp {env_example} {env_file}")
    else:
        print(f"[WARN] {env_example} also not found")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

if found_keys and not missing_keys:
    print("\n[OK] Configuration complete! You can now run:")
    print("  cd refactored")
    print("  python main_hybrid_fci_llm.py     # For GPT-3.5")
    print("  python main_hybrid_fci_zephyr.py  # For Zephyr")
    print("\n  cd ../Neuro-Symbolic-Reasoning")
    print("  python run_pipeline.py            # Full pipeline")
else:
    print("\n1. Configure missing API keys (see 'HOW TO FIX' above)")
    print("2. Run this script again to verify")
    print("3. Read API_KEY_SETUP.md for detailed instructions")

print("\n" + "=" * 80)
