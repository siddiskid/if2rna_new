"""Quick script to check the structure of test_results.pkl"""
import pickle
import sys

results_file = 'results/if2rna_models/baseline_resnet_log/test_results.pkl'

print(f"Loading {results_file}...")
with open(results_file, 'rb') as f:
    results = pickle.load(f)

print(f"\nType: {type(results)}")
print(f"\nKeys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")

# Print structure
if isinstance(results, dict):
    for key, value in results.items():
        if isinstance(value, list):
            print(f"\n{key}: list of length {len(value)}")
            if len(value) > 0:
                print(f"  First item type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"  First item keys: {value[0].keys()}")
        elif isinstance(value, dict):
            print(f"\n{key}: dict with keys {value.keys()}")
        else:
            print(f"\n{key}: {type(value)}")
