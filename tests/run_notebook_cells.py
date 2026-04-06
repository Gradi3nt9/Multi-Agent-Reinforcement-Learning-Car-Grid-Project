import json
import os
import sys
from pathlib import Path

# Resolve repository root relative to this test script so the runner is portable.
repo_root = Path(__file__).resolve().parent.parent
nb_path = repo_root / 'PolicyNetwork.ipynb'
if not nb_path.exists():
    print("Notebook not found:", nb_path)
    sys.exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
if len(cells) < 3:
    print(f"Notebook has only {len(cells)} cells; need at least 3 for this smoke test")
    sys.exit(1)

# Combine the first three cells' source into a single script
script_lines = []
for i in range(3):
    src = cells[i].get('source', [])
    # source entries may be strings; join them
    for line in src:
        script_lines.append(line)

script = '\n'.join(script_lines)

print("--- Environment info ---")
print("Python:", sys.version.replace('\n', ' '))
try:
    import torch
    print("PyTorch:", torch.__version__)
except Exception:
    print("PyTorch not available. Install torch to run the notebook smoke test: https://pytorch.org/")

print("--- Executing concatenated script from first 3 cells ---")
# Execute in a controlled namespace
namespace = {}
try:
    exec(script, namespace)
except Exception as e:
    import traceback
    print("Error while executing script:")
    traceback.print_exc()
    sys.exit(2)

print("--- Execution finished successfully ---")
