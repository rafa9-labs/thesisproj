"""Comprehensive audit: deprecated files vs current pipeline."""
import ast, os, glob

def get_defs(path):
    """Return dict {name: (kind, line, file)} from a Python file."""
    if not os.path.exists(path):
        return {}
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return {}
    defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defs[node.name] = ("FN", node.lineno, path)
        elif isinstance(node, ast.ClassDef):
            defs[node.name] = ("CL", node.lineno, path)
    return defs

# Collect ALL current definitions
current_files = []
for pattern in [
    "pipeline/*.py",
    "pipeline/backtester/*.py",
    "pipeline/tuning/*.py",
    "models/*.py",
    "ui/*.py",
    "config.py",
]:
    current_files.extend(glob.glob(pattern))

current_defs = {}
for f in current_files:
    if "__pycache__" in f:
        continue
    defs = get_defs(f)
    for name, (kind, line, path) in defs.items():
        current_defs.setdefault(name, []).append((kind, line, path))

# Collect ALL deprecated definitions
deprecated_files = ["tuningNoWFO.py", "utilsNoWFO.py"]
deprecated_defs = {}
for f in deprecated_files:
    defs = get_defs(f)
    for name, (kind, line, path) in defs.items():
        deprecated_defs.setdefault(name, []).append((kind, line, path))

# Find what's in deprecated but NOT in current
# Filter out private/dunder helpers that are implementation details
missing = []
for name, locations in sorted(deprecated_defs.items()):
    if name in current_defs:
        continue
    # Skip truly private helpers (double underscore or very short)
    if name.startswith("__") and name.endswith("__"):
        continue
    # Show it
    files = set(os.path.basename(loc[2]) for loc in locations)
    missing.append((name, files))

print("=" * 70)
print("  MISSING FROM CURRENT PIPELINE (in deprecated but not refactored)")
print("=" * 70)
for name, files in sorted(missing):
    print(f"  {name:50s} <- {', '.join(sorted(files))}")

print(f"\n  TOTAL MISSING: {len(missing)} functions/classes")

# Show what IS covered
covered = []
for name in sorted(deprecated_defs.keys()):
    if name in current_defs:
        covered.append(name)
print(f"\n  ALREADY COVERED: {len(covered)} functions/classes")
print(f"  (e.g.: {', '.join(covered[:10])}...)")