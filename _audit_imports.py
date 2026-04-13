"""Audit all utilsNoWFO imports across the codebase."""
import re, os
from collections import defaultdict

# Map: symbol -> list of files that import it
symbol_files = defaultdict(list)

for root, dirs, files in os.walk(".", topdown=True):
    # Skip hidden dirs, .git, node_modules
    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname).replace("\\", "/")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        # Pattern 1: from utilsNoWFO import a, b, c
        for m in re.finditer(r"from\s+utilsNoWFO\s+import\s*\(([^)]+)\)", content, re.DOTALL):
            imports = m.group(1)
            for sym in re.findall(r"(\w+)", imports):
                symbol_files[sym].append(fpath)

        # Pattern 2: from utilsNoWFO import x as y (single line)
        for m in re.finditer(r"from\s+utilsNoWFO\s+import\s+([^\n(]+)", content):
            imports = m.group(1)
            for sym in re.findall(r"(\w+)", imports):
                if sym not in ("import", "as"):
                    symbol_files[sym].append(fpath)

# Sort by frequency
print(f"{'Symbol':50s} {'#Files':>6s}  Files")
print("=" * 120)
for sym, files in sorted(symbol_files.items(), key=lambda x: -len(x[1])):
    unique = sorted(set(files))
    print(f"{sym:50s} {len(unique):6d}  {', '.join(unique)}")
print(f"\nTotal unique symbols imported: {len(symbol_files)}")