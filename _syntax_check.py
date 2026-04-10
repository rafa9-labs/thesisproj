import py_compile, sys
files = [
    "ui/validators.py", "ui/charts.py", "ui/results.py",
    "ui/controls.py", "ui/state.py", "ui/dashboard.py", "app.py",
]
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except py_compile.PyCompileError as e:
        print(f"FAIL: {f} -> {e}")
        ok = False
sys.exit(0 if ok else 1)