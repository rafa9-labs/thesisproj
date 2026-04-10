"""Quick dependency audit for WSL2 venv."""
import importlib, sys

deps = [
    'pandas', 'numpy', 'sklearn', 'xgboost', 'optuna', 'streamlit',
    'joblib', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'tensorflow',
    'shap', 'statsmodels', 'httpx', 'requests', 'websocket', 'tqdm',
    'graphviz', 'tabulate',
]

missing = []
for d in deps:
    try:
        importlib.import_module(d)
        print(f'  OK  {d}')
    except Exception as e:
        missing.append(d)
        print(f'  MISS {d}: {e}')

if missing:
    print(f'\nTO INSTALL: pip install {" ".join(missing)}')
else:
    print('\nAll deps OK!')