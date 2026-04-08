"""Quick smoke test for UI imports."""
from ui.state import AppState, AVAILABLE_MODELS, DATA_FILES
from ui.controls import render_data_tab, render_model_tab
from ui.dashboard import render_dashboard

print("Models:", AVAILABLE_MODELS)
print("Data:", list(DATA_FILES.keys()))
print("Settings type:", type(AppState.get_settings()).__name__)
print("\nAll UI imports OK!")