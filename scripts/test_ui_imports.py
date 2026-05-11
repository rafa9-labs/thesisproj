"""Quick smoke test for UI imports."""
from ui.state import AppState, AVAILABLE_MODELS, DATA_FILES
from ui.controls import get_all_params, render_nav_bar, render_tab_content
from ui.dashboard import render_dashboard

print("Models:", AVAILABLE_MODELS)
print("Data:", list(DATA_FILES.keys()))
print("Settings type:", type(AppState.get_settings()).__name__)
print("Public API: get_all_params, render_nav_bar, render_tab_content")
print("\nAll UI imports OK!")