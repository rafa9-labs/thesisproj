"""
S15.1 Branding audit -- verify KodaQuant name replaces all stale "FX ML Backtester" references.

Tests:
1. No stale name in any Python source file
2. No stale name in any shell/batch/spec/config file
3. KodaQuant appears in expected key files
4. API config reports correct app_name
5. run_server.py uses KodaQuant branding
6. PyInstaller spec contains KodaQuant data dir reference
"""
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STALE_PATTERNS = ["FX ML Backtester", "FOREX ML BACKTESTER"]
SKIP_GLOBS = {"__pycache__", ".git", "node_modules", "venv", ".venv",
              ".venv-wsl", "dist", "build", "release", "results",
              ".pytest_cache", ".mypy_cache", ".feature_cache",
              "news_cache", "data", ".tox", "package-lock.json"}

THIS_FILE = Path(__file__).name


def _should_skip(path: str) -> bool:
    parts = Path(path).parts
    return any(part in SKIP_GLOBS for part in parts)


def _walk_source_files(extensions):
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_GLOBS]
        for f in files:
            ext = f.rsplit(".", 1)[-1] if "." in f else ""
            if ext in extensions:
                yield os.path.join(root, f)


class TestBrandingNoStaleReferences:
    """No source file should contain the old "FX ML Backtester" branding."""

    PY_EXTENSIONS = {"py", "pyx"}
    CONFIG_EXTENSIONS = {"bat", "spec", "yml", "yaml", "cfg", "toml", "ini"}
    TS_EXTENSIONS = {"ts", "tsx", "mts", "cts"}
    WEB_EXTENSIONS = {"css", "json", "html", "svg", "js"}

    @pytest.mark.parametrize("ext", ["py", "bat", "spec", "yml", "ts", "tsx", "css", "json", "html"])
    def test_no_stale_name_in_files(self, ext):
        files_checked = 0
        for filepath in _walk_source_files({ext}):
            if Path(filepath).name == THIS_FILE:
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except Exception:
                continue
            for pattern in STALE_PATTERNS:
                assert pattern not in content, (
                    f"Stale branding '{pattern}' found in {filepath}"
                )
            files_checked += 1
        assert files_checked > 0, f"No .{ext} files found to check"

    def test_no_stale_name_in_run_server(self):
        path = PROJECT_ROOT / "run_server.py"
        content = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            assert pattern not in content, f"Stale '{pattern}' in run_server.py"

    def test_no_stale_name_in_forex_pipeline_spec(self):
        path = PROJECT_ROOT / "forex_pipeline.spec"
        content = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            assert pattern not in content, f"Stale '{pattern}' in forex_pipeline.spec"

    def test_no_stale_name_in_readme(self):
        path = PROJECT_ROOT / "README.md"
        content = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            assert pattern not in content, f"Stale '{pattern}' in README.md"


class TestBrandingKodaquantPresent:
    """KodaQuant branding must appear in expected locations."""

    def test_kodaquant_in_api_config(self):
        from api.config import settings
        assert settings.app_name == "KodaQuant API", (
            f"Expected app_name='KodaQuant API', got '{settings.app_name}'"
        )

    def test_kodaquant_in_run_server(self):
        path = PROJECT_ROOT / "run_server.py"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "KodaQuant missing from run_server.py"
        assert "[KodaQuant] Starting server" in content, (
            "run_server.py startup message not updated to KodaQuant"
        )

    def test_kodaquant_in_forex_pipeline_spec(self):
        path = PROJECT_ROOT / "forex_pipeline.spec"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "KodaQuant missing from forex_pipeline.spec"
        assert "%APPDATA%/KodaQuant/" in content, (
            "spec file still has old data dir path"
        )

    def test_kodaquant_in_main_py(self):
        path = PROJECT_ROOT / "api" / "main.py"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, (
            "KodaQuant missing from api/main.py docstring"
        )

    def test_kodaquant_in_readme(self):
        path = PROJECT_ROOT / "README.md"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "KodaQuant missing from README"
        assert "KODAQUANT" in content, (
            "ASCII diagram header not updated to KODAQUANT"
        )
        assert "forex-pipeline" not in content, (
            "README still has stale 'forex-pipeline' tree name"
        )

    def test_kodaquant_in_frontend_package_json(self):
        path = PROJECT_ROOT / "frontend" / "package.json"
        content = path.read_text(encoding="utf-8")
        assert '"kodaquant"' in content.lower(), (
            "frontend package.json name should be kodaquant"
        )

    def test_kodaquant_in_electron_main(self):
        path = PROJECT_ROOT / "electron" / "main.ts"
        if not path.exists():
            pytest.skip("electron/main.ts not found")
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "KodaQuant missing from electron/main.ts"

    def test_kodaquant_in_electron_tray(self):
        path = PROJECT_ROOT / "electron" / "tray.ts"
        if not path.exists():
            pytest.skip("electron/tray.ts not found")
        content = path.read_text(encoding="utf-8")
        assert '"KodaQuant"' in content, (
            "electron/tray.ts still has old tray label/tooltip"
        )

    def test_kodaquant_in_electron_builder_yml(self):
        path = PROJECT_ROOT / "electron-builder.yml"
        content = path.read_text(encoding="utf-8")
        assert 'KodaQuant' in content, (
            "electron-builder.yml still has old product name"
        )


class TestBrandingDocs:
    """Documentation files should reflect the KodaQuant brand."""

    def test_claude_md_consistent(self):
        path = PROJECT_ROOT / "CLAUDE.md"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "CLAUDE.md missing KodaQuant"
        # CLAUDE.md explicitly says "forex pipeline" as internal shorthand — this is acceptable
        # The branding test just confirms the name is present

    def test_news_init_no_stale(self):
        path = PROJECT_ROOT / "news" / "__init__.py"
        content = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            assert pattern not in content, f"Stale '{pattern}' in news/__init__.py"

    def test_schemas_init_no_stale(self):
        path = PROJECT_ROOT / "schemas" / "__init__.py"
        content = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            assert pattern not in content, f"Stale '{pattern}' in schemas/__init__.py"

    def test_streamlit_removed(self):
        assert not (PROJECT_ROOT / "app.py").exists(), (
            "Streamlit app.py should not exist (removed in Sprint 8B)"
        )
        assert not (PROJECT_ROOT / "ui").is_dir(), (
            "Streamlit ui/ directory should not exist (removed in Sprint 8B)"
        )
        assert not (PROJECT_ROOT / "launch_ui.bat").exists(), (
            "launch_ui.bat should not exist (removed in Sprint 8B)"
        )

    def test_about_dialog_no_thesisproj(self):
        path = PROJECT_ROOT / "frontend" / "src" / "components" / "shared" / "AboutDialog.tsx"
        content = path.read_text(encoding="utf-8")
        assert "thesisproj" not in content, (
            "AboutDialog.tsx should not link to thesisproj repo"
        )
        assert "kodaquant" in content.lower(), (
            "AboutDialog.tsx should link to kodaquant repo"
        )

    def test_setup_md_uses_kodaquant(self):
        path = PROJECT_ROOT / "SETUP.md"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "SETUP.md missing KodaQuant in header"
        for pattern in STALE_PATTERNS:
            assert pattern not in content, f"SETUP.md has stale '{pattern}'"
        assert "Streamlit" not in content, "SETUP.md should not reference Streamlit"

    def test_roadmap_md_s15_complete(self):
        path = PROJECT_ROOT / "ROADMAP.md"
        content = path.read_text(encoding="utf-8")
        assert "## Sprint 15: KodaQuant Branding" in content, (
            "ROADMAP.md S15 header missing or stale"
        )
        assert "✅ COMPLETE" in content or "✅ DONE" in content, (
            "ROADMAP.md S15 not marked complete"
        )

    def test_project_plan_superseded(self):
        path = PROJECT_ROOT / "PROJECT_PLAN.md"
        content = path.read_text(encoding="utf-8")
        assert "KodaQuant" in content, "PROJECT_PLAN.md missing KodaQuant"
        assert "Superseded" in content, "PROJECT_PLAN.md not marked as superseded"


class TestS152IconAssets:
    """S15.2 — KodaQuant icon assets must exist with correct content."""

    BUILD_DIR = PROJECT_ROOT / "build"
    PUBLIC_DIR = PROJECT_ROOT / "frontend" / "public"

    def test_icon_svg_exists(self):
        path = self.BUILD_DIR / "icon.svg"
        assert path.exists(), "build/icon.svg missing"
        content = path.read_text(encoding="utf-8")
        assert "<polygon" in content, "icon.svg must contain polygon elements"
        assert "#00E5FF" in content, "icon.svg must use KodaQuant cyan (#00E5FF)"
        assert "#050608" in content, "icon.svg must use KodaQuant dark (#050608)"

    def test_icon_svg_no_stale_text(self):
        path = self.BUILD_DIR / "icon.svg"
        content = path.read_text(encoding="utf-8")
        has_text_tag = "<text" in content
        upper = content.upper()
        has_fx_in_text = "FX" in upper and has_text_tag
        assert not has_fx_in_text, (
            "icon.svg must not contain 'FX' in a <text> element"
        )

    def test_icon_png_exists(self):
        path = self.BUILD_DIR / "icon.png"
        assert path.exists(), "build/icon.png missing"
        assert path.stat().st_size > 1000, "icon.png too small (<1KB)"

    def test_icon_ico_exists(self):
        path = self.BUILD_DIR / "icon.ico"
        assert path.exists(), "build/icon.ico missing"
        assert path.stat().st_size > 1000, "icon.ico too small (<1KB)"

    def test_icon_ico_multi_resolution(self):
        import struct
        path = self.BUILD_DIR / "icon.ico"
        data = path.read_bytes()
        _, _, image_count = struct.unpack_from("<HHH", data, 0)
        assert image_count >= 4, (
            f"icon.ico has only {image_count} images in header, expected >=4"
        )

    def test_favicon_svg_no_stale_text(self):
        path = self.PUBLIC_DIR / "favicon.svg"
        assert path.exists(), "frontend/public/favicon.svg missing"
        content = path.read_text(encoding="utf-8").upper()
        assert "<text" not in path.read_text(encoding="utf-8"), (
            "favicon.svg must not contain text elements"
        )

    def test_favicon_svg_colors(self):
        path = self.PUBLIC_DIR / "favicon.svg"
        content = path.read_text(encoding="utf-8")
        assert "#00E5FF" in content, "favicon.svg must use cyan (#00E5FF)"
        assert "#050608" in content, "favicon.svg must use dark (#050608)"

    def test_favicon_ico_exists(self):
        path = self.PUBLIC_DIR / "favicon.ico"
        assert path.exists(), "frontend/public/favicon.ico missing"

    def test_installer_bmps_exist(self):
        header = self.BUILD_DIR / "installer-header.bmp"
        sidebar = self.BUILD_DIR / "installer-sidebar.bmp"
        assert header.exists(), "build/installer-header.bmp missing"
        assert sidebar.exists(), "build/installer-sidebar.bmp missing"

    def test_electron_builder_icon_ref(self):
        path = PROJECT_ROOT / "electron-builder.yml"
        content = path.read_text(encoding="utf-8")
        assert "icon.ico" in content, "electron-builder.yml must reference icon.ico"

    def test_icon_generation_script_exists(self):
        path = PROJECT_ROOT / "scripts" / "generate_icons.py"
        assert path.exists(), "scripts/generate_icons.py missing"

    def test_icon_png_is_valid_rgba_image(self):
        from PIL import Image
        path = self.BUILD_DIR / "icon.png"
        img = Image.open(path)
        assert img.mode == "RGBA", f"icon.png mode should be RGBA, got {img.mode}"
        assert img.size == (1024, 1024), f"icon.png size should be 1024x1024, got {img.size}"
