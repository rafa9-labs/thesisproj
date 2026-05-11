"""Validate Electron desktop shell, build config, and packaging prerequisites.

These tests verify the Electron shell source, TypeScript compilation,
electron-builder config, and asar contents WITHOUT actually running
the full build (which is expensive).
"""
import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
ELECTRON_DIR = os.path.join(PROJECT_ROOT, "electron")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
ELECTRON_DIST_DIR = os.path.join(FRONTEND_DIR, "electron-dist")


class TestElectronSource:
    """Verify Electron TypeScript source files exist and are well-formed."""

    REQUIRED_TS_FILES = [
        "main.ts",
        "preload.ts",
        "python.ts",
        "health.ts",
        "splash.ts",
        "menu.ts",
        "tray.ts",
        "utils.ts",
    ]

    @pytest.mark.parametrize("filename", REQUIRED_TS_FILES, ids=REQUIRED_TS_FILES)
    def test_source_file_exists(self, filename):
        path = os.path.join(ELECTRON_DIR, filename)
        assert os.path.isfile(path), f"Missing Electron source: {path}"

    def test_tsconfig_exists(self):
        path = os.path.join(ELECTRON_DIR, "tsconfig.json")
        assert os.path.isfile(path)

    def test_tsconfig_outdir_points_to_frontend(self):
        import json as _json

        with open(os.path.join(ELECTRON_DIR, "tsconfig.json")) as f:
            tsconfig = _json.load(f)
        outdir = tsconfig.get("compilerOptions", {}).get("outDir", "")
        assert "electron-dist" in outdir, f"outDir should contain 'electron-dist', got: {outdir}"
        assert ".." in outdir or "frontend" in outdir, f"outDir should output into frontend/, got: {outdir}"


class TestPackageJson:
    """Verify frontend package.json Electron configuration."""

    @pytest.fixture()
    def pkg(self):
        with open(os.path.join(FRONTEND_DIR, "package.json")) as f:
            return json.load(f)

    def test_main_field_resolves_in_asar(self, pkg):
        main = pkg.get("main", "")
        assert not main.startswith("../"), f"main must not use ../ (fails in asar), got: {main}"
        assert "electron-dist/main.js" in main, f"main should be electron-dist/main.js, got: {main}"

    def test_electron_compile_script(self, pkg):
        scripts = pkg.get("scripts", {})
        assert "electron:compile" in scripts
        assert "tsconfig.json" in scripts["electron:compile"]

    def test_electron_run_script(self, pkg):
        scripts = pkg.get("scripts", {})
        assert "electron:run" in scripts
        run_cmd = scripts["electron:run"]
        assert not run_cmd.startswith("electron ../"), f"electron:run should not use ../, got: {run_cmd}"
        assert "electron-dist/main.js" in run_cmd

    def test_electron_pack_script(self, pkg):
        scripts = pkg.get("scripts", {})
        assert "electron:pack" in scripts
        assert "electron-builder" in scripts["electron:pack"]

    def test_electron_dist_script(self, pkg):
        scripts = pkg.get("scripts", {})
        assert "electron:dist" in scripts
        assert "electron-builder" in scripts["electron:dist"]

    def test_electron_in_dev_deps(self, pkg):
        dev_deps = pkg.get("devDependencies", {})
        assert "electron" in dev_deps, "electron missing from devDependencies"
        assert "electron-builder" in dev_deps, "electron-builder missing from devDependencies"


class TestElectronBuilderConfig:
    """Verify electron-builder.yml is well-formed."""

    @pytest.fixture()
    def yml_path(self):
        return os.path.join(PROJECT_ROOT, "electron-builder.yml")

    def test_config_file_exists(self, yml_path):
        assert os.path.isfile(yml_path)

    def test_files_include_electron_dist(self, yml_path):
        with open(yml_path) as f:
            content = f.read()
        assert "electron-dist/**/*" in content, "electron-dist not in files pattern"

    def test_no_backward_path_in_files(self, yml_path):
        with open(yml_path) as f:
            content = f.read()
        assert "../electron-dist" not in content, "Should not reference ../electron-dist in files"

    def test_extra_resources_backend(self, yml_path):
        with open(yml_path) as f:
            content = f.read()
        assert "fx_backend" in content, "fx_backend not in extraResources"
        assert "backend" in content, "backend target not in extraResources"

    def test_extra_resources_csv_data(self, yml_path):
        with open(yml_path) as f:
            content = f.read()
        assert "csv_data" in content, "csv_data not in extraResources"

    def test_nsis_config(self, yml_path):
        with open(yml_path) as f:
            content = f.read()
        assert "oneClick: false" in content, "NSIS must not be one-click"

    def test_no_sign_for_dev_builds(self, yml_path):
        with open(yml_path) as f:
            content = f.read()
        assert "signAndEditExecutable: false" in content, "signAndEditExecutable should be false for dev"


class TestBuildArtifacts:
    """Verify build output artifacts when they exist."""

    def test_electron_dist_exists_after_compile(self):
        if not os.path.isdir(ELECTRON_DIST_DIR):
            pytest.skip("Run 'npm run electron:compile' first")
        required = ["main.js", "preload.js", "python.js", "health.js", "splash.js", "utils.js"]
        for f in required:
            path = os.path.join(ELECTRON_DIST_DIR, f)
            assert os.path.isfile(path), f"Missing compiled Electron JS: {path}"

    def test_icon_ico_exists(self):
        ico = os.path.join(BUILD_DIR, "icon.ico")
        assert os.path.isfile(ico), f"Missing app icon: {ico}"

    def test_icon_png_exists(self):
        png = os.path.join(BUILD_DIR, "icon.png")
        assert os.path.isfile(png), f"Missing app icon PNG: {png}"


class TestPythonBackendBundle:
    """Verify PyInstaller backend bundle when it exists."""

    BACKEND_DIR = os.path.join(PROJECT_ROOT, "dist", "fx_backend")

    def test_fx_backend_exe_exists(self):
        if not os.path.isdir(self.BACKEND_DIR):
            pytest.skip("Run 'scripts\\build_python.bat' first")
        exe = os.path.join(self.BACKEND_DIR, "fx_backend.exe")
        assert os.path.isfile(exe), f"Missing backend exe: {exe}"
        size_mb = os.path.getsize(exe) / (1024 * 1024)
        assert size_mb > 1, f"Backend exe too small ({size_mb:.1f} MB), build may be broken"

    def test_no_torch_in_bundle(self):
        if not os.path.isdir(self.BACKEND_DIR):
            pytest.skip("Run 'scripts\\build_python.bat' first")
        torch_files = []
        for root, dirs, files in os.walk(self.BACKEND_DIR):
            for f in files:
                if "torch" in f.lower() and "torch" in root.lower():
                    torch_files.append(os.path.join(root, f))
        assert len(torch_files) == 0, f"Found {len(torch_files)} torch files in bundle (should be excluded)"

    def test_backend_bundle_size_reasonable(self):
        if not os.path.isdir(self.BACKEND_DIR):
            pytest.skip("Run 'scripts\\build_python.bat' first")
        total_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(self.BACKEND_DIR)
            for f in files
        )
        size_mb = total_size / (1024 * 1024)
        assert size_mb < 3000, f"Backend bundle too large ({size_mb:.0f} MB), check excludes"
        assert size_mb > 100, f"Backend bundle too small ({size_mb:.0f} MB), may be incomplete"


class TestReleaseArtifacts:
    """Verify release output when it exists."""

    RELEASE_DIR = os.path.join(PROJECT_ROOT, "release")

    def test_installer_exists(self):
        if not os.path.isdir(self.RELEASE_DIR):
            pytest.skip("Run 'npm run electron:dist' first")
        installers = [f for f in os.listdir(self.RELEASE_DIR) if f.endswith(".exe") and "Setup" in f]
        assert len(installers) >= 1, f"No NSIS installer found in {self.RELEASE_DIR}"

    def test_unpacked_app_exists(self):
        unpacked = os.path.join(self.RELEASE_DIR, "win-unpacked")
        if not os.path.isdir(unpacked):
            pytest.skip("Run 'npm run electron:pack' first")
        exe = os.path.join(unpacked, "KodaQuant.exe")
        assert os.path.isfile(exe), f"Missing unpacked exe: {exe}"

    def test_asar_contains_entry_point(self):
        unpacked = os.path.join(self.RELEASE_DIR, "win-unpacked")
        if not os.path.isdir(unpacked):
            pytest.skip("Run 'npm run electron:pack' first")
        asar_path = os.path.join(unpacked, "resources", "app.asar")
        if not os.path.isfile(asar_path):
            pytest.skip("No app.asar found")
        try:
            result = subprocess.run(
                ["npx", "asar", "list", asar_path],
                capture_output=True, text=True, timeout=30,
                cwd=FRONTEND_DIR,
            )
            listing = result.stdout
        except Exception:
            pytest.skip("asar tool not available")
        assert "electron-dist/main.js" in listing, "Entry point not found in asar"
        assert "electron-dist/preload.js" in listing, "preload.js not found in asar"
        assert "dist/index.html" in listing, "React frontend not found in asar"

    def test_backend_in_resources(self):
        unpacked = os.path.join(self.RELEASE_DIR, "win-unpacked")
        if not os.path.isdir(unpacked):
            pytest.skip("Run 'npm run electron:pack' first")
        backend_exe = os.path.join(unpacked, "resources", "backend", "fx_backend.exe")
        assert os.path.isfile(backend_exe), "Python backend exe not in resources/backend/"

    def test_csv_data_in_resources(self):
        unpacked = os.path.join(self.RELEASE_DIR, "win-unpacked")
        if not os.path.isdir(unpacked):
            pytest.skip("Run 'npm run electron:pack' first")
        csv_dir = os.path.join(unpacked, "resources", "csv_data")
        assert os.path.isdir(csv_dir), "csv_data directory not in resources/"
        csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        assert len(csvs) >= 3, f"Expected 3+ CSV files, found {len(csvs)}"


class TestMainTsPaths:
    """Verify path resolution logic in compiled main.js."""

    def test_no_frontend_dist_in_prod_path(self):
        main_js = os.path.join(ELECTRON_DIST_DIR, "main.js")
        if not os.path.isfile(main_js):
            pytest.skip("Run 'npm run electron:compile' first")
        with open(main_js) as f:
            content = f.read()
        assert '"frontend"' not in content or '"dist"' not in content.split('"frontend"')[0][:200], \
            "main.ts still references 'frontend/dist' in production path — should be just 'dist'"