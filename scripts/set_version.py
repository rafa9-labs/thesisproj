"""
KodaQuant — Single-source-of-truth version setter.

Usage:
    python scripts/set_version.py <major.minor.patch> [--build <number>]

Example:
    python scripts/set_version.py 1.1.0 --build 42

Updates:
  1. frontend/package.json         — "version" field
  2. electron-builder.yml          — artifactName template (version comes from package.json)
  3. forex_pipeline.spec           — console title

Does NOT:
  - Create git tags (do that manually: git tag v1.1.0)
  - Push to GitHub (do that manually)
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PKG = ROOT / "frontend" / "package.json"
SPEC_FILE = ROOT / "forex_pipeline.spec"


def set_package_json(version: str) -> None:
    pkg = json.loads(FRONTEND_PKG.read_text(encoding="utf-8"))
    old = pkg.get("version", "?")
    pkg["version"] = version
    FRONTEND_PKG.write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  [OK] frontend/package.json: {old} -> {version}")


def set_spec_file(version: str) -> None:
    text = SPEC_FILE.read_text(encoding="utf-8")
    text = re.sub(
        r"(console\s*=\s*CONSOLE\(.*?version\s*=\s*['\"]).*?(['\"])",
        rf"\g<1>{version}\g<2>",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(KodaQuant\s+v)\d+\.\d+\.\d+",
        rf"\g<1>{version}",
        text,
    )
    SPEC_FILE.write_text(text, encoding="utf-8")
    print(f"  [OK] forex_pipeline.spec: version -> {version}")


def main():
    parser = argparse.ArgumentParser(description="Set KodaQuant version across all config files")
    parser.add_argument("version", help="Version string (e.g. 1.2.3)")
    parser.add_argument("--build", type=int, default=None, help="Optional build number (not written, for reference only)")
    args = parser.parse_args()

    semver = re.match(r"^\d+\.\d+\.\d+$", args.version)
    if not semver:
        print(f"ERROR: Invalid version '{args.version}'. Must be semver (e.g. 1.2.3)")
        sys.exit(1)

    version = args.version
    print(f"Setting version to {version}" + (f" (build {args.build})" if args.build else ""))
    print()

    set_package_json(version)
    set_spec_file(version)

    print()
    print("Next steps:")
    print(f"  git tag v{version}")
    print(f"  git push origin {args.version} --tags")
    print(f"  scripts\\build_electron.bat")


if __name__ == "__main__":
    main()