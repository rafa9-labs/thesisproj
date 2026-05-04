"""Strip docstrings from Python source files to reduce binary size and obscure logic.

Usage:
    python scripts/strip_docstrings.py [--dry-run]

Walks the project source tree, AST-parses each .py file, removes all
function/class/module docstrings, and writes the modified source back.
Only processes files under pipeline/, api/, models/, news/, rl/, schemas/.

With --dry-run, prints what would be changed without modifying files.
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ["pipeline", "api", "models", "news", "rl", "schemas"]


class DocstringStripper(ast.NodeTransformer):
    def _strip_docstring(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Constant,))
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        return self._strip_docstring(node)

    def visit_FunctionDef(self, node):
        return self._strip_docstring(node)

    def visit_ClassDef(self, node):
        return self._strip_docstring(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def strip_file(filepath: Path, dry_run: bool = False) -> bool:
    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    new_tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_source = ast.unparse(new_tree)
    except Exception:
        return False

    if new_source == source:
        return False

    if not dry_run:
        filepath.write_text(new_source + "\n", encoding="utf-8")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    total = 0
    changed = 0

    for src_dir in SOURCE_DIRS:
        root = PROJECT_ROOT / src_dir
        if not root.is_dir():
            continue
        for py_file in root.rglob("*.py"):
            total += 1
            if strip_file(py_file, dry_run=dry_run):
                changed += 1
                rel = py_file.relative_to(PROJECT_ROOT)
                action = "would strip" if dry_run else "stripped"
                print(f"  {action}: {rel}")

    print(f"\n{changed}/{total} files {'would be ' if dry_run else ''}modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())