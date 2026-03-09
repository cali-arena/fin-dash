"""
CI/QA: Ensure .py, .yml, .yaml, .md, .json files are valid UTF-8.

Walks the repository and attempts to decode each matching file as UTF-8.
Reports any file that fails decode with (path, byte_offset, error).
Exit 0 if all clean; exit 1 and print offending files otherwise.

Also checks for weird characters in Python source that can break parsing
(e.g. invisible Unicode, wrong quotes, replacement char U+FFFD).
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Repo root = parent of qa/
ROOT = Path(__file__).resolve().parents[1]

# Extensions to validate as UTF-8
TEXT_EXTENSIONS = {".py", ".yml", ".yaml", ".md", ".json"}

# Directories to skip (binary or external)
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
}

# Characters that are problematic in Python source (can break parsing or display)
# U+FFFD = replacement char; U+200B = zero-width space; U+FEFF = BOM (ok at start)
PROBLEMATIC_IN_PY = (
    "\ufffd",  # replacement character
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM only bad in middle of file
)


def should_skip(path: Path) -> bool:
    if path.name.startswith(".") and path.name != ".yml":
        return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def check_utf8(path: Path) -> list[tuple[int, str]]:
    """Attempt to read file as UTF-8. Return list of (byte_offset, error_msg) or empty if ok."""
    errors: list[tuple[int, str]] = []
    try:
        raw = path.read_bytes()
    except OSError as e:
        return [(0, str(e))]
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as e:
        errors.append((e.start, f"invalid UTF-8: {e.reason} at byte {e.start}"))
    return errors


def check_python_weird_chars(path: Path, content: str) -> list[tuple[int, str]]:
    """Check for problematic invisible/weird chars in Python source. Returns (1-based line, msg)."""
    errors: list[tuple[int, str]] = []
    for i, line in enumerate(content.splitlines(), start=1):
        for j, char in enumerate(line):
            if char in PROBLEMATIC_IN_PY:
                name = repr(char)
                if char == "\ufeff" and j == 0:
                    continue  # BOM at line start is often intentional
                if char == "\ufeff":
                    errors.append((i, f"BOM in middle of line (col {j + 1})"))
                else:
                    errors.append((i, f"problematic character {name} at column {j + 1}"))
    # If file has content, try parsing to catch any remaining parse-breaking chars
    if content.strip():
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append((e.lineno or 0, f"SyntaxError: {e.msg}"))
    return errors


def rewrite_as_utf8(path: Path) -> bool:
    """Try to read as latin-1 and write as UTF-8. Return True if rewritten."""
    try:
        raw = path.read_bytes()
        raw.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    try:
        text = path.read_bytes().decode("latin-1")
        path.write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check (and optionally fix) UTF-8 encoding of text files.")
    parser.add_argument("--fix", action="store_true", help="Rewrite non-UTF-8 files as UTF-8 (latin-1 decode)")
    args = parser.parse_args(argv)
    fix_mode = getattr(args, "fix", False)

    violations: list[tuple[Path, list[tuple[int, str]]]] = []
    fixed: list[Path] = []
    for ext in TEXT_EXTENSIONS:
        for path in sorted(ROOT.rglob(f"*{ext}")):
            if should_skip(path) or not path.is_file():
                continue
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                continue
            utf8_errors = check_utf8(path)
            if utf8_errors and fix_mode and ext in {".py", ".yml", ".yaml", ".md", ".json"}:
                if rewrite_as_utf8(path):
                    fixed.append(rel)
                    continue
            if utf8_errors:
                violations.append((rel, utf8_errors))
                continue
            if ext == ".py":
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                weird = check_python_weird_chars(path, text)
                if weird:
                    violations.append((rel, weird))

    if fixed:
        for rel in fixed:
            print(f"Fixed (rewritten as UTF-8): {rel}")
    if not violations:
        return 0
    print("ERROR: The following files have encoding or problematic character issues:", file=sys.stderr)
    for rel, errs in violations:
        print(f"  {rel}", file=sys.stderr)
        for offset_or_line, msg in errs:
            print(f"    -> {offset_or_line}: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
