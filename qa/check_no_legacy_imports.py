"""
CI/QA: Fail if app/, etl/, or models/ import from legacy or old module names.

Forbidden in canonical code:
- from legacy. / import legacy
- from src. / import src
- from pipelines. / import pipelines

Exit 0 if clean; exit 1 and print offending lines otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Canonical roots to check (repo root = parent of qa/)
ROOT = Path(__file__).resolve().parents[1]
DIRS = ["app", "etl", "models"]

# Lines that indicate a forbidden import (excluding comments and strings)
PATTERN = re.compile(
    r"^\s*(from\s+(legacy\.|src\.|pipelines\.)|import\s+(legacy|src|pipelines)\b)",
    re.MULTILINE,
)


def main() -> int:
    violations: list[tuple[str, int, str]] = []
    for dir_name in DIRS:
        d = ROOT / dir_name
        if not d.is_dir():
            continue
        for py in d.rglob("*.py"):
            try:
                text = py.read_text(encoding="utf-8")
            except Exception:
                continue
            rel = py.relative_to(ROOT)
            for m in PATTERN.finditer(text):
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.start())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end].strip()
                # Skip if whole line is comment
                if line.startswith("#"):
                    continue
                line_no = text[: m.start()].count("\n") + 1
                violations.append((str(rel), line_no, line))

    if not violations:
        return 0
    print("ERROR: app/, etl/, or models/ must not import from legacy, src, or pipelines.", file=sys.stderr)
    for path, line_no, line in violations:
        print(f"  {path}:{line_no}: {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
