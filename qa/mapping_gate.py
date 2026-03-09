"""
Quality gate: unmapped mapping ratio. Fail QA if unmapped ratio > 1%.
Deterministic: reads qa/unmapped_keys.meta.json (written by etl/transform_curated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UNMAPPED_RATIO_THRESHOLD = 0.01  # 1%


def run(qa_dir: Path) -> tuple[bool, str]:
    meta_path = qa_dir / "unmapped_keys.meta.json"
    if not meta_path.exists():
        return True, "No unmapped_keys.meta.json (run ETL transform first); gate skipped."
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    total = int(meta.get("total_raw_rows", 0))
    unmapped_rows = int(meta.get("unmapped_rows", 0))
    unmapped_keys = int(meta.get("unmapped_keys", 0))
    ratio = unmapped_rows / max(total, 1)
    message = f"Unmapped keys: {unmapped_keys}, rows: {unmapped_rows} / {total} ({ratio:.2%})"
    if ratio > UNMAPPED_RATIO_THRESHOLD:
        return False, f"QA FAIL: unmapped ratio {ratio:.2%} > {UNMAPPED_RATIO_THRESHOLD:.0%}. {message}"
    return True, message


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality gate: fail if unmapped ratio > 1%")
    parser.add_argument("--qa-dir", default="qa", type=Path)
    args = parser.parse_args()
    ok, msg = run(args.qa_dir)
    print(msg)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
