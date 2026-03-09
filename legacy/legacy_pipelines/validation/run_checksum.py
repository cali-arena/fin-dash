"""
Top-level validation pipeline runner: contract check (Step 1) then firm-level recompute (Step 2).
Later steps will compare against Excel. Reproducible caching via data/cache/{dataset_version}/qa/.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    """Run contract_check then recompute_firm_level. Exit on first failure."""
    root = Path.cwd()
    logger.info("run_checksum: starting (root=%s)", root)

    # Step 1: contract_check
    r1 = subprocess.run(
        [sys.executable, "-m", "pipelines.validation.contract_check", "--root", str(root)],
        cwd=str(root),
        capture_output=False,
    )
    if r1.returncode != 0:
        logger.error("contract_check failed with exit code %s", r1.returncode)
        return r1.returncode
    logger.info("contract_check passed")

    # Step 2: recompute_firm_level
    r2 = subprocess.run(
        [sys.executable, "-m", "pipelines.validation.recompute_firm_level", "--root", str(root)],
        cwd=str(root),
        capture_output=False,
    )
    if r2.returncode != 0:
        logger.error("recompute_firm_level failed with exit code %s", r2.returncode)
        return r2.returncode
    logger.info("recompute_firm_level passed")

    logger.info("run_checksum: done")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
