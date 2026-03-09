"""
Optional git info for versioning. Must not crash; returns None on failure.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_GIT_CWD = Path(__file__).resolve().parent.parent.parent
_TIMEOUT = 2


def get_git_sha() -> str | None:
    """
    Run git rev-parse HEAD. Return stripped sha or None if unavailable.
    Catches FileNotFoundError, CalledProcessError, TimeoutExpired.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=_GIT_CWD,
        )
        if out.returncode == 0 and out.stdout:
            sha = out.stdout.strip()
            if len(sha) == 40:
                return sha
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def is_git_dirty() -> bool | None:
    """
    Run git status --porcelain. True if non-empty, False if empty, None if git unavailable.
    Must not crash.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=_GIT_CWD,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
