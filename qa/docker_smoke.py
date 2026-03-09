"""
Docker sanity smoke: bring up app with compose, wait for HTTP, check DB file, then tear down.

- Runs: docker compose up --build -d
- Polls http://localhost:8501 for up to 60s
- Checks /workspace/analytics.duckdb exists inside app container
- On failure: prints last 200 lines of app logs
- Always: docker compose down

Exit 0 on success, 1 on failure. Cross-platform (Python).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

try:
    import urllib.request
except ImportError:
    urllib.request = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_CMD = ["docker", "compose"]
SERVICE = "app"
URL = "http://localhost:8501"
POLL_SECONDS = 60
POLL_INTERVAL = 2
LOG_TAIL = 200
DUCKDB_PATH_IN_CONTAINER = "/workspace/analytics.duckdb"


def run(cmd: list[str], cwd: Path | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        capture_output=capture,
        text=True,
        timeout=300,
    )


def compose_up() -> bool:
    print("Running: docker compose up --build -d")
    r = run(COMPOSE_CMD + ["up", "--build", "-d"])
    if r.returncode != 0:
        print(r.stderr or r.stdout or "compose up failed")
        return False
    return True


def wait_for_http() -> bool:
    if urllib.request is None:
        print("Warning: urllib.request not available, skipping HTTP check")
        return True
    deadline = time.monotonic() + POLL_SECONDS
    while time.monotonic() < deadline:
        try:
            req = urllib.request.urlopen(URL, timeout=5)
            req.read()
            req.close()
            print(f"  {URL} responded OK")
            return True
        except Exception:
            time.sleep(POLL_INTERVAL)
    print(f"  Timeout: {URL} did not respond within {POLL_SECONDS}s")
    return False


def check_duckdb_exists() -> bool:
    # exec: test -f /workspace/analytics.duckdb
    r = run(COMPOSE_CMD + ["exec", "-T", SERVICE, "test", "-f", DUCKDB_PATH_IN_CONTAINER])
    if r.returncode != 0:
        print(f"  {DUCKDB_PATH_IN_CONTAINER} missing or not a file in container")
        return False
    print(f"  {DUCKDB_PATH_IN_CONTAINER} exists in container")
    return True


def print_logs() -> None:
    print(f"\n--- Last {LOG_TAIL} lines of app logs ---")
    r = run(COMPOSE_CMD + ["logs", f"--tail={LOG_TAIL}", SERVICE], capture=False)
    if r.returncode != 0:
        print("(could not get logs)")


def compose_down() -> None:
    print("Running: docker compose down")
    run(COMPOSE_CMD + ["down"], capture=False)


def main() -> int:
    if run(COMPOSE_CMD + ["version"], capture=True).returncode != 0:
        print("docker compose not available or failed", file=sys.stderr)
        return 1
    try:
        if not compose_up():
            print_logs()
            return 1
        if not wait_for_http():
            print_logs()
            return 1
        if not check_duckdb_exists():
            print_logs()
            return 1
        print("Docker smoke: OK")
        return 0
    finally:
        compose_down()


if __name__ == "__main__":
    raise SystemExit(main())
