# Finance Dashboard

Streamlit dashboard for asset management and distribution analytics. All data access goes through `app.data.data_gateway` (re-export of `app.data_gateway`).

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Or: `make run`. Open `http://localhost:8501`.

## Deploy (Streamlit Cloud)

- **Main file path:** `app/main.py`
- **Root:** Run from repo root; all paths are relative (no Windows-specific paths).
- **Canonical deps:** `requirements.txt` (Streamlit Cloud uses this; keep `anthropic` here for Claude).
- **Claude:** Uses Streamlit secrets (`ANTHROPIC_API_KEY`); no UI key entry. Redeploy or reboot app after changing secrets.
- See **docs/DEPLOY.md** for structure, checklist, and validation.

## Guardrail: No Direct Data Access

Pages must not use DuckDB or Parquet directly. Run before dev/CI:

```bash
python tools/guardrails/check_no_direct_data_access.py
```

Or: `make guardrail`

- Exit 0: clean
- Exit 2: violations found (file, line, snippet, fix message)

With `DEV_MODE=1`, the checker runs at Streamlit startup and stops the app if violations exist.

## QA: No legacy imports in canonical code

`app/`, `etl/`, and `models/` must not import from `legacy`, `src`, or `pipelines`. Run:

```bash
python qa/check_no_legacy_imports.py

streamlit run app/main.py
```

Exit 1 if any file in those directories has a forbidden import; exit 0 if clean. Use this in CI to prevent import drift.

## QA: Encoding (UTF-8 for Docker/Linux)

All `.py`, `.yml`, `.yaml`, `.md`, and `.json` files must be valid UTF-8 with no problematic characters. Run:

```bash
python qa/check_encoding.py
```

- Exit 0: all checked files are valid UTF-8 and (for `.py`) parse cleanly with no invisible/weird characters.
- Exit 1: reports each offending file with byte offset or line number and reason (invalid UTF-8, or problematic character).

Use this in CI (e.g. before build) so images and Linux runs don’t hit encoding errors.

## Repository

- **.gitignore** at root excludes Python cache, venvs, local `analytics.duckdb`, large parquet outputs, and secrets so the repo stays clean for production and deploy.
