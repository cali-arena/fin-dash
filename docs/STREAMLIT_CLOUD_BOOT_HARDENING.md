# Streamlit Cloud Boot — Final Hardening Summary

## 1. Search results (entire repo)

| Pattern | Location | Runs at import? |
|--------|-----------|------------------|
| **OPENAI_API_KEY** | Not present in app | N/A |
| **ANTHROPIC_API_KEY** | `app/config/llm_config.py` (env + st.secrets in `get_anthropic_api_key`) | No — only when function is **called** |
| | `app/llm/claude_client.py` (env, st.secrets, constant) | No — module not loaded at startup |
| | `tests/` (monkeypatch) | No — tests only |
| **st.secrets** | `app/config/llm_config.py` | No — inside `get_anthropic_api_key()` |
| | `app/observability/debug_panel.py`, `app/ui/observability.py`, `app/perf_budget.py`, `app/guardrails/no_adhoc_agg.py` | No — inside functions; all wrapped in try/except |
| | `app/llm/claude_client.py` | No — module not loaded at startup |
| **os.getenv / os.environ.get** | Various (DEV_MODE, DUCKDB_PATH, APP_DATA_BACKEND, etc.) | Yes — but only for config flags; no LLM key read at startup |
| **OpenAI(** | `app/services/llm_client.py` line 79 | No — inside `_call_openai()` only |
| **Anthropic(** | `app/services/llm_client.py` line 59; `app/llm/claude.py` 25; `app/llm/claude_client.py` 62 | No — all inside request-time functions |
| **import anthropic / from anthropic** | `app/services/llm_client.py` (inside `_call_claude`); `app/llm/claude.py`; `app/llm/claude_client.py` | No — llm_client is lazy; app.llm not loaded at startup |
| **from openai import OpenAI** | `app/services/llm_client.py` (inside `_call_openai`) | No — lazy inside function |

## 2. What runs at import time (startup chain)

- **main.py** imports: `app.config.contract`, `app.config.tab1_defaults`, `app.data_contract`, `app.data.data_gateway`, `app.pages.dynamic_report`, `app.pages.nlq_chat`, `app.pages.visualisations`, `app.state`, `app.ui.filters`, `app.ui.theme`.
- **app.pages.nlq_chat** imports: `app.data.data_gateway`, `app.nlq.*`, `app.state`, `app.ui.*`, and **app.services.llm_client** (with `except Exception` → stubs if it fails).
- **app.services.llm_client** top-level: only `json`, `logging`, `typing` — no provider imports, no env, no secrets.
- **app.llm** (claude, claude_client, llm_config) is **not** imported by main or by nlq_chat. Only loaded lazily via `app.llm.__getattr__` if something does `from app.llm import ...` (no startup path does).

So: **no LLM key, no provider client, and no provider package is used or read at import/startup.**

## 3. Remaining “risky” lines (safe because not on startup path)

| File | Line / pattern | Why it does not affect boot |
|------|----------------|-----------------------------|
| `app/config/llm_config.py` | `os.environ.get("ANTHROPIC_API_KEY")`, `_st.secrets.get("ANTHROPIC_API_KEY")` | Module not imported at startup; only used by app.llm.claude when explicitly called. |
| `app/llm/claude_client.py` | Same + `ANTHROPIC_API_KEY` constant | app.llm.claude_client not imported at startup (app.llm is lazy). |
| `app/llm/claude.py` | `get_anthropic_api_key()`, `from anthropic import Anthropic` | app.llm.claude not imported at startup. |
| `app/services/llm_client.py` | `from anthropic import Anthropic`, `from openai import OpenAI`, `Anthropic(api_key=...)`, `OpenAI(api_key=...)` | All inside `_call_claude` / `_call_openai`; only run when user clicks Generate with key. |

No changes required to these for boot safety; they are already isolated from startup.

## 4. Startup-time dependency for LLM — removed / isolated

- **Removed:** None (there was no startup-time LLM dependency in the main/nlq_chat chain).
- **Isolated:** All LLM usage is behind:
  - **app.services.llm_client** — only stdlib at import; provider code and clients only inside request-time functions.
  - **app.llm** — not imported by main or pages; lazy `__getattr__` so claude/claude_client/llm_config load only when `app.llm` is accessed (no startup access).

## 5. Boot with no LLM key

- **main** and **nlq_chat** do not import **app.llm** or **app.config.llm_config**.
- **llm_client** does not read env or secrets; it only receives `api_key` from the caller (session state).
- Session state for LLM is initialized with empty key; provider is only called after the user enters a key and clicks Generate.

So the app **can boot with no LLM key configured anywhere** (no env, no secrets, no disk).

## 6. Missing optional dependency → UI error only after submit

- In **app.services.llm_client**, `_call_claude` / `_call_openai` do `from anthropic import Anthropic` / `from openai import OpenAI` **inside** the function.
- On `ImportError` they raise `LLMError("... package is not installed in this deployment.")`.
- That is caught in the Market Intelligence path and shown via `_set_nlq_response(..., error=e.message)` in the result area.
- So a missing optional dependency leads to a **UI error only after the user submits** (clicks Generate with that provider); it does not affect boot or other tabs.

## 7. No top-level page code crash from Market Intelligence

- **nlq_chat.render()** is wrapped in try/except; any uncaught exception calls `st.exception(e)` and does not re-raise.
- LLM session state (provider, model, key) is initialized at the start of the page; no provider or key read at module level.
- Market Intelligence logic runs only when `run_clicked` is True and `route_to_market` is True; then we validate key/model and call `generate_market_intelligence` inside try/except and set error in the response.
- So **enabling Market Intelligence or opening the tab does not run any code that can crash the app**; only the Generate path can fail, and failures are contained to the result panel.

---

## Final cleaned files (changes made in this pass)

1. **app/config/llm_config.py** — Docstring updated to state: do not import from main or startup pages; Intelligence Desk uses llm_client only; `get_anthropic_api_key()` must not raise.
2. **app/main.py** — Comment added above the try block: "All app imports below: any failure is caught so Cloud shows real error instead of 'Oh no.'"
3. **docs/STREAMLIT_CLOUD_BOOT_HARDENING.md** — This file (audit and summary).

No code paths were removed; only comments and documentation were added to lock the intended boot contract.

---

## Exact reason the Cloud boot should now succeed

1. **No LLM on startup path**  
   main.py and the page modules it imports (including nlq_chat) do not import app.llm or app.config.llm_config. The only LLM-related import is app.services.llm_client, which at import time only uses stdlib (json, logging, typing). So no provider SDK, no env key, and no st.secrets are used at import.

2. **Import failure is visible**  
   All app imports in main.py are inside a single try/except. Any import failure (e.g. missing dependency, broken module) is caught and shown with `st.error` and traceback instead of the generic “Oh no. Error running app.”

3. **Intelligence Desk cannot kill the app**  
   The nlq_chat page’s render() wraps the real implementation in try/except and never re-raises. So any uncaught error on that tab only fills the result area and does not crash the process.

4. **LLM key is UI-only**  
   The only place that uses an API key for Market Intelligence is the llm_client, which receives it as an argument from session state. Nothing at startup reads OPENAI_API_KEY or ANTHROPIC_API_KEY.

5. **Missing SDK is request-time only**  
   If anthropic or openai is missing, the failure happens only when the user clicks Generate with that provider, and the error is shown in the UI; boot is unaffected.

Together, this gives a **single, clear boot path with no LLM or secret dependency**, **visible errors if something in the import chain fails**, and **no crash from the Intelligence Desk or Market Intelligence** even when enabled or used.
