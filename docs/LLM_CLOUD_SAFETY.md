# LLM / Intelligence Desk — Cloud-Safe Startup and Runtime

## Root cause summary

The Streamlit Cloud **"Oh no. Error running app."** occurs when an uncaught exception happens **before** the app can render. The most likely causes are:

1. **Import-time failure** — One of the modules imported by `app/main.py` (e.g. `app.pages.nlq_chat`, `app.data.data_gateway`, `app.data_contract`, or their transitive imports) raises during import. That can be:
   - Missing or incompatible dependency (e.g. `anthropic`, `openai`, `duckdb`)
   - Code that runs at module level and touches env/secrets or files (e.g. missing manifest)
   - Syntax or import errors in a dependency

2. **No LLM secrets at startup** — The app must not require `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in environment or `st.secrets` to boot. The Intelligence Desk uses **UI-only** API key (session state).

**Files responsible for crash (if LLM-related):**
- `app/main.py` — imports `app.pages.nlq_chat`; if any import in the chain fails, the crash occurs here unless caught.
- `app/pages/nlq_chat.py` — imports `app.services.llm_client`; if `llm_client` raised on import (e.g. missing package), the chain would fail. Now hardened with broad `except Exception` and stub fallbacks.
- `app/services/llm_client.py` — no top-level provider imports; only lazy imports inside `_call_claude` / `_call_openai`. Does not touch env or secrets.

**Non-LLM crash candidates:** `app.data_contract` (missing data file), `app.data_gateway` (missing manifest), or a missing/incompatible system dependency.

---

## Exact files changed (this pass)

| File | Change |
|------|--------|
| `app/pages/nlq_chat.py` | (1) Catch any `Exception` when importing `llm_client` (not just `ImportError`). (2) Wrap entire page in `render()` → try/except → `_render_intelligence_desk()` so uncaught exceptions show inline and never crash the app. |
| `app/services/llm_client.py` | Clearer missing-package messages: "Anthropic/OpenAI package is not installed in this deployment." |
| `docs/LLM_CLOUD_SAFETY.md` | New: audit summary, validation checklist, safeguards. |

---

## Code diffs / inserted blocks

### app/pages/nlq_chat.py

**1. LLM client import — catch any exception:**
```python
# LLM client: optional; must not crash app if missing or broken (cloud-safe)
try:
    from app.services.llm_client import LLMError, generate_data_narrative, generate_market_intelligence
except Exception:
    generate_market_intelligence = None
    generate_data_narrative = None
    LLMError = Exception
```

**2. render() wrapper — never crash app:**
```python
def render(state: FilterState, contract: dict[str, Any]) -> None:
    """Intelligence Desk: mode, presets, input, single response area. Never crashes app."""
    try:
        _render_intelligence_desk(state, contract)
    except Exception as e:
        st.error("Intelligence Desk encountered an error. Other tabs are unaffected.")
        st.exception(e)

def _render_intelligence_desk(state: FilterState, contract: dict[str, Any]) -> None:
    """Intelligence Desk UI implementation."""
    ...
```

### app/services/llm_client.py

**ImportError messages:**
- `"Anthropic SDK not installed. Add 'anthropic' to requirements."` → `"Anthropic package is not installed in this deployment."`
- `"OpenAI SDK not installed. Add 'openai' to requirements."` → `"OpenAI package is not installed in this deployment."`

---

## Dependency changes

- **requirements.txt** — No change. Already has `anthropic>=0.39` and `openai>=1.0`.
- If a deployment omits one package, the app still boots; when the user selects that provider and clicks Generate, they see the clear "package is not installed in this deployment" message.

---

## Local / cloud simulation (checks performed)

| Check | Result |
|-------|--------|
| No env secrets | Intelligence Desk does not read `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` at startup. Key only from UI → session state. |
| No user key entered | Market Intelligence path checks `api_key` and `model`; shows inline warning and does not call provider. |
| Invalid key | Provider call wrapped in try/except; `LLMError` or generic exception → `_set_nlq_response(..., error=...)` → shown in result area. |
| Missing SDK package | `_call_claude` / `_call_openai` do lazy `from anthropic` / `from openai`; on `ImportError` raise `LLMError` with deployment message. |
| Successful request path | When provider, model, key, and prompt are set and user clicks Generate, `generate_market_intelligence()` is called; result and provider_meta shown. |
| Page load without clicking | Session state inits (provider, model, key); no LLM call until "Generate response" is clicked. |
| Switch to Market Intelligence without key | Caption shows "Setup required"; on Generate we show "LLM settings are required..." and do not call provider. |

---

## Final manual validation checklist

- [ ] **App boots in Streamlit Cloud** with no LLM secrets in env or Cloud secrets.
- [ ] **Intelligence Desk tab** opens without error.
- [ ] **Data Questions mode** still works (governed query, verified result, optional narrative if key in session).
- [ ] **Market Intelligence with no key** — show warning in result area, no crash.
- [ ] **Invalid key** — show clean error in result area (e.g. "API request failed: check your API key").
- [ ] **Valid key** — returns response and shows provider/model in caption.
- [ ] **Switching providers** (Claude ↔ OpenAI) works; model list and Apply update session state.
- [ ] **Repeated queries** update the result panel correctly.
- [ ] **Other tabs** (Executive Dashboard, Investment Commentary) unaffected by Intelligence Desk errors.

---

## Final safeguard note — remaining risks

1. **Import failure before our try/except** — If an exception is raised during `import streamlit as st` or in a module loaded before the `try` block in `main.py`, Streamlit may still show the generic "Error running app." We already wrapped all `app.*` imports in that try block so the first failure in the import chain should be caught and displayed.
2. **Data contract / missing files** — If `get_data_contract_cached()` raises (e.g. missing `data/agg/firm_monthly.parquet` or `analytics.duckdb`), that is now caught in `main()` and shown as a clear message; if the failure is during import of a module that itself loads data at import time (not the case in current code), it would be caught by the main import try/except.
3. **Streamlit Cloud runner** — In some environments the runner might catch exceptions before our handlers; checking the **Streamlit Cloud → Logs** for the app is required to see the real traceback if the UI still shows "Oh no."

---

## Architecture after fix

1. **Cloud-safe startup** — No provider client created at startup; no secret read at startup; optional `llm_client` import with stub fallbacks.
2. **UI-only API key** — Provider, model, API key in LLM settings; stored only in `st.session_state`.
3. **Lazy provider adapter** — `app/services/llm_client.py`: `generate_market_intelligence()` and `generate_data_narrative()`; provider imports and client creation only inside the request path.
4. **Button-gated execution** — Market Intelligence runs only when mode is Market Intelligence, provider/model/key set, prompt non-empty, and user clicked "Generate response."
5. **Graceful errors** — All failures surface as `st.error` / `st.warning` in the result area; page-level try/except ensures Intelligence Desk never crashes the app.
