# Smoke test (app + exports + empty state)

Use this after `make up` or `make run` to confirm the app and exports work.

1. **Start the app**
   - Run: `make up` (or `docker compose up --build`).
   - Or locally: `make run` (or `streamlit run app/main.py`).
   - Open: http://localhost:8501.

2. **Open the app**
   - Sidebar shows data version (or placeholder). No startup errors in the browser or terminal.

3. **Tab 3 — sample NLQ and export**
   - Go to **NLQ Chat**.
   - Enter a sample question, e.g. **"NNB by channel last 12 months"**.
   - Click **Run**.
   - Confirm a table (and optional chart) appears.
   - Click **Download CSV (current view)** and confirm a CSV file downloads (e.g. `tab3_nlq_result__rows-N__....csv`).

4. **Empty state behaviour**
   - In the sidebar (or Tab 1 filters), set a **date range or filter** that yields no rows (e.g. a future date range or a channel that does not exist).
   - Confirm the UI shows an empty state message (e.g. "No data under current filters" or "No rows returned") and a hint (e.g. "Widen date range or relax filters") instead of an error stack.

Pass: app loads, Tab 3 NLQ returns a result, CSV export downloads, and empty filters show a safe empty state.
