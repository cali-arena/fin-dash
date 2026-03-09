# Validation Triage Summary

Source: `qa/validation_report.csv` (regenerated from `python qa/validate_vs_data_summary.py`)

---

## Counts

| Item | Value |
|------|--------|
| Total rows | 20 |
| Flagged rows (`any_fail` = True) | 0 |

---

## Counts by fail_reason

| fail_reason | Count | Acceptable? |
|-------------|--------|-------------|
| *(empty — pass)* | 19 | — |
| SKIP_INCOMPLETE_COVERAGE | 1 | Yes |

---

## Remaining fails (acceptable SKIPs only)

After fixes, remaining non-pass rows are **only acceptable SKIPs**:

- **SKIP_INCOMPLETE_COVERAGE (1):** Summary has a month_end with no matching row in actual (firm) data; coverage incomplete. Label: *Month missing from actual data; coverage incomplete. Widen date range or ensure data for all months.*

No **FORMULA_MISMATCH**, **MISSING_DATA** (first-month/zero begin_aum), or other non-SKIP failures.  
*MISSING_DATA* and *SKIP_INCOMPLETE_COVERAGE* are the only acceptable categories for “expected” remaining rows when present.

---

## How to regenerate

From repo root (with `PYTHONPATH` or `pip install -e .`):

```bash
python qa/validate_vs_data_summary.py --curated-dir data/curated --qa-dir qa
```

Exit 0 = validation passed (no unacceptable failures). Exit 2 = one or more rows above threshold (e.g. FORMULA_MISMATCH).
