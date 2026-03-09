# Dataset version (fingerprint)

Deterministic fingerprint for caching and reproducibility: same inputs + same pipeline ⇒ same `dataset_version` ⇒ same numbers every run.

---

## Inputs

1. **Raw bytes of source-of-truth file(s)** — see [Canonical input bytes](#canonical-input-bytes) below.
2. **`pipeline_version`** — string identifying pipeline code/config (e.g. git SHA, version tag, or build id). Must change whenever pipeline logic or config changes.

## Output

```text
dataset_version = sha256(hex(input_bytes) + "\n" + pipeline_version)
```

- `input_bytes`: canonical byte sequence (see below).
- Result is a 64-character lowercase hex string (SHA-256).

**Copy-pastable definition:**

- Compute `input_bytes` (concatenation of file bytes in fixed order).
- Compute `content = hexlify(input_bytes).decode("ascii") + "\n" + pipeline_version`.
- Compute `dataset_version = hashlib.sha256(content.encode("utf-8")).hexdigest()`.

## Rules

- Any change in **input file bytes** (add/remove/rewrite byte in any source file) must change `dataset_version`.
- Any change in **pipeline code or config** must change `pipeline_version`, hence `dataset_version`.
- Order of files and encoding of the combined content must be fixed and documented so the fingerprint is deterministic and reproducible.

## Where stored

- **Path:** `data/.version.json`
- **Contents (example):**

```json
{
  "dataset_version": "<64-char hex>",
  "pipeline_version": "<string>",
  "input_files": ["DATA_RAW.csv", "DATA_SUMMARY.csv", "DATA_MAPPING.csv", "ETF.csv", "EXECUTIVE_SUMMARY.csv"],
  "computed_at": "<ISO8601 UTC timestamp>"
}
```

## How used

- **Cache keys:** Use `dataset_version` (and optionally `pipeline_version`) as part of cache keys so caches invalidate when inputs or pipeline change.
- **Run metadata:** Persist `dataset_version` and `pipeline_version` with runs for audit and reproducibility.
- **Reproducibility:** Same `dataset_version` + same pipeline run ⇒ same outputs; “same numbers every run” for that version.

---

## Canonical input bytes (current: CSV inputs)

Source of truth: `data/input/*.csv` per `docs/data_contract.md`.

**Canonical `input_bytes`** = concatenation of **raw file bytes** of the required CSVs in this **fixed order**:

1. `DATA_RAW.csv`
2. `DATA_SUMMARY.csv`
3. `DATA_MAPPING.csv`
4. `ETF.csv`
5. `EXECUTIVE_SUMMARY.csv`

- Read each file in binary mode and append its bytes in order. No normalisation (no strip, no re-encoding). Missing file ⇒ fingerprint computation fails (do not run).
- Order is part of the contract; changing it changes `dataset_version`.

---

## If we switch back to a single .xlsx

If the source of truth becomes **one .xlsx file** instead of multiple CSVs:

- **Canonical `input_bytes`** = raw bytes of that single `.xlsx` file (whole file, read in binary mode).
- Keep the same formula: `dataset_version = sha256(hex(input_bytes) + "\n" + pipeline_version)`.
- Update `data/.version.json` and any code that builds `input_bytes` to use the xlsx path and single-file logic. Document the change in this file and in the data contract.
