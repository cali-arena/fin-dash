# Metrics policy

Deterministic policies for rate metrics (guards, inf handling, clamp). Config: `configs/metrics_policy.yml`. Output: `curated/metrics_monthly.parquet` with policy applied; compliance is checked by `pipelines.metrics.metrics_policy_gate`.

---

## Required inputs at grain

Metrics are computed from a single table at grain **(path_id, slice_id, month_end)** with:

| Column      | Role    | Description |
|------------|---------|-------------|
| path_id    | grain   | Drill path id |
| slice_id   | grain   | Slice id within path |
| month_end  | grain   | Month-end date |
| begin_aum  | measure | Beginning AUM (denominator for rates) |
| end_aum    | measure | Ending AUM |
| nnb        | measure | Net new business |
| nnf        | measure | Net new base fees |

- **Grain** must be unique: one row per (path_id, slice_id, month_end).
- **Measures** must be numeric. Missing (NaN) is allowed; division and guards apply as below.

---

## Guard modes

### begin_aum_guard

When **begin_aum ≤ threshold** (default 0), the following rates are **guarded** so we never interpret division-by-low-AUM as meaningful:

- **ogr** (organic growth rate) = nnb / begin_aum  
- **market_impact_rate** = market_pnl / begin_aum  
- **total_growth_rate** = (end_aum − begin_aum) / begin_aum  

**Modes:**

- **nan** (default): set the rate to **NaN** where begin_aum ≤ threshold. Downstream logic and UI can treat NaN as “not applicable.”
- **zero**: set the rate to **0.0** where begin_aum ≤ threshold. Use when you want a numeric value (e.g. for aggregations) instead of missing.

### fee_yield_guard

When **nnb ≤ threshold** (default 0), **fee_yield** = nnf / nnb is guarded:

- **nan**: set fee_yield to **NaN** where nnb ≤ threshold.
- **zero**: set fee_yield to **0.0**.
- **cap**: set fee_yield to **cap_value** (e.g. 0.0 or a configured constant) where nnb ≤ threshold.

---

## Clamp behavior (warn vs hard)

After guards and inf handling, rate metrics can be **clamped** to a [min, max] range per metric (e.g. ogr, market_impact_rate, total_growth_rate, fee_yield).

- **warn_only**  
  - Stored **values are not changed**; they may lie outside [min, max].  
  - A **clamped flag** column (e.g. `ogr_clamped_flag`) is set to **True** wherever the value is outside the cap range.  
  - Use for monitoring and QA without altering numbers.

- **hard_clamp**  
  - Values are **clipped** to [min, max].  
  - The clamped flag is **True** wherever clipping was applied.  
  - Use when downstream systems require values to stay in range.

---

## Why inf → NaN always

Division can produce **±∞** (e.g. positive value / 0 or overflow). Policy treats **inf_handling.mode** as fixed:

- **All ±∞ are converted to NaN** before clamp.

Reasons:

1. **Consistency**: Downstream logic and UI can assume “no inf”; NaN is the single “missing / undefined” signal.
2. **Storage and serialization**: Inf can be problematic in Parquet/JSON and across tools; NaN is well supported.
3. **Guards and clamp**: Guards already force NaN or 0 for low denominators; inf can still appear from other divisions (e.g. before guards). Converting inf to NaN keeps one clear rule: “no inf in output.”

The compliance gate (**metrics_policy_gate**) checks that **no ±inf** remain in metric columns and that recomputed samples match stored values; inf→NaN is part of that contract.
