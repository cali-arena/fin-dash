# Star schema contract — dashboard model

**Contract version**: 2026-03-03  
**Scope**: Authoritative grain, dimension keys, join rules, and data quality expectations for the dashboard star model. Any change that alters grain, join keys, or uniqueness guarantees is a breaking change and must be versioned.

---

## 1) Objective

- **Fact**: The curated fact table at **monthly grain** is the single source for all dashboard KPIs. No dashboard metric is computed from any table other than this fact (and dimensions joined to it).
- **Dimensions**: Dimension tables supply **stable join keys** and **human-readable attributes**. They are unique on their natural keys. The dashboard joins fact to dimensions only via the keys defined in this contract.

---

## 2) Fact grain (authoritative)

**Table**: `fact_monthly` (curated).

**Grain** — one row per:

| Grain component    | Type   | Description                          |
|--------------------|--------|--------------------------------------|
| `month_end`        | date   | Month-end date                       |
| `product_ticker`   | string | Product/ETF ticker                   |
| `preferred_label`  | string | Channel display label (from mapping) |
| `src_country`      | string | Source country                       |
| `product_country`  | string | Product country                      |
| `segment`          | string | Segment                              |
| `sub_segment`      | string | Sub-segment                          |

**Definition**:

```
fact_monthly grain =
  month_end + product_ticker + preferred_label + src_country + product_country + segment + sub_segment
```

This grain is **authoritative**. All downstream models and dashboards must treat it as the only valid grain for fact_monthly. Measures (e.g. asset_under_management, net_new_business, net_new_base_fees) are aggregated to this grain.

---

## 3) Dimension keys (natural vs surrogate)

| Dimension    | Natural key              | Type   | Surrogate key (optional) | Notes |
|-------------|---------------------------|--------|---------------------------|--------|
| **dim_time**    | `month_end`               | date   | `time_id` (int) optional  | Not required for this dashboard. |
| **dim_channel** | `preferred_label`        | string | `channel_id` optional     | Stable hash of preferred_label if used. |
| **dim_product** | `product_ticker`          | string | `product_id` optional     | Stable hash of product_ticker if used. |
| **dim_geo**     | `country_key` (or equivalent) | string | —                     | Single geography dimension. See role-playing below. |

**Geo and role-playing**:

- The fact has two country attributes: `src_country` and `product_country`.
- Either:
  - **Option A**: One `dim_geo` with natural key `country_key` (or normalized `country_code` / `country_name`). The fact joins twice to the same dimension: once for source (`fact_monthly.src_country -> dim_geo.country_key`), once for product (`fact_monthly.product_country -> dim_geo.country_key`). Role is implied by which fact column is used in the join.
  - **Option B**: Two dimension tables (e.g. `dim_src_geo`, `dim_product_geo`) with the same structure, each keyed by the same natural key. Both options are valid; the contract only requires that join rules (Section 5) be followed consistently.

---

## 4) SCD stance

- **Default**: **SCD Type 1** (overwrite). Dimension attributes are updated in place; no history is kept. This is the stance for the current dashboard.
- **Type 2** is **not** in scope for the current contract. The following are documented as **future Type 2 triggers** if history is required later:
  - **Segment / sub_segment reclassification**: Historical reclassification of a product or channel into a different segment or sub_segment.
  - **Channel hierarchy changes**: Changes to channel_l1 / channel_l2 or preferred_label that must be preserved with effective dates.
  - **Product or geography attribute changes**: Changes to product or country attributes that reporting must track over time.

If Type 2 is adopted, this contract will be versioned and updated to define effective/expiry dates and join rules for point-in-time correctness.

---

## 5) Join rules (must be consistent)

All fact-to-dimension joins use **natural keys** unless a surrogate is explicitly adopted. The following rules are **mandatory** and must be implemented consistently by any consumer (dashboards, semantic layer, BI tool).

| Fact column           | Dimension       | Dimension key column | Role (if applicable) |
|-----------------------|-----------------|----------------------|------------------------|
| `fact_monthly.month_end`       | dim_time        | month_end            | —                      |
| `fact_monthly.preferred_label` | dim_channel     | preferred_label      | —                      |
| `fact_monthly.product_ticker`  | dim_product     | product_ticker       | —                      |
| `fact_monthly.src_country`     | dim_geo         | country_key          | source                 |
| `fact_monthly.product_country` | dim_geo         | country_key          | product                |

- Joins are **equi-joins** on the keys above. No optional/outer join is required for the fact grain keys; dimensions must contain keys present in the fact for the chosen scope (or the contract must explicitly allow missing dimension keys and define behavior).
- Role-playing for geo: the same dimension table is joined twice to the fact using `src_country` and `product_country` respectively; role (source vs product) is determined by the join, not by an extra column in the fact.

---

## 6) Data quality expectations

The following are **contractual** and must be enforced (by pipeline validation or downstream checks):

| Rule | Scope | Requirement |
|------|--------|--------------|
| **No nulls in fact grain** | fact_monthly | No null in any of: month_end, product_ticker, preferred_label, src_country, product_country, segment, sub_segment. |
| **Uniqueness of fact grain** | fact_monthly | At most one row per (month_end, product_ticker, preferred_label, src_country, product_country, segment, sub_segment). |
| **Dimension uniqueness** | dim_time, dim_channel, dim_product, dim_geo | Each dimension is unique on its natural key (month_end, preferred_label, product_ticker, country_key respectively). |

Violations of these expectations must be reported (e.g. in pipeline gates or meta) and resolved before the dataset is considered valid for the dashboard.

---

*End of star schema contract.*
