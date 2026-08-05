Portfolio-wide rules live in the root CLAUDE.md. This file covers
ws_Finance_Analysis-specific context only.

---

## Project identity

- **Workspace**: `ws_Finance_Analysis`
- **GitHub repo**: `https://github.com/alisaghilutfi/Fabric-Analytics-Projects`
- **Git folder**: `/ws_Finance_Analysis` (top-level folder in repo)
- **Fabric Git branch**: `dev-fabric-sync` → PR → `main`
- **Fabric account**: `alisaghi_fabric@alisaghi2015gmail.onmicrosoft.com`
- **Pattern**: follows the same conventions as `ws_USGS_Earthquake` in this repo

---

## Architecture

Three-layer medallion. All artifacts in `ws_Finance_Analysis/` folder.

```
Bronze Lakehouse  (lh_Finance_Bronze)  — raw CSV → Delta tables
Silver Lakehouse  (lh_Finance_Silver)  — cleaned star schema tables
Semantic Model    (sm_Finance)         — DirectLake on Silver
Report            (rpt_Finance)        — 4 pages, Layout Trifecta
DataPipeline      (pl_Finance)         — orchestrates Bronze → Silver notebooks
```

### Star schema (Silver)

```
fact_transactions  ←→  dim_customer   (customer_id)
fact_transactions  ←→  dim_date       (transaction_date → Date)
fact_transactions  ←→  dim_channel    (channel)
fact_transactions  ←→  dim_merchant   (merchant_category)
```

---

## Data quality issues to handle in Silver notebook

1. **Duplicate transaction_ids**: 69 exact full-row duplicates → `dropDuplicates()` before write
2. **Dirty `channel` values**: `M@bile App`, leading/trailing spaces → `trim()` + `regexp_replace()`
3. **Dirty `currency` values**: `inr`, `inR` → `upper()` → normalize to `INR`
4. **Negative `amount` values**: 9 rows across Deposit/Transfer/Loan EMI/Card Payment/Bill Payment → flag as `is_reversal = True`, keep in fact
5. **`fee_amount` nulls**: 24 rows → `fillna(0.0)`
6. **Date format**: `dd-MM-yyyy` strings → parse with `to_date(col, 'dd-MM-yyyy')`
7. **1,017 customers with no transactions**: valid — keep in dim_customer, they just have no fact rows
