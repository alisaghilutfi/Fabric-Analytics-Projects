# CLAUDE.md — ws_Finance_Analysis

This file is read automatically by Claude Code in every session.
Do not remove or rename it.

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

## Naming conventions

| Artifact type    | Prefix  | Example                    |
|------------------|---------|----------------------------|
| Lakehouse        | `lh_`   | `lh_Finance_Bronze`        |
| Notebook         | `nb_`   | `nb_Finance_Bronze`        |
| Semantic model   | `sm_`   | `sm_Finance`               |
| Report           | `rpt_`  | `rpt_Finance`              |
| DataPipeline     | `pl_`   | `pl_Finance`               |
| Measures table   | `_`     | `_Measures`                |

---

## Fabric artifact format rules (CRITICAL)

### Notebook files
Every notebook needs TWO files:

**`.platform`** (metadata):
```json
{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
  "metadata": {
    "type": "Notebook",
    "displayName": "nb_Finance_Bronze"
  },
  "config": {
    "version": "2.0",
    "logicalId": "<UUID>"
  }
}
```

**`notebook-content.py`** (content):
```python
# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "<lakehouse_id>",
# META       "default_lakehouse_name": "lh_Finance_Bronze",
# META       "default_lakehouse_workspace_id": "<workspace_id>",
# META       "known_lakehouses": [{"id": "<lakehouse_id>"}]
# META     }
# META   }
# META }

# CELL ********************
<code here>

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
```

### Lakehouse files
Every lakehouse needs `.platform`, `lakehouse.metadata.json`, `shortcuts.metadata.json`, `alm.settings.json`.

### TMDL relationship syntax (CRITICAL — do NOT use bracket notation)
```
relationship <name>
    fromColumn: fact_transactions.customer_id
    toColumn: dim_customer.customer_id
```
NOT `fromColumn: fact_transactions[customer_id]` — that will break import.

### PBIR report JSON
The `config` field in report visuals is **escaped JSON string** (JSON inside JSON):
```json
"config": "{\"version\":\"5.1.0\",\"singleVisual\":{\"visualType\":\"card\",...}}"
```
Every quote inside config is backslash-escaped. Never write raw nested JSON here.

### DirectLake partition syntax
```tmdl
partition fact_transactions = entity
    mode: directLake
    source
        entityName: fact_transactions
        schemaName: dbo
        expressionSource: 'DirectLake - lh_Finance_Silver'
```

### Date relationship
`fact_transactions.transaction_date` is `dateTime`. `dim_date.Date` is `dateTime`.
Relationship **must** include:
```tmdl
joinOnDateBehavior: DatePartOnly
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

---

## DAX rules

- ALL measures use `VAR / RETURN` pattern
- Display folders: `KPIs`, `Amount`, `Transaction Volume`, `Time Intelligence`, `Fees & Tax`, `Fraud & Risk`
- All measures get `///` description for Copilot readiness
- Raw columns hidden in report view; only measures and key dims exposed
- No implicit measures anywhere
- `DIVIDE()` used for all division (never `/`) — third arg = 0

---

## Design: powerbi.tips Layout Trifecta

- **Scrim**: single background image (Figma/PowerPoint export), not individual shapes
- **Page layout**: 1280×720, left nav panel (200px), top header (80px), content zone
- **Theme**: Power Designer export — applied as SharedResources base theme in `report.json`
- **Colors**: max 4 intentional colors, consistent semantic meaning
- **Typography**: DIN for callouts/KPIs, Segoe UI for labels
- **4 pages**: Overview, Customers, Transactions, Trends

---

## Known Fabric gotchas

- `delta.columnMapping.mode=name` must be set AT WRITE TIME — retrofitting breaks DirectLake framing
- `ALTER TABLE RENAME COLUMN` is unsupported in T-SQL analytics engine — renames via Spark SQL only
- `%pip` is blocked in pipeline execution — use Fabric Environment for extra packages
- Null GUID `00000000-0000-0000-0000-000000000000` in pipeline JSON causes silent failures — always use real workspace GUIDs
- `logicalId` values in `.platform` files must be stable UUIDs — never regenerate them after first Fabric sync
