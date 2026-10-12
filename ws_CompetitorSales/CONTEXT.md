# ws_CompetitorSales — CONTEXT.md
*ProjectPlanner-authored · 2026-10-05 · v3 — confirmed and locked*

---

## Architecture Reference

All item decisions (Lakehouse, DataPipeline, 3 Notebooks, SemanticModel, Report),
wave order, deployment strategy, and acceptance criteria are fixed in:

- `_projects/analyze-sales-performance-market/docs/architecture-handoff.md` (Revision 1)
- `_projects/analyze-sales-performance-market/docs/deployment-handoff.md`

Do not redesign anything those docs decided. The sections below cover only what
Studio does not.

---

## Workspace

| Property | Value |
|---|---|
| Workspace name | `ws_CompetitorSales` |
| Workspace ID | `bdf49d9a-d2e4-444b-b666-735d2460b5cb` |
| Capacity | `fc905db7-b7f5-42ed-a8e5-e6e809f8bd8e` |
| Git repo | `alisaghilutfi/Fabric-Analytics-Projects` |
| Git folder | `/ws_CompetitorSales` |
| Branch | `dev-fabric-sync` |

---

## Naming Conventions

| Studio name | Fabric item name |
|---|---|
| `lakehouse` | `lh_CompetitorSales` |
| `ingestion-pipeline` | `pl_CompetitorSales` |
| `notebook-bronze` | `nb_CompetitorSales_Bronze` |
| `notebook-silver` | `nb_CompetitorSales_Silver` |
| `notebook-gold` | `nb_CompetitorSales_Gold` |
| `notebook-date` | `nb_CompetitorSales_Date` |
| `semantic-model` | `sm_CompetitorSales` |
| `report` | `rpt_CompetitorSales` |
| `data-agent` | `agent_CompetitorSales` |

> Variable Library — dropped; not required for this project.

### Semantic Model Table Names (Marco Russo / SQLBI convention)

- Fact tables: plural (e.g. `Sales`, `Orders`, `Transactions`)
- Dimension tables: singular (e.g. `Product`, `Customer`, `Date`, `Geography`)
- Measures table: `_Measures` (leading underscore, sorts to top)
- `entityName` in TMDL always uses the physical Gold table name (e.g. `gold_fact_sales`)
- Display name in TMDL uses the convention above (e.g. `table Sales`)

---

## Source Schema (confirmed)

### fact_sales (6 regional CSVs — UNION ALL in Silver)

| Column | Type | Notes |
|---|---|---|
| `ProductID` | number | FK → dim_product |
| `Date` | date | Transaction date |
| `Zip` | text | FK → dim_geography |
| `Units` | number | Units sold |
| `Revenue` | number | Pre-calculated — do NOT re-derive from Units × Price |

International files have an additional `Country` column — retain in Silver UNION,
use to enrich dim_geography join where Zip is ambiguous across countries.

### dim_product (bi_dimensions.xlsx — Product Details sheet)

| Column | Type | Notes |
|---|---|---|
| `PerformanceID` | number | PK — same key as Sales[ProductID]; rename to `ProductID` in Silver |
| `Product` | text | |
| `Segment` | text | |
| `Category` | text | |
| `ManufacturerID` | number | FK → dim_manufacturer |
| `Price` | text | Contains currency — inspect at Bronze; parse to numeric in Silver |

### dim_manufacturer (bi_dimensions.xlsx — Manufacturer sheet)

| Column | Type | Notes |
|---|---|---|
| `ManufacturerID` | number | PK |
| `Manufacturer` | text | |
| `Logo` | text | Hyperlink URL — keep in Gold, hide in semantic model |

Source is wide format (15 columns) — unpivot to one row per manufacturer in Silver
(architecture-handoff.md AC-8).

### dim_geography (bi_dimensions.xlsx — Geography sheet)

| Column | Type | Notes |
|---|---|---|
| `Zip` | text | PK |
| `City` | text | |
| `State` | text | |
| `Region` | text | |
| `District` | text | |
| `Country` | text | |

3 metadata header rows skipped at Bronze read (AC-5).
No lat/lon enrichment — 90,395 unique cities makes geocoding impractical.
Map visual uses Power BI named-field geocoding (Country + State).

---

## 1 · Semantic Model Plan

### Star schema (Gold output)

fact_sales → dim_product → dim_manufacturer (snowflake; two-hop path intentional)
fact_sales → dim_geography
fact_sales → dim_date


### Relationships

| From | To | Cardinality | Active |
|---|---|---|---|
| `fact_sales[ProductID]` | `dim_product[ProductID]` | Many-to-One | ✅ |
| `dim_product[ManufacturerID]` | `dim_manufacturer[ManufacturerID]` | Many-to-One | ✅ |
| `fact_sales[Zip]` | `dim_geography[Zip]` | Many-to-One | ✅ |
| `fact_sales[Date]` | `dim_date[Date]` | Many-to-One | ✅ |

> Snowflake (Sales → Product → Manufacturer) is correct at 1.8M rows with DirectLake.
> Two-hop cross-filter performance is fine at this scale.

### Date dimension

- Notebook: `nb_CompetitorSales_Date` (PySpark, same pattern as
  `ws_USGS_Earthquake/nb_USGS_Earthquake_Date`)
- Scope: `2016-01-01` → `2022-12-31` (dataset is 2017–2021; 1-year buffer each side)
- Calendar year only — no fiscal year columns
- Write to: `lh_CompetitorSales/Tables/dbo/DateDimension`

Columns to include:

| Column | Notes |
|---|---|
| `Date` | PK — relationship to fact_sales |
| `Year` | |
| `Quarter` | e.g. "Q1" |
| `MonthNumber` | Integer |
| `MonthName` | e.g. "January" |
| `MonthYear` | e.g. "Jan 2020" — axis label |
| `MonthYearCode` | Integer e.g. 202001 — sort column for MonthYear |
| `Day` | |
| `DayOfWeek` | e.g. "Monday" |
| `IsWeekend` | Boolean |
| `YearOffset` | Current year = 0 |

### _Measures calculated table

All measures in a single `_Measures` table. Hide the helper `Column` column.

#### Display folder: Sales Performance

```dax
/// Total pre-calculated revenue across all sales transactions in the current filter
/// context. Revenue is pre-calculated in source — not derived from Units × Price.
Revenue = SUM('fact_sales'[Revenue])
    formatString: $ #,##0

/// Total units sold across all sales transactions in the current filter context.
Units Sold = SUM('fact_sales'[Units])
    formatString: #,0

/// Average revenue per transaction in the current filter context.
Avg Revenue per Transaction = DIVIDE([Revenue], [Transaction Count])
    formatString: $ #,##0.00
```

#### Display folder: Market Share

```dax
/// Selected manufacturer's revenue as a proportion of total revenue across all
/// manufacturers in the current filter context. Returns 0 if total is zero.
Market Share % =
VAR _SelectedRev = [Revenue]
VAR _TotalRev = CALCULATE([Revenue], ALL('dim_manufacturer'))
RETURN DIVIDE(_SelectedRev, _TotalRev, 0)
    formatString: 0.0%

/// Revenue versus prior year. Returns BLANK() if no prior year data.
Revenue YoY Change =
VAR _Current = [Revenue]
VAR _Prior = CALCULATE([Revenue], DATEADD('dim_date'[Date], -1, YEAR))
RETURN _Current - _Prior
    formatString: $ +#,##0;$ -#,##0;0

/// Revenue YoY change as a percentage of prior year.
Revenue YoY % =
VAR _Current = [Revenue]
VAR _Prior = CALCULATE([Revenue], DATEADD('dim_date'[Date], -1, YEAR))
RETURN DIVIDE(_Current - _Prior, _Prior, BLANK())
    formatString: +0.0%;-0.0%

/// Market share versus prior year market share.
Market Share YoY Change =
VAR _Current = [Market Share %]
VAR _Prior = CALCULATE([Market Share %], DATEADD('dim_date'[Date], -1, YEAR))
RETURN _Current - _Prior
    formatString: +0.0%;-0.0%

/// Manufacturer rank by revenue among all manufacturers in the current filter context.
/// 1 = highest revenue.
Revenue Rank =
RANKX(ALL('dim_manufacturer'), [Revenue], , DESC, Dense)
    formatString: 0
```

#### Display folder: Volume

```dax
/// Count of all sales transactions in the current filter context.
Transaction Count = COUNTROWS('fact_sales')
    formatString: #,0

/// Count of distinct products with at least one sale in the current filter context.
Products Sold = DISTINCTCOUNT('fact_sales'[ProductID])
    formatString: #,0

/// Count of distinct manufacturers with revenue in the current filter context.
Manufacturers Active = DISTINCTCOUNT('dim_product'[ManufacturerID])
    formatString: #,0
```

#### Display folder: Geography

```dax
/// Count of distinct countries with at least one sale in the current filter context.
Countries with Sales = DISTINCTCOUNT('dim_geography'[Country])
    formatString: #,0
```

### Hidden columns

| Table | Hide |
|---|---|
| `fact_sales` | `ProductID`, `Zip` |
| `dim_product` | `ProductID` (after rename), `ManufacturerID` |
| `dim_manufacturer` | `ManufacturerID`, `Logo` |
| `dim_geography` | `Zip` |
| `dim_date` | `MonthNumber`, `MonthYearCode`, `YearOffset` |

---

## 2 · AI-Readiness (Phases 2–4)

**Phase 2 — Semantic Model Hygiene**

- 2a: Column visibility — per hidden columns table above
- 2b: Data categories — set `dataCategory: "Date"` on `fact_sales[Date]` and
  `dim_date[Date]`
- 2c: Measures — VAR/RETURN throughout, `///` descriptions as above, display folders
  as above

**Phase 3 — Business Context**

Table descriptions (apply via `powerbi-modeling-mcp → batch_table_operations`):

- `fact_sales`: "Regional sales transactions from 6 countries (USA, Canada, Germany,
  Japan, Mexico, Nigeria). One row per sale. ~1.8M transactions, 2017–2021. Joined to
  Products via ProductID and Geography via Zip. Revenue is pre-calculated in source."
- `dim_product`: "Product catalogue from bi_dimensions.xlsx — Product Details sheet.
  One row per product. Covers 15 manufacturers across Segment and Category hierarchies.
  Price parsed to numeric in Silver."
- `dim_manufacturer`: "Manufacturer reference, unpivoted from wide 15-column source.
  One row per manufacturer. Sintec is the focal manufacturer for market share analysis."
- `dim_geography`: "Geographic lookup. One row per Zip code. Covers 6 countries.
  176k rows; 3 metadata header rows skipped on load. No lat/lon — map uses named-field
  geocoding."
- `dim_date`: "Calendar date dimension, 2016-01-01 to 2022-12-31. One row per day.
  Generated via PySpark in nb_CompetitorSales_Date."

Synonyms: DirectLake limitation — defer to Power BI Desktop → Tools → Language & Q&A
→ Manage Synonyms after build.

**Phase 4 — AI Layer**

- 4a: Configure "Prep data for AI" after semantic model is stable; focus Copilot on
  `fact_sales`, `_Measures`, and `dim_manufacturer`
- 4b: `agent_CompetitorSales` — in scope for v1; configure after report is validated.
  Seed example queries:
  - "What is Sintec's market share in 2020?"
  - "Which country had the highest revenue in 2019?"
  - "How did Sintec's sales compare to competitors in Germany?"
  - "Which product category drives the most revenue?"

---

## 3 · Power BI Report Plan

### Pages

| # | Page | Purpose |
|---|---|---|
| 1 | Sales Overview | Company-wide KPIs — revenue, units, YoY trends |
| 2 | Market Share | Manufacturer share breakdown and trend |
| 3 | Regional Performance | Revenue by country and state/region |
| 4 | Product Analysis | Revenue by Category, Segment, top products |

### Layout Trifecta (powerbi.tips / Mike Carlo)

- **Scrim:** Design title strip + left slicer panel + content zone containers in
  Figma or PowerPoint. Export as PNG per page. Set as page background — avoids
  individual shape overhead.
- **Grid:** F-pattern. Top strip: page title. Left panel (~200px): Country, Year,
  Manufacturer slicers — consistent across all pages. Main area: primary visual
  top-left, supporting visual top-right, detail visual bottom.
- **Theme:** Max 4 colours. Sintec → primary accent. Competitors → neutral grey.
  YoY positive → good (green). YoY negative → bad (red). Generate via powerbi.tips
  Theme Generator; apply as report theme JSON.

### Visuals by page

**Page 1 — Sales Overview**
- Card strip: Revenue, Units Sold, Transaction Count, Revenue YoY %
- Line chart: Revenue over time (`dim_date[Month Year]` × `[Revenue]`) — all
  manufacturers or Sintec vs. rest as legend series
- Combo chart: Revenue YoY % (line, secondary axis) + Units Sold (column, primary)
- Slicers: Country, Year

**Page 2 — Market Share**
- Bar chart: Market Share % by Manufacturer (sorted descending) — Sintec highlighted
  with primary accent; competitors in neutral
- Line chart: Market Share % trend — top 5 manufacturers as legend series
- Card: Revenue Rank for selected manufacturer
- Slicers: Country, Category, Year

**Page 3 — Regional Performance**
- Filled map: Revenue by Country + State (Power BI named-field geocoding — no
  coordinates required)
- Clustered bar: Revenue by Country (sorted descending)
- Clustered bar: Revenue by Region/District — top 10 filter
- Slicers: Manufacturer, Year

**Page 4 — Product Analysis**
- Matrix: Category × Manufacturer — Revenue values, data bars conditional formatting
- Bar chart: Top 10 Products by Revenue
- Donut: Revenue split by Segment
- Slicers: Manufacturer, Country, Year

---

## 4 · Open Items

| # | Item | Owner |
|---|---|---|
| OQ-1 | `Price` in dim_product is type text — inspect at Bronze; confirm whether it contains currency symbol before Silver parse | FabricEngineer |

---

## 5 · What Exists So Far

| Item | Status |
|---|---|
| Task Flow Studio artifacts | ✅ Complete |
| Source schema | ✅ Confirmed |
| Fabric workspace | ✅ Created — ID `bdf49d9a-d2e4-444b-b666-735d2460b5cb` |
| Git integration | ✅ Connected — `/ws_CompetitorSales` on `dev-fabric-sync` |
| CONTEXT.md | ✅ This file |
| `nb_CompetitorSales_Date` | ⚠️ DateDimension write pending — capacity limit hit; fix committed (saveAsTable + lakehouse attachment in metadata) |
| `nb_CompetitorSales_Bronze` | ✅ Complete — 1,798,121 sales rows + 3 dimension tables |
| `nb_CompetitorSales_Silver` | ✅ Complete — silver_sales 1,798,121 rows, AC-7 and AC-8 pass |
| `nb_CompetitorSales_Gold` | ✅ Complete — Date NULL fixed (to_date MM/dd/yyyy), ZipCountry composite key added, pipeline runs clean |
| `sm_CompetitorSales` | ✅ Complete — ID: `8570540e-a83e-4a5b-b356-e252eeedbb8e`, 4 relationships, 12 measures, DAX verified (Revenue $695M, Transactions 1,798,121, Sintec market share 17%) |
| `rpt_CompetitorSales` | ✅ Complete — 4 pages complete, all slicers functional (Country, Year, Manufacturer), ZipCountry relationship resolved |
| `agent_CompetitorSales` | ❌ Not yet created |

---

## 6 · Next Session Starts At

AI-Readiness pass (Phase 2–4 per Section 2 of this file):
1. Add table descriptions via plugin:powerbi-authoring (5 tables)
2. Set data categories on Geography columns (Country, State)
3. Configure "Prep data for AI" in sm_CompetitorSales settings
4. Create agent_CompetitorSales with 4 seed example queries from Section 2


---

## Last Session

*2026-10-05 — ProjectPlanner: CONTEXT.md authored. Workspace created manually by Ali
in Fabric portal (Git sync confirmed). FabricEngineer blocked on workspace creation
due to InsufficientScopes on Fabric Core MCP connector — OAuth reconnect needed or
workspace created manually. Workspace ID confirmed:
`bdf49d9a-d2e4-444b-b666-735d2460b5cb`.*

*2026-10-06 — Bronze/Silver/Gold notebooks executed successfully. All AC criteria met:
AC-7 (silver_sales 1,798,121 rows), AC-8 (silver_manufacturer 15 rows), AC-10
(0 orphaned fact rows). DateDimension written (2,557 rows). OPTIMIZE complete on
gold_fact_sales. Key fixes applied during run: Bronze xlsx header=1 for Product
Details, manual iloc unpivot for Manufacturer, Silver manufacturer changed to
pass-through. dim_date column names confirmed as camelCase (MonthNumber,
MonthYearCode etc). Next: Wave 4 — sm_CompetitorSales.*

*2026-10-06 — Waves 1–4 substantially complete. Gold validated (1,798,121 rows,
0 orphans, OPTIMIZE done). Pipeline clean. sm_CompetitorSales created
(ID: 2d55bd9f-5512-40de-803b-6d5be8fc4eb0) with snowflake schema, all 12 measures,
display folders. DateDimension blocked by HTTP 430 capacity limit — fix committed
(saveAsTable + lakehouse metadata attachment). Key fixes this session: Bronze xlsx
header=1, manual iloc Manufacturer unpivot (14 rows), Silver/Gold overwriteSchema,
dim_date camelCase columns, definition.pbism missing from initial TMDL, abfss path
format requires workspace/lakehouse name not ID.*

*2026-10-08 — sm_CompetitorSales rebuilt via Fabric UI after Git-synced version failed
Power BI service registration (404). All 4 relationships and 12 measures added via
powerbi-modeling-mcp REST API (XMLA write was enabled but REST API used as fallback).
Date relationship required DateType cast in Gold notebook (gold_fact_sales[Date] was
String); column type also corrected in TMDL (string → dateTime). DAX verified:
Revenue $695M, 1,798,121 transactions, Sintec 17% market share. TMDL exported to
Git — gold_* table names replace old dim_*/fact_sales naming.*

*2026-10-11 — ZipCountry composite key added to resolve duplicate Zip in Geography
(176k rows, 6 countries). Manual notebook edits applied directly in Fabric UI (dbo.
prefix reverted, ZipCountry added). Pipeline ran successfully — 1,798,121 rows,
gold_dim_geography and gold_fact_sales both have ZipCountry. Blocker: Date column NULL
in gold_fact_sales — DateType cast failing. Semantic model table renames confirmed live
via plugin:powerbi-authoring XMLA. Report has 4 pages but Geography and Date slicers
not yet functional.*

*2026-10-12 — Wave 5 complete. Date NULL fixed (to_date MM/dd/yyyy format).
ZipCountry composite key resolves duplicate Zip across 6 countries —
Sales[ZipCountry]→Geography[ZipCountry] relationship live via XMLA.
EDA notebook (nb_CompetitorSales_EDA) built and validated: 0 NULLs, 0 PK
duplicates, 0 Sales→Product orphans, 55,594 Sales→Geography orphans (known,
documented). rpt_CompetitorSales 4 pages live: Sales Overview, Market Share,
Regional Performance, Product Analysis. All slicers functional. README.md
and ws_CompetitorSales/Readme.md written. Analyst Mindset Framework applied.
Next: AI-Readiness pass.*

---
*Instructions layer: ProjectPlanner (Claude.ai) · 2026-10-05*
*Execution layer: FabricEngineer (Claude Code / VS Code)*
*Branch: dev-fabric-sync — do not push directly to main*
