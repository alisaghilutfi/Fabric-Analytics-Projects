# CONTEXT — ws_AaltoEE_Forecast

> **How to use this file**
> ProjectPlanner reads the Instructions layer at session start to understand
> the current state and approved next steps. FabricEngineer reads it before
> acting and writes the Last Session layer at session end. Both layers are
> updated after every session.

---

## Instructions layer

*(Written by ProjectPlanner — read by FabricEngineer before acting)*

### Workspace purpose

Net sales forecasting prototype built for the Aalto EE Data Analyst/Engineer
role application. Combines three data sources — actuals (Jeeves), accruals
(project delivery system), and CRM pipeline (Dynamics 365) — into a governed
medallion pipeline with a DirectLake semantic model, Power BI report, and
Fabric Data Agent.

This is a completed prototype. The Fabric workspace is live at
`ws_AaltoEE_Forecast`. Further development should focus on productionising
the ingestion layer (Dataverse shortcut, SharePoint folder connector) and
adding the forecast adjustment input workflow.

### Architecture summary

```
Files/Bronze/      ← Excel source files (manual upload for prototype)
silver_net_sales   ← 795 rows · €14,635,067.81 · Jan–Aug 2026 actuals
silver_accruals    ← 313 rows · €6,616,025.78 · Sep–Dec 2026 forecast
silver_pipeline    ← 126 rows · €828,900.00 · weighted CRM pipeline
gold_dim_bu        ← 4 rows (ExEd, Qualification, University, Unassigned)
gold_dim_date      ← 60 rows (Jan 2024 – Dec 2028, monthly)
gold_fact_net_sales
gold_fact_accruals
gold_fact_pipeline
gold_forecast_snapshot ← appended per pipeline run, version: YYYYMMDD_HHMMSS
```

### Semantic model

- Name: `sm_AaltoEE_Forecast`
- Mode: DirectLake on Gold Delta tables
- 8 DAX measures in `_Measures` table across 3 display folders
- 3 RLS roles: Finance (no filter), Programme_Manager_ExEd (ExEd BU only),
  Ranking_Team (defined, not yet assigned)
- Relationships: Dim Date → 3 fact tables · Dim BU → 3 fact tables
- No date relationship on gold_forecast_snapshot (intentional — queried by
  version timestamp, not calendar month)

### Pipeline

- `pl_AaltoEE_Build` orchestrates: ingest → transform → validate
- Runtime: ~6 minutes
- Validation: 11/11 acceptance checks pass against source control totals
- Snapshot: each run appends ~43 rows to gold_forecast_snapshot

### Known gaps — approved next steps

1. **Production ingestion** — replace manual Excel Bronze uploads with:
   - Dataverse shortcut for CRM pipeline data
   - SharePoint folder connector for actuals and survey exports
   Both connectors are available in the Fabric capacity (verified).

2. **Forecast adjustment input** — add a governed adjustment fact table
   allowing Finance to submit project-month corrections with reason codes.
   Approval workflow via Power Apps + Dataverse.

3. **Snapshot comparison report page** — add a Page 4 to rpt_AaltoEE_Forecast
   showing version-over-version forecast variance using the snapshot table.

4. **Root README update** — add ws_AaltoEE_Forecast entry to the repo root
   README.md Projects section.

### Data quality flags — do not auto-resolve

These flags are intentionally preserved for Finance review. Do not write
code that silently removes or corrects them without documented Finance approval:

- `NEGATIVE_NET_SALES` (16 rows) — credit notes, legitimate
- `ZERO_NET_SALES` (102 rows) — deferred revenue, legitimate
- `UNASSIGNED_BU` (4 rows, O7401201, €42,550) — mapping candidate: ExEd
- `MISSING_PROBABILITY` (2 rows) — treat as 0% until Finance assigns
- `ALLOCATION_EXCEEDS_100PCT` (1 row, Opp 111) — fix required in Dynamics 365

### Agent demo questions (confirmed answers from Gold layer)

- "What is the total full-year forecast for 2026?" → ~€22,072,000
- "What is the accrued forecast for September 2026?" → ~€1,866,000
- "What is the total weighted pipeline value for 2026?" → ~€821,000

---

## Last session layer

*(Written by FabricEngineer at session end — summarises what was done)*

### Session: September 2026 — prototype build complete

**What was completed:**

- `nb_AaltoEE_01_ingest` — Bronze → Silver, all three sources, validation
  flags, 11/11 acceptance checks pass
- `nb_AaltoEE_02_transform` — Silver → Gold star schema, Dim BU, Dim Date
  (2024–2028), three Gold fact tables, forecast snapshot cell added
- `nb_AaltoEE_03_validate` — 11 acceptance checks, all pass:
  - silver_net_sales: 795 rows · €14,635,067.81 ✓
  - silver_accruals: 313 rows · €6,616,025.78 ✓
  - silver_pipeline: 126 rows · €828,900.00 ✓
  - Referential integrity: 0 orphan rows across all BU and Date joins ✓
- `pl_AaltoEE_Build` — pipeline created, run succeeded (5m 22s), all three
  activities green
- `sm_AaltoEE_Forecast` — DirectLake semantic model, 6 relationships,
  8 DAX measures, 3 RLS roles, gold_forecast_snapshot added as table
- `rpt_AaltoEE_Forecast` — 3 pages (Overview, Monthly Forecast, Pipeline
  Analysis), Dim BU slicer on all pages, year filter 2026 on pages 1–2,
  Closing Year slicer on page 3 (uses gold_fact_pipeline[Closing_Year]
  to avoid Dim Date range mismatch)
- `agent_AaltoEE_Forecast` — published (workspace only), 3 demo questions
  confirmed against Gold model

**Fixes applied during build:**

- Deduplication defect: restored €341,225.79 of legitimate accrual data
  by removing Table.Distinct step that excluded Amount column from key
- BU filter propagation: created Dim BU table, related to all three fact
  tables, replaced raw BU columns with Dim BU[BU_Name] in all slicers
- Dim Date extended: 12 rows (2026 only) → 60 rows (2024–2028) to
  accommodate pipeline closing dates
- Period_Type added to Dim Date: Historical / Actuals / Forecast / Future

**Unresolved items (carry to next session):**

- Production ingestion not yet implemented — Excel files in Bronze manually
  uploaded. Dataverse shortcut and SharePoint folder connector available
  but not configured (requires Aalto EE credentials not available)
- Snapshot comparison report page (Page 4) not yet built
- Root README.md not yet updated with ws_AaltoEE_Forecast entry
- docs/images folder not yet created — architecture PNGs to be added

**Git state:**

- Branch: dev-fabric-sync
- Folder: /ws_AaltoEE_Forecast
- Fabric sync: initial commit pending
- README.md: generated, not yet committed
- CONTEXT.md: this file — not yet committed
