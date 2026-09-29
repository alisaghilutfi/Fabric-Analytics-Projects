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
- 9 DAX measures in `_Measures` table across 4 display folders
- Relationships: Dim Date → 3 fact tables · Dim BU → 3 fact tables
- No date relationship on gold_forecast_snapshot (intentional — queried by
  version timestamp, not calendar month)
- README.md sections to keep current: intro counts, What Was Built table,
  Measure Library table, Report Pages section.

### RLS roles (live model — verified 2026-09-28)

| Role | Filter |
|---|---|
| Finance | No filter (full read) |
| Programme_Manager_ExEd | BU_Name = "ExEd Programs BU" |
| Programme_Manager_Qualification | BU_Name = "Qualification Programs BU" |
| Programme_Manager_University | BU_Name = "University Programs BU" |

Note: `Ranking_Team` role documented in previous CONTEXT.md is absent
from the live model — removed from documentation.

### Pipeline

- `pl_AaltoEE_Build` orchestrates: ingest → transform → validate
- Runtime: ~6 minutes
- Validation: 11/11 acceptance checks pass against source control totals
- Snapshot: each run appends ~43 rows to gold_forecast_snapshot

### Approved next steps (priority order)

**Step 1 — AI-readiness enrichment ✅ COMPLETE (2026-09-28)**
See Last Session layer for full details.

**Known gap — Synonyms ⚠️**
Not writable at this DirectLake compatibility level via TMDL or
translation API. Workaround: Power BI Desktop → Tools → Language
& Q&A → Manage Synonyms. Deferred — low priority until Copilot
usage is validated.

**Step 2 — Production ingestion (BLOCKED — requires Aalto EE credentials)**
- Dataverse shortcut for CRM pipeline data
- SharePoint folder connector for actuals and survey exports

**Step 3 — Forecast adjustment input**
Governed adjustment fact table with Finance input via Power Apps +
Dataverse. Approval workflow with reason codes.

**Step 4 — Root README update**
Add ws_AaltoEE_Forecast entry to repo root README.md Projects section.

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

### Session: 2026-09-28 — AI-readiness enrichment + Page 4 build

**What was completed:**

1. AI-readiness enrichment of `sm_AaltoEE_Forecast` via powerbi-modeling-mcp:
   - 42 raw columns hidden across 6 tables
   - `dataCategory: "Date"` set on 7 DateTime columns
   - Table descriptions added to all 7 tables (Copilot/agent-ready)
   - All 8 measure descriptions already present — no changes needed
   - Synonyms blocked at DirectLake compatibility level — workaround documented
   - Agent validation: all 3 demo questions pass ✅

2. Page 4 — Snapshot Comparison built in `rpt_AaltoEE_Forecast`:
   - Title, 2 KPI cards, 2 slicers (Snapshot_Version dropdown, Period_Type tile),
     line chart (forecast by month per version), stacked bar chart (forecast by
     component per version)
   - New measure `Snapshot Amount EUR` added for correct per-version filter context
     on comparison charts; KPI cards retain `Latest Snapshot Total EUR`
   - Canvas: 1920×1080, FitToPage, Fluent2 theme inherited from existing pages

3. powerbi-modeling-mcp upgraded from 0.1.9 (Downloads) to 1.0.0 (VS Code
   extension) in both `claude_desktop_config.json` and Claude Code. `--accept-eula`
   flag added. Stale 0.1.9 processes removed.

**Commits:** `dc2c828` (Page 4), `acf4ca0` (Snapshot Amount EUR measure + chart fix)

**What exists so far:**

| Artifact | Type | Status |
|---|---|---|
| `nb_AaltoEE_01_ingest` | Notebook | ✅ Complete |
| `nb_AaltoEE_02_transform` | Notebook | ✅ Complete |
| `nb_AaltoEE_03_validate` | Notebook | ✅ Complete — 11/11 checks pass |
| `pl_AaltoEE` | DataPipeline | ✅ Complete — ~6 min, all green |
| `sm_AaltoEE_Forecast` | Semantic model | ✅ Complete — DirectLake, 9 measures, 3 RLS roles, AI-ready |
| `rpt_AaltoEE_Forecast` | Report | ✅ Complete — 4 pages |
| `agent_AaltoEE_Forecast` | Data Agent | ✅ Published — 3 demo questions confirmed |
| `docs/images/` | Docs | ✅ Added — architecture PNGs committed |
| Synonyms | Semantic model | ⚠️ Blocked — DirectLake compatibility level |
| Production ingestion connectors | Config | ❌ Not started — requires Aalto EE credentials |

**Blockers:**
- Production ingestion not yet implemented — Dataverse shortcut and SharePoint
  folder connector require Aalto EE credentials not available in prototype environment
- Synonyms not writable at this DirectLake compatibility level — workaround:
  Power BI Desktop → Tools → Language & Q&A → Manage Synonyms
