# ws_USGS_Earthquake — Project Context
> Last updated: 2026-09-08 by ProjectPlanner (claude.ai)
> Next agent: FabricEngineer (Claude Code)

---

## Project Summary

End-to-end Microsoft Fabric analytics solution ingesting live USGS seismic event data through a Bronze → Silver → Gold medallion architecture, surfaced in a 4-page Power BI Direct Lake report. Fully operational and pipeline-idempotent. Now entering Phase 2: Fabric Data Agent layer on top of sm_USGS_Earthquake.

---

## Workspace

| Property | Value |
|---|---|
| Workspace name | `ws_USGS_Earthquake` |
| Workspace ID | `39caa3ab-a964-45c2-bd5b-7d46ad66c985` |
| Capacity | `alisaghif2capacity` — F2, North Europe |
| Git branch | `dev-fabric-sync` |
| Git folder | `/ws_USGS_Earthquake` |
| GitHub connection | `Ali-GitHub-Lab-Classic` |
| GitHub PAT | Regenerated 2026-07-26, no expiration |

Note: Capacity was listed as "Trial" in prior context — confirmed F2 (alisaghif2capacity) as of 2026-09-08. Fabric Data Agents confirmed available at F2 (validated via ws_DS_BankChurn — agent_DS_BankChurn live on same capacity).

---

## Artifacts

| Artifact | Type | ID | Notes |
|---|---|---|---|
| `lh_USGS_Earthquake` | Lakehouse | `fda3c40e-3b18-4230-b5a8-18334d845083` | Default schema: dbo |
| `nb_USGS_Earthquake_Bronze` | Notebook | `75f3d0b1-b6c1-401b-a8b6-5ff5fb852bb5` | API ingest |
| `nb_USGS_Earthquake_Silver` | Notebook | `de29c7b4-3987-41fc-a12a-54bb031e09b7` | MERGE upsert |
| `nb_USGS_Earthquake_Gold` | Notebook | `1601fa81-8b09-48d3-a9ba-107f338481dd` | Vectorized pandas_udf, MERGE upsert |
| `nb_USGS_Earthquake_Date` | Notebook | `2beef350-49b8-4b40-b665-2de3a752fb0c` | Date dim, one-time setup |
| `nb_USGS_Earthquake` | Notebook | `63e42046-96df-4372-a558-3474a6c2cbc7` | SCRATCH — DELETE THIS |
| `env_USGS_Earthquake` | Environment | `445540a6-feb0-43af-b11c-93e8b08e8bb0` | reverse_geocoder 1.5.1, Runtime 1.3 |
| `pl_USGS_Earthquake` | DataPipeline | `da73eec0-8422-4a21-9956-0185255710d5` | Bronze->Silver->Gold, -7d rolling |
| `sm_USGS_Earthquake` | SemanticModel | `856acf3d-ddbc-4ed7-87a8-559018c12f05` | Direct Lake, 3 tables, 17 measures |
| `rpt_USGS_Earthquake` | Report | `21c62478-fd88-4c21-b79b-b20f6383b2c3` | 4 pages, PBIR format |

Phase 2 artifacts to be created (not yet in workspace):

| Artifact | Type | Notes |
|---|---|---|
| `agent_USGS_SeismicActivity` | Data Agent | Grounds into sm_USGS_Earthquake, _Measures table only |
| `agent_USGS_GeographicImpact` | Data Agent | Grounds into sm_USGS_Earthquake, _Measures + Country Code/Place Description columns |
| `agent_USGS_SignificanceAnalyst` | Data Agent | Grounds into sm_USGS_Earthquake, _Measures table only |

---

## Data Layer

### Tables (all in lh_USGS_Earthquake.dbo)

| Table | Rows | Notes |
|---|---|---|
| `earthquake_events_silver` | 9,164 | Deduplicated, MERGE upsert on id |
| `earthquake_events_gold` | 9,164 | Deduplicated, MERGE upsert on id, columnMapping.mode=name |
| `DateDimension` | 1,461 | 2024-01-01 to 2027-12-31 |

### Gold Table — ABFSS Path

    abfss://39caa3ab-a964-45c2-bd5b-7d46ad66c985@onelake.dfs.fabric.microsoft.com/fda3c40e-3b18-4230-b5a8-18334d845083/Tables/dbo/earthquake_events_gold

### Gold Table Schema

    id               string
    longitude        double
    latitude         double
    elevation        double
    title            string
    place_description string
    sig              bigint
    mag              double
    magType          string
    time             timestamp
    updated          timestamp
    country_code     string
    sig_class        string    <- typo fixed from sig_calss via ALTER TABLE RENAME COLUMN

### Key Engineering Fixes Applied
- Silver/Gold both use `DeltaTable.merge().whenNotMatchedInsertAll()` — idempotent on re-run
- Gold uses `@pandas_udf` vectorized reverse geocoder (not row-serial UDF)
- Gold written directly to ABFSS path to prevent Direct Lake path drift
- `delta.columnMapping.mode = name` set at write time (not retrofitted)
- `sig_calss` typo renamed to `sig_class` via `ALTER TABLE RENAME COLUMN`
- `%pip` replaced by `env_USGS_Earthquake` for pipeline-safe library loading

---

## Semantic Model — sm_USGS_Earthquake

**Mode:** Direct Lake  
**Connection:** `Fabric-ws_USGS_Earthquake-sm_USGS_Earthquake` (via powerbi-modeling-mcp)

### Tables

| Table | Description |
|---|---|
| `Earthquake Events` | One row per seismic event from Gold layer. Direct Lake partition on `earthquake_events_gold`. |
| `Date` | Calendar dimension 2024-2027. Direct Lake partition on `DateDimension`. |
| `_Measures` | Calculated partition. 17 DAX measures across 5 display folders. |

### Relationship

    Date[Date] (1) ---- (*) Earthquake Events[Event Time]
    joinOnDateBehavior: DatePartOnly
    CrossFilteringBehavior: OneDirection

Note: DatePartOnly required because Event Time is TimestampType, Date is DateType.

### Hidden Columns (field picker)
**Earthquake Events:** ID, Elevation, Last Updated, Significance (raw score), Magnitude (raw), Latitude, Longitude  
**Date:** Day, MonthNumber, Weekday, IsWeekend, Quarter

### Sort-By Columns
- `Date[MonthName]` sorts by `Date[MonthNumber]`
- `Date[DayOfWeek]` sorts by `Date[Weekday]`

### Measure Library (17 measures)

**Magnitude folder:**
- `Avg Magnitude` — AVERAGE(mag), format 0.00
- `Max Magnitude` — MAX(mag), format 0.00
- `Min Magnitude` — MIN(mag), format 0.00
- `Magnitude Band` — SWITCH(TRUE(), <2="Micro", <4="Minor", <6="Moderate", "Strong")

**Significance folder:**
- `% High Significance` — DIVIDE(COUNTROWS FILTER sig_class="High", COUNTROWS), 0.0%
- `% Moderate Significance` — same pattern, 0.0%
- `% Low Significance` — same pattern, 0.0%
- `Avg Significance` — AVERAGE(sig), #,0
- `Max Significance` — MAX(sig), #,0

**Time folder:**
- `Earliest Event Date` — MIN(Event Time), dd MMM yyyy
- `Latest Event Date` — MAX(Event Time), dd MMM yyyy

**Time Intelligence folder:**
- `Total Earthquakes MTD` — TOTALMTD([Total Earthquakes], 'Date'[Date]), #,0
- `Total Earthquakes PMTD` — CALCULATE([Total Earthquakes], PREVIOUSMONTH('Date'[Date])), #,0, hidden
- `Total Earthquakes MOM %` — VAR/RETURN DIVIDE(Current-Prev, Prev), +0.0%;-0.0%
- `Avg Magnitude MTD` — TOTALMTD([Avg Magnitude], 'Date'[Date]), 0.00
- `7-Day Rolling Earthquakes` — CALCULATE DATESINPERIOD -7 DAY, #,0

**Volume folder:**
- `Total Earthquakes` — COUNTROWS('Earthquake Events'), #,0

All measures: VAR/RETURN pattern, Copilot descriptions populated, display folders set.

---

## Report — rpt_USGS_Earthquake (4 pages)

### Page 1 — Overview Dashboard
- Report title text box: "USGS Earthquake Activity Dashboard" (#094780, bold, 28pt)
- 5 KPI cards: Total Events, Avg Magnitude, Highest Magnitude, Peak Significance, High Significance %
- Line chart: Total Earthquakes by Date (x=Date[Date], y=Total Earthquakes)
- Combo chart: Daily Seismic Activity — Magnitude vs. Volume (x=Date[Date], col=Avg Magnitude, line=Total Earthquakes)
- Page background: #F3F4F6

### Page 2 — Time Intelligence
- Date range slicer (Between style, Date[Date])
- 5 KPI cards: Total Events, MTD Events, MOM % Change, Avg Magnitude, Avg Magnitude MTD
- Line chart: "7-Day Rolling Earthquake Count"
- Bar chart: "Monthly Earthquake Volume"
- Donut chart: "Significance Class Distribution"
- Page background: #F3F4F6
- Note: MTD/MOM measures show -- when full date range selected — requires filtered month context (correct DAX behaviour)

### Page 3 — Geographic Distribution
- Map: "Global Earthquake Distribution" (lat=Latitude, long=Longitude, size=Total Earthquakes, legend=Significance Class)
- Country Code dropdown slicer (search enabled)
- Bar chart: "Top Countries by Earthquake Volume" (y=Country Code, x=Total Earthquakes, Top 15 filter)
- 100% stacked bar: "Significance Distribution by Country" (y=Country Code, x=% Low/Moderate/High Significance)
- Page background: #F3F4F6

### Page 4 — Event Detail
- Significance Class list slicer
- Magnitude Type dropdown slicer
- Table "Seismic Event Log": Title, Event Time, Magnitude (conditional format green->yellow->red), Significance Class, Country Code, Place Description
- Sorted by Event Time DESC
- Row totals: Off
- Page background: #F3F4F6

---

## Pipeline — pl_USGS_Earthquake

    Bronze (retry:2, timeout:1h) -> Silver (timeout:2h) -> Gold (timeout:6h)
    workspaceId: 39caa3ab-a964-45c2-bd5b-7d46ad66c985
    start_date: @formatDateTime(addDays(utcNow(), -7), 'yyyy-MM-dd')
    end_date:   @formatDateTime(utcNow(), 'yyyy-MM-dd')

All three notebooks receive `start_date`. Silver uses it to locate the Bronze file. Gold uses `to_timestamp(lit(start_date))` for explicit timestamp comparison.

---

## Phase 2 — Fabric Data Agent Layer (NEW)

### Architecture
Three domain agents, each grounded in sm_USGS_Earthquake, scoped to a distinct analytical concern. Mirrors the Aalto EE multi-agent pattern (governance-first: measures as single source of truth, raw tables not exposed to agents).

### Agent 1 — agent_USGS_SeismicActivity
**Focus:** Time-series trends, event volume, magnitude  
**Data exposure:** `_Measures` table only (Total Earthquakes, Avg/Max/Min Magnitude, Magnitude Band, MTD/MOM measures, 7-Day Rolling)  
**Do NOT expose:** Raw `earthquake_events_gold` table  
**System prompt:**

    You are a seismic activity analyst assistant. You help users understand earthquake frequency and magnitude trends using data from the USGS (United States Geological Survey).

    The dataset covers global earthquake events with daily granularity. Key metrics to know:
    - Total Earthquakes: count of recorded seismic events
    - Avg Magnitude: mean strength on logarithmic scale (each +1.0 = ~31.6x more energy released)
    - Max Magnitude: strongest single event in the filter context
    - Earliest/Latest Event Date: temporal bounds of the selection

    You can answer questions about:
    - Daily, weekly, monthly earthquake counts and trends
    - Magnitude distributions and extremes
    - Changes in seismic activity over time periods
    - Comparisons between time windows

    Always state the time range your answer covers. When reporting magnitude, remind users it is a logarithmic scale. Ground every answer in the measures — do not estimate or hallucinate counts.

    You do not have access to geographic or location data — no country codes, place descriptions, or regional breakdowns. If asked any question about which country, region, or location had the most or fewest earthquakes, respond: "I don't have geographic data. Please ask agent_USGS_GeographicImpact for location-based analysis."

### Agent 2 — agent_USGS_GeographicImpact
**Focus:** Location, country-level patterns, spatial distribution  
**Data exposure:** `_Measures` + `Earthquake Events` columns: Country Code, Place Description, Latitude, Longitude  
**System prompt:**

    You are a geographic earthquake impact analyst. You help users understand
    where earthquakes are occurring and which regions are most affected,
    using USGS global seismic event data.

    The dataset includes country codes (ISO 2-letter), place descriptions,
    latitude, and longitude for each event.

    You can answer questions about:
    - Which countries or regions recorded the most earthquakes
    - Geographic patterns in seismic activity
    - Filtering activity by specific countries or regions

    Always use ISO country codes when referring to countries (e.g. JP, US, TR).
    Do not infer political context from earthquake locations. Ground every
    answer in the data — do not estimate geographic distributions.

### Agent 3 — agent_USGS_SignificanceAnalyst
**Focus:** Impact scoring, risk classification, significance trends  
**Data exposure:** `_Measures` table only (% High/Moderate/Low Significance, Avg Significance, Max Significance)  
**System prompt:**

    You are a seismic significance analyst. You help users understand the
    impact and risk profile of earthquake events using the USGS significance
    scoring system.

    The USGS significance score combines magnitude, Modified Mercalli Intensity
    (MMI), felt reports, and estimated impact. Higher scores indicate more
    impactful events.

    Key metrics to know:
    - Significance: raw USGS score (numeric)
    - Significance Class: Low (<100), Moderate (100-499), High (>=500)
    - % High Significance: proportion of events classified as High
    - Avg Significance / Max Significance: distribution measures

    You can answer questions about:
    - What percentage of events were High significance in a period
    - Peak significance events and when they occurred
    - Comparing significance profiles across time windows
    - Risk profiling by significance class distribution

    Always distinguish between the raw significance score and the classification
    tier. When % High Significance is elevated, contextualize what that means
    in human impact terms.

### Phase 2 Build Order
1. Delete `nb_USGS_Earthquake` scratch notebook (ID: 63e42046) — clear the workspace
2. Verify `sig_class` column is correctly named in `earthquake_events_gold` via SQL analytics endpoint — confirm the ALTER TABLE fix is live
3. Create `agent_USGS_SeismicActivity` — expose `_Measures` only, paste system prompt, test independently
4. Create `agent_USGS_GeographicImpact` — expose `_Measures` + geographic columns, test
5. Create `agent_USGS_SignificanceAnalyst` — expose `_Measures` only, test
6. Update PROJECTS.md and commit

### Reference — agent_DS_BankChurn pattern
Built in ws_DS_BankChurn on same F2 capacity. Confirmed: Fabric Data Agents available at F2. Exposes both `_Measures` and raw prediction table. USGS agents should expose measures only (except GeographicImpact which needs Country Code and Place Description columns).

---

## Outstanding Items

| Priority | Item | Notes |
|---|---|---|
| **Blocker** | Delete `nb_USGS_Earthquake` scratch notebook | ID: 63e42046 — do this first |
| High | Create agent_USGS_SeismicActivity | Phase 2 step 3 |
| High | Create agent_USGS_GeographicImpact | Phase 2 step 4 |
| High | Create agent_USGS_SignificanceAnalyst | Phase 2 step 5 |
| Medium | Branded header strip on all report pages | Dark rectangle #094780, 40px, full width — deferred by Ali |
| Medium | Drill-through: Geographic -> Event Detail | Wire map/country bar to drill to Page 4 filtered by country |
| Medium | Date notebook guard in pipeline | `nb_USGS_Earthquake_Date` not wired into pipeline as prerequisite |
| Low | Tooltip page for map | Show Title, Magnitude, Significance Class, Place Description on hover |
| Low | Custom theme JSON | Replace CY25SU12 default with intentional Lotusoftware palette |
| Low | Time Intelligence measures info button | button explaining MTD requires filtered date context |

---

## Key Learnings & Gotchas

- `CREATE OR REPLACE TABLE AS SELECT` resets column mapping — always write with `delta.columnMapping.mode=name` at write time, not retrofitted
- `DROP TABLE` + `RENAME TABLE` changes ABFSS path — Direct Lake loses framing. Use `mode("overwrite")` with `.save(ABFSS_PATH)` instead
- `sig_class` column required `ALTER TABLE SET TBLPROPERTIES` before `RENAME COLUMN` — column mapping must be enabled first
- `joinOnDateBehavior: DatePartOnly` required for TimestampType -> DateType relationship — zero rows match otherwise
- `%pip` blocked in pipeline context — use Fabric Environment for library pre-installation
- `powerbi-modeling-mcp` BatchUpdate requires `tableName` inside `updateDefinition`, not just as top-level param
- Direct Lake `Significance Class` column framing failed after multiple overwrites — fixed by writing directly to ABFSS path with column mapping baked in from scratch
- MSAL token is in-process only by default — `SerializableTokenCache` must be persisted to disk for Claude Desktop subprocess launches to authenticate silently
- Fabric Data Agents: available at F2, confirmed on same capacity (alisaghif2capacity). Do not expose raw fact tables to agents — expose measures only to keep answers governed.

---

## MCP Server — fabric_mcp_server.py

**Location:** `C:\Users\alisa\fabric-mcp\fabric_mcp_server.py`  
**Conda env:** `fabric-mcp` (Python 3.11)  
**Token cache:** `~/.fabric_mcp_token_cache.json` (persistent, ~90-day refresh token)  
**Auth fix applied:** `_acquire_token()` called in `main()` before `stdio_server()` opens — device code prints to visible terminal, not swallowed by Claude Desktop subprocess  
**Tools:** `list_workspaces`, `list_workspace_artifacts`, `get_artifact_details`, `trigger_semantic_model_refresh`  
**Claude Desktop config must use:**

    "command": "C:\\ProgramData\\Anaconda3\\envs\\fabric-mcp\\python.exe"

Not `"python"` — base conda env is Python 3.8.5 which lacks `mcp` package.

---

## Design Principles Applied

- powerbi.tips Layout Trifecta awareness (scrim deferred, containers functional)
- VAR/RETURN DAX throughout
- All measures have Copilot `///` descriptions
- Hidden raw columns, display folders, sort-by-column set
- Star schema: fact (Earthquake Events) -> dim (Date)
- Report page names match tab labels (no redundant page titles on Pages 3/4)
- Governance-first agent design: measures as single source of truth, raw tables not exposed to agents

---

## Last Session

> Owner: FabricEngineer (Claude Code)
> Overwrite this section after every execution session.

2026-09-08 — ProjectPlanner planning session (claude.ai). Confirmed F2 capacity supports Fabric Data Agents (validated via ws_DS_BankChurn on same alisaghif2capacity). Designed three-agent architecture on sm_USGS_Earthquake: agent_USGS_SeismicActivity, agent_USGS_GeographicImpact, agent_USGS_SignificanceAnalyst. System prompts drafted for all three. Build order defined. No Fabric items created this session — execution begins next session with FabricEngineer.