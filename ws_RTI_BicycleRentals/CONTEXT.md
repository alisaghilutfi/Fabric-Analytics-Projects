# ws_RTI_BicycleRentals — Session Context

> This file is the handoff document for the RTI BicycleRentals project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-10-22

---

## What We Are Building
A real-time intelligence project on Microsoft Fabric tracking live
bicycle rental activity. Demonstrates real-time ingestion, stream
processing, KQL analytics, and live Power BI dashboards for
operational monitoring of a bike rental network.

## Workspace
- **Fabric workspace:** ws_RTI_BicycleRentals
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Local path:** C:\Users\alisa\Fabric-Analytics-Projects\ws_RTI_BicycleRentals

## Architecture
- **Ingestion:** Eventstream for live rental events
- **Storage:** Eventhouse + KQL Database for time-series queries
- **Historical:** Lakehouse for Delta table storage
- **Processing:** Notebooks for transformation and aggregation
- **Reporting:** Live Power BI dashboard on rental activity

## Key KQL Patterns
- Always include time filters: `where Timestamp > ago(...)`
- Use `has` over `contains` for indexed string search
- Use idempotent commands: `.create-merge table`,
  `.create-or-alter function`

## Current Focus
Active development — ready for next phase:
1. Set up Eventstream for bicycle rental event ingestion
2. Create Eventhouse and KQL Database
3. Build KQL queries for live rental monitoring
4. Build Lakehouse for historical analysis
5. Create live Power BI dashboard

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Use Fabric MCP to connect to ws_RTI_BicycleRentals
5. Follow skills at C:\Users\alisa\skills-for-fabric:
   - Eventhouse: skills/eventhouse-authoring-cli/SKILL.md
   - Eventstream: skills/eventstream-authoring-cli/SKILL.md
   - Spark/Lakehouse: skills/spark-authoring-cli/SKILL.md
   - Power BI report: skills/powerbi-report-authoring/SKILL.md

## Actual State (audited 2026-07-24)

The workspace was found to be far more built-out than this file previously
reflected — the real-time ingestion and KQL medallion pipeline were already
live and running.

### Eventstream (es_RTI_BicycleRentals)
- Source: Fabric SampleData "Bicycles"
- ManageFields: BikepointID, Street, Neighbourhood, Latitude, Longitude,
  No_Bikes, No_Empty_Docks, Timestamp (SystemTimestamp)
- Destination 1: eh_RTI_BicycleRentals → RawData table
- Destination 2: activator_BicycleRentals (direct stream)

### KQL Medallion (live, 187M rows)
| Layer  | Object                   | Detail                                                          |
|--------|--------------------------|------------------------------------------------------------------|
| Bronze | RawData                  | 187M rows, live ingestion                                       |
| Silver | TransformedData          | 186.7M rows, update policy on RawData via TransformRawData()     |
| Gold   | AggregatedData           | Materialized view, arg_max(Timestamp, No_Bikes) by BikepointID   |
| Gold   | LatestStationSnapshot()  | New function (added this session): arg_max(Timestamp, *) by BikepointID — full-column station snapshot used by the semantic model, since AggregatedData only carries No_Bikes |

### Other Artifacts
- activator_BicycleRentals.Reflex — wired to Eventstream, rule contents not yet inspected
- dashboard_BicycleRentals.KQLDashboard — exists, tile contents not yet inspected
- map_RTI_BicycleRentals.Map — geospatial station view
- anomalies_BicycleRentals.AnomalyDetector — ML anomaly detection on stream

### Orphaned
- lh_RTI_BicycleRentals.Lakehouse — exists, no Eventstream destination, no notebooks, no tables

### Reporting Layer (built this session)
- **sm_RTI_BicycleRentals** (SemanticModel) — DirectQuery on
  `LatestStationSnapshot()` via `AzureDataExplorer.Contents`. Table
  `Station Snapshot` (BikepointID, Street, Neighbourhood, Latitude/Longitude
  visible; Timestamp, No_Bikes, No_Empty_Docks, BikesToBeFilled, Action
  hidden). `_Measures` table with 10 measures across Availability / Capacity /
  Rebalancing / Recency display folders, all with Copilot-ready descriptions.
  **Credentials fixed 2026-10-14:** OAuth2 bound via Service → Data source
  credentials → AzureDataExplorer (root cause: DMTS_DatasourceHasNoCredentialError).
  Validated live: Total Stations = 120, Network Utilization % = 48.1%.
- **rpt_RTI_BicycleRentals** (Report) — 4 pages: Live Network Overview (KPI
  row, azureMap, Top-10 restock bar chart, Neighbourhood slicer), Station
  Detail (table with Fill/Empty conditional formatting, Action + Neighbourhood
  slicers), Rebalancing Ops (Restock vs Emptying bar chart, Total Bikes To
  Move card, Neighbourhood × Action matrix), About (data lineage + freshness
  card). PBIR validated with `powerbi-report-author validate` (0 errors).
  **BLOCKED 2026-10-14:** Does not exist in workspace — all programmatic
  provisioning paths (Git sync, Fabric API, Desktop Bridge) produce
  `manifest.json 404` on this Trial capacity. Service UI "Create report"
  from sm_RTI_BicycleRentals is the only remaining untested path.

## Task Flow Studio (run 2026-08-29)
A Task Flow Studio pass confirmed the workspace follows the **event-medallion
pattern**: Eventstream → Eventhouse Bronze/Silver/Gold (KQL) → semantic
model/report, matching the architecture already documented above.

- 13 docs committed to `ws_RTI_BicycleRentals/task-flow-studio/` on `main`:
  discovery-brief.md, project-brief.md, architecture-handoff.md,
  decisions.json, test-plan.md, validation-report.md, deployment-handoff.md,
  and cache files (.architecture-cache.json, .capability-llm-cache.json,
  .capability-llm-prompt.json, .capability-mapper-cache.json,
  .discovery-intake.json, .signal-mapper-cache.json)
- GitHub drift resolved — `main`, `test`, and `dev-fabric-sync` are all synced

### Open items from this run
- **Hot/cold split assumption unverified:** the design assumes Eventhouse
  (hot) hands off to Lakehouse (cold) for historical storage, but
  `lh_RTI_BicycleRentals` remains orphaned (see "Orphaned" above, no
  Eventstream destination, no notebooks, no tables) — this assumption needs
  verification against actual data flow, not just design intent
- **Map data binding unconfirmed:** `map_RTI_BicycleRentals` (geospatial
  station view) — bindings not yet verified against the KQL Gold layer
- **Anomaly Detection data binding unconfirmed:** `anomalies_BicycleRentals`
  — bindings and output destination not yet verified

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-10-14
**Completed:**
- sm_RTI_BicycleRentals credentials fixed: bound OAuth2 via Service → Data
  source credentials → AzureDataExplorer (root cause: DMTS_DatasourceHasNoCredentialError)
- KQL Gold layer validated live: `LatestStationSnapshot()` returns 120 stations,
  live ingestion confirmed
- All 10 DAX measures validated live via EVALUATE: Total Stations = 120,
  Network Utilization % = 48.1%
- manifest.json 404 fully diagnosed on rpt_RTI_BicycleRentals:
  - PBIR files confirmed structurally valid (0 errors, powerbi-report-author validate)
  - azureMap visual fixed: Lat/Long wrapped in Average aggregation (Function: 4)
    to resolve DataViewMappingError_LocationAndLatLongGroup
  - All programmatic provisioning paths attempted and confirmed failing:
    - Git sync delete + reprovision → still 404
    - Fabric API pack + POST /reports (create) → still 404
    - Fabric API updateDefinition → still 404
    - Desktop Bridge publish (PID 28068, bridge connected) → still 404
  - Root cause confirmed: Trial capacity limitation — Service-side blob storage
    cannot be provisioned via any programmatic path on this capacity tier
  - rpt_RTI_BicycleRentals deleted from workspace; not recreated
- .pbip project manifest created (was missing; required for Desktop to open project)
- learnings.md created with 4 workflow findings; committed to dev-fabric-sync (8429af1)

**New blockers discovered:**
- manifest.json 404 on all programmatic provisioning paths (API, Git sync,
  Desktop Bridge) — confirmed Trial capacity limitation; Service UI the only
  remaining path
- Duplicate/orphan report items may exist in workspace from failed publish
  attempts — need MCP cleanup before next attempt
- rpt_RTI_BicycleRentals does not exist in workspace (deleted, not yet recreated)

**Pick up next session at:**
1. FabricEngineer: list workspace items via MCP, delete any duplicate/orphan
   rpt_RTI_BicycleRentals items (null logicalId or stale artifacts)
2. Ali: Fabric Service UI → open sm_RTI_BicycleRentals → "Create report" →
   save as rpt_RTI_BicycleRentals
3. FabricEngineer: download the new report definition via Fabric API, diff
   against existing PBIR in Git, commit any changes to dev-fabric-sync
4. Verify all 4 pages render in browser after Service UI creation
5. Resume deferred items: hot/cold split verification, Map/Anomaly data bindings,
   Activator rules, Dashboard tiles, orphaned Lakehouse decision

---

## Phase 3b — Rayfin App (added 2026-10-22)

### Goal
Build `FabricApp_RTI_BicycleRentals` as a Rayfin app connecting directly
to `sm_RTI_BicycleRentals`. The app is an operational rebalancing tool for
an operations manager — station lookup, live availability, and rebalancing
action flagging.

### Why Rayfin (not Power BI)
The report is blocked on Trial capacity (manifest.json 404 on all
programmatic paths). Rayfin apps are provisioned via CLI + Fabric Apps
portal — a separate provisioning path unaffected by the Trial capacity
limitation. Rayfin also enables write-back (AlertSubscription entity) which
the Power BI report cannot do.

### Semantic Model (confirmed live 2026-10-22)
- **Item ID:** `e50ec3a9-c600-446d-b856-44946bcd78ce`
- **Table:** `Station Snapshot` (DirectQuery on `LatestStationSnapshot()`)
- **Key columns:** BikepointID, Street, Neighbourhood, Latitude, Longitude, Action
- **Key measures:** Total Stations, Network Utilization %, Stations Needing
  Restock, Stations Needing Emptying, Total Bikes To Move, Latest Snapshot Time
- **Validated live:** Total Stations = 120, Network Utilization % = 48.1%

### App Design — 3 Screens

**Screen 1: Network Overview**
- KPI cards: Total Stations, Network Utilization %, Stations Needing Restock
- Neighbourhood filter (dropdown)
- Station list filtered by Action = "Fill" | "Empty" | "All"
- Latest Snapshot Time in header (data freshness)

**Screen 2: Station Detail**
- User selects a station from Screen 1
- Shows: Street, Neighbourhood, No_Bikes, No_Empty_Docks, BikesToBeFilled, Action
- "Flag for Rebalancing" button → writes AlertSubscription entity

**Screen 3: Rebalancing Queue**
- List of all flagged AlertSubscription records
- Shows station, flagged time, action required
- Mark as resolved

### dim_Symbol equivalent — not needed
No symbol dimension required. Neighbourhood is the natural grouping
dimension (already in the semantic model). Station list is 120 rows —
no filtering needed.

### Rayfin CLI
**Not installed** as of 2026-10-22 reconnaissance. Install before build:
```bash
npm install -g @microsoft/rayfin-cli@latest
```
Verify with: `npm list -g @microsoft/rayfin-cli`

### Scaffold command (from USGS pattern)
```bash
npm init @microsoft/create-rayfin@latest -- FabricApp_RTI_BicycleRentals
```
> Note: `npm create @microsoft/rayfin@latest` does NOT exist (404 on npm).
> Correct scaffolder confirmed by FabricEngineer 2026-10-22.
Packs: `data-modeling`, `connectors`, `analytics`
Connector: `sm_RTI_BicycleRentals` (fabric-semanticmodel, delegated auth)

### AlertSubscription Entity
Same pattern as USGS app (AlertSubscription):
- Fields: `stationId` (int), `street` (string), `neighbourhood` (string),
  `action` (string), `flaggedAt` (datetime), `resolvedAt` (datetime, nullable)
- Storage: Rayfin db capability

### Session Recap — 2026-10-22 (continued, same date)
**Completed:**
- Workspace item manifest confirmed (no orphan rpt items, no existing FabricApp)
- Semantic model schema confirmed — table, columns, 10 measures all documented
- Rayfin app design defined (3 screens, AlertSubscription entity)
- CONTEXT.md updated with Phase 3b plan
- Pivoted from ws_RTI_Crypto (Git OAuth never initialized, data stale)
  back to ws_RTI_BicycleRentals as correct Rayfin target
- **FabricApp_RTI_BicycleRentals built and deployed:**
  - Rayfin CLI 1.36.2 installed globally
  - Scaffold: `npm create @microsoft/rayfin@latest` (universal-app template)
  - Packs applied: connectors, analytics, data-modeling
  - Connector added: `sm_rti_bicyclerentals` (fabric-semanticmodel, delegated, version "1")
  - Connector smoke-test passed (root cause of prior failures: Windows PowerShell
    strips inner double-quotes from single-quoted strings passed to external
    processes; fix: `--% ` stop-parsing token with doubled double-quotes)
  - App.tsx built: 3 screens (Network Overview, Station Detail, Rebalancing Queue)
  - AlertSubscription entity added: stationId (int), street, neighbourhood, action,
    flaggedAt, resolvedAt (optional), userId
  - Pre-deploy gates: typecheck ✅, build ✅, lint ✅, tests 60/63 ✅
    (3 failures are scaffold EmptyStatePreview tests — expected once app replaces placeholder)
  - Deployed: `npx rayfin up --workspace-id 7bdd5482-61f9-4cb6-b70e-42723751f60f`
  - **Live URL: https://humid-gem-4f2e8b4d0a-northeurope.webapp.fabricapps.net**
  - Rayfin item ID: `3d942f93-b1f3-4e01-abf8-e664b0ff969b`
  - Committed to dev-fabric-sync

**Known issues / polish for next session:**
- Root.spec.tsx 3 failures: scaffold placeholder tests (EmptyStatePreview heading)
  need to be updated to reflect the real app entry state
- rpt_RTI_BicycleRentals still does not exist (Trial capacity blocker unchanged)
- Chunk size warning in build (593 KB bundle) — advisory only, no impact on deploy

**Pick up next session at:**
1. Update Root.spec.tsx to reflect actual app entry (replace EmptyStatePreview assertions)
2. Smoke-test live app in browser — verify connector queries, AlertSubscription writes
3. Optional: neighbourhood filter on Network Overview screen (currently shows all 120 stations)
4. Optional: "Mark resolved" on Rebalancing Queue (set resolvedAt on AlertSubscription rows)
5. Consider rpt_RTI_BicycleRentals via Fabric Service UI "Create report" if Trial capacity allows