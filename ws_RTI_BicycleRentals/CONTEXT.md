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
FabricApp_RTI_BicycleRentals is live. Open items:
- Root.spec.tsx scaffold placeholder tests (3 failures — non-blocking, update to reflect real app entry)
- Optional: neighbourhood filter on Network Overview, mark-resolved flow on Rebalancing Queue
- rpt_RTI_BicycleRentals still does not exist in workspace (Trial capacity limitation — not the active focus)

**Operational note:** `es_RTI_BicycleRentals` Eventstream may go inactive and stop feeding the KQL pipeline.
If KQL row counts are stale, restart via Fabric portal → open `es_RTI_BicycleRentals` → **Activate all → Now**.

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

### Reporting Layer
- **sm_RTI_BicycleRentals** (SemanticModel) — DirectQuery on
  `LatestStationSnapshot()` via `AzureDataExplorer.Contents`. Table
  `Station Snapshot` (BikepointID, Street, Neighbourhood, Latitude/Longitude
  visible; Timestamp, No_Bikes, No_Empty_Docks, BikesToBeFilled, Action
  hidden). `_Measures` table with 10 measures across Availability / Capacity /
  Rebalancing / Recency display folders, all with Copilot-ready descriptions.
  **Credentials fixed 2026-10-14:** OAuth2 bound via Service → Data source
  credentials → AzureDataExplorer (root cause: DMTS_DatasourceHasNoCredentialError).
  Validated live: Total Stations = 120, Network Utilization % = 48.1%.
  **Station History table added 2026-10-31:** DirectQuery KQL partition —
  `TransformedData | where Timestamp > ago(2h) | summarize total_bikes = sum(No_Bikes) by bin(Timestamp, 15m)`.
  Powers the sparkline chart in `FabricApp_RTI_BicycleRentals` via DAX
  `EVALUATE 'Station History'`. **Do not remove or rename this table —
  the Rayfin app depends on it.**
- **rpt_RTI_BicycleRentals** (Report) — PBIR validated (0 errors). **BLOCKED:**
  Does not exist in workspace — all programmatic provisioning paths (Git sync,
  Fabric API, Desktop Bridge) produce `manifest.json 404` on Trial capacity.
  Not the active focus; deferred.
- **FabricApp_RTI_BicycleRentals** (Rayfin App) — 3-screen operational tool
  connecting directly to `sm_RTI_BicycleRentals`. Screens: Network Overview
  (KPI cards + 120-station table), Station Detail (per-station stats + flag
  action), Rebalancing Queue (Fill/Empty stations + flag button). AlertSubscription
  entity for write-back. Rayfin item ID: `3d942f93-b1f3-4e01-abf8-e664b0ff969b`.
  **Live URL: https://humid-gem-4f2e8b4d0a-northeurope.webapp.fabricapps.net**

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
**Date:** 2026-10-22

**Completed:**
- es_RTI_BicycleRentals Eventstream restarted via Fabric portal → Activate all → Now
  (had gone inactive; KQL row counts were stale)
- sm_RTI_BicycleRentals connector smoke-tested: EVALUATE query against `Station Snapshot`
  confirmed live (Total Stations = 120, Network Utilization % = 48.1%)
  — root cause of prior smoke-test failures: Windows PowerShell strips inner double-quotes
  from single-quoted strings passed to external processes; fix: `--% ` stop-parsing token
  with doubled double-quotes: `npx rayfin --% connector invoke ... --input "{""query"":""...""}"`
- FabricApp_RTI_BicycleRentals built and deployed:
  - Scaffold: `npm create @microsoft/rayfin@latest` (universal-app template confirmed working)
  - Rayfin CLI 1.36.2; packs: connectors, analytics, data-modeling
  - Connector: sm_rti_bicyclerentals (fabric-semanticmodel, delegated auth, version "1")
  - App.tsx: 3 screens — Network Overview (KPI cards + 120-station table),
    Station Detail (per-station stats + flag action), Rebalancing Queue (Fill/Empty + flag)
  - AlertSubscription entity: stationId (int), street, neighbourhood, action,
    flaggedAt, resolvedAt (optional), userId
  - KPI label truncation fixed: 4 short labels (AVG BIKES, LOW STOCK %, UTILIZATION %,
    LAST SNAPSHOT) replace long default names
  - Pre-deploy gates: typecheck ✅, build ✅, lint ✅, 60/63 tests ✅
    (3 failures are scaffold EmptyStatePreview placeholder tests — non-blocking)
  - Deployed: `npx rayfin up --workspace-id 7bdd5482-61f9-4cb6-b70e-42723751f60f`
  - Live URL: https://humid-gem-4f2e8b4d0a-northeurope.webapp.fabricapps.net
  - Rayfin item ID: 3d942f93-b1f3-4e01-abf8-e664b0ff969b
- docs/images/ created with 4 screenshots; README updated with Live App section
- All code committed to dev-fabric-sync

**Pick up next session at:**
1. Update Root.spec.tsx: replace EmptyStatePreview placeholder assertions with
   real app entry assertions (heading "Bicycle Network Monitor", KPI cards present)
2. Smoke-test live app in browser — verify connector queries and AlertSubscription writes
3. Optional: neighbourhood filter dropdown on Network Overview screen
4. Optional: "Mark resolved" flow on Rebalancing Queue (set resolvedAt on AlertSubscription rows)
5. rpt_RTI_BicycleRentals: try Fabric Service UI "Create report" from sm_RTI_BicycleRentals

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