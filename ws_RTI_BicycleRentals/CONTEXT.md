# ws_RTI_BicycleRentals — Session Context

> This file is the handoff document for the RTI BicycleRentals project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-08-29

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
- **rpt_RTI_BicycleRentals** (Report) — 4 pages: Live Network Overview (KPI
  row, azureMap, Top-10 restock bar chart, Neighbourhood slicer), Station
  Detail (table with Fill/Empty conditional formatting, Action + Neighbourhood
  slicers), Rebalancing Ops (Restock vs Emptying bar chart, Total Bikes To
  Move card, Neighbourhood × Action matrix), About (data lineage + freshness
  card). PBIR validated with `powerbi-report-author validate` (0 errors).

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
**Date:** 2026-08-29
**Completed:**
- Ran a Task Flow Studio pass over the workspace; confirmed the
  event-medallion pattern (Eventstream → Eventhouse Bronze/Silver/Gold →
  semantic model/report) matches actual implementation
- Committed 13 Task Flow Studio docs to
  `ws_RTI_BicycleRentals/task-flow-studio/` on `main`
- Resolved GitHub drift — `main`, `test`, and `dev-fabric-sync` branches back
  in sync

**Left unfinished:**
- Hot/cold split assumption (Eventhouse → Lakehouse historical handoff) not
  verified against actual data flow — `lh_RTI_BicycleRentals` still orphaned
- Map (`map_RTI_BicycleRentals`) data bindings not confirmed
- Anomaly Detection (`anomalies_BicycleRentals`) data bindings not confirmed
- Carried over from 2026-07-24: DAX measure validation (EVALUATE over the 10
  measures), visual report review in Desktop/Service, Activator rule logic
  inspection, Dashboard tile inspection, decision on orphaned Lakehouse

**New blockers discovered:**
- None

**Pick up next session at:**
- Verify hot/cold split assumption (Eventhouse → Lakehouse)
- Confirm Map and Anomaly Detection data bindings
- Validate the 10 measures live (DAX EVALUATE) and the report visually
  (Desktop reload + screenshot review)
- Inspect activator_BicycleRentals rule logic and document
- Inspect dashboard_BicycleRentals tile definitions and document
- Decide: wire lh_RTI_BicycleRentals for historical Delta layer or remove
- Investigate anomalies_BicycleRentals output destination