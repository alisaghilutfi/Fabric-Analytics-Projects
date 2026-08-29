# Existing Fabric Workspace Called

> event-medallion architecture on Microsoft Fabric | 2026-08-29 | VALIDATED ✅

## The Problem

I have an existing Fabric workspace called ws_RTI_BicycleRentals built around Real-Time Intelligence. It contains an Eventhouse (eh_RTI_BicycleRentals) with one KQL database, an Eventstream (es_RTI_BicycleRentals), an Activator (activator_BicycleRentals), an Anomaly Detection item (anomalies_BicycleRentals), a Real-Time Dashboard (dashboard_BicycleRentals), a Map (map_RTI_BicycleRentals), a Lakehouse (lh_RTI_BicycleRentals), a Semantic Model (sm_RTI_BicycleRentals), and a Report (rpt_RTI_BicycleRentals). The workspace is connected to a GitHub repo on the dev-fabric-sync branch, last synced 7/27/2026. The Fabric capacity is PAYG in North Europe. I want you to read what is already deployed, document the current RTI architecture, identify any gaps, and describe what a complete end-to-end deployment would look like. Do not create any new items.

## What We Built

### Fabric Items (7 items, 2 deployment waves)

| Wave | Items |
|------|-------|
| 1: Wave 1 | eventhouse |
| 2: Wave 2 | eventstream, kql-queryset, real-time-dashboard, semanticmodel, reflex, lakehouse |

## Why This Architecture

### Task Flow: event-medallion

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | Eventhouse | Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics |
| Ingestion | Eventstream | Real-time/streaming velocity requires Eventstream for continuous data ingestion |
| Processing | KQL Queryset | KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration |
| Visualization | Real-Time Dashboard | Sub-second streaming data requires Real-Time Dashboard for live operational monitoring |

## How to Deploy

**Tool:** fabric-cicd
**Script:** `deploy/deploy-existing-fabric-workspace-called.py`

```bash
cd _projects/existing-fabric-workspace-called/deploy
python deploy-existing-fabric-workspace-called.py
```

**Status:** Artifacts generated — deploy when ready.

## Validation Summary

| Check | Result |
|-------|--------|
| All 7 items generated | ✅ |
| Structural validation passed | ✅ |
| Deploy artifacts complete | ✅ |
| Live data-flow validation | ⏳ Pending deployment |

### Edge Cases Identified

- Eventhouse storage quota exceeded — verify ingestion pauses with clear error
- Concurrent KQL queries during heavy ingestion — verify query latency stays acceptable
- Source disconnects mid-ingestion — verify eventstream reconnects or alerts
- Malformed event payload — verify eventstream handles schema drift gracefully
- KQL query against empty table — verify graceful empty result (no error)

## What's Next

- Deploy to live Fabric workspace when ready
- Run data-flow validation after deployment
- Connect source systems (CRM, ERP, spreadsheets)
