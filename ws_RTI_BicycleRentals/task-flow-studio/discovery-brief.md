## Discovery Brief

**Project:** existing-fabric-workspace-called
**Date:** 2026-08-29

### Problem Statement

> I have an existing Fabric workspace called ws_RTI_BicycleRentals built around Real-Time Intelligence. It contains an Eventhouse (eh_RTI_BicycleRentals) with one KQL database, an Eventstream (es_RTI_BicycleRentals), an Activator (activator_BicycleRentals), an Anomaly Detection item (anomalies_BicycleRentals), a Real-Time Dashboard (dashboard_BicycleRentals), a Map (map_RTI_BicycleRentals), a Lakehouse (lh_RTI_BicycleRentals), a Semantic Model (sm_RTI_BicycleRentals), and a Report (rpt_RTI_BicycleRentals). The workspace is connected to a GitHub repo on the dev-fabric-sync branch, last synced 7/27/2026. The Fabric capacity is PAYG in North Europe. I want you to read what is already deployed, document the current RTI architecture, identify any gaps, and describe what a complete end-to-end deployment would look like. Do not create any new items.

### 4 V's Assessment

| V | Value | Source |
|---|-------|--------|
| Volume | Not sized numerically; bike/dock telemetry, likely low-to-moderate GB/day | inferred |
| Velocity | Real-time streaming — Eventstream + Eventhouse + Real-Time Dashboard | user |
| Variety | Single streaming source (bike/dock events) plus a Lakehouse for curated data | inferred |
| Versatility | Mixed — low-code RTI authoring alongside Lakehouse/Semantic Model path | inferred |

### Inferred Signals

| Signal | Value | Confidence | Source |
|--------|-------|------------|--------|
| Real-time / Streaming | Event analytics | high | "real-time" |
| Machine Learning | Anomaly / pattern detection | high | "anomaly detection" item |
| Alerting & Triggers | Reactive automation | high | Activator + Anomaly Detection items present |
| Layered / Historical Store | Lakehouse alongside Eventhouse | medium | Lakehouse item present |
| Semantic Self-Service BI | Governed reporting | high | Semantic Model + Report items |

### Task Flow Candidates

| Candidate | Score | Why It Fits |
|-----------|-------|-------------|
| event-medallion | 8 | Real-time ingest already paired with a Lakehouse (hot + cold path) |
| event-analytics | 6 | Core RTI loop: Eventstream → Eventhouse → Dashboard already deployed |
| basic-machine-learning-models | 5 | Anomaly Detection item implies ML/pattern-discovery workload |

### Architectural Judgment Calls

- Treat this as an audit, not a build — no new items; existing item inventory is authoritative.
- Assume Eventhouse → Lakehouse is a shortcut-transform hot/cold split pending verification during design.
- GitHub sync (dev-fabric-sync, last synced 7/27/2026) implies drift risk to check in Design.
- Map and Anomaly Detection items are non-standard RTI additions — confirm their data bindings during Design.

### Confirmed with User

- Problem statement, deployed item inventory, and PAYG/North Europe capacity context taken as stated.
- 4 V's above are inferred from the described architecture, not separately elicited — flagged for correction during design if wrong.
- Required capabilities identified: streaming-ingest, low-latency-event-store, semantic-self-service, alerting-trigger, historical-medallion-store (coverage 0.40 after LLM augmentation).
- Scope is read-only documentation + gap analysis; explicit instruction to create no new items carried forward to Design/Deploy phases.
