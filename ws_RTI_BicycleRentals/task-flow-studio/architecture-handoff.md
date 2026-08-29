---
project: existing-fabric-workspace-called
task_flow: event-medallion
created: 2026-08-29
items: 7
deployment_waves: 2
---

# Architecture Handoff — existing-fabric-workspace-called

**Task flow:** event-medallion  
**Date:** 2026-08-29  
**Items:** 7 across 2 waves

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | Eventhouse | Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics |
| Ingestion | Eventstream | Real-time/streaming velocity requires Eventstream for continuous data ingestion |
| Processing | KQL Queryset | KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration |
| Visualization | Real-Time Dashboard | Sub-second streaming data requires Real-Time Dashboard for live operational monitoring |
| Parameterization | Not yet determined | No parameterization signals detected — provide environment_count and deployment_tool |
| Skillset | Not yet determined | No skillset signals detected — provide skillset or team_composition |
| API | N/A | N/A — no API layer needed for this task flow |
| Semantic Model Query Mode | N/A | Not applicable for this task flow |

## Items to Deploy

```yaml
items:
  - id: 1
    name: eventhouse
    type: Eventhouse
    skillset: Eventhouse
    depends_on: []
    purpose: "Time-series/streaming store for KQL Databases"
  - id: 2
    name: eventstream
    type: Eventstream
    skillset: Eventstream
    depends_on: [Eventhouse]
    purpose: "Real-time streaming ingestion for Real-time data"
  - id: 3
    name: kql-queryset
    type: KQLQueryset
    skillset: KQLQueryset
    depends_on: [Eventhouse]
    purpose: "KQL-based data exploration for Transformations"
  - id: 4
    name: real-time-dashboard
    type: KQLDashboard
    skillset: KQLDashboard
    depends_on: [Eventhouse]
    purpose: "Live streaming dashboard for Live monitoring"
  - id: 5
    name: semanticmodel
    type: "Semantic Model"
    skillset: SemanticModel
    depends_on: []
    purpose: "Required by capability mapper (not in event-medallion default)"
  - id: 6
    name: reflex
    type: Activator
    skillset: Reflex
    depends_on: []
    purpose: "Required by capability mapper (not in event-medallion default)"
  - id: 7
    name: lakehouse
    type: Lakehouse
    skillset: Lakehouse
    depends_on: []
    purpose: "Required by capability mapper (not in event-medallion default)"
```

## Deployment Order

```yaml
waves:
  - id: 1
    name: "Wave 1"
    items: [eventhouse]
    parallel: false
  - id: 2
    name: "Wave 2"
    items: [eventstream, kql-queryset, real-time-dashboard, semanticmodel, reflex, lakehouse]
    blocked_by: [1]
    parallel: true
```

## Acceptance Criteria

```yaml
acceptance_criteria:
  - id: AC-1
    type: structural
    criterion: "eventhouse exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=Eventhouse | verify eventhouse"
    target: eventhouse
  - id: AC-2
    type: structural
    criterion: "eventstream exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=Eventstream | verify eventstream"
    target: eventstream
  - id: AC-3
    type: structural
    criterion: "kql-queryset exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=KQLQueryset | verify kql-queryset"
    target: kql-queryset
  - id: AC-4
    type: structural
    criterion: "real-time-dashboard exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=KQLDashboard | verify real-time-dashboard"
    target: real-time-dashboard
  - id: AC-5
    type: structural
    criterion: "semanticmodel exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=SemanticModel | verify semanticmodel"
    target: semanticmodel
  - id: AC-6
    type: structural
    criterion: "reflex exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=Reflex | verify reflex"
    target: reflex
  - id: AC-7
    type: structural
    criterion: "lakehouse exists and is accessible"
    verify: "REST API GET /workspaces/{id}/items?type=Lakehouse | verify lakehouse"
    target: lakehouse
```

## Alternatives Considered

| # | Decision | Option Rejected | Why Rejected |
|---|----------|-----------------|--------------|
| 1 | Storage | Lakehouse | Not selected — Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics |
| 2 | Storage | Warehouse | Not selected — Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics |
| 3 | Storage | SQL Database | Not selected — Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics |
| 4 | Ingestion | Pipeline | Not selected — Real-time/streaming velocity requires Eventstream for continuous data ingestion |
| 5 | Ingestion | Copy Job | Not selected — Real-time/streaming velocity requires Eventstream for continuous data ingestion |
| 6 | Ingestion | Dataflow Gen2 | Not selected — Real-time/streaming velocity requires Eventstream for continuous data ingestion |
| 7 | Processing | Notebook | Not selected — KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration |
| 8 | Processing | Spark Job Definition | Not selected — KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration |
| 9 | Processing | Dataflow Gen2 | Not selected — KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration |
| 10 | Visualization | Power BI Report | Not selected — Sub-second streaming data requires Real-Time Dashboard for live operational monitoring |
| 11 | Visualization | KQL Dashboard | Not selected — Sub-second streaming data requires Real-Time Dashboard for live operational monitoring |
| 12 | Visualization | Direct Lake | Not selected — Sub-second streaming data requires Real-Time Dashboard for live operational monitoring |

## Trade-offs

| # | Trade-off | Benefit | Cost | Mitigation |
|---|-----------|---------|------|------------|
| 1 | Eventhouse chosen over Lakehouse | Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics | Lakehouse remains available if requirements change | Switch via sign-off revision |
| 2 | Eventhouse chosen over Warehouse | Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics | Warehouse remains available if requirements change | Switch via sign-off revision |
| 3 | Eventhouse chosen over SQL Database | Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics | SQL Database remains available if requirements change | Switch via sign-off revision |
| 4 | Eventhouse chosen over Cosmos DB Database | Real-time streaming velocity requires Eventhouse for sub-second ingestion and hot-path analytics | Cosmos DB Database remains available if requirements change | Switch via sign-off revision |
| 5 | Eventstream chosen over Copy Job | Real-time/streaming velocity requires Eventstream for continuous data ingestion | Copy Job remains available if requirements change | Switch via sign-off revision |
| 6 | Eventstream chosen over Data Pipeline | Real-time/streaming velocity requires Eventstream for continuous data ingestion | Data Pipeline remains available if requirements change | Switch via sign-off revision |
| 7 | Eventstream chosen over Dataflow Gen2 | Real-time/streaming velocity requires Eventstream for continuous data ingestion | Dataflow Gen2 remains available if requirements change | Switch via sign-off revision |
| 8 | KQL Queryset chosen over Notebook | KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration | Notebook remains available if requirements change | Switch via sign-off revision |
| 9 | KQL Queryset chosen over Spark Job Definition | KQL skillset defaults to KQL Queryset for ad-hoc log and time-series exploration | Spark Job Definition remains available if requirements change | Switch via sign-off revision |
| 10 | Real-Time Dashboard chosen over Semantic Model | Sub-second streaming data requires Real-Time Dashboard for live operational monitoring | Semantic Model remains available if requirements change | Switch via sign-off revision |
| 11 | Real-Time Dashboard chosen over Report | Sub-second streaming data requires Real-Time Dashboard for live operational monitoring | Report remains available if requirements change | Switch via sign-off revision |

## Deployment Strategy

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Workspace Approach | Dev/Test/Prod workspaces | Environment isolation for safe promotion |
| Environments | Dev + Prod (minimum) | Standard two-environment promotion pattern |
| CI/CD Tool | fabric-cicd Python package | Standard deployment tool for Fabric items |
| Parameterization | Environment Variables | Default single-environment configuration |
| Branching Model | Git-based with dev/test/prod branches | Standard branch-per-environment promotion |

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                           ARCHITECTURE: existing-fabric-workspace-called (event-medallion)                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                      ║
║   ░░ INGESTION (Wave 2) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ║
║   ┌─────────────────────┐                                                                                            ║
║   │ eventstream         │                                                                                            ║
║   │ (Eventstream)       │                                                                                            ║
║   │ [W2]                │                                                                                            ║
║   └─────────────────────┘                                                                                            ║
║              │                                                                                                       ║
║              ▼                                                                                                       ║
║                                                                                                                      ║
║   ░░ STORAGE (Wave 1, Wave 2) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ║
║   ┌─────────────────────┐    ┌─────────────────────┐                                                                 ║
║   │ eventhouse          │    │ lakehouse           │                                                                 ║
║   │ (Eventhouse)        │    │ (Lakehouse)         │                                                                 ║
║   │ [W1]                │    │ [W2]                │                                                                 ║
║   └─────────────────────┘    └─────────────────────┘                                                                 ║
║              │                                                                                                       ║
║              ▼                                                                                                       ║
║                                                                                                                      ║
║   ░░ PROCESSING (Wave 2) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ║
║   ┌─────────────────────┐    ┌─────────────────────┐                                                                 ║
║   │ kql-queryset        │    │ semanticmodel       │                                                                 ║
║   │ (KQLQueryset)       │    │ (Semantic Model)    │                                                                 ║
║   │ [W2]                │    │ [W2]                │                                                                 ║
║   └─────────────────────┘    └─────────────────────┘                                                                 ║
║              │                                                                                                       ║
║              ▼                                                                                                       ║
║                                                                                                                      ║
║   ░░ SERVING (Wave 2) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ║
║   ┌─────────────────────┐                                                                                            ║
║   │ real-time-dashboard │                                                                                            ║
║   │ (KQLDashboard)      │                                                                                            ║
║   │ [W2]                │                                                                                            ║
║   └─────────────────────┘                                                                                            ║
║              │                                                                                                       ║
║              ▼                                                                                                       ║
║                                                                                                                      ║
║   ░░ ALERTING (Wave 2) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ║
║   ┌─────────────────────┐                                                                                            ║
║   │ reflex              │                                                                                            ║
║   │ (Activator)         │                                                                                            ║
║   │ [W2]                │                                                                                            ║
║   └─────────────────────┘                                                                                            ║
║                                                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   No blockers                                                                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

- Diagram reference: diagrams/event-medallion.md