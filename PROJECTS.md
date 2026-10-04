# Project Registry

> This file is maintained by agents. After every work session, the
> executing agent updates the relevant project's Last Session and
> Status fields. Do not edit manually unless correcting an error.

Last updated: 2026-10-04

---

## ws_Finance_Analysis
**Purpose:** End-to-end Power BI vibe-coding project with Claude Code; also the practice project for a three-environment Fabric/GitHub CI/CD setup  
**Status:** Complete  
**Current focus:** None — open for future enhancements (Rayfin, client-driven report additions)  
**Last session:** Holistic `rpt_Finance` redesign via agentic authoring loop (powerbi-authoring skill + Desktop Bridge): Layout Trifecta applied to all three pages (Overview, Transactions, Trends), Customers page removed with KPIs redistributed, teal/gray/near-white scrim zones established, x-axis labels polished manually in Desktop, `dash_Finance_Analysis` dashboard created in Fabric Service, dashboard screenshot embedded in README.  
**Next session:** Open for future enhancements (Rayfin, client-driven report additions)  
**Blockers:** None known  
- GitHub Actions workflow live: `.github/workflows/fabric-refresh.yml`
  - Triggers `sm_Finance` semantic model refresh on every push to `main`
  - Target: `ws_Finance_Analysis_Prod` (workspace ID: c0f9d7bf-7649-43d5-9fff-065f454db778)
  - Dataset ID: 69576dc1-8364-4e37-bc7c-77650ef8264c
  - Auth: Service Principal `sp-fabric-cicd` (client ID: f0128254-14f0-44d8-9e17-09c567e742d2), client credentials flow, no MSAL dependency
  - SP role: Contributor on Prod workspace
  - Secret rotation due: January 2027
  - All three branches (main, test, dev-fabric-sync) in sync as of 2026-08-03

---

## ws_DS_BankChurn
**Purpose:** Data science / ML — customer churn prediction  
**Status:** Active — full stack complete including Data Agent and DataPipeline. Open: Fabric Source Control sync, scheduled refresh  
**Last updated:** 2026-09-30  
**Current focus:** AI-readiness and pipeline polish complete; pending Fabric Source Control sync to apply TMDL changes  
**Last session (2026-09-29/30):** Full pipeline re-run (10,000 rows, champion_BankChurn v2, val_roc_auc: 0.8495); AI-readiness remediation on sm_DS_BankChurn (8 raw columns hidden → all 20 hidden, table description, predictions → Churn Prediction rename, 5 DAX references updated); Data Agent validated with 5 benchmarks; pl_DS_BankChurn DataPipeline built (3-notebook Succeeded chain, logicalIds, null GUID workspaceId); lakehouse table renamed customer_churn_test_predictions → churn_predictions (physical), TMDL display name updated to 'Churn Predictions', 23 DAX references updated in _Measures.tmdl.  
**Next session:** Configure scheduled refresh on sm_DS_BankChurn; Fabric portal → Source Control → Update all (apply TMDL changes to live model); investigate powerbi-modeling-mcp write gate  
**Blockers:** powerbi-modeling-mcp v1.0.0 write gate hardcoded — all MCP writes blocked; TMDL-direct is working pattern. Fabric Source Control sync required to apply committed TMDL changes to live model.  

---

## ws_RTI_BicycleRentals
**Purpose:** Real-time intelligence project — live bicycle rental station monitoring  
**Status:** Active  
**Current focus:** Event-medallion pattern confirmed via Task Flow Studio; open items are hot/cold split verification and Map/Anomaly Detection data-binding confirmation  
**Last session (2026-08-29):** Ran a Task Flow Studio pass over the workspace — confirmed the event-medallion pattern (Eventstream → Eventhouse Bronze/Silver/Gold → semantic model/report). Committed 13 Task Flow Studio docs (discovery brief, project brief, architecture handoff, decisions, test plan, validation report, deployment handoff, caches) to `ws_RTI_BicycleRentals/task-flow-studio/` in `main`. Resolved GitHub drift — all branches (main, test, dev-fabric-sync) now synced.  
**Next session:** Verify the hot/cold split assumption (Eventhouse → Lakehouse handoff for historical data); confirm Map (`map_RTI_BicycleRentals`) and Anomaly Detection (`anomalies_BicycleRentals`) data bindings; still-open from prior session: validate the 10 semantic model measures live via DAX, visually review the report in Desktop/Service, inspect Activator rule logic and Dashboard tiles  
**Blockers:** None known  

---

## ws_USGS_Earthquake
**Purpose:** End-to-end USGS seismic analytics — medallion lakehouse, Direct Lake semantic model, 4-page Power BI report, three domain-scoped Fabric Data Agents, and Ontology item  
**Status:** Active — Phase 3 in progress  
**Current focus:** Phase 3b Rayfin (deferred), report polish still open  
**Last session:** Git sync fixed for all three DataAgents (logicalId byte-swap root cause). Phase 3a complete: onto_USGS_Earthquake created with Earthquake Events + Date entities, validated against live data.  
**Next session:** Phase 3b — Rayfin app, OR report polish (branded header, drill-through)  
**Blockers:** None  

---

## ws_AaltoEE_Forecast
**Purpose:** Net sales forecasting prototype for Aalto EE — actuals (Jeeves),
accruals (project delivery), and CRM pipeline (Dynamics 365) in a governed
medallion pipeline with DirectLake semantic model, 4-page Power BI report,
and Fabric Data Agent  
**Status:** Prototype complete — open for productionisation  
**Current focus:** Production ingestion (Dataverse shortcut + SharePoint
connector) not yet implemented — blocked on Aalto EE credentials  
**Last session (2026-09-28):** AI-readiness enrichment of sm_AaltoEE_Forecast
(42 columns hidden, 7 date categories, 7 table descriptions, agent validation
passed); Page 4 Snapshot Comparison built; Snapshot Amount EUR measure added
for correct version-comparison filter context; powerbi-modeling-mcp upgraded
to 1.0.0. Commits: dc2c828, acf4ca0  
**Next session:** Production ingestion when Aalto EE credentials available;
otherwise synonym workaround via Power BI Desktop  
**Blockers:** Production ingestion requires Aalto EE credentials — not
available in prototype environment. Synonyms blocked at DirectLake
compatibility level.

---

## Adding a New Project
When a new workspace is created, add a new section above following
this exact template:

## ws_<name>
**Purpose:** <what this workspace is for>  
**Status:** Active / Reference / Paused  
**Current focus:** <what we are working on right now>  
**Last session:** <what was done last time>  
**Next session:** <where to pick up>  
**Blockers:** <anything blocking progress>

---

## Archived
Workspaces kept for reference only — not active portfolio items.

| Workspace | Reason |
|---|---|
| ws_RTI_Crypto | Superseded by ws_RTI_BicycleRentals |
| ws_DigitalTwin_Bus | Deprioritised |
| ws_AgenticLab_* | Exam/trial workspaces |
| ws_AdventureWorks | Sample content only |
| ws_Ecommerce_Olist | Replaced by ws_Ecommerce (Whiskique) |
| ws_Contoso | Sample content only |
| ws_dp700_*, ws_dp600 | DP exam prep — retired |
