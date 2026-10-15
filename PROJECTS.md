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
**Status:** Blocked — rpt_RTI_BicycleRentals not provisioned; all programmatic paths fail on Trial capacity  
**Current focus:** Service UI "Create report" from sm_RTI_BicycleRentals — only remaining untested provisioning path  
**Last session (2026-10-14):** Fixed sm_RTI_BicycleRentals OAuth2 credentials (DMTS_DatasourceHasNoCredentialError). Validated all 10 DAX measures live (Total Stations = 120, Network Utilization = 48.1%). Full manifest.json 404 diagnosis on rpt_RTI_BicycleRentals: PBIR files are valid (0 errors), but all programmatic provisioning paths confirmed failing on Trial capacity — Git sync, Fabric API (POST /reports + updateDefinition), and Desktop Bridge (PID 28068, bridge connected) all produce 404. Report deleted from workspace; not recreated. .pbip project manifest created. learnings.md created with 4 findings (commit 8429af1).  
**Next session:** (1) FabricEngineer: list workspace items via MCP and delete any duplicate/orphan rpt_RTI_BicycleRentals artifacts; (2) Ali: Fabric Service UI → sm_RTI_BicycleRentals → "Create report" → save as rpt_RTI_BicycleRentals; (3) FabricEngineer: download new report definition via API, diff against PBIR in Git, commit to dev-fabric-sync  
**Blockers:** manifest.json 404 on all programmatic provisioning paths — confirmed Trial capacity limitation. Duplicate/orphan report items may exist in workspace from failed publish attempts — need MCP cleanup. rpt_RTI_BicycleRentals does not currently exist in workspace.  

---

## ws_USGS_Earthquake
**Purpose:** End-to-end USGS seismic analytics — medallion lakehouse, Direct Lake semantic model, 4-page Power BI report, three domain-scoped Fabric Data Agents, Ontology item, and Rayfin Fabric App  
**Status:** Active — Phase 3b in progress  
**Current focus:** Phase 3b — FabricApp_USGS_Earthquake scaffold complete; connector install + entity + UI pending  
**Last session (2026-10-15):** Source control audit (dashboard delete blocked, Bronze phantom diff pending). Phase 3b scaffold: FabricApp_USGS_Earthquake created with create-rayfin@1.36.2; data-modeling/connectors/analytics packs enabled; sm_usgs_earthquake connector wired; connector packages not yet installed.  
**Next session:** Step 11 — install connector packages; Step 12 — inspect sm_usgs_earthquake; Step 13 — generate entities; build app UI; rayfin up  
**Blockers:** sm connector packages not installed (Step 11 pending); dashboard_USGS_Earthquake manual portal delete pending; nb_Bronze phantom diff Undo pending in Fabric Source Control  

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

## ws_CompetitorSales
**Purpose:** Competitive sales intelligence — Sintec vs. 13 manufacturers 
across 6 countries, ~1.8M transactions 2014–2021  
**Status:** Active — report complete, AI-Readiness pending  
**Current focus:** AI-Readiness pass (table descriptions, data categories, 
agent_CompetitorSales)  
**Last session (2026-10-13):** SQLBI table naming reverted — physical Gold names
confirmed live and in Git. datedimension[MonthYear] sortByColumn set to
MonthYearCode; datedimension[MonthName] sortByColumn set to MonthNumber (both
via XMLA, committed). Visual titles applied across all 4 report pages.
rpt_CompetitorSales source control committed. docs/images/ folder created,
dashboard_CompetitorSales.png added, README updated with image reference.
HARNESS.md dash_ prefix corrected to dashboard_. Retrospective completed —
6 workflow improvement findings documented.  
**Next session:** AI-Readiness pass — table descriptions (5 tables), data
categories on Geography columns, Prep data for AI, agent_CompetitorSales  
**Blockers:** None

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
