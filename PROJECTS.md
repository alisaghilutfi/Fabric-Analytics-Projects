# Project Registry

> This file is maintained by agents. After every work session, the
> executing agent updates the relevant project's Last Session and
> Status fields. Do not edit manually unless correcting an error.

Last updated: 2026-08-11

---

## ws_AdventureWorks
**Purpose:** Medallion architecture practice — Bronze/Silver/Gold lakehouse  
**Status:** Active  
**Current focus:** Semantic model and Power BI report layer  
**Last session:** Built Bronze, Silver, Gold notebooks and DataPipeline.
Semantic model scaffolded. Report file created.  
**Next session:** Validate semantic model measures, build Power BI report pages  
**Blockers:** None known  

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
**Status:** Active — full stack complete including Data Agent. Open: DataPipeline orchestration, table rename (customer_churn_test_predictions → Churn Predictions), scheduled refresh  
**Last updated:** 2026-08-20  
**Current focus:** Pipeline orchestration and report polish  
**Last session:** Fixed Geography visual on Churn Overview (pie chart by country), added Bronze ingestion metadata logging (ingestion_metadata Delta table), created and published agent_DS_BankChurn (Fabric Data Agent grounded on sm_DS_BankChurn), Power BI Pro license purchased for alisaghi_fabric account.  
**Next session:** Run full pipeline end-to-end after notebooks were re-run — verify customer_churn_test_predictions is current; consider DataPipeline orchestration for the notebook sequence  
**Blockers:** None known  

---

## ws_RTI_Crypto
**Purpose:** Real-time intelligence — live crypto prices via Eventhouse and KQL  
**Status:** Git blocked — support ticket pending  
**Current focus:** Blocked on Microsoft support; no Fabric work planned until resolved  
**Last session:** Git integration failed with `Git_GitProviderCommitRejectedByPolicy` (request ID `1e57bbf7-fa24-4989-b4e4-caf3f5048441`); workspace disconnected from Git. Support ticket filed with Microsoft. Documented artifact inventory (eh_RTI_Crypto Eventhouse + KQLDatabase, es_RTI_Crypto Eventstream, lh_RTI_Crypto Lakehouse, nb_RTI_Crypto, nb_RTI_Crypto_Automated, auto-generated compaction notebook) in CONTEXT.md.  
**Next session:** Check support ticket status before any Git operation on this workspace  
**Blockers:** Workspace disconnected from Git — `Git_GitProviderCommitRejectedByPolicy`, Microsoft support ticket pending  

---

## ws_Ecommerce_Olist
**Purpose:** End-to-end Fabric analytics on the Olist Brazilian e-commerce dataset — medallion architecture with Power BI reporting  
**Status:** Discovered, synced — build not started  
**Current focus:** Full audit of existing artifacts before adding new work  
**Last session:** Discovered 2026-07-27 — pre-existing workspace already synced to GitHub with a full stack (lh_Ecommerce_Olist Lakehouse; wh_Ecommerce_Olist Warehouse with Gold schema Fact_Sales, Dim_Customers, Dim_Products, Dim_Sellers, Dim_Date, Agg_Customer_Intelligence; nb_Ecommerce_Olist_Bronze/Silver; pl_Ecommerce_Olist DataPipeline; sm_Ecommerce_Olist SemanticModel; rpt_Ecommerce_Olist Report), but no CONTEXT.md existed. Wrote CONTEXT.md to establish baseline.  
**Next session:** Audit Gold schema grain/relationships, build DAX measure library on sm_Ecommerce_Olist, review report pages  
**Blockers:** None known  

---

## ws_RTI_BicycleRentals
**Purpose:** Real-time intelligence project — live bicycle rental station monitoring  
**Status:** Active  
**Current focus:** Reporting layer complete; validation and drill-down of Activator/Dashboard/Anomaly Detector still open  
**Last session:** Audited the workspace and found the KQL medallion (Bronze/Silver/Gold), Eventstream, Activator, Dashboard, Map, and Anomaly Detector already live — CONTEXT.md had been stale. Added a `LatestStationSnapshot()` KQL function (Gold layer, full columns). Built and deployed `sm_RTI_BicycleRentals` (DirectQuery semantic model, 10 measures) and `rpt_RTI_BicycleRentals` (4-page report, PBIR-validated). Synced to git via Fabric's commitToGit (branch `dev-fabric-sync`, commit `a716dab`).  
**Next session:** Validate the 10 measures live via DAX, visually review the report in Desktop/Service, inspect Activator rule logic and Dashboard tiles, decide on the orphaned Lakehouse, investigate Anomaly Detector output  
**Blockers:** None known  

---

## ws_USGS_Earthquake
**Purpose:** Portfolio documentation project  
**Status:** Active  
**Current focus:** Documentation completed  
**Last session:** USGS_Earthquake portfolio documentation generated  
**Next session:** TBD  
**Blockers:** None known  

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