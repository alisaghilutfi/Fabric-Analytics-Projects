# Project Registry

> This file is maintained by agents. After every work session, the
> executing agent updates the relevant project's Last Session and
> Status fields. Do not edit manually unless correcting an error.

Last updated: 2026-07-26

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
**Purpose:** End-to-end Power BI vibe-coding project with Claude Code  
**Status:** Active  
**Current focus:** Power BI dashboard on finance_transactions.csv and customers.csv  
**Last session:** Project initiated with datasets and business requirements loaded  
**Next session:** Build semantic model, create report pages  
**Blockers:** None known  

---

## ws_DS_BankChurn
**Purpose:** Data science / ML — customer churn prediction  
**Status:** Active — semantic model and report complete, pipeline pending  
**Last updated:** 2026-07-26  
**Current focus:** Pipeline orchestration and report polish  
**Last session:** Reorganized semantic model measures into standalone `_Measures` table (10 DAX measures, 4 display folders), hid 11 raw columns on customer_churn_test_predictions, built rpt_DS_BankChurn PBIR report (3 pages, 15 visuals).  
**Next session:** Fix empty Geography bar chart on Churn Overview page; consider DataPipeline orchestration for the notebook sequence  
**Blockers:** None known  

---

## ws_RTI_Crypto
**Purpose:** Real-time intelligence — live crypto prices via Eventhouse and KQL  
**Status:** Active  
**Current focus:** Portfolio documentation completed  
**Last session:** RTI_Crypto portfolio documentation generated  
**Next session:** TBD  
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

## ws_dp600
**Purpose:** DP-600 exam preparation workspace  
**Status:** Reference  
**Current focus:** Reference only — exam passed  
**Last session:** Not applicable  
**Next session:** Not applicable  
**Blockers:** Not applicable  

---

## ws_AgenticLab
**Purpose:** Agentic AI experimentation  
**Status:** Active  
**Current focus:** Not yet started in this session cycle  
**Last session:** Not recorded  
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