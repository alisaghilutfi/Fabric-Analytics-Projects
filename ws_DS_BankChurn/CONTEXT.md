# ws_DS_BankChurn — Session Context

> This file is the handoff document for the DS BankChurn project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-26

---

## What We Are Building
A data science and machine learning project on Microsoft Fabric
predicting customer churn for a bank. Demonstrates end-to-end ML
workflow: data ingestion, feature engineering, model training,
evaluation, and Power BI reporting on predictions.

## Workspace
- **Fabric workspace:** ws_DS_BankChurn
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Local path:** C:\Users\alisa\Fabric-Analytics-Projects\ws_DS_BankChurn

## Architecture
- **Pattern:** Medallion (Bronze/Silver/Gold) + ML layer
- **Compute:** Spark notebooks via Fabric Data Engineering
- **ML framework:** PySpark MLlib or scikit-learn via mssparkutils
- **Output:** Predictions table in Gold layer → semantic model →
  Power BI churn dashboard

## Current Focus
Active development — ready for next phase:
1. Profile source data and assess quality
2. Build Bronze ingestion notebook
3. Build Silver feature engineering notebook
4. Train churn prediction model in Gold notebook
5. Expose predictions via semantic model and Power BI report

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Use Fabric MCP to connect to ws_DS_BankChurn
5. Follow skills at C:\Users\alisa\skills-for-fabric:
   - Spark/Lakehouse: skills/spark-authoring-cli/SKILL.md
   - SQL Warehouse: skills/sqldw-authoring-cli/SKILL.md
   - Semantic model: skills/semantic-model-authoring/SKILL.md
   - Power BI report: skills/powerbi-report-authoring/SKILL.md

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-07-26
**Completed:**
- Semantic model measures reorganized into standalone `_Measures` table
  (10 DAX measures across 4 display folders), 11 raw columns hidden
  on customer_churn_test_predictions
- rpt_DS_BankChurn PBIR report built — 3 pages, 15 visuals
- CONTEXT.md and PROJECTS.md updated to reflect actual state

**Left unfinished:**
- Geography bar chart on Churn Overview page is empty (needs redesign)
- customer_churn_test_predictions table not renamed to business-readable name
- No DataPipeline artifact / orchestration
- No scheduled refresh on sm_DS_BankChurn

**New blockers discovered:**
- None

**Pick up next session at:**
- Fix Geography bar chart on Churn Overview page
- Consider DataPipeline orchestration for the notebook sequence

---

## Actual state as of 2026-07-26

### Workspace ID: e82dfb36-dba0-483b-8860-67b2a08d0487

### Artifacts (12 total):
- lh_DS_BankChurn (Lakehouse + SQL Endpoint auto-paired)
- nb_DS_BankChurn_transformData — downloads churn.csv, cleans,
  engineers features, writes churn_clean Delta table
- nb_DS_BankChurn_TrainRegisterML — trains RFC1/RFC2/LightGBM,
  evaluates on val set ROC-AUC, registers champion programmatically
  as champion_BankChurn
- nb_DS_BankChurn_Predictions — loads champion_BankChurn, scores
  churn_test, writes customer_churn_test_predictions with
  columnMapping.mode=name
- bank-churn-experiment (MLExperiment)
- rfc1_sm, rfc2_sm, lgbm_sm (MLModel — tutorial naming, not renamed)
- champion_BankChurn (MLModel — programmatically selected champion)
- sm_DS_BankChurn — Direct Lake on customer_churn_test_predictions,
  _Measures table with 10 DAX measures across 4 display folders
  (Volume, Churn Rate, Geography, Risk Signals), 11 raw columns hidden
- rpt_DS_BankChurn — PBIR format, 3 pages (Churn Overview, Risk Profile,
  Model Performance), 15 visuals

### Known issues / open items:
- MLModel names (rfc1_sm, rfc2_sm, lgbm_sm) use tutorial convention
  with _sm suffix — future projects will use model_ prefix
- customer_churn_test_predictions table name is not business-readable —
  rename to 'Churn Predictions' in a future session
- Geography bar chart on Churn Overview page is empty — visual needs
  redesign to use a dimension column instead of measures as categories
- No DataPipeline artifact — notebooks run manually; pipeline
  orchestration not yet implemented
- No scheduled refresh configured on sm_DS_BankChurn

### Naming convention note:
MLModel names follow Microsoft tutorial convention (rfc1_sm, rfc2_sm,
lgbm_sm). Future projects will use model_ prefix for ML model artifacts.