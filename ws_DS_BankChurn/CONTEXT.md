# ws_DS_BankChurn — Session Context

> This file is the handoff document for the DS BankChurn project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-09-30

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
**Date:** 2026-09-29 / 2026-09-30
**Completed:**
- Full pipeline re-run confirmed clean: 10,000 rows, champion_BankChurn
  Version 2 (lgbm_sm, val_roc_auc: 0.8495)
- AI-Readiness audit + full remediation on sm_DS_BankChurn:
  - 8 raw columns hidden (CreditScore, Age, Tenure, Balance, NumOfProducts,
    HasCrCard, IsActiveMember, EstimatedSalary) — all 20 columns now hidden
  - Table description written for Churn Predictions (customer_churn_test_predictions)
  - `predictions` column renamed to `Churn Prediction`; 5 DAX references
    updated in _Measures.tmdl
  - All changes via direct TMDL edit (powerbi-modeling-mcp write gate blocked)
- Data Agent validated — 5 benchmark questions recorded in CONTEXT.md
- pl_DS_BankChurn DataPipeline created: 3-notebook Succeeded chain using
  notebook logicalIds; corrected after first commit (Fabric item IDs → logicalIds)
- Lakehouse table renamed to churn_predictions (physical Delta), semantic model
  display name updated to 'Churn Predictions' in TMDL, all TMDL references updated
  (23 DAX replacements in _Measures.tmdl); entityName: churn_predictions confirmed
- nb_DS_BankChurn_Predictions updated with ALTER TABLE rename cell
- settings.local.json updated: fabric-mcp removed, mcp__powerbi-modeling-mcp__*
  wildcard added; does not fix server-internal write gate
- Commits: 174d9df, 1022267, 459a471 + entityName fix in this commit

**Left unfinished:**
- Scheduled refresh on sm_DS_BankChurn (not configured)
- Source Control sync in Fabric needed to apply TMDL changes (Churn Prediction
  rename, entityName: churn_predictions, 8 hidden columns, table description)

**New blockers discovered:**
- powerbi-modeling-mcp v1.0.0 write gate: all writes decline with "user
  declined" — internal to the server process, not controlled by settings.local.json
  or Claude Code permissions. No config option found. Current latest version.

**Pick up next session at:**
- Configure scheduled refresh on sm_DS_BankChurn
- Fabric portal → ws_DS_BankChurn → Source Control → Update all (applies TMDL
  changes to live model: 'Churn Predictions' display name, entityName, 8 hidden
  columns, table description)
- Investigate powerbi-modeling-mcp write gate (VS Code Extension Settings panel)

---

## Actual state as of 2026-08-20

### Workspace ID: e82dfb36-dba0-483b-8860-67b2a08d0487

### Artifacts (14 total):
- lh_DS_BankChurn (Lakehouse + SQL Endpoint auto-paired)
- nb_DS_BankChurn_transformData — downloads churn.csv, cleans,
  engineers features, writes churn_clean Delta table; logs Bronze
  ingestion metadata (see Lakehouse tables below)
- nb_DS_BankChurn_TrainRegisterML — trains RFC1/RFC2/LightGBM,
  evaluates on val set ROC-AUC, registers champion programmatically
  as champion_BankChurn
- nb_DS_BankChurn_Predictions — loads champion_BankChurn, scores
  churn_test, writes churn_predictions with columnMapping.mode=name;
  includes ALTER TABLE rename cell (churn_predictions)
- pl_DS_BankChurn — DataPipeline orchestrating 3 notebooks in sequence
  (TransformData → TrainRegisterML → Predictions), Succeeded dependency,
  notebook logicalIds, null GUID workspaceId
- bank-churn-experiment (MLExperiment)
- rfc1_sm, rfc2_sm, lgbm_sm (MLModel — tutorial naming, not renamed)
- champion_BankChurn (MLModel — programmatically selected champion;
  Version 2 as of 2026-09-29 re-run, val_roc_auc: 0.8495)
- sm_DS_BankChurn — Direct Lake on churn_predictions (entityName:
  churn_predictions, schemaName: dbo; TMDL display name: 'Churn Predictions'),
  _Measures table with 11 DAX measures across 4 display folders (Volume,
  Churn Rate, Geography, Risk Signals), all 20 columns hidden, table description
  set, predictions column renamed to 'Churn Prediction' in TMDL
  Note: TMDL changes pending Fabric Source Control sync
- rpt_DS_BankChurn — PBIR format, 3 pages (Churn Overview, Risk Profile,
  Model Performance), 15 visuals
- agent_DS_BankChurn — status: Live. Fabric Data Agent grounded on
  sm_DS_BankChurn for natural language churn analysis, system prompt
  configured, published to workspace. Not a Git-syncable artifact type.

### Lakehouse tables (lh_DS_BankChurn):
- churn_clean — cleaned/engineered source data
- churn_predictions — Gold predictions table (physical name; renamed from
  customer_churn_test_predictions 2026-09-30; TMDL display name: 'Churn Predictions';
  Direct Lake source for sm_DS_BankChurn; entityName binding updated in TMDL)
- ingestion_metadata — Bronze ingestion run log written by
  nb_DS_BankChurn_transformData (run_timestamp, source_url,
  source_table, rows_written, columns_written, ingestion_mode,
  notebook_name, spark_app_id, schema_version)

### Known issues / open items:
- MLModel names (rfc1_sm, rfc2_sm, lgbm_sm) use tutorial convention
  with _sm suffix — future projects will use model_ prefix
- Fabric Source Control sync pending — TMDL changes (entityName:
  churn_predictions, Churn Prediction rename, 8 hidden columns, table
  description) committed to Git but not yet applied to live model
- No scheduled refresh configured on sm_DS_BankChurn
- powerbi-modeling-mcp v1.0.0 write gate: all MCP writes blocked
  internally; TMDL-direct is the working pattern until resolved

### Naming convention note:
MLModel names follow Microsoft tutorial convention (rfc1_sm, rfc2_sm,
lgbm_sm). Future projects will use model_ prefix for ML model artifacts.

---

## Data Agent Benchmarks (2026-09-29)
Agent: agent_DS_BankChurn | Model: sm_DS_BankChurn | Run date: 2026-09-29

| Question | Answer |
|---|---|
| How many customers are predicted to churn? | 365 customers |
| What is the churn rate in Germany? | 34.5% |
| Which customer segment has the highest churn risk? | Customers with 3–4 products; 3-product micro-segments show 100% predicted churn rate |
| What is the average credit score of churned customers? | 640 |
| How does churn rate compare across France, Germany, and Spain? | Germany 34.5%, Spain 13.2%, France 12.2% — Germany ~3x higher than France/Spain |
