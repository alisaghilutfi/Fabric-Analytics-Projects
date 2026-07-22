# ws_DS_BankChurn — Session Context

> This file is the handoff document for the DS BankChurn project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-22

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
**Date:** 2026-07-22
**Completed:**
- Git integration confirmed
- CONTEXT.md created — ready for active development

**Left unfinished:**
- Bronze ingestion notebook not yet built
- Feature engineering not yet started
- Model training not yet started

**New blockers discovered:**
- None

**Pick up next session at:**
- Profile source data
- Build Bronze ingestion notebook