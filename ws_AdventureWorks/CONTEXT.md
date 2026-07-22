# ws_AdventureWorks — Session Context

> This file is the handoff document for the AdventureWorks project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-22

---

## What We Are Building
A full medallion architecture (Bronze/Silver/Gold) on Microsoft Fabric
for the AdventureWorks dataset, ending with a Power BI report and
semantic model. This is a practice project to demonstrate end-to-end
Fabric engineering with best practices baked in.

## Workspace
- **Fabric workspace:** ws_AdventureWorks
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Local path:** C:\Users\alisa\Fabric-Analytics-Projects\ws_AdventureWorks

## What Exists So Far
| Item | Type | Status |
|---|---|---|
| lh_AdventureWorks.Lakehouse | Lakehouse | Created |
| nb_AdventureWorks_Bronze.Notebook | Notebook | Created |
| nb_AdventureWorks_Silver.Notebook | Notebook | Created |
| nb_AdventureWorks_Gold.Notebook | Notebook | Created |
| nb_AdventureWorks_Date.Notebook | Notebook | Created |
| pl_AdventureWorks.DataPipeline | Pipeline | Created |
| sm_AdventureWorks.SemanticModel | Semantic Model | Scaffolded |
| rpt_AdventureWorks | Power BI Report | Created |

## Data Model
- **Architecture:** Star schema over Gold layer
- **Fact table:** FactSalesOrder
- **Dimensions:** DimProduct, DimCustomer, DimDate
- **Known issue:** DimDate not yet marked as proper date table —
  time intelligence functions may not work until fixed
- **Measures:** None explicit yet — aggregation is implicit

## Current Focus
Validate and complete the semantic model:
1. Mark DimDate as date table
2. Add explicit measures (Sales Total, Order Count, etc.)
3. Build Power BI report pages on top of the model

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Use Fabric MCP to connect to ws_AdventureWorks
5. Use powerbi-modeling-mcp for semantic model work
6. Follow skills at C:\Users\alisa\skills-for-fabric:
   - Semantic model: skills/semantic-model-authoring/SKILL.md
   - Power BI report: skills/powerbi-report-authoring/SKILL.md
   - Spark/Lakehouse: skills/spark-authoring-cli/SKILL.md

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-07-22  
**Completed:**
- Built Bronze, Silver, Gold, Date notebooks
- Scaffolded semantic model and pipeline
- Created report file skeleton

**Left unfinished:**
- DimDate not marked as date table
- No explicit measures created
- Report pages not built

**New blockers discovered:**
- None

**Pick up next session at:**
- Mark DimDate as date table in semantic model
- Add core measures: Sales Total, Order Count, Average Order Value
- Build Overview report page