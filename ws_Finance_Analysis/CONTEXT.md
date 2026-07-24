# ws_Finance_Analysis — Session Context

> This file is the handoff document for the Finance Analysis project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-22

---

## What We Are Building
An end-to-end Power BI dashboard built entirely inside VS Code using
Claude Code, Fabric, and GitHub. Practice project combining vibe-coding
methodology, Power Designer data modeling, and Rayfin for Fabric app
deployment. Demonstrates the full Lotusoftware consulting delivery stack.

## Workspace
- **Fabric workspace:** ws_Finance_Analysis
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Local path:** C:\Users\alisa\Fabric-Analytics-Projects\ws_Finance_Analysis
- **Power BI project:** C:\Lotusoftware\Power BI_Projects\Power BI_Finance_Analysis

## Source Data
| File | Description |
|---|---|
| customers.csv | Customer master data |
| finance_transactions.csv | Transaction records |

## Business Requirements
- Stakeholder deliverable: executive Power BI dashboard
- Reference documents: Business Requirements.docx in project folder
- Key visuals requested: Executive Summary, Monthly State Tooltip,
  Semantic Model view, Transaction Details

## What Exists So Far
| Item | Type | Status |
|---|---|---|
| Finance Analysis.pbix | Power BI Desktop file | Exists |
| Business Requirements.docx | Requirements doc | Exists |
| customers.csv | Source data | Exists |
| finance_transactions.csv | Source data | Exists |
| Executive Summary.png | Reference screenshot | Exists |
| Monthly State Tooltip.png | Reference screenshot | Exists |
| Semantic Model.png | Reference screenshot | Exists |
| Transaction Details.png | Reference screenshot | Exists |

## Data Model
- **Architecture:** TBD — to be designed with SAP PowerDesigner
- **Fact table:** TBD (likely finance_transactions)
- **Dimensions:** TBD (likely customers, date)
- **Measures:** TBD

## Current Focus
1. Design dimensional model using SAP PowerDesigner
2. Build semantic model in Fabric from the design
3. Create Power BI report pages matching reference screenshots
4. Deploy via Rayfin (Fabric App) for stakeholder access

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Use Fabric MCP to connect to ws_Finance_Analysis
5. Use powerbi-modeling-mcp for semantic model and DAX work
6. Follow skills at C:\Users\alisa\skills-for-fabric:
   - Power BI planning: skills/powerbi-report-planning/SKILL.md
   - Power BI design: skills/powerbi-report-design/SKILL.md
   - Power BI authoring: skills/powerbi-report-authoring/SKILL.md
   - Semantic model: skills/semantic-model-authoring/SKILL.md
7. Reference screenshots in the images/ folder for visual targets

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-07-22  
**Completed:**
- Project initiated
- Datasets and business requirements confirmed
- Vibe-coding workflow discussed with Claude

**Left unfinished:**
- Dimensional model not yet designed
- Semantic model not yet built in Fabric
- Report pages not yet created

**New blockers discovered:**
- None

**Pick up next session at:**
- Open SAP PowerDesigner and design star schema
- Create semantic model in ws_Finance_Analysis
- Build Executive Summary report page first