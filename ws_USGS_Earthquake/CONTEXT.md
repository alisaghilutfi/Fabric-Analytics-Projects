# ws_USGS_Earthquake — Session Context

> This file is the handoff document for the USGS Earthquake project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-22

---

## What We Are Building
A data analytics and portfolio project on Microsoft Fabric ingesting
live USGS earthquake data via REST API, processing it through a
medallion architecture, and exposing insights via KQL analytics and
a Power BI dashboard. Demonstrates real-world API ingestion, geospatial
data handling, and end-to-end Fabric engineering.

## Workspace
- **Fabric workspace:** ws_USGS_Earthquake
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Local path:** C:\Users\alisa\Fabric-Analytics-Projects\ws_USGS_Earthquake

## Data Source
- **API:** USGS Earthquake Hazards Program REST API
- **Endpoint:** https://earthquake.usgs.gov/fdsnws/event/1/
- **Data:** Real-time and historical earthquake events globally
- **Key fields:** magnitude, location, depth, time, coordinates

## Architecture
- **Ingestion:** Notebook calling USGS REST API → Bronze Lakehouse
- **Bronze:** Raw API response stored as Delta tables
- **Silver:** Cleaned, typed, deduplicated earthquake records
- **Gold:** Aggregated insights — by region, magnitude range, time period
- **Analytics:** KQL queries for time-series and geospatial analysis
- **Reporting:** Power BI dashboard with map visuals and magnitude trends

## Current Focus
Active development — ready for next phase:
1. Validate Bronze ingestion notebook against live USGS API
2. Build Silver cleaning and typing notebook
3. Build Gold aggregation notebook
4. Create KQL queries for time-series analysis
5. Build Power BI dashboard with map and magnitude visuals

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Use Fabric MCP to connect to ws_USGS_Earthquake
5. Follow skills at C:\Users\alisa\skills-for-fabric:
   - Spark/Lakehouse: skills/spark-authoring-cli/SKILL.md
   - Eventhouse: skills/eventhouse-authoring-cli/SKILL.md
   - Semantic model: skills/semantic-model-authoring/SKILL.md
   - Power BI report: skills/powerbi-report-authoring/SKILL.md

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-07-22
**Completed:**
- Portfolio documentation generated
- Git integration confirmed
- CONTEXT.md created — ready for active development

**Left unfinished:**
- Silver and Gold notebooks not yet validated
- KQL queries not yet written
- Power BI dashboard not yet built

**New blockers discovered:**
- None

**Pick up next session at:**
- Validate Bronze ingestion notebook against live USGS API
- Build Silver cleaning notebook