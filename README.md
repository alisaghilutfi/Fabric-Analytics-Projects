# Fabric Analytics Projects

End-to-end Microsoft Fabric analytics projects built by 
[Ali Saghi](https://www.linkedin.com/in/ali-saghi-/), 
DP-600 certified Microsoft Fabric Analytics Engineer and 
founder of [Lotusoftware](https://lotusoftware.hashnode.dev) 
— a data analytics consultancy in Vantaa, Finland.

Each project demonstrates a full analytics workload on Microsoft 
Fabric: from raw data ingestion through medallion architecture 
to semantic models, Power BI reports, and real-time dashboards.

---

## Projects

### ws_AdventureWorks
**Type:** Data Engineering + Business Intelligence  
**Stack:** Medallion architecture (Bronze/Silver/Gold), Spark 
notebooks, DataPipeline, DirectLake semantic model, Power BI report  
**Status:** Active development  
**Focus:** End-to-end star schema over AdventureWorks sales data, 
built entirely from VS Code using Claude Code and Fabric Skills.

---

### ws_Finance_Analysis
**Type:** Business Intelligence  
**Stack:** Medallion architecture, DirectLake semantic model, 
Power BI report, vibe-coding methodology  
**Status:** Active development  
**Focus:** Finance analytics dashboard built via Claude Code 
without touching the Fabric UI — demonstrated in 
[this blog post](https://lotusoftware.hashnode.dev/vibe-coding-an-end-to-end-finance-analytics-platform-in-microsoft-fabric-with-claude-code).

---

### ws_DS_BankChurn
**Type:** Data Science + Machine Learning  
**Stack:** Spark notebooks, PySpark MLlib, medallion architecture, 
semantic model, Power BI churn dashboard  
**Status:** Active development  
**Focus:** Customer churn prediction model with end-to-end ML 
pipeline from raw data to Power BI predictions report.

---

### ws_RTI_Crypto
**Type:** Real-Time Intelligence  
**Stack:** Eventstream, Eventhouse, KQL Database, Lakehouse, 
Spark notebooks, KQL dashboards  
**Status:** Active  
**Focus:** Live crypto price ingestion every few seconds, 
processed in-stream, stored for historical analysis, 
exposed via KQL queries and dashboards.

---

### ws_RTI_BicycleRentals
**Type:** Real-Time Intelligence  
**Stack:** Eventstream, Eventhouse, KQL Database, Lakehouse, 
Power BI live dashboard  
**Status:** Active development  
**Focus:** Real-time operational monitoring of a bicycle 
rental network via live event stream processing.

---

### ws_USGS_Earthquake
**Type:** Data Engineering + Analytics  
**Stack:** REST API ingestion, medallion architecture, 
Spark notebooks, KQL analytics, Power BI map visuals  
**Status:** Active development  
**Focus:** Live USGS earthquake data ingested via REST API, 
processed through Bronze/Silver/Gold layers, with geospatial 
Power BI dashboard.

---

## Development Stack

| Layer | Tool |
|---|---|
| AI coding agent | Claude Code for VS Code |
| Fabric operations | Fabric MCP |
| Power BI / DAX | powerbi-modeling-mcp |
| Fabric grounding | skills-for-fabric (CLAUDE.md) |
| Context management | HARNESS.md + PROJECTS.md + CONTEXT.md |
| Azure auth | Azure CLI v2.88.0 |
| Planning | claude.ai (Fabric & Power BI MCP project) |

All projects follow the two-harness agentic workflow described 
in this 
[blog post](https://lotusoftware.hashnode.dev/stop-re-prompting-how-i-built-a-two-harness-agentic-workflow-for-microsoft-fabric).

---

## Workflow Architecture

This repo serves as the **context harness** for all Fabric 
development sessions:

- [`HARNESS.md`](./HARNESS.md) — master architecture and 
  tool reference
- [`PROJECTS.md`](./PROJECTS.md) — living project registry, 
  updated by agents after every session
- `ws_<name>/CONTEXT.md` — per-project session handoff 
  document, read at session start and written at session end

---

## About Lotusoftware

Lotusoftware is a data analytics consultancy founded by 
Ali Saghi in Vantaa, Finland. Specializing in Microsoft 
Fabric, Power BI, and agentic BI systems.

- Blog: [lotusoftware.hashnode.dev](https://lotusoftware.hashnode.dev)
- LinkedIn: [ali-saghi-](https://www.linkedin.com/in/ali-saghi-/)
- GitHub: [alisaghilutfi](https://github.com/alisaghilutfi)
- X: [@alis05111](https://x.com/alis05111)