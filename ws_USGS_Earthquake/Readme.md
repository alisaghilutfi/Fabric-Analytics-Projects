# USGS Earthquake Analytics — Microsoft Fabric End-to-End Pipeline

> **Lotusoftware Portfolio Project** · Microsoft Fabric · PySpark · DAX · Power BI · MCP  
> Certified: DP-600 Microsoft Fabric Analytics Engineer

---

## Overview

An end-to-end analytics solution built entirely in Microsoft Fabric that ingests live seismic event data from the USGS Earthquake API, processes it through a Bronze → Silver → Gold medallion architecture on OneLake, and surfaces it in a 4-page Power BI Direct Lake report — all automated by a Data Factory pipeline and extensible via a custom MCP server.

**Key capabilities demonstrated:**
- Medallion Lakehouse architecture (Delta Lake on OneLake) with idempotent MERGE-based upserts
- PySpark data engineering with vectorized `@pandas_udf` reverse geocoding at Spark scale
- Direct Lake semantic model with 17-measure DAX library across 5 display folders
- Time intelligence measures (MTD, MOM %, 7-Day Rolling) with `DatePartOnly` relationship
- Automated orchestration via Data Factory pipeline with parameterised -7d rolling date window
- Custom MCP server (MSAL device code auth, persistent token cache, Fabric/Power BI REST APIs)
- 4-page Power BI report: Overview Dashboard, Time Intelligence, Geographic Distribution, Event Detail

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Microsoft Fabric Workspace                    │
│                         ws_USGS_Earthquake                           │
│                                                                      │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │   Data        │    │   Lakehouse   │    │   Semantic Model     │  │
│  │   Pipeline    │───▶│ lh_USGS_      │───▶│ sm_USGS_Earthquake   │  │
│  │ pl_USGS_      │    │ Earthquake    │    │ (Direct Lake)        │  │
│  │ Earthquake    │    │               │    └──────────┬───────────┘  │
│  └──────────────┘    │ Files/        │               │               │
│         │            │  └─ raw JSON  │    ┌──────────▼───────────┐  │
│    ┌────▼────┐        │               │    │   Power BI Report    │  │
│    │ Bronze  │        │ Tables/dbo/   │    │ pbir_USGS_Earthquake │  │
│    │Notebook │───────▶│  ├─ silver   │    │ 4 pages              │  │
│    └────┬────┘        │  ├─ gold     │    └──────────────────────┘  │
│    ┌────▼────┐        │  └─ Date     │                               │
│    │ Silver  │        │    Dimension │                               │
│    │Notebook │───────▶│              │                               │
│    └────┬────┘        └───────────────┘                              │
│    ┌────▼────┐                                                       │
│    │  Gold   │  @pandas_udf vectorized reverse geocoding             │
│    │Notebook │  MERGE upsert → country_code + sig_class              │
│    └─────────┘                                                       │
└─────────────────────────────────────────────────────────────────────┘

External: USGS Earthquake API → https://earthquake.usgs.gov/fdsnws/event/1/
```

---

## Workspace Artifacts

| Artifact | Type | Description |
|---|---|---|
| `lh_USGS_Earthquake` | Lakehouse | OneLake storage — raw files + Delta tables |
| `nb_USGS_Earthquake_Bronze` | Notebook | API ingest → raw JSON to Files/ |
| `nb_USGS_Earthquake_Silver` | Notebook | JSON parse, reshape, timestamp cast → Delta MERGE upsert |
| `nb_USGS_Earthquake_Gold` | Notebook | Vectorized reverse geocode, sig classification → Delta MERGE upsert |
| `nb_USGS_Earthquake_Date` | Notebook | Date dimension generation 2024–2027 (one-time setup) |
| `env_USGS_Earthquake` | Environment | Spark environment with `reverse_geocoder 1.5.1` pre-installed |
| `pl_USGS_Earthquake` | Data Pipeline | Orchestrates Bronze → Silver → Gold, -7d rolling window |
| `sm_USGS_Earthquake` | Semantic Model | Direct Lake, 3 tables, 17 DAX measures |
| `pbir_USGS_Earthquake` | Report | 4-page Power BI report |

---

## Data Flow

### Bronze — Raw Ingest
Parameterised by `start_date` / `end_date` (injected by pipeline as `utcNow() - 7d` → `utcNow()`). Calls the USGS GeoJSON feed and writes raw `features` array to:
```
/lakehouse/default/Files/{start_date}_earthquake_data.json
```

### Silver — Reshape & Type-Cast
Reads Bronze file with explicit `StructType` schema (no inference). Explodes nested GeoJSON geometry and properties, converts epoch-millisecond timestamps to `TimestampType`. **MERGE upsert on `id`** — idempotent on re-run.

**Silver schema:** `id`, `longitude`, `latitude`, `elevation`, `title`, `place_description`, `sig`, `mag`, `magType`, `time` (timestamp), `updated` (timestamp)

### Gold — Enrich & Classify
Reads Silver (filtered to `time > to_timestamp(start_date)`), applies two enrichments:

1. **Country code** via vectorized `@pandas_udf` using `rg.search(list_of_tuples, mode=2)` — batch KD-tree lookup, not row-serial
2. **Significance classification** via Spark `when()`:
   - `sig < 100` → Low
   - `100 ≤ sig < 500` → Moderate
   - `sig ≥ 500` → High

**MERGE upsert on `id`** into `earthquake_events_gold`. Column mapping enabled (`delta.columnMapping.mode = name`).

---

## Semantic Model

**Mode:** Direct Lake (OneLake → `earthquake_events_gold` + `DateDimension`)

**Relationship:**
```
Date[Date] (1) ──── (*) Earthquake Events[Event Time]
joinOnDateBehavior: DatePartOnly  ← bridges DateType → TimestampType
```

**17-Measure DAX Library:**

| Folder | Measure | Format |
|---|---|---|
| Volume | Total Earthquakes | #,0 |
| Magnitude | Avg Magnitude | 0.00 |
| Magnitude | Max Magnitude | 0.00 |
| Magnitude | Min Magnitude | 0.00 |
| Magnitude | Magnitude Band | text |
| Significance | % High Significance | 0.0% |
| Significance | % Moderate Significance | 0.0% |
| Significance | % Low Significance | 0.0% |
| Significance | Avg Significance | #,0 |
| Significance | Max Significance | #,0 |
| Time | Earliest Event Date | dd MMM yyyy |
| Time | Latest Event Date | dd MMM yyyy |
| Time Intelligence | Total Earthquakes MTD | #,0 |
| Time Intelligence | Total Earthquakes PMTD *(hidden)* | #,0 |
| Time Intelligence | Total Earthquakes MOM % | +0.0%;-0.0% |
| Time Intelligence | Avg Magnitude MTD | 0.00 |
| Time Intelligence | 7-Day Rolling Earthquakes | #,0 |

All measures use VAR/RETURN pattern, include descriptions, and are organised into display folders. Raw/technical columns hidden from report field pickers.

---

## Report Pages

| Page | Purpose | Key Visuals |
|---|---|---|
| Overview Dashboard | Executive summary — all-time KPIs | 5 KPI cards, daily line chart, magnitude vs. volume combo chart |
| Time Intelligence | Temporal analysis with MTD/MOM context | Date slicer, 5 KPI cards, 7-day rolling line, monthly bar, significance donut |
| Geographic Distribution | Global spread via reverse geocoding | Bubble map, top countries bar, significance distribution stacked bar, country slicer |
| Event Detail | Drill-through event explorer | Magnitude conditional formatting, slicers, sortable event log table |

---

## Pipeline Orchestration

```
Bronze_Notebook (retry: 2, timeout: 1h)
    └──[Succeeded]──▶ Silver_Notebook (timeout: 2h)
                          └──[Succeeded]──▶ Gold_Notebook (timeout: 6h)
```

All three activities parameterised with `start_date = utcNow() - 7d`. Pipeline is fully idempotent — safe to re-run on the same date window due to MERGE-based upserts in Silver and Gold.

---

## MCP Server (`fabric_mcp_server.py`)

Custom Python MCP server exposing Fabric REST API operations to Claude Desktop.

**Authentication:** MSAL device code flow with **persistent token cache** (`~/.fabric_mcp_token_cache.json`) — authenticates once, loads silently on subsequent launches (~90-day refresh token lifetime).

**Conda env:** `fabric-mcp` (Python 3.11)  
**Dependencies:** `mcp 1.28.1`, `msal 1.37.0`, `httpx 0.28.1`

| Tool | API | Description |
|---|---|---|
| `list_workspaces` | Fabric v1 | List workspaces with optional name filter |
| `list_workspace_artifacts` | Fabric v1 | List items by workspace, optional type filter |
| `get_artifact_details` | Fabric v1 / Power BI v1 | Routed by item type |
| `trigger_semantic_model_refresh` | Power BI v1 | POST refresh, returns RequestId |

---

## Setup & Deployment

### Prerequisites
- Microsoft Fabric capacity (F2 minimum for Direct Lake)
- Python 3.11 + conda
- Claude Desktop with MCP support

### One-Time Setup

```bash
# 1. Clone repo
git clone https://github.com/alisagihilutfi/Fabric-Analytics-Projects.git

# 2. Connect ws_USGS_Earthquake folder to a Fabric workspace via Git integration

# 3. Run Date dimension notebook (one-time)
# Execute nb_USGS_Earthquake_Date — populates DateDimension table

# 4. Install MCP server dependencies
conda create -n fabric-mcp python=3.11
conda activate fabric-mcp
pip install mcp msal httpx

# 5. Register fabric-mcp in Claude Desktop config
# Set command to full path: C:\ProgramData\Anaconda3\envs\fabric-mcp\python.exe
```

### Running the Pipeline

Trigger `pl_USGS_Earthquake` manually or schedule daily at 06:00 UTC. On first run, backfills 7 days. All subsequent re-runs are idempotent.

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| Delta MERGE over append | Idempotent pipeline re-runs — append causes silent row inflation |
| Vectorized `@pandas_udf` for geocoding | `rg.search(list, mode=2)` is O(n log n) batch vs O(n) row-serial UDF |
| Fabric Environment for `reverse_geocoder` | `%pip` is disabled in pipeline context — pre-install via Environment |
| ABFSS path write for Gold table | Prevents path drift after DROP/RENAME operations that break Direct Lake framing |
| `joinOnDateBehavior: DatePartOnly` | Bridges `TimestampType` (Gold) → `DateType` (DateDimension) without physical column rewrite |
| Persistent MSAL token cache | Prevents device code auth deadlock when Claude Desktop launches MCP server as subprocess |
| `delta.columnMapping.mode = name` | Required for `ALTER TABLE RENAME COLUMN` on existing Delta tables |

---

## Tech Stack

![Microsoft Fabric](https://img.shields.io/badge/Microsoft_Fabric-0078D4?style=flat&logo=microsoft&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)
![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-003366?style=flat)
![MCP](https://img.shields.io/badge/MCP_Server-Custom-purple?style=flat)

- **Fabric:** Lakehouse, Notebooks (Synapse PySpark), Data Factory, Spark Environment, Direct Lake Semantic Model, Power BI Report
- **Storage:** OneLake (Delta Lake / Parquet), column mapping enabled
- **Compute:** Synapse Spark 3.5 / Delta 3.2, Power BI Analysis Services
- **Auth:** MSAL device code flow, persistent SerializableTokenCache
- **Languages:** Python 3.11, DAX, PySpark, JSON, TMDL

---

## Author

**Ali Saghi** · DP-600 Microsoft Fabric Analytics Engineer · Lotusoftware  
[GitHub](https://github.com/alisagihilutfi)

---

*Data source: [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/fdsnws/event/1/) — public domain seismic event data.*
