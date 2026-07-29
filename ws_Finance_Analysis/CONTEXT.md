# ws_Finance_Analysis — Session Context

> This file is the handoff document for the Finance Analysis project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-29

---

## What We Are Building
An end-to-end Power BI dashboard built entirely inside VS Code using
Claude Code, Fabric, and GitHub. Practice project combining vibe-coding
methodology, Power Designer data modeling, and Rayfin for Fabric app
deployment. Demonstrates the full Lotusoftware consulting delivery stack.

## Workspace
- **Fabric workspace:** ws_Finance_Analysis
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Local path:** C:\Users\alisa\DS-ML-DL\Fabric-Analytics-Projects\ws_Finance_Analysis
- **Power BI project:** C:\Lotusoftware\Power BI_Projects\Power BI_Finance_Analysis

## Source Data
| File | Description |
|---|---|
| customers.csv | Customer master data — 5,000 rows, profiled, 0 nulls |
| finance_transactions.csv | Transaction records — 50,069 rows, profiled, 8 known data quality issues |

Full profile and remediation plan: `docs/data-profile.md`.

## Business Requirements
- Stakeholder deliverable: executive Power BI dashboard
- Reference documents: Business Requirements.docx in project folder
- Key visuals requested: Executive Summary, Monthly State Tooltip,
  Semantic Model view, Transaction Details

## What Exists So Far (verified against repo filesystem + git history, 2026-07-29)

Built in a single session on **2026-07-18** (commits `3070771`…`a669c0a`, 12:43–16:42).
Git-side build is complete; **not yet verified against the live Fabric workspace**
(see Instructions for Executing Agent below — that verification is the next session's
first task).

| Item | Type | Status |
|---|---|---|
| Finance Analysis.pbix | Power BI Desktop file | Exists |
| Business Requirements.docx | Requirements doc | Exists |
| customers.csv / finance_transactions.csv | Source data | Exists, profiled |
| docs/data-profile.md | Data profile + DQ remediation plan | Exists |
| lh_Finance_Bronze | Lakehouse | Scaffolded in git |
| lh_Finance_Silver | Lakehouse | Scaffolded in git |
| nb_Finance_Bronze | Notebook — raw CSV → Delta | Built in git (pandas CSV read, `dbo` schema prefix) |
| nb_Finance_Silver | Notebook — cleaned star schema | Built in git (DQ fixes from data-profile.md applied) |
| pl_Finance | DataPipeline — orchestrates Bronze → Silver | Built in git, wired to real Fabric object IDs |
| sm_Finance | SemanticModel — DirectLake on Silver | **Live in Fabric, verified 2026-07-29.** 6 tables (`fact_transactions`, `dim_customer`, `dim_date`, `dim_channel`, `dim_merchant`, `_Measures`) with **22 DAX measures** across 6 display folders (KPIs, Amount, Transaction Volume, Time Intelligence, Fees & Tax, Fraud & Risk) — table/measure list confirmed live via powerbi-modeling-mcp, matches git exactly. DirectLake mode confirmed (`targetStorageMode: Abf`), refreshable, created 2026-07-18T11:34 UTC. |
| rpt_Finance | Report — Layout Trifecta | **Live in Fabric, verified 2026-07-29.** PBIR format, correctly bound to sm_Finance dataset. 4 pages / 12 visuals per git; page-by-page visual rendering not yet checked (needs Desktop/Service open). |
| pl_Finance | DataPipeline | **Live in Fabric, verified 2026-07-29** (exists, correct workspace binding). Internal step definitions not inspected via MCP — Fabric MCP's artifact-details call doesn't expose pipeline activity JSON. |
| lh_Finance_Bronze / lh_Finance_Silver | Lakehouses + SQL endpoints | **Live in Fabric, verified 2026-07-29.** Silver lakehouse GUID (`d726b863-...`) matches the ID hardcoded in git's `expressions.tmdl` exactly. |
| Rayfin app deployment | Stakeholder access layer | **Not started** |

## Data Model
- **Architecture:** Built — see `docs/data-profile.md` § Star Schema Design
- **Fact table:** `fact_transactions` (50,000 rows after dedup of 69 exact duplicates)
- **Dimensions:** `dim_customer` (5,000), `dim_date` (1,461), `dim_channel` (7), `dim_merchant` (14)
- **Measures:** 22 DAX measures in `_Measures` table (see sm_Finance.SemanticModel/definition/tables/_Measures.tmdl) — count verified live via powerbi-modeling-mcp on 2026-07-29
- **Note:** `account_id` intentionally kept as a fact attribute, not a separate dimension (no attributes beyond ID)

## Current Focus
Documentation was out of sync with actual repo state as of 2026-07-29 — this file and
PROJECTS.md previously described the project as not-yet-started despite the full build
existing in git since 2026-07-18. That gap has been corrected, and live Fabric state has
been verified against git (see recap below) — all 5 artifacts (2 lakehouses, semantic
model, report, pipeline) exist live and match git for everything MCP tooling can inspect.

**Remaining before genuinely new build work:**
- Visually confirm the 4 report pages render correctly (needs Desktop/Service, not MCP-inspectable)
- Spot-check a few DAX measures execute correctly against live data (values look sane, no errors)
- Inspect pl_Finance's actual pipeline steps (not exposed by current Fabric MCP artifact-details call)
- Rayfin app deployment for stakeholder access — not started at all

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Use Fabric MCP to connect to ws_Finance_Analysis and verify live state:
   - Confirm sm_Finance semantic model exists, tables/relationships/measures match `sm_Finance.SemanticModel/definition/`
   - Confirm rpt_Finance report exists with 4 pages matching `rpt_Finance.Report/definition/`
   - Confirm pl_Finance pipeline exists and references valid notebook/lakehouse object IDs
   - Confirm lh_Finance_Bronze / lh_Finance_Silver lakehouses exist with expected tables
5. Only after live-state verification, use powerbi-modeling-mcp for any new semantic model/DAX work
6. Follow skills at C:\Users\alisa\skills-for-fabric:
   - Power BI planning: skills/powerbi-report-planning/SKILL.md
   - Power BI design: skills/powerbi-report-design/SKILL.md
   - Power BI authoring: skills/powerbi-report-authoring/SKILL.md
   - Semantic model: skills/semantic-model-authoring/SKILL.md
7. Reference screenshots in the images/ folder for visual targets
8. See CLAUDE.md in this folder for naming conventions, TMDL/PBIR format rules, and known Fabric gotchas

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-07-29
**Completed:**
- Audited repo filesystem and git history against CONTEXT.md / PROJECTS.md; found both
  stale — described project as not-started despite a complete build (Bronze/Silver
  notebooks, star schema, semantic model, 4-page report, pipeline) committed to git on
  2026-07-18. Rewrote this file and the PROJECTS.md entry to reflect actual state.
- Verified live Fabric workspace (`ws_Finance_Analysis`, id `61549e76-...`) against git
  via fabric-mcp + powerbi-modeling-mcp:
  - All 5 artifact types present live: lh_Finance_Bronze, lh_Finance_Silver (+ SQL
    endpoints), sm_Finance, rpt_Finance, pl_Finance
  - lh_Finance_Silver's live GUID matches the ID hardcoded in git's `expressions.tmdl` exactly
  - sm_Finance live table list (6 tables) and measure list (22 measures) matches git
    exactly; DirectLake mode confirmed (`targetStorageMode: Abf`)
  - rpt_Finance correctly bound to sm_Finance as its dataset, PBIR format confirmed
  - Corrected an earlier miscount: it's 22 measures, not 25 (a grep artifact — 3 `///`
    description lines contained the word "measure" in prose)

**Left unfinished:**
- Report pages not visually inspected (Fabric MCP can't render/screenshot; needs
  Desktop or Service)
- DAX measures not spot-checked for correct output values, only existence/naming
- pl_Finance's internal pipeline steps not inspected (current Fabric MCP artifact-details
  call doesn't expose pipeline activity JSON — would need a different tool or the Fabric portal)
- Rayfin app deployment not started

**New blockers discovered:**
- None. Git/live parity for everything MCP-inspectable is confirmed — the earlier
  concern about a "fragile sync process" (many `fix:` commits) did not manifest as
  drift; those commits already resolved the issues before the final sync.

**Pick up next session at:**
- Open rpt_Finance in Power BI Service or Desktop, visually confirm all 4 pages render
  and the 12 visuals show sane data
- Spot-check 2-3 measures (e.g. Total Amount, % Fraud Rate) for correct values
- Then decide: polish/fix existing build vs. start Rayfin deployment
