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
Now also serving as the practice project for a three-environment CI/CD
setup (Dev/Test/Prod) across Fabric + GitHub.

## Workspace

### CI/CD environment map (as of 2026-07-29)
The single `ws_Finance_Analysis` workspace was split into three environments,
each a distinct Fabric workspace connected to its own GitHub branch:

| Fabric workspace | GitHub branch | Sync status |
|---|---|---|
| ws_Finance_Analysis_Dev (renamed from ws_Finance_Analysis) | `dev-fabric-sync` | Existing, active dev environment |
| ws_Finance_Analysis_Test | `test` | Created 2026-07-29, connected to folder `/ws_Finance_Analysis`, synced at commit `f1807664` (7/7 artifacts) |
| ws_Finance_Analysis_Prod | `main` | Created 2026-07-29, connected to folder `/ws_Finance_Analysis`, synced at commit `f1807664` (7/7 artifacts) |

- `test` branch created on GitHub from `main`, pushed `fb5039c..f180766`
- All references elsewhere in this file to "ws_Finance_Analysis" as a Fabric
  workspace now mean **ws_Finance_Analysis_Dev** unless stated otherwise —
  that is where the day-to-day build work in this file's history happened
  and continues to happen.

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
CI/CD environment setup (Dev/Test/Prod) is in progress. Three Fabric workspaces now
exist mapped to three GitHub branches (see table above); Test and Prod are freshly
synced at commit `f1807664` with all 7 artifacts. **CI/CD Step 4 is not yet done:**
`.github/` workflow files have been generated but **not committed to the repo** —
do not assume any GitHub Actions automation is live until that commit lands.

**Remaining CI/CD work:**
- Commit the generated `.github/` workflow files (Step 4)
- Decide and document the promotion flow (dev-fabric-sync → test → main) and whether
  it's manual PR-based or automated
- Validate that a change pushed to `test` actually lands correctly in
  ws_Finance_Analysis_Test, and same for `main` → ws_Finance_Analysis_Prod

**Still open from the build-QA track (unchanged since last session):**
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
**Date:** 2026-07-31 (session 3 — sm_Finance model cleanup + PBIR fix)
**Completed:**
- Connected live to `ws_Finance_Analysis_Dev` / `sm_Finance` via powerbi-modeling-mcp (XMLA)
- **Group 1 — column cleanup** (`batch_column_operations`, 1 transaction, 6/6 succeeded):
  hid `_Measures[Column]`, `fact_transactions[is_fraud_bool]`, `fact_transactions[is_reversal]`,
  `fact_transactions[risk_score]` (also set `summarizeBy: none`); set
  `dim_customer[annual_income]` to `summarizeBy: none`; hid `dim_customer[second_name]`
- **Group 2 — table display renames** (`batch_table_operations` BatchRename, 1 transaction,
  5/5 succeeded): `fact_transactions`→`Transactions`, `dim_customer`→`Customers`,
  `dim_date`→`Date`, `dim_channel`→`Channel`, `dim_merchant`→`Merchant`. Confirmed TOM
  auto-cascades renames into dependent DAX — verified all 22 measures in `_Measures`
  already referenced the new table names with zero manual edits needed
- **Group 3 — table descriptions** (`table_operations` Update ×5, all succeeded): added
  `///` descriptions to Transactions, Customers, Date, Channel, Merchant; partition names
  and `sourceLineageTag` confirmed unchanged (DirectLake framing intact)
- **PBIR report fix:** scanned `rpt_Finance.Report/definition/pages/**/visual.json` for
  stale `Entity`/`queryRef` references to the old table names — found 17 `Entity` +
  matching `queryRef` occurrences across 11 visual.json files (`dim_channel`/`dim_merchant`
  had zero references). Applied find-and-replace (31 total string replacements), then
  validated JSON syntax on all 11 files — all passed
- **Local git sync gap caught before commit:** discovered the MCP writes went straight to
  the live XMLA endpoint and never touched local TMDL — `git status` showed zero changes
  under `sm_Finance.SemanticModel/` despite the live model being fully updated. Ran
  `database_operations` `ExportToTmdlFolder` to pull the live model back down; this
  correctly renamed the table `.tmdl` files (`dim_date.tmdl` deleted, `Date.tmdl` created,
  etc.) and captured all Group 1–3 changes as real diffs
- Committed and pushed to `dev-fabric-sync`; opened PR into `test`

**Left unfinished:**
- Live re-verification of the exported TMDL against the git-tracked semantic model
  definition not yet done in a fresh session (only spot-checked `Transactions.tmdl` and
  `Customers.tmdl` for `isHidden`/`summarizeBy`/description content)
- Everything carried over from sessions 1–2: report page visual QA (now doubly relevant
  since visuals were touched), measure spot-checks, pipeline step inspection, CI/CD Step 4
  (`.github/` workflow commit), promotion-flow documentation, Rayfin deployment

**New blockers discovered:**
- None — but note for future sessions: any `powerbi-modeling-mcp` write against a Fabric
  workspace is live-only; it does **not** appear in `git status` until an explicit
  `ExportToTmdlFolder` pulls the model back into the repo. Always export before committing
  when a session includes semantic-model MCP writes.

**Pick up next session at:**
- Confirm the PR from `dev-fabric-sync` → `test` merged cleanly and Test workspace
  reflects the renamed tables/hidden columns/descriptions
- Resume CI/CD Step 4 and the carried-over build-QA track

---

### Previous Session Recap (2026-07-29, session 2 — CI/CD environment setup)
**Completed:**
- Renamed the Fabric workspace `ws_Finance_Analysis` → `ws_Finance_Analysis_Dev`
- Created `ws_Finance_Analysis_Test` in Fabric, connected to GitHub branch `test`,
  folder `/ws_Finance_Analysis`, synced at commit `f1807664` — all 7 artifacts
- Created `ws_Finance_Analysis_Prod` in Fabric, connected to GitHub branch `main`,
  folder `/ws_Finance_Analysis`, synced at commit `f1807664` — all 7 artifacts
- Created `test` branch on GitHub from `main`, pushed `fb5039c..f180766`
- Branch/workspace map established: `dev-fabric-sync` → Dev, `test` → Test, `main` → Prod
- Updated this file and PROJECTS.md to document the new environment topology

**Left unfinished:**
- **CI/CD Step 4 not done:** `.github/` workflow files generated but not committed
- No promotion-flow documentation yet (how changes move dev → test → prod)
- No end-to-end validation that a push to `test`/`main` actually deploys correctly
  to the corresponding Fabric workspace
- Everything carried over from session 1 (report page visual QA, measure spot-checks,
  pipeline step inspection, Rayfin deployment) — untouched this session

**New blockers discovered:**
- None

**Pick up next session at:**
- Commit the generated `.github/` workflow files (CI/CD Step 4)
- Test the promotion flow with a trivial change pushed through dev → test → main
- Resume the build-QA track (report visual check, measure spot-check) once CI/CD
  scaffolding is confirmed stable

---

### Previous Session Recap (2026-07-29, session 1 — doc audit + live verification)
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
