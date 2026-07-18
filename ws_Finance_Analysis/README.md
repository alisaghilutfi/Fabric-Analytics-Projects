# ws_Finance_Analysis

End-to-end Finance Analytics Dashboard built in Microsoft Fabric using the vibe-coding workflow:
VS Code + Claude Code → GitHub → Fabric Git sync.

## Architecture

```
Bronze Lakehouse (lh_Finance_Bronze)
  └── raw_customers, raw_transactions        ← CSVs from GitHub raw URLs

Silver Lakehouse (lh_Finance_Silver)
  └── dim_date, dim_customer, dim_channel,
      dim_merchant, fact_transactions        ← cleaned star schema

Semantic Model (sm_Finance)
  └── DirectLake on lh_Finance_Silver
      22 DAX measures across 6 folders

Report (rpt_Finance)
  └── 4 pages: Overview, Customers,
      Transactions, Trends
      powerbi.tips Layout Trifecta design

DataPipeline (pl_Finance)
  └── Bronze → Silver notebook orchestration
```

## Data Quality Issues Resolved

See `docs/data-profile.md` for the full audit.

| Issue | Rows affected | Fix |
|---|---|---|
| Duplicate transaction_ids | 69 | dropDuplicates() |
| Dirty channel values | 770+ | trim + normalize |
| Dirty currency (inr/inR) | 605 | upper() |
| fee_amount nulls | 24 | fill 0.0 |
| Column typo (fisrt_name) | all | rename |
| Negative amounts | 9 | flag is_reversal |

## Setup Steps

1. Clone this repo locally
2. Create Fabric workspace `ws_Finance_Analysis`
3. Connect Fabric Git integration: repo `alisaghilutfi/Fabric-Analytics-Projects`, folder `/ws_Finance_Analysis`, branch `dev-fabric-sync`
4. Sync from Git → Fabric creates all artifacts
5. Open `nb_Finance_Bronze` in Fabric → attach `lh_Finance_Bronze` as default lakehouse → Run
6. Open `nb_Finance_Silver` → attach `lh_Finance_Silver` as default lakehouse → Run
7. Update `expressions.tmdl` with real Workspace ID and Lakehouse ID for `lh_Finance_Silver`
8. Sync semantic model → verify DirectLake framing
9. Run `pl_Finance` pipeline to validate end-to-end

## Key Files

| File | Purpose |
|---|---|
| `CLAUDE.md` | Claude Code context — read automatically in every VS Code session |
| `docs/data-profile.md` | Full data audit and star schema rationale |
| `nb_Finance_Bronze.Notebook/` | Raw CSV ingestion notebook |
| `nb_Finance_Silver.Notebook/` | Star schema build + DQ remediation |
| `sm_Finance.SemanticModel/` | TMDL semantic model (22 measures) |
| `pl_Finance.DataPipeline/` | Orchestration pipeline |

## Post-sync steps (in Fabric UI)

After Git sync creates the artifacts:
- Notebooks: attach correct default lakehouse in notebook settings
- Pipeline: update `<WORKSPACE_ID>` placeholders in `pipeline-content.json`
- Semantic model: update `<WORKSPACE_ID>` and `<LAKEHOUSE_ID>` in `expressions.tmdl`

## Conventions

- Branches: `dev-fabric-sync` → PR → `main`
- Naming: `lh_` (Lakehouse), `nb_` (Notebook), `sm_` (Semantic Model), `rpt_` (Report), `pl_` (Pipeline)
- DAX: VAR/RETURN throughout, display folders, Copilot descriptions on all measures
- Design: powerbi.tips Layout Trifecta
