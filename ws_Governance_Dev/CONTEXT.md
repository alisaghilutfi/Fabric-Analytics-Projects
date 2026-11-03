# ws_Governance_Dev

## Status
Phase 2 complete — full pipeline end-to-end (Bronze → Silver → Gold → Score)

## Architecture
Scanner API + Activity Events API + Refresh History API
→ Bronze/Silver/Gold Lakehouse (lh_Governance)
→ DirectLake semantic model (sm_Governance)
→ Rayfin app (FabricApp_Governance) in Fabric-Apps-Projects

## SP / Auth
- App: sp-governance-scanner
- Client ID: 4c046cb3-0862-4862-b228-5fee8381deca
- Tenant ID: c8cb697b-07d1-4c61-a5d9-bbe5ad099fe8
- Secret name: governance-scanner-key-1
- Security group: sg-fabric-scanner
- Tenant settings: read-only admin APIs + detailed metadata + DAX/mashup (applied 2026-11-02)

## Items

| Item | Type | Status |
|------|------|--------|
| lh_Governance | Lakehouse | Complete — ID `4478b3d6-2e64-43ee-b120-3caa722abf48` |
| nb_Governance_Bronze | Notebook | Complete — logicalId `605504c4-9581-4dac-9ee4-fdbee7f74435` |
| nb_Governance_Silver | Notebook | Complete — logicalId `f645c71e-7e21-4c30-9359-6b69a0407279` |
| nb_Governance_Gold | Notebook | Complete — logicalId `1f77ebc4-fd06-40ae-a179-c606d4c45757` |
| nb_Governance_Score | Notebook | Complete — logicalId `84c8553a-ced6-4f21-afa4-46094baffe61` |
| pl_Governance_Ingest | DataPipeline | Not yet created |
| sm_Governance | SemanticModel | Not yet created |
| FabricApp_Governance | Fabric App | Not yet created |

## Current Focus / Blockers
- `pl_Governance_Ingest` not yet created — orchestrates all four notebooks in sequence
- `sm_Governance` not yet created — DirectLake semantic model on Gold tables
- `FabricApp_Governance` not yet created — Rayfin app surfacing health scores

## Known Gotchas
- `GET /groups/{wsId}/datasets/{id}/refreshes` requires workspace membership — use `/admin/datasets/{id}/refreshes` instead (covered by read-only admin APIs tenant setting)
- DirectLake models return 0 refresh history records (no scheduled refresh by design) — Cell 6 kept for client tenants with import-mode models
- Scanner `getInfo` returns 10 fields per dataset; `isRefreshable` not included at current scan depth (`datasetSchema=false`)

## Last Session — 2026-11-03
- `nb_Governance_Silver` built and validated: 28 workspaces, 62 items, 687 activity events
  - Fixed `WorkSpaceName` casing bug in activity events schema (field is `WorkspaceId` not `WorkSpaceId`)
- `nb_Governance_Gold` built and validated: 28 workspaces, 62 items, 16 activity summaries
- `nb_Governance_Score` built and validated — full health rules engine live:
  - 28 scored workspaces, 149 rule results across 3 active rules
  - NO_SENSITIVITY_LABEL: 62 violations (data unavailable at current scanner depth)
  - SINGLE_ADMIN: 24 violations (proxy via distinct active users)
  - NO_GIT_CONNECTION: 25 violations (data unavailable at current scanner depth)
  - UNUSED_REPORT: 0 — needs 90 days of accumulated activity data
  - REFRESH_FAILING: skipped — DirectLake tenant, no refresh history
  - Score range: 0 (PersonalWorkspace/high violation count) to 100 (empty personal workspace)
  - Active portfolio workspaces score 82-92
- Full Bronze -> Silver -> Gold -> Score pipeline runs end-to-end
- Next session starts at: build `pl_Governance_Ingest` to orchestrate all four notebooks, then `sm_Governance`
