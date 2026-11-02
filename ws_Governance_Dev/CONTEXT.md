# ws_Governance_Dev

## Status
Phase 1 complete — Bronze layer live

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
| nb_Governance_Silver | Notebook | Not yet created |
| nb_Governance_Gold | Notebook | Not yet created |
| nb_Governance_Score | Notebook | Not yet created |
| pl_Governance_Ingest | DataPipeline | Not yet created |
| sm_Governance | SemanticModel | Not yet created |
| FabricApp_Governance | Fabric App | Not yet created |

## Current Focus / Blockers
- Silver notebook not yet built — next step is `nb_Governance_Silver` (parse `raw_json` from all three Bronze tables into typed Silver Delta tables)

## Known Gotchas
- `GET /groups/{wsId}/datasets/{id}/refreshes` requires workspace membership — use `/admin/datasets/{id}/refreshes` instead (covered by read-only admin APIs tenant setting)
- DirectLake models return 0 refresh history records (no scheduled refresh by design) — Cell 6 kept for client tenants with import-mode models
- Scanner `getInfo` returns 10 fields per dataset; `isRefreshable` not included at current scan depth (`datasetSchema=false`)

## Last Session — 2026-11-02
- SP `sp-governance-scanner` registered, `sg-fabric-scanner` created, all three admin API tenant settings enabled
- `ws_Governance_Dev` created, Git-connected to `dev-fabric-sync` `/ws_Governance_Dev`
- `lh_Governance` Lakehouse created via Fabric Core MCP (ID `4478b3d6-2e64-43ee-b120-3caa722abf48`)
- SP smoke test passed — status 200, 28 workspaces returned; `nb_Governance_SmokeTest` created, run, then deleted (secret never committed)
- `nb_Governance_Bronze` provisioned as Fabric item folder (`nb_Governance_Bronze.Notebook/` + `.platform` + `notebook-content.py`); logicalId `605504c4-9581-4dac-9ee4-fdbee7f74435`; `lh_Governance` wired as default lakehouse via `# META dependencies.lakehouse` block
- Bronze layer first run complete:
  - `bronze_activity_events`: 687 rows (yesterday's tenant activity events)
  - `bronze_scan_workspaces`: 28 rows (full tenant scan, one row per workspace)
  - `bronze_refresh_history`: 0 rows — all 33 semantic models are DirectLake; admin endpoint confirmed correct but returns empty for DirectLake tenant
- Cell 6 endpoint corrected from `/groups/{ws}/datasets/{id}/refreshes` to `/admin/datasets/{id}/refreshes`
- Next session starts at: build `nb_Governance_Silver`
