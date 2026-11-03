# ws_Governance_Dev

## Status
Phase 4 complete — sm_Governance DirectLake semantic model synced and validated

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
| pl_Governance_Ingest | DataPipeline | Complete — logicalId `979d10b1-b1d1-8880-4737-a1fe5699952a` |
| sm_Governance | SemanticModel | Complete — ID `2514647e-c2a4-4eee-984a-e26da08dccd0` |
| FabricApp_Governance | Fabric App | Not yet created |

## Current Focus / Blockers
- `FabricApp_Governance` not yet created — Rayfin app surfacing health scores

## Known Gotchas
- `GET /groups/{wsId}/datasets/{id}/refreshes` requires workspace membership — use `/admin/datasets/{id}/refreshes` instead (covered by read-only admin APIs tenant setting)
- DirectLake models return 0 refresh history records (no scheduled refresh by design) — Cell 6 kept for client tenants with import-mode models
- Scanner `getInfo` returns 10 fields per dataset; `isRefreshable` not included at current scan depth (`datasetSchema=false`)
- Pipeline provisioning via Git fails — Fabric cannot resolve notebook logicalIds during Update all; always create DataPipelines in Fabric UI, then export to Git via Fabric Core MCP
- `Git_GitProviderCommitRejectedByPolicy` on Source Control commit — caused by Fabric OAuth session degradation; workaround is Fabric Core MCP export + FabricEngineer git push
- `modifiedSince` API rejects all tested timestamp formats (ISO-8601 with/without milliseconds) — Cell 3 uses full scan; incremental deferred
- `nb_Governance_Bronze` is permanently Uncommitted in Fabric Source Control — contains real secret; Git version has placeholder; intentional and permanent
- `definition.pbism` for semantic models must contain only `{"version":"1.0"}` — no artifacts array; artifacts array is PBIP/report schema and causes `Workload_FailedToParseFile` on Source Control Update all
- XMLA-created models get a live ObjectId with null LogicalId in Git — causes duplicate artifact conflict on Update all; fix is delete live artifact + recycle bin, then reprovision via Git Update all
- After XMLA session + ExportToTmdlFolder, always do Source Control Update all (not Commit) to provision the Git-anchored version; the XMLA artifact and the Git-provisioned artifact are treated as separate items by Fabric

## Last Session — 2026-11-03 (Session 4)
- `sm_Governance` DirectLake semantic model built via XMLA, exported to TMDL, and synced to Fabric after resolving three provisioning errors:
  - (1) Duplicate artifact conflict (XMLA ObjectId vs null LogicalId) — fixed by deleting live artifact and reprovisioning via Git Update all
  - (2) Missing `definition.pbism` — fixed with correct `{"version":"1.0"}` schema (no artifacts array)
  - (3) Merge conflict on `dfcdf5f` measure swap — resolved via Accept incoming changes
- `[Items Without Sensitivity Label]` replaced with `[Single Admin Workspaces]` — `sensitivity_label` column does not exist at current scanner depth (`datasetSchema=false`)
- DAX validation passed: 56 workspaces · avg score 63.6 · 294 critical violations · 30 below 80 · 0 unused items · 50 single-admin workspaces
- `gold_items` shows yellow warning in model view — expected, caused by inactive relationship `gold_health_rule_results[item_id] → gold_items[item_id]`; not blocking
- Commits: `af9698e` (initial TMDL), `5b84593` (docs), `73bebe6` (definition.pbism attempt), `e26c7b2` (fix pbism schema), `b1e5de6` (correct pbism), `dfcdf5f` (Single Admin Workspaces measure)
- Next session starts at: build `FabricApp_Governance` Rayfin app

## Session 3 — 2026-11-03
- `pl_Governance_Ingest` created in Fabric UI (logicalId `979d10b1-b1d1-8880-4737-a1fe5699952a`):
  - 4 TridentNotebook activities: Bronze -> Silver -> Gold -> Score (linear chain, Success dependency)
  - `CLIENT_SECRET` pipeline parameter wired to Bronze activity Base parameters
  - Daily schedule set: 06:00 UTC
  - Exported to Git via Fabric Core MCP (secret redacted from `defaultValue` in committed JSON)
- `Git_GitProviderCommitRejectedByPolicy` blocker hit on Fabric Source Control commit — OAuth session degradation; bypassed via Fabric Core MCP export + FabricEngineer git push
- Git-authored pipeline provisioning confirmed broken — Fabric cannot resolve notebook logicalIds from `.platform` files during Update all; workaround documented in Known Gotchas
- `modifiedSince` incremental scan dropped from Cell 3 — API rejects all tested timestamp formats; full scan (28 workspaces) used; deferred to phase 2
- Pipeline run validated end-to-end: Bronze 1.20s + Silver 1.20s + Gold 2.51s + Score 2.37s = ~7.3s total (warm Spark sessions); expected ~3-5 min cold start on first daily run
- `nb_Governance_Bronze` permanently Uncommitted in Fabric (contains real secret) — Git version has placeholder; intentional
- Next session starts at: build `sm_Governance` DirectLake semantic model on Gold tables
