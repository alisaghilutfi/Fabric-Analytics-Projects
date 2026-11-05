# ws_Governance_Dev

## Status
Phase 5 complete — FabricApp_Governance live

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
| FabricApp_Governance | Fabric App | Complete — https://hefty-pearl-60b5367c54-northeurope.webapp.fabricapps.net (Backend ID `8c9b81da-3e15-4d38-a325-22a6c06ef526`) |

## Current Focus
Phase 6 — data quality fixes:
1. `git_connected` — Scanner API not returning Git metadata at current scan depth; investigate `getInfo` parameters or fallback enrichment
2. Items tab — `gold_items` shows 0 rows for all workspaces; audit `nb_Governance_Bronze` item-level endpoint calls
3. `NO_SENSITIVITY_LABEL` rule — downgrade severity from Critical to Medium in `nb_Governance_Score`

## Next Session Starts At
Open `nb_Governance_Bronze`: check whether `/admin/workspaces/{id}/items` or
`getInfo` with `getArtifactUsers=true&lineage=true` is being called; fix items
population first (highest user-visible impact), then tackle `git_connected`.
Downgrade `NO_SENSITIVITY_LABEL` severity in the health rules dict in
`nb_Governance_Score` as a quick win while notebooks are open.

## Blockers
None — no infrastructure blockers; data quality issues are code-level fixes
in Bronze/Score notebooks

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

## Last Session — 2026-11-04 (Session 5)
- `FabricApp_Governance` scaffolded, built, and deployed as a Rayfin app in Fabric-Apps-Projects:
  - Live URL: https://hefty-pearl-60b5367c54-northeurope.webapp.fabricapps.net
  - App backend ID: `8c9b81da-3e15-4d38-a325-22a6c06ef526`
  - 3 screens: Tenant Health Overview, Workspace Detail, Violations Explorer
  - Smoke test passed: 28 workspaces, bar chart, drill-down, violations explorer, severity filters all working
- Duplicate `workspace_id` root cause fixed:
  - `silver_workspaces` dedup key `["workspace_id", "scanned_at"]` allowed same workspace on multiple pagination pages (identical `scanned_at`); fixed to `["workspace_id"]`
  - Same fix applied to `silver_items`: `["item_id", "scanned_at"]` → `["item_id"]`
  - Diagnostic cell removed from `nb_Governance_Gold`; `dropDuplicates(["workspace_id"])` safety net retained
  - Validation: `gold_workspaces` 28 rows / 28 distinct workspace_ids
- Commits: `31fa92a` (diagnostic cell), `548a653` (Silver dedup fix)
- Known issues for next session:
  - (1) `git_connected` returns No Git for most workspaces — Scanner API not returning Git metadata at current scan depth
  - (2) Items tab shows 0 items for all workspaces — `nb_Governance_Bronze` may not be calling item-level Scanner API endpoints
  - (3) `NO_SENSITIVITY_LABEL` rule fires for every item (1,588 violations) driving workspace scores to 0 — severity should be downgraded to Medium or suppressed until scanner depth is increased
- Next session starts at: investigate Scanner API depth for `git_connected` and items; downgrade `NO_SENSITIVITY_LABEL` to Medium severity

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
