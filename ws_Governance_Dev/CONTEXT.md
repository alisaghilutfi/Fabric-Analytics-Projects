# ws_Governance_Dev

## Status
Phase 1 — SP smoke test pending

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
- Tenant settings: read-only admin APIs + detailed metadata + DAX/mashup ✅ (applied 2026-11-02)

## Items
- lh_Governance — Lakehouse (not yet created)
- nb_Governance_Bronze — Bronze ingest notebook (not yet created)
- nb_Governance_Silver — Silver transform (not yet created)
- nb_Governance_Gold — Gold schema load (not yet created)
- nb_Governance_Score — Health rules engine (not yet created)
- pl_Governance_Ingest — Orchestration pipeline (not yet created)
- sm_Governance — DirectLake semantic model (not yet created)

## Last Session — 2026-11-02
- ws_Governance_Dev created, Git-connected to dev-fabric-sync /ws_Governance_Dev
- SP sp-governance-scanner registered, sg-fabric-scanner created, all three tenant settings enabled
- Next: create lh_Governance via Fabric portal, then run SP smoke test notebook
