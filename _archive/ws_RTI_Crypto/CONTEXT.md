# ws_RTI_Crypto — Session Context

> This file is the handoff document for the RTI_Crypto project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-10-22

---

## What We Are Building
A real-time intelligence project on Microsoft Fabric tracking live
crypto prices via Eventhouse and KQL.

## Workspace
- **Fabric workspace:** ws_RTI_Crypto
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Git branch:** dev-fabric-sync

## ⚠️ Git Integration BLOCKED — Root Cause Identified 2026-10-22

The workspace cannot commit to GitHub via Fabric Git integration.

**Root cause (confirmed 2026-10-22):** Fabric has no stored GitHub OAuth
token for this workspace. The `/git-credentials` page in Fabric settings
renders blank — no token was ever successfully issued. All other workspaces
work because they were connected before this issue appeared and have cached
tokens. This workspace was never successfully initialized.

**Error history:**
- Request ID `1e57bbf7` (2026-07-27) — first occurrence
- Request ID `ecda6951` (2026-10-22) — confirmed same root cause
- Request ID `f097518f` (2026-10-22) — confirmed same root cause

**Fix when revisiting:** Disconnect workspace → reconnect via Workspace
settings → Git integration → watch for GitHub OAuth popup during reconnect
— that popup is what issues the token. If popup is suppressed or skipped,
the token will not be created and the block will persist.

**This is NOT a branch policy issue.** The `protect-main` ruleset only
targets `main`, and `dev-fabric-sync` has no protection rules. The block
is purely a missing OAuth credential on the Fabric side.

## Artifacts (as last known in workspace)
| Artifact | Type | Notes |
|---|---|---|
| `eh_RTI_Crypto` | Eventhouse | |
| `eh_RTI_Crypto` | KQLDatabase | |
| `es_RTI_Crypto` | Eventstream | Not Git-committable (Fabric limitation, unrelated to the policy block) |
| `lh_RTI_Crypto` | Lakehouse | |
| `nb_RTI_Crypto` | Notebook | |
| `nb_RTI_Crypto_Automated` | Notebook | |
| *(unnamed)* | Notebook | Auto-generated compaction notebook |

No semantic model, no report, no pipeline exist in this workspace yet.

## Current Focus
Blocked on Git reconnection. No further Fabric work should be planned
against this workspace's Git sync until Microsoft support resolves the
`Git_GitProviderCommitRejectedByPolicy` error.

## Instructions for Executing Agent
When starting a session on this project:
1. Read this file in full
2. Read PROJECTS.md for current status and blockers
3. Read HARNESS.md for tool and authentication reference
4. Check whether the Git support ticket has been resolved before
   attempting any Git-related operation on this workspace
5. Use Fabric MCP to connect to ws_RTI_Crypto directly (bypassing Git)
   if live inspection or non-Git work is needed

## Session Recap Template
When finishing a session, replace the section below with actual results:

### Last Session Recap
**Date:** 2026-10-22
**Completed:**
- Full schema reconnaissance via FabricEngineer (kusto + Fabric Core MCP)
- Confirmed workspace items: Eventhouse, KQL Database, Eventstream,
  Lakehouse, 3 Notebooks — no semantic model, no report
- Confirmed data: 617,460 rows in Crypto_Price, 615 symbols, stale since
  2026-01-30 (ingestion not running)
- Root cause of Git block identified: missing Fabric OAuth token (not a
  branch policy issue as previously thought)
- Workspace disconnected from Git and re-archived
- Pivoted Rayfin app build to ws_RTI_BicycleRentals (working Git sync,
  live semantic model, 187M rows)

**Left unfinished:**
- Git OAuth initialization — requires disconnect/reconnect with OAuth popup
- Ingestion restart — nb_RTI_Crypto needs manual run to resume live data
- Semantic model — does not exist, needs to be built before any reporting

**Pick up next session at:**
1. Fix Git OAuth: Workspace settings → Git integration → Disconnect →
   Reconnect → watch for GitHub OAuth popup → complete authorization
2. Restart ingestion: run nb_RTI_Crypto in Fabric portal
3. Build sm_RTI_Crypto (DirectLake on Gold_Crypto_Daily)
4. Build FabricApp_RTI_Crypto (Rayfin, 3 screens)
