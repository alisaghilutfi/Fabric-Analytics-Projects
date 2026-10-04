# ws_RTI_Crypto — Session Context

> This file is the handoff document for the RTI_Crypto project.
> The executing agent reads this at session start and writes a recap
> at session end. Do not edit manually unless correcting an error.

Last updated: 2026-07-27

---

## What We Are Building
A real-time intelligence project on Microsoft Fabric tracking live
crypto prices via Eventhouse and KQL.

## Workspace
- **Fabric workspace:** ws_RTI_Crypto
- **GitHub repo:** alisaghilutfi/Fabric-Analytics-Projects
- **Git branch:** dev-fabric-sync

## ⚠️ Git Integration BLOCKED

The workspace is currently **disconnected from Git**. Git sync failed with:

```
Error:      Git_GitProviderCommitRejectedByPolicy
Request ID: 1e57bbf7-fa24-4989-b4e4-caf3f5048441
```

A Microsoft support ticket is pending on this issue. Until it is
resolved, do **not** attempt to reconnect or force-sync this workspace —
changes made in the workspace will not reflect in this repo, and this
repo's `ws_RTI_Crypto/` folder will not reflect the live workspace
(hence no artifact files present here, only this CONTEXT.md).

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
**Date:** 2026-07-27
**Completed:**
- Documented the Git integration block (`Git_GitProviderCommitRejectedByPolicy`,
  request ID `1e57bbf7-fa24-4989-b4e4-caf3f5048441`) and current artifact
  inventory in this CONTEXT.md

**Left unfinished:**
- Git reconnection — pending Microsoft support ticket resolution

**New blockers discovered:**
- Workspace disconnected from Git; cannot commit/sync until support
  resolves the policy rejection

**Pick up next session at:**
- Check support ticket status before any Git operation on this workspace
- Once unblocked, reconnect Git and verify the artifact inventory above
  against the live workspace
