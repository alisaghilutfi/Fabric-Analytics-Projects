---
name: sync-status
description: Read-only report of repo vs Fabric sync state — branch status, changed Fabric artifacts per workspace folder, and placeholder ID scan. Use before committing, before editing TMDL, or when the user asks about sync or drift.
---
Read-only. Never stage, commit, push, or edit files.
1. git fetch origin, then git status -sb. Report ahead/behind.
2. List changed and untracked files, grouped by top-level ws_* folder.
3. Placeholder scan of tracked Fabric files:
   git grep -nE '"logicalId"[[:space:]]*:[[:space:]]*"00000000-0000-0000-0000-000000000000"|ONT_WorkspaceId[[:space:]]*=[[:space:]]*00000000-0000-0000-0000-000000000000|_PLACEHOLDER' -- . ':(exclude)*.md' ':(exclude).githooks/*' ':(exclude).claude/*'
   Report every hit with file and line.
   Zero GUIDs in workspaceId fields are valid same-workspace references and are not flagged.
4. For each ws_* folder with changes, remind the user to check that workspace's Fabric Source control panel (uncommitted live changes) before committing locally.
5. End with a one-line verdict: CLEAN, LOCAL CHANGES, BEHIND ORIGIN, or PLACEHOLDERS FOUND.