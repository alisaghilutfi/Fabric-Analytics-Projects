---
description: Read-only session-start report for one workspace -- status, blockers, and next step verbatim. Does not execute anything until you confirm.
argument-hint: <workspace-name-without-ws_prefix>
---

Workspace: ws_$ARGUMENTS

READ-ONLY. Do not run az, claude mcp or any execution command; the only commands allowed are the read-only git log calls in step 3.

**Input handling.** If $ARGUMENTS is empty or the folder `ws_$ARGUMENTS` does not exist at the repo root:
- Glob `ws_*/` at the repo root (exclude `_archive`).
- List the workspace folders found.
- STOP. Ask the user which workspace to open.

1. Read `PROJECTS.md` and find the `ws_$ARGUMENTS` block. Read it in full. If PROJECTS.md has no `ws_$ARGUMENTS` block, say so and continue.
2. Read `ws_$ARGUMENTS/CONTEXT.md` in full.
3. Run these two read-only git commands (PowerShell: quote the pathspec):
     git log -1 --date=short --format="%ad %h %s" -- "ws_$ARGUMENTS/CONTEXT.md"
     git log -1 --date=short --format="%ad %h %s" -- "ws_$ARGUMENTS" ":(exclude)ws_$ARGUMENTS/CONTEXT.md"
   If the second command returns a commit more recent than the first, print:
   "WARNING: ws_$ARGUMENTS changed after CONTEXT.md was last updated (<date> <sha> <subject>); the recap may be stale."
   Note: Fabric workspace-sync commits (e.g. "Committing N items from workspace") count as changes; print the subject so you can judge.
   Do not parse dates out of the document text; use only the git log output.
4. Report status in 3-5 bullets covering: what exists and is working, what is pending, what is blocked.
5. State any hard blockers that prevent work from starting.
6. Search CONTEXT.md and the `ws_$ARGUMENTS` block of PROJECTS.md for lines or sections containing
   "Next session", "Next session starts at", or "Pick up next session at".
   Print each one found, with its source file (CONTEXT.md or PROJECTS.md).
   If both files contain one and they differ, add: "CONTEXT.md and PROJECTS.md disagree."
   If neither file contains any of these phrases, say:
   "No next-step section found in CONTEXT.md or PROJECTS.md."
7. If the fabric-explorer agent is available, offer: "Run fabric-explorer for a live item inventory of ws_$ARGUMENTS? (yes/no)"

STOP after the report. Do not execute any step until the user confirms which step to take.