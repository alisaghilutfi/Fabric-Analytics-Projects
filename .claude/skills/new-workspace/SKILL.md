---
name: new-workspace
description: Scaffold the repo side of a new Fabric workspace (folder, CONTEXT.md, PROJECTS.md row) following repo naming rules. Use when the user starts a new Fabric project.
argument-hint: <workspace-name-without-ws_prefix>
disable-model-invocation: true
---
Workspace name: ws_$ARGUMENTS
1. Check the name against the naming conventions in the repo CLAUDE.md. Stop and ask if it doesn't comply or if the folder already exists.
2. Create ws_$ARGUMENTS/CONTEXT.md with these headings and empty bodies:
   ## Project identity / ## Architecture / ## What exists so far / ## Last session / ## Blockers
   Fill Project identity with: workspace name, repo, Git folder /ws_$ARGUMENTS, branch dev-fabric-sync.
3. Add a row for the workspace to PROJECTS.md, matching the existing table format, status "Scaffolded".
4. Do NOT create Fabric item folders, .platform files, or any IDs. Fabric items are created in the workspace and arrive via Git sync — never hand-authored with placeholder IDs.
5. Tell the user the manual steps: create the workspace in Fabric on the F2 capacity, then connect Workspace settings → Git integration to folder /ws_$ARGUMENTS on dev-fabric-sync.
6. Show the diff and wait for confirmation before any commit.