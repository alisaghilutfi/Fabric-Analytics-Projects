---
description: Trigger the session-recap-writer agent for a named workspace. Updates CONTEXT.md, PROJECTS.md, and READMEs, then commits and pushes to dev-fabric-sync.
argument-hint: <workspace-name-without-ws_prefix>
---

Invoke the `session-recap-writer` agent for workspace `ws_$ARGUMENTS`.

Pass it:
- Workspace name: `ws_$ARGUMENTS`
- Session summary: ask me for a brief summary of what was completed this session before proceeding, unless I have already provided one in this message.

The agent handles all file updates and the commit. Do not write or commit anything yourself — delegate entirely to the agent.
