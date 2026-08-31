# Portfolio Workflow Reference

This document captures the two-harness agentic workflow, session
protocols, report design process, and hard-won lessons from building
the Fabric Analytics Projects portfolio. It is the authoritative
reference for all future sessions.

Read this at the start of any planning or execution session before
making decisions about architecture, tooling, or process.

---

## The Two-Harness System

```
ProjectPlanner (Claude Desktop)     FabricEngineer (Claude Code)
         │                                    │
         │  refreshes + writes instructions   │
         ▼                                    │
   CONTEXT.md ──────────────────────────────▶ │ reads at session start
         │                                    │
         │                          executes via MCP + CLIs
         │                                    │
         │◀────────────────── files recap ────┘
         │
   PROJECTS.md  ◀──────────── updates status
```

**ProjectPlanner** — architectural decisions, report design consultation,
DAX review, content generation, and pre-session brief refresh. Runs in
Claude Desktop (with filesystem, powerbi-modeling-mcp, fabric-mcp, and
fabric-rti-mcp MCPs). Never executes.

**FabricEngineer** — execution, file edits, Git operations, MCP tool
calls. Runs in Claude Code (VS Code). Never makes architectural decisions
without ProjectPlanner input.

**Context files are the handoff mechanism.** CONTEXT.md is written by
ProjectPlanner (instructions layer) and FabricEngineer (recap layer).
PROJECTS.md is the portfolio-wide status registry maintained by agents.

---

## Session Protocols

### Pre-session brief refresh (ProjectPlanner)

Run this in Claude Desktop **before** starting a FabricEngineer session.
This replaces manually drafting the CONTEXT.md instructions layer.
Claude Desktop reads the files directly from disk via the filesystem MCP.

```
Read these two files in order:
1. PROJECTS.md — find the ws_<name> block in full
2. ws_<name>/CONTEXT.md — read the Last Session section only

Then propose a rewritten Instructions Layer for ws_<name>/CONTEXT.md:
- Update "Current Focus" to reflect what last session confirmed is done
- Update "Next Session Starts At" to the next concrete step
- Resolve blockers: remove ones last session cleared, add any new ones
- Do NOT modify the Last Session section — FabricEngineer owns that

Output the proposed Instructions Layer for my review.
Do not write anything to disk until I confirm.
```

Once confirmed, write the updated Instructions Layer to
ws_<name>/CONTEXT.md in place. Then hand off to FabricEngineer
using the start-of-session prompt below.

---

### Start-of-session prompt (FabricEngineer)

```
Read PROJECTS.md and ws_<name>/CONTEXT.md in full.

1. State current project status in 3-5 bullets
   (what exists, what is broken, what is pending)
2. State any hard blockers that prevent work
3. Confirm the next step from the "Next session starts at"
   section of CONTEXT.md — do not invent a new step unless
   that section is empty or a blocker makes it impossible

Do not begin any execution until I confirm the proposed step.
```

### End-of-session prompt (FabricEngineer)

```
Session is complete. Update the project docs as follows:

1. In ws_<name>/CONTEXT.md:
   - Overwrite the entire "Last Session" section with today's
     recap. Do not add a new "Previous Session" section —
     the old one is preserved in git history.
   - In the "What Exists So Far" table, update any rows whose
     status changed this session. Do not add new rows unless
     a new artifact was created.
   - If any blocker was resolved, remove it from the Current
     Focus section. If a new blocker was discovered, add it.

2. In PROJECTS.md:
   - Update the ws_<name> status line and open items to reflect
     current state. Remove items that are done.

3. Then run:
   git add .
   git commit -m "<imperative summary, 50 chars max>"
   git push origin dev-fabric-sync

Do not push to main directly.
Confirm commit hash when done.
```

---

## CONTEXT.md Maintenance Rules

**Two layers, two different retention policies:**

**Instructions layer** (What We Are Building, Architecture, What Exists,
Blockers, MCP Servers, Agent Instructions):
→ Update in place. Prune resolved items. Never append.
→ This section stays roughly the same length over time.

**Last Session layer** (recap):
→ Overwrite every session. Keep only the most recent entry.
→ Git history is the archive. Do not accumulate sessions inline.

**Who writes what:**
- Instructions layer → ProjectPlanner writes this (via pre-session brief refresh prompt in Claude Desktop)
- Last Session layer → FabricEngineer writes this

---

## Git Branch Strategy

```
dev-fabric-sync  →  PR  →  main
                            │
                     backfill both:
                     dev-fabric-sync
                     test
```

**After every PR merge to main:**

```powershell
git pull origin main
git push origin dev-fabric-sync
git checkout test
git pull origin main
git push origin test
git checkout dev-fabric-sync
```

**Rules:**
- All writes go to `dev-fabric-sync`
- Promote via PR only — never push directly to `main`
- Batch 2-3 related commits per PR to reduce overhead
- `protect-main` ruleset active (Restrict deletions, Block force
  pushes, Require PR, 0 approvals)

---

## Report Design Workflow

### Phase 1 — Plugin and Bridge setup (one-time per machine)

```powershell
# Register marketplace and install plugins
claude plugin marketplace add microsoft/skills-for-fabric
claude plugin install powerbi-authoring@fabric-collection
claude plugin install fabric-authoring@fabric-collection
claude plugin install fabric-consumption@fabric-collection

# Install Desktop Bridge CLI
npm install -g @microsoft/powerbi-desktop-bridge-cli

# Verify
claude plugin list
powerbi-desktop --version
powerbi-report-author --version
```

**Power BI Desktop version requirement:** 2.155.756.0 (June 2026) or later.

### Phase 2 — Open report via PBIP (every session)

Do not open `.pbix` or `definition.pbir` — the Bridge cannot see those.

Create `rpt_<name>.pbip` in the workspace folder:

```json
{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
  "version": "1.0",
  "artifacts": [
    {
      "report": {
        "path": "rpt_<name>.Report"
      }
    }
  ]
}
```

Open the `.pbip` in Desktop. When prompted:
- "Set up remote model" → select workspace + existing semantic model
- "Metadata don't match" → Overwrite the current model

### Phase 3 — Verify Bridge connection

```powershell
powerbi-desktop status
```

Required output: `reportDir` populated (not null), `pages` listing
actual page names. If `reportDir` is null, the wrong file type is open.

### Phase 4 — Agentic redesign prompt

```
The Power BI Desktop Bridge is connected and ready.
reportDir: <path to rpt_<name>.Report>

Use powerbi-report-design to produce a design brief first.
Save it to ws_<name>/docs/design-brief-rpt_<name>.md.
Wait for my approval before implementing.

After approval, use powerbi-report-authoring to implement.
After each page:
1. Validate with powerbi-report-author validate
2. Reload with powerbi-desktop reload
3. Screenshot with powerbi-desktop screenshot
4. Review and adjust if needed
5. Move to next page only when satisfied

Do not ask about pixel positions.
Make all layout decisions yourself.
Do not commit until all pages are complete and verified.
```

### Phase 5 — After Desktop session cleanup

Desktop auto-modifies these files on every reload — always reset before
committing:

```powershell
git checkout -- ws_<name>/sm_<name>.SemanticModel/definition/database.tmdl
git checkout -- ws_<name>/sm_<name>.SemanticModel/definition/model.tmdl
```

These are permanently in `.gitignore`:

```
**/.pbi/
**/diagramLayout.json
```

---

## Design Standards

### Layout Trifecta (powerbi.tips — Mike Carlo)

Every report page must have three zones:

| Zone | Color | Position (1280×720 canvas) |
|---|---|---|
| Left panel (slicers) | `#F3F2F1` | x=0, width=200, full height |
| Header bar | `#0F6C74` | x=200, y=0, width=1080, height=56 |
| Content area | `#FAFAFA` | x=200, y=56, width=1080, height=664 |

Content zone safe area: x=220 to x=1268, y=68 to y=708 (20px margins).

### Lotusoftware theme

File: `Lotusoftware_Base_Theme.json` at repo root.

Deployment path:
```
rpt_<name>.Report/
  StaticResources/
    RegisteredResources/
      Lotusoftware_Base_Theme.json  ← Fabric renames with hash on sync
```

`report.json` must reference it as `customTheme` under `RegisteredResources`
— not `baseTheme` under `SharedResources`. Fabric auto-renames the file
with a hash suffix on first sync. Update `report.json` to match the
hashed filename after first Fabric sync.

### Background scrim PNG generation

Use Python Pillow at exactly 1280×720px — not PowerPoint (DPI mismatch):

```python
from PIL import Image, ImageDraw
img = Image.new("RGB", (1280, 720), "#FFFFFF")
draw = ImageDraw.Draw(img)
draw.rectangle([0, 0, 199, 719], fill="#F3F2F1")    # left panel
draw.rectangle([200, 0, 1279, 55], fill="#0F6C74")  # header
draw.rectangle([200, 56, 1279, 719], fill="#FAFAFA") # content
img.save("background.png")
```

Set as Canvas background (not Wallpaper), Stretch fit, 0% transparency.

---

## TMDL and PBIR Rules

### TMDL
- UTF-8 without BOM — use Python `open(path, 'w', encoding='utf-8', newline='\n')`
- Free-floating `//` comments cause `InvalidLineType` — only `///` descriptions attached to measures/columns are valid
- `expressions.tmdl` for DirectLake: no comment lines of any kind, exact tab indentation required
- Relationship syntax: dot notation (`fromColumn: fact.column`) not bracket notation
- After any `powerbi-modeling-mcp` write session, run `database_operations ExportToTmdlFolder` before Git commits

### PBIR
- Visual JSON `position` fields: `x`, `y`, `z`, `height`, `width`
- In Power BI Desktop Format panel: `Horizontal` = Y position, `Vertical` = X position (confusingly named)
- TOM auto-cascades table renames into DAX; PBIR visual JSON files do not — find-and-replace `Entity` references manually after table renames
- `powerbi-report-author validate` after every batch of PBIR edits

### DirectLake
- `delta.columnMapping.mode=name` must be set at Delta table write time — cannot be retrofitted
- `joinOnDateBehavior: DatePartOnly` required when joining `TimestampType` to `DateType` columns
- Null GUID `00000000-0000-0000-0000-000000000000` as `workspaceId` in pipeline JSON is valid for same-workspace notebook references across Dev/Test/Prod

---

## Known Fabric Gotchas

| Gotcha | Resolution |
|---|---|
| `Dataset.ReadWrite.All` is Delegated-only | No Application permission variant exists for Power BI Service |
| GitHub Secret must store the Value string | Not the Secret ID GUID — AADSTS7000215 if wrong |
| Token scope for Power BI refresh | `https://analysis.windows.net/powerbi/api/.default` not `fabric.microsoft.com/.default` |
| Fabric auto-commits during active Git sessions | Rebase your commits on top — never force push |
| Theme file renamed with hash on Fabric sync | Update `report.json` to reference hashed filename under `RegisteredResources` |
| `.pbix` open in Desktop → Bridge `reportDir: null` | Must open `.pbip` file, not `.pbix` or `definition.pbir` |
| Desktop reload touches `database.tmdl` / `model.tmdl` | Always `git checkout --` these files before committing |
| `powerbi-report-author` is not an npm package | It comes bundled with `powerbi-authoring@fabric-collection` plugin |
| `@pbir/cli` does not exist on npm | The correct package is `@microsoft/powerbi-desktop-bridge-cli` |
| `list_workspace_artifacts` returns Eventhouse and KQL Database with identical `displayName` | Always filter on `type` before acting |

---

## DAX Standards

- All measures in a dedicated `_Measures` calculated table
- VAR/RETURN pattern for all measures with more than one expression step
- Display folders required — minimum groupings: Volume, Amount, Time, Risk
- `DIVIDE()` instead of `/` for all division (safe division, returns 0 not error)
- `///` Copilot description on every measure
- Hide all raw source columns — expose only measures and display-ready columns
- No implicit measures

---

## Folder Structure Standard

```
ws_<name>/
├── docs/
│   ├── images/
│   │   └── dashboard-overview.png   ← add when report is complete
│   ├── design-brief-rpt_<name>.md   ← add before agentic redesign
│   └── data-profile.md              ← add if data dictionary exists
├── data/                            ← add only if source files exist locally
├── <Fabric artifact folders>/
├── rpt_<name>.pbip                  ← required for Desktop Bridge
├── CLAUDE.md                        ← workspace-specific agent rules
├── CONTEXT.md                       ← agent session handoff document
└── README.md                        ← human-facing documentation
```

**Rule:** create folders when content warrants them. Never pre-create
empty folders — Git does not track them.

---

## Naming Conventions

| Artifact | Prefix | Example |
|---|---|---|
| Lakehouse | `lh_` | `lh_Finance_Bronze` |
| Notebook | `nb_` | `nb_Finance_Silver` |
| DataPipeline | `pl_` | `pl_Finance` |
| SemanticModel | `sm_` | `sm_Finance` |
| Report | `rpt_` | `rpt_Finance` |
| Dashboard | `dash_` | `dash_Finance_Analysis` |
| ML Model | `model_` | `model_BankChurn` |
| Deployment Pipeline | `dp_` | `dp_Finance_Analysis` |

---

> The prompt is not the product. The harness is. Context organized
> here means agents execute correctly on the first try, not the fifth.
> Every session that files its recap makes the next session faster
> and more accurate.

*This document is maintained by the ProjectPlanner agent after sessions
that establish new workflow patterns. Update it when a new lesson is
confirmed — not speculatively. Last updated: 2026-08-31.*
