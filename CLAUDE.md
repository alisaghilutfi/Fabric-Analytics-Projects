# Fabric-Analytics-Projects — repo rules

Global guardrails: ~/.claude/CLAUDE.md (source: private repo dev-environment).

## Project overview

Portfolio-wide rules for all Fabric workspaces in this repo.
Read automatically by Claude Code in every session.
Workspace-specific context lives in ws_<name>/CONTEXT.md.

Session protocols, git branch flow and MCP routing: see HARNESS.md.
Portfolio status: see PROJECTS.md.

---

## Tech stack

Microsoft Fabric (Lakehouse, PySpark notebooks, DataPipeline, Eventhouse/KQL for real-time workspaces), DirectLake semantic models authored as TMDL, PBIR reports, Windows 11 + PowerShell 5.1, Git integration on dev-fabric-sync.
Tool versions: see HARNESS.md.

---

## Coding conventions

### Naming conventions

| Artifact type    | Prefix  | Example                    |
|------------------|---------|----------------------------|
| Lakehouse        | `lh_`   | `lh_Finance_Bronze`        |
| Notebook         | `nb_`   | `nb_Finance_Bronze`        |
| Semantic model   | `sm_`   | `sm_Finance`               |
| Report           | `rpt_`  | `rpt_Finance`              |
| DataPipeline     | `pl_`   | `pl_Finance`               |
| Measures table   | `_`     | `_Measures`                |
| Dashboard        | `dashboard_` | `dashboard_Finance_Analysis` |
| ML Model         | `model_`     | `model_BankChurn`            |
| Deployment Pipeline | `dp_`    | `dp_Finance_Analysis`        |

---

### Fabric artifact format rules (CRITICAL)

#### Notebook files
Every notebook needs TWO files:

**`.platform`** (metadata):
```json
{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
  "metadata": {
    "type": "Notebook",
    "displayName": "nb_Finance_Bronze"
  },
  "config": {
    "version": "2.0",
    "logicalId": "<UUID>"
  }
}
```

**`notebook-content.py`** (content):
```python
# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "<lakehouse_id>",
# META       "default_lakehouse_name": "lh_Finance_Bronze",
# META       "default_lakehouse_workspace_id": "<workspace_id>",
# META       "known_lakehouses": [{"id": "<lakehouse_id>"}]
# META     }
# META   }
# META }

# CELL ********************
<code here>

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
```

#### Lakehouse files
Every lakehouse needs `.platform`, `lakehouse.metadata.json`, `shortcuts.metadata.json`, `alm.settings.json`.

#### TMDL relationship syntax (CRITICAL — do NOT use bracket notation)
```
relationship <name>
    fromColumn: fact_transactions.customer_id
    toColumn: dim_customer.customer_id
```
NOT `fromColumn: fact_transactions[customer_id]` — that will break import.

#### PBIR report JSON
The `config` field in report visuals is **escaped JSON string** (JSON inside JSON):
```json
"config": "{\"version\":\"5.1.0\",\"singleVisual\":{\"visualType\":\"card\",...}}"
```
Every quote inside config is backslash-escaped. Never write raw nested JSON here.

#### DirectLake partition syntax
```tmdl
partition fact_transactions = entity
    mode: directLake
    source
        entityName: fact_transactions
        schemaName: dbo
        expressionSource: 'DirectLake - lh_Finance_Silver'
```

#### Date relationship
`fact_transactions.transaction_date` is `dateTime`. `dim_date.Date` is `dateTime`.
Relationship **must** include:
```tmdl
joinOnDateBehavior: DatePartOnly
```

---

### DAX rules

- ALL measures use `VAR / RETURN` pattern
- Display folders: workspace-specific; minimum groupings `Volume`, `Amount`, `Time`, `Risk` (see HARNESS.md)
- All measures get `///` description for Copilot readiness
- Raw columns hidden in report view; only measures and key dims exposed
- No implicit measures anywhere
- `DIVIDE()` used for all division (never `/`) — third arg = 0

---

### Design: powerbi.tips Layout Trifecta

- **Scrim**: single background image (Python Pillow script (1280×720px — never PowerPoint, DPI mismatch)), not individual shapes
- **Page layout**: 1280×720, left nav panel (200px), top header (56px), content zone
- **Theme**: Power Designer export — apply as customTheme under RegisteredResources in `report.json` (not baseTheme under SharedResources; the filename gains a hash on first Fabric sync — update the reference then). Some existing reports still reference baseTheme/SharedResources; migrate them one at a time with Desktop validate. Do not bulk-edit report.json.
- **Colors**: max 4 intentional colors, consistent semantic meaning
- **Typography**: DIN for callouts/KPIs, Segoe UI for labels
- **Pages**: workspace-specific; see ws_<name>/CONTEXT.md. (Finance: Overview, Transactions, Trends.)

---

### Known Fabric gotchas

- `delta.columnMapping.mode=name` must be set AT WRITE TIME — retrofitting breaks DirectLake framing
- `ALTER TABLE RENAME COLUMN` is unsupported in T-SQL analytics engine — renames via Spark SQL only
- `%pip` is blocked in pipeline execution — use Fabric Environment for extra packages
- Null GUID (00000000-0000-0000-0000-000000000000) as workspaceId in pipeline activity JSON is valid and confirmed working for same-workspace notebook references — resolves correctly across Dev/Test/Prod without hardcoding environment-specific IDs.
- `logicalId` values in `.platform` files must be stable UUIDs — never regenerate them after first Fabric sync
- Theme file renamed with hash on Fabric sync →
  Apply as customTheme under RegisteredResources in report.json
  (not baseTheme under SharedResources). Update report.json to
  reference hashed filename after first Fabric sync.

---

## Common commands

`/sync-status` -- read-only repo-vs-Fabric sync report: branch state, changed artifacts per workspace folder, placeholder-ID scan (project skill).
`/new-workspace <name>` -- scaffold the repo side of a new workspace: folder, CONTEXT.md, PROJECTS.md row (project command).
`/start` -- session-start report: MCP, Azure login, Git state, PROJECTS.md status (user-level command).
`/pause-capacity` -- check, pause or resume the F2 capacity; asks for confirmation first (user-level command).
`powerbi-report-author validate` -- validate PBIR edits.
`powerbi-desktop status | reload | screenshot` -- Bridge CLI commands.
Branch backfill after a PR merge: see HARNESS.md.

---

## Directories and files to avoid

`_archive/` — not active workspace content.
`.claude/scratchpads/` — session-only, not committed.
`**/.pbi/` — gitignored; always excluded.
`**/diagramLayout.json` — gitignored; always excluded.
`**/localSettings.json` — gitignored; always excluded.
`**/editorSettings.json` — gitignored; always excluded.
Never regenerate `logicalId` values after first Fabric sync.
Reset `database.tmdl` / `model.tmdl` after Desktop sessions (see HARNESS.md Phase 5).
