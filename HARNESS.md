# Ali Saghi — Fabric & Power BI Development Harness

## Who I Am
- Name: Ali Saghi
- Role: DP-600 certified Microsoft Fabric Analytics Engineer
- Company: Lotusoftware (sole proprietor, data analytics consultancy)
- Location: Vantaa, Finland

## The Two-Harness System

This repository is the central context store for all Fabric and Power BI
development work. It serves as the "second brain" that both humans and
agents read from and write back to after every session.

### Harness 1 — Context (This Repo)
Stores and organizes all project context so agents never start from zero.

Structure:
- `HARNESS.md` — this file, the master architecture reference
- `PROJECTS.md` — registry of all active projects, statuses, and blockers
- `ws_<name>/CONTEXT.md` — per-project instructions page, written and
  updated by agents after each session

### Harness 2 — Execution (VS Code + Claude Code)
Executes work against Microsoft Fabric and Power BI via MCP servers.

Tools:
- Claude Code for VS Code (agent mode)
- Fabric MCP — workspace and item management
- powerbi-modeling-mcp — semantic model authoring and DAX execution
- pylance mcp server — Python/PySpark assistance
- skills-for-fabric (cloned at C:\Users\alisa\skills-for-fabric) —
  CLAUDE.md loaded automatically, grounding all Fabric operations

### The Handoff Rule
Every agent session MUST follow this loop:

1. READ the relevant `ws_<name>/CONTEXT.md` at session start
2. READ `PROJECTS.md` to check current status and blockers
3. DO the work using Fabric MCP and skills-for-fabric
4. WRITE a recap back to `ws_<name>/CONTEXT.md` when done:
   - What was completed
   - What was left unfinished
   - Any new blockers discovered
   - Where to pick up next session
5. UPDATE `PROJECTS.md` status for this project

No session ends without steps 4 and 5. Output without a filed report
is work the system never learns from.

## Authentication
Azure CLI authenticated against:
- Tenant: Default Directory
- Subscription: AzureSubscription_PAYG
- Account: alisaghi_fabric

To refresh if token expires:
```bash
az login --use-device-code
az account get-access-token --resource https://api.fabric.microsoft.com
```

## Fabric Workspaces
| Workspace | Purpose | Status |
|---|---|---|
| ws_AdventureWorks | Medallion architecture practice project | Active |
| ws_Finance_Analysis | Power BI vibe-coding with Claude Code | Active |
| ws_DS_BankChurn | Data science / ML project | Active |
| ws_RTI_Crypto | Real-time intelligence, Eventhouse, KQL | Active |
| ws_USGS_Earthquake | Portfolio documentation project | Active |
| ws_RTI_BicycleRentals | Real-time intelligence project | Active |
| ws_dp600 | DP-600 exam preparation workspace | Reference |
| ws_AgenticLab | Agentic AI experimentation | Active |

## Agent Roles
Three agents operate in this system, each with a narrow job:

**FabricEngineer** — execution agent
Reads the instructions page, does the Fabric/Power BI work via MCP,
files the recap. Lives in Claude Code (VS Code). Uses skills-for-fabric.

**ProjectPlanner** — planning agent
Reviews meeting notes, blockers, and status. Helps plan next steps for
any active project. Lives in claude.ai (Fabric & Power BI MCP project).

**PortfolioBuilder** — output agent
Takes completed project work and drafts portfolio case studies,
documentation, and consulting deliverables for Lotusoftware. Lives in
claude.ai (Lotusoftware & Business Finland project).

## Skills Reference
All Fabric operations follow the skills in `C:\Users\alisa\skills-for-fabric`:
- Medallion architecture → `agents/FabricDataEngineer.agent.md`
- Power BI reports → `skills/powerbi-report-authoring/SKILL.md`
- Semantic models → `skills/semantic-model-authoring/SKILL.md`
- SQL Warehouse → `skills/sqldw-authoring-cli/SKILL.md`
- Spark/Lakehouse → `skills/spark-authoring-cli/SKILL.md`
- KQL/Eventhouse → `skills/eventhouse-authoring-cli/SKILL.md`

## GitHub Repos
- `alisaghilutfi/Fabric-Analytics-Projects` — this repo, all Fabric work
- `alisaghilutfi/fabric-mcp` — custom Python Fabric MCP server

## Key Principle
The prompt is not the product. The harness is.
Context organized here means agents execute correctly on the first try,
not the fifth. Every session that files its recap makes the next session
faster and more accurate.