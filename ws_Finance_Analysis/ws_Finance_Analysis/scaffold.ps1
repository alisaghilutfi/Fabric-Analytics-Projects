# scaffold.ps1
# Run this ONCE from C:\Lotusoftware\Power_BI_Finance_Analysis (or wherever you clone the repo)
# Creates the full ws_Finance_Analysis folder structure ready for Fabric Git sync
#
# Usage:
#   cd C:\Users\alisa\<your-repo-clone>
#   .\ws_Finance_Analysis\scaffold.ps1

$base = "ws_Finance_Analysis"

$dirs = @(
    # Bronze Lakehouse
    "$base\lh_Finance_Bronze.Lakehouse",

    # Silver Lakehouse
    "$base\lh_Finance_Silver.Lakehouse",

    # Notebooks
    "$base\nb_Finance_Bronze.Notebook",
    "$base\nb_Finance_Silver.Notebook",

    # DataPipeline
    "$base\pl_Finance.DataPipeline",

    # Semantic Model
    "$base\sm_Finance.SemanticModel\definition\tables",

    # Report
    "$base\rpt_Finance.Report\definition\pages",
    "$base\rpt_Finance.Report\StaticResources\SharedResources\BaseThemes",

    # Data (CSVs committed here, Bronze notebook reads from GitHub raw URLs)
    "$base\data"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "Created: $dir"
}

Write-Host ""
Write-Host "Scaffold complete. Open this folder in VS Code and start Claude Code." -ForegroundColor Green
Write-Host "Copy customers.csv and finance_transactions.csv into $base\data\" -ForegroundColor Yellow
