# setup-env.ps1
# Run this ONCE to set your Azure environment variables permanently.
# After running, close and reopen PowerShell for variables to take effect.
# These values are stored on YOUR machine only — never in the repo.
# Usage: .\setup-env.ps1

Write-Host ""
Write-Host "==> Setting Fabric capacity environment variables" -ForegroundColor Cyan
Write-Host "    These are stored on your machine only - not in git." -ForegroundColor Yellow
Write-Host ""

[System.Environment]::SetEnvironmentVariable(
    "AZURE_SUBSCRIPTION_ID",
    "967210d2-4ed2-47cd-aa8b-1e9a982502f8",
    "User"
)

[System.Environment]::SetEnvironmentVariable(
    "AZURE_FABRIC_RG",
    "Fabric_AI_Lab",
    "User"
)

[System.Environment]::SetEnvironmentVariable(
    "AZURE_FABRIC_CAPACITY",
    "alisaghif2capacity",
    "User"
)

Write-Host "    AZURE_SUBSCRIPTION_ID = 967210d2-4ed2-47cd-aa8b-1e9a982502f8" -ForegroundColor Green
Write-Host "    AZURE_FABRIC_RG       = Fabric_AI_Lab" -ForegroundColor Green
Write-Host "    AZURE_FABRIC_CAPACITY = alisaghif2capacity" -ForegroundColor Green
Write-Host ""
Write-Host "==> Done. Close and reopen PowerShell, then verify with:" -ForegroundColor Cyan
Write-Host "    echo" '$env:AZURE_SUBSCRIPTION_ID' -ForegroundColor White
