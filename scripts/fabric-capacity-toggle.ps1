# fabric-capacity-toggle.ps1
# Quick one-command toggle: pauses if running, resumes if paused.
# Run at the START and END of every Fabric session.
# Usage: .\scripts\fabric-capacity-toggle.ps1
#
# Required environment variables (set once via setup-env.ps1):
#   AZURE_SUBSCRIPTION_ID
#   AZURE_FABRIC_RG
#   AZURE_FABRIC_CAPACITY

$ErrorActionPreference = "Stop"

# ---- READ FROM ENVIRONMENT ----
$SubscriptionId = $env:AZURE_SUBSCRIPTION_ID
$ResourceGroup  = $env:AZURE_FABRIC_RG
$CapacityName   = $env:AZURE_FABRIC_CAPACITY
# --------------------------------

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "    $msg" -ForegroundColor Red }

# Guard: ensure env vars are set
if (-not $SubscriptionId -or -not $ResourceGroup -or -not $CapacityName) {
    Write-Fail "Missing environment variables. Run setup-env.ps1 first."
    Write-Host "  Required: AZURE_SUBSCRIPTION_ID, AZURE_FABRIC_RG, AZURE_FABRIC_CAPACITY" -ForegroundColor Yellow
    exit 1
}

# Check az login
$azCheck = az account show 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Step "Not logged into Azure. Running az login..."
    az login
}

# Get current capacity state
$resource = az resource show `
    --resource-group $ResourceGroup `
    --name $CapacityName `
    --resource-type "Microsoft.Fabric/capacities" `
    --subscription $SubscriptionId | ConvertFrom-Json

$state = $resource.properties.state
Write-Step "$CapacityName — current status: $state"

if ($state -eq "Active") {
    Write-Step "Pausing capacity..."
    az fabric capacity suspend `
        --resource-group $ResourceGroup `
        --capacity-name $CapacityName `
        --subscription $SubscriptionId | Out-Null
    Write-OK "Paused. No charges until next resume."
} elseif ($state -in @("Paused", "Suspended")) {
    Write-Step "Resuming capacity..."
    az fabric capacity resume `
        --resource-group $ResourceGroup `
        --capacity-name $CapacityName `
        --subscription $SubscriptionId | Out-Null
    Write-OK "Active. Ready for Fabric work."
    Write-Host "    Remember to run this script again when you finish." -ForegroundColor Yellow
} else {
    Write-Host "    Unexpected state: $state" -ForegroundColor Yellow
    Write-Host "    Check: https://portal.azure.com" -ForegroundColor Cyan
}
