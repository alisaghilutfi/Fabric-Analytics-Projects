# fabric-capacity-monitor.ps1
# Monitors your Fabric F2 capacity:
#   - Status check with cost estimate
#   - Alert if running too long
#   - Optional auto-pause
#   - Optional continuous watch mode
#
# Usage:
#   .\scripts\fabric-capacity-monitor.ps1              # status check only
#   .\scripts\fabric-capacity-monitor.ps1 -AutoPause   # pause if > $MaxHours
#   .\scripts\fabric-capacity-monitor.ps1 -Resume      # force resume
#   .\scripts\fabric-capacity-monitor.ps1 -Watch       # check every 30 min
#
# Required environment variables (set once via setup-env.ps1):
#   AZURE_SUBSCRIPTION_ID
#   AZURE_FABRIC_RG
#   AZURE_FABRIC_CAPACITY

param(
    [switch]$AutoPause,
    [switch]$Resume,
    [switch]$Watch,
    [int]$MaxHours   = 4,
    [int]$WatchEvery = 30
)

$ErrorActionPreference = "Stop"

# ---- READ FROM ENVIRONMENT ----
$SubscriptionId = $env:AZURE_SUBSCRIPTION_ID
$ResourceGroup  = $env:AZURE_FABRIC_RG
$CapacityName   = $env:AZURE_FABRIC_CAPACITY
$CostPerHourUSD = 0.36   # F2 pay-as-you-go rate
# --------------------------------

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    WARNING: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "    $msg" -ForegroundColor Red }

# Guard: ensure env vars are set
if (-not $SubscriptionId -or -not $ResourceGroup -or -not $CapacityName) {
    Write-Fail "Missing environment variables. Run setup-env.ps1 first."
    Write-Host "  Required: AZURE_SUBSCRIPTION_ID, AZURE_FABRIC_RG, AZURE_FABRIC_CAPACITY" -ForegroundColor Yellow
    exit 1
}

function Get-CapacityStatus {
    az resource show `
        --resource-group $ResourceGroup `
        --name $CapacityName `
        --resource-type "Microsoft.Fabric/capacities" `
        --subscription $SubscriptionId | ConvertFrom-Json
}

function Show-Status($capacity) {
    $state = $capacity.properties.state
    $sku   = $capacity.sku.name
    $color = if ($state -eq "Active") { "Green" } else { "Yellow" }
    Write-Host ""
    Write-Host "  Capacity : $CapacityName" -ForegroundColor White
    Write-Host "  SKU      : $sku (2 CUs)" -ForegroundColor White
    Write-Host "  Status   : $state" -ForegroundColor $color
    Write-Host "  Rate     : ~`$$CostPerHourUSD USD/hr when active" -ForegroundColor White
    Write-Host ""
}

function Pause-Capacity {
    Write-Step "Pausing $CapacityName"
    az fabric capacity suspend `
        --resource-group $ResourceGroup `
        --capacity-name $CapacityName `
        --subscription $SubscriptionId | Out-Null
    Write-OK "Capacity paused. No further charges until resumed."
}

function Resume-Capacity {
    Write-Step "Resuming $CapacityName"
    az fabric capacity resume `
        --resource-group $ResourceGroup `
        --capacity-name $CapacityName `
        --subscription $SubscriptionId | Out-Null
    Write-OK "Capacity active. Remember to pause when done."
}

function Check-AndAlert($capacity) {
    $state = $capacity.properties.state
    if ($state -ne "Active") {
        Write-OK "Capacity is $state — not accruing charges."
        return
    }

    $updatedAt    = [datetime]$capacity.systemData.lastModifiedAt
    $hoursRunning = [math]::Round(((Get-Date) - $updatedAt).TotalHours, 1)
    $costSoFar    = [math]::Round($hoursRunning * $CostPerHourUSD, 2)

    Write-OK "Capacity is ACTIVE"
    Write-Host "  Last state change : $updatedAt" -ForegroundColor White
    Write-Host "  Hours running     : ~$hoursRunning hrs" -ForegroundColor White
    Write-Host "  Cost this session : ~`$$costSoFar USD" -ForegroundColor White

    if ($hoursRunning -gt $MaxHours) {
        Write-Warn "Running for $hoursRunning hours — limit is $MaxHours hrs"
        if ($AutoPause) {
            Pause-Capacity
        } else {
            Write-Warn "Run with -AutoPause to pause automatically."
        }
    } else {
        $hoursLeft = [math]::Round($MaxHours - $hoursRunning, 1)
        Write-OK "Within limit — ~$hoursLeft hrs remaining before alert."
    }
}

# ---- MAIN ----
Write-Step "Fabric Capacity Monitor — $CapacityName"

$azCheck = az account show 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Step "Not logged into Azure. Running az login..."
    az login
}

if ($Resume) {
    Resume-Capacity
    exit 0
}

if ($Watch) {
    Write-Step "Watch mode — checking every $WatchEvery minutes. Ctrl+C to stop."
    while ($true) {
        $capacity = Get-CapacityStatus
        Show-Status $capacity
        Check-AndAlert $capacity
        Write-Host "  Next check in $WatchEvery minutes..." -ForegroundColor DarkGray
        Start-Sleep -Seconds ($WatchEvery * 60)
    }
}

$capacity = Get-CapacityStatus
Show-Status $capacity
Check-AndAlert $capacity
