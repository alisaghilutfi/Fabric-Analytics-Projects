# Fabric Capacity Scripts

PowerShell helpers for managing the `alisaghif2capacity` Fabric F2 capacity — used to avoid unnecessary charges when the capacity is left running idle.

## setup-env.ps1

Sets the Azure environment variables (`AZURE_SUBSCRIPTION_ID`, `AZURE_FABRIC_RG`, `AZURE_FABRIC_CAPACITY`) required by the other two scripts. Run once per machine; values are stored locally via `[System.Environment]::SetEnvironmentVariable` (User scope), never committed to the repo.

```powershell
.\scripts\setup-env.ps1
```

Close and reopen PowerShell afterward for the variables to take effect.

## fabric-capacity-toggle.ps1

One-command toggle: pauses the capacity if it's active, resumes it if it's paused. Intended to be run at the start and end of every Fabric work session.

```powershell
.\scripts\fabric-capacity-toggle.ps1
```

## fabric-capacity-monitor.ps1

Checks capacity status and estimated cost, with optional auto-pause and continuous watch mode.

```powershell
.\scripts\fabric-capacity-monitor.ps1              # status check only
.\scripts\fabric-capacity-monitor.ps1 -AutoPause    # pause if running longer than -MaxHours (default 4)
.\scripts\fabric-capacity-monitor.ps1 -Resume       # force resume
.\scripts\fabric-capacity-monitor.ps1 -Watch        # recheck every -WatchEvery minutes (default 30)
```

## Requirements

- Azure CLI (`az`) installed and available on PATH
- Logged in via `az login` (scripts will prompt if not)
- `setup-env.ps1` run at least once on the machine
