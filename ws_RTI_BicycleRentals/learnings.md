# ws_RTI_BicycleRentals — Learnings

## PBIR reports built via powerbi-modeling-mcp without Desktop validation fail to render in the Fabric Service

**Date:** 2026-10-14

PBIR reports authored entirely through the `powerbi-modeling-mcp` CLI and committed to Git via Source Control sync can fail to render in the Fabric Service with `manifest.json 404` errors in the browser console. The report loads indefinitely and never displays.

**Root cause:** The PBIR JSON definition files are structurally valid and pass `powerbi-report-author validate` with 0 errors. The issue is a Service-side blob corruption: the Fabric item exists but its internal manifest is missing or unresolvable because it was never provisioned through a Desktop save or a direct Fabric API upload. Git sync alone does not generate a valid Service manifest.

**Fix:**
1. Delete the broken report item from the Fabric Service.
2. Use `powerbi-report-author pack --raw --mode create` to serialize the local PBIR into a Fabric create-report payload.
3. POST to the Fabric Reports API (`/v1/workspaces/{ws}/reports`) to reprovision the report with a valid Service artifact. Poll LRO to Succeeded.
4. Validate the report loads in the browser (no manifest.json 404, all pages render).

**Alternative fix if the Fabric API is unavailable:** Open the local `.pbip` project in Power BI Desktop and save — Desktop regenerates the PBIR files with a valid manifest. Then publish to the workspace from Desktop (`File → Publish → Publish to Power BI`).

**Always open and save in Desktop or publish via Fabric API after code-based PBIR authoring.** Do not rely on Git Source Control sync alone to provision a renderable Service artifact.

**Rebuild in Desktop if `manifest.json 404` errors appear in the browser console.**

---

## azureMap visual: Location (Category) and explicit Lat/Long conflict

**Date:** 2026-10-14

The `azureMap` visual throws a `RENDER_FAILED` error (`DataViewMappingError_LocationAndLatLongGroup`) when the `Category` role contains a location field (e.g., BikepointID) and the `Y`/`X` roles contain raw (unaggregated) `Column` expressions for Latitude and Longitude.

**Error message:** "Remove Location to display latitude and longitude pairs. Alternatively, you can also keep Location and set the aggregate for Latitude and Longitude to Average."

**Fix:** Keep the `Category` role populated (it is a required role for `azureMap`) and wrap the Latitude and Longitude column expressions in `Aggregation` with `Function: 4` (Average). Update `queryRef` to `Average(Table.Column)` and keep `nativeQueryRef` as the column name.

**Note:** The `Category` role is required by the `azureMap` PBIR schema — removing it triggers `PBIR_ROLE_REQUIRED_MISSING` validation error. Always use the Average aggregation pattern for Lat/Long when a Location identifier is also bound.

---

## Power BI Desktop Store version uses PBIDesktopStore.exe, not PBIDesktop.exe

**Date:** 2026-10-14

The `powerbi-report-author` CLI searches standard paths for `PBIDesktop.exe` and does not automatically find the Microsoft Store version, which installs to `C:\Program Files\WindowsApps\Microsoft.MicrosoftPowerBIDesktop_<version>_x64__8wekyb3d8bbwe\bin\PBIDesktop.exe`.

**Fix:** Set `PBI_DESKTOP_PATH` to the full path resolved via:
```powershell
$pkg = Get-AppxPackage -Name "*MicrosoftPowerBIDesktop*"
$exe = Get-ChildItem "$($pkg.InstallLocation)\bin" -Filter "PBIDesktop.exe"
$env:PBI_DESKTOP_PATH = $exe.FullName
```

The `PBIDesktopStore.exe` stub in `WindowsApps` resolves correctly for `--status` checks but is rejected by the CLI's open/attach command.

---

## definition.pbir must be byConnection for Fabric API publish, byPath for Git/Desktop

**Date:** 2026-10-14

The Fabric Reports API rejects `definition.pbir` with `byPath` dataset reference. Every `pack → POST /reports` or `pack → POST /reports/{id}/updateDefinition` cycle requires:
1. Temporarily change `definition.pbir` to `byConnection` with the live semantic model GUID.
2. Pack and publish.
3. Revert `definition.pbir` to `byPath` (`../sm_<Name>.SemanticModel`) for Git commit and local Desktop use.

The `byPath` form is the canonical committed form. The `byConnection` form is a publish-time transformation only.
