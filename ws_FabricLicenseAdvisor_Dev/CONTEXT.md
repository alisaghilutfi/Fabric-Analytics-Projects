# ws_FabricLicenseAdvisor_Dev — Context

## Purpose
Interactive Rayfin Fabric App that guides users through Microsoft Fabric licensing decisions
and outputs a recommended license path + monthly cost estimate.
Target users: Fabric admins, consultants, BI managers evaluating Fabric.
Gallery submission target: Fabric Apps Gallery (community.fabric.microsoft.com).

## Architecture
- **Repo split:** Workspace items -> Fabric-Analytics-Projects (this repo); Rayfin app -> Fabric-Apps-Projects/apps/FabricApp_FabricLicenseAdvisor/
- **Capacity:** alisaghif2capacity (F2 PAYG, North Europe)
- **Lakehouse:** lh_pricing (Delta tables only -- no shortcuts)
- **Semantic model:** sm_FabricLicenseAdvisor (item ID: 19433fad-1ba4-4f6d-b7d1-5f8c77827ca9)

## Lakehouse tables

| Table | Description |
|---|---|
| pricing_sku | F2-F8192 capacity SKUs -- CUs, PAYG $/month, reserved $/month, feature flags |
| pricing_user_licenses | Free / Pro / PPU -- $/user/month, capability matrix |
| pricing_storage | OneLake and mirrored storage pricing |

## App screens (Rayfin)

| Screen | Route | Description |
|---|---|---|
| License Wizard | default | Decision tree: workload type -> user count -> viewer count -> recommended SKU |
| Cost Estimator | /estimate | Interactive calculator: SKU + users + storage -> monthly total (PAYG vs reserved) |
| SKU Reference | /skus | Full SKU table with CUs, prices, feature flags |
| F64 Break-Even | /breakeven | Calculator: how many viewers before F64 reserved beats per-user Pro licenses |

## Pricing source
RADACAD "How much does Microsoft Fabric really cost -- November 2026" + MS Learn /fabric/enterprise/licenses.
US list prices. Region disclaimer shown in UI.
Last verified: 2026-11-16.

## Deployment
- **Workspace ID:** 9629e4d5-9acb-499e-b936-1fd7a3b0241d
- **Backend workspace ID:** b23af3d8-bda7-4a5f-822f-ba900bd4fde5
- **Semantic model item ID:** 19433fad-1ba4-4f6d-b7d1-5f8c77827ca9
- **Live URL:** https://brief-fawn-30a395c0ef-northeurope.webapp.fabricapps.net
- **Rayfin item ID:** 17b217db-4e21-430a-a533-5aa99cc39a2d

## Last Session

**Date:** 2026-11-16

**Completed:**
- Implemented all 4 Rayfin app screens from scratch: SkuReference, CostEstimator, BreakEven, LicenseWizard.
- Fixed DAX SORT() anti-pattern -- replaced with TOPN(999, ...).
- App fully live at https://brief-fawn-30a395c0ef-northeurope.webapp.fabricapps.net.

**Pick up next at:** UI polish pass; consider gallery submission to Fabric Apps Gallery.
