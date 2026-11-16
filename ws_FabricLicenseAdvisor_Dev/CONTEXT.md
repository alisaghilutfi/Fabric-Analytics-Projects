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
- **Semantic model:** sm_FabricLicenseAdvisor (to be created after tables are loaded)

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
- **Live URL:** (pending first deploy)
- **Rayfin item ID:** (pending)
