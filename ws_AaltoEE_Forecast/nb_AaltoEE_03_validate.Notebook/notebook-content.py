# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "f40e0030-a5ab-40aa-ba4b-90edd23ba8d0",
# META       "default_lakehouse_name": "lh_AaltoEE",
# META       "default_lakehouse_workspace_id": "167ca0e1-7e56-40f5-8a99-ccf78ec0ac81",
# META       "known_lakehouses": [
# META         {
# META           "id": "f40e0030-a5ab-40aa-ba4b-90edd23ba8d0"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# nb_AaltoEE_03_validate
# Purpose: Run acceptance checks on Gold tables
# Validates row counts, monetary totals, referential integrity,
# and data quality flags against source control values
# ----------------------------------------------------------------

from pyspark.sql import functions as F
from datetime import datetime

print(f"Validation session started: {datetime.now()}")
print("Running acceptance checks on Gold tables...")
print()

# Source control values from reevaluation document
SOURCE_CONTROL = {
    "actuals_rows": 795,
    "actuals_total": 14635067.81,
    "accruals_rows": 313,
    "accruals_total": 6616025.78,
    "pipeline_rows": 126,
    "pipeline_weighted_this_year": 828900.00
}

results = []

def check(name, actual, expected, tolerance=1.0):
    passed = abs(actual - expected) <= tolerance
    results.append(passed)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | {name}")
    print(f"         Got: {actual:,.2f}  Expected: {expected:,.2f}")
    return passed

print("─" * 55)
print("CHECK 1 — Actuals reconciliation")
print("─" * 55)
df_actuals = spark.table("gold_fact_net_sales")
check("Row count", df_actuals.count(), SOURCE_CONTROL["actuals_rows"], 0)
check("Total Net Sales EUR",
    df_actuals.agg(F.sum("Net_Sales")).collect()[0][0],
    SOURCE_CONTROL["actuals_total"])

neg_rows = df_actuals.filter(F.col("Net_Sales") < 0).count()
zero_rows = df_actuals.filter(F.col("Net_Sales") == 0).count()
print(f"  ℹ Negative rows preserved: {neg_rows} (expected 16)")
print(f"  ℹ Zero rows preserved: {zero_rows} (expected 102)")

print()
print("─" * 55)
print("CHECK 2 — Accruals reconciliation")
print("─" * 55)
df_accruals = spark.table("gold_fact_accruals")
check("Row count", df_accruals.count(), SOURCE_CONTROL["accruals_rows"], 0)
check("Total Accrued Forecast EUR",
    df_accruals.agg(F.sum("Accrued_Net_Sales_Forecast")).collect()[0][0],
    SOURCE_CONTROL["accruals_total"])

unassigned = df_accruals.filter(F.col("BU") == "Unassigned").count()
unassigned_total = df_accruals.filter(
    F.col("BU") == "Unassigned"
).agg(F.sum("Accrued_Net_Sales_Forecast")).collect()[0][0]
print(f"  ℹ Unassigned BU rows: {unassigned} (expected 4)")
print(f"  ℹ Unassigned BU total: €{unassigned_total:,.2f} (expected €42,550.00)")

print()
print("─" * 55)
print("CHECK 3 — Pipeline reconciliation")
print("─" * 55)
df_pipeline = spark.table("gold_fact_pipeline")
check("Row count", df_pipeline.count(), SOURCE_CONTROL["pipeline_rows"], 0)
check("Weighted Value This Year EUR",
    df_pipeline.agg(F.sum("Weighted_Value_This_Year")).collect()[0][0],
    SOURCE_CONTROL["pipeline_weighted_this_year"])

missing_prob = df_pipeline.filter(F.col("Validation_Flag") == "MISSING_PROBABILITY").count()
alloc_exceed = df_pipeline.filter(F.col("Validation_Flag") == "ALLOCATION_EXCEEDS_100PCT").count()
print(f"  ⚠ Missing probability: {missing_prob} opportunities (require Finance review)")
print(f"  ⚠ Allocation exceeds 100%: {alloc_exceed} opportunity (require Finance review)")

print()
print("─" * 55)
print("CHECK 4 — Referential integrity")
print("─" * 55)
# Check all BU values exist in gold_dim_bu
dim_bu_values = [r.BU_Name for r in spark.table("gold_dim_bu").select("BU_Name").collect()]

for table, col in [
    ("gold_fact_net_sales", "BU"),
    ("gold_fact_accruals", "BU"),
    ("gold_fact_pipeline", "BU")
]:
    orphan_count = spark.table(table).filter(
        ~F.col(col).isin(dim_bu_values)
    ).count()
    status = "✓ PASS" if orphan_count == 0 else "✗ FAIL"
    print(f"  {status} | {table}[BU] → gold_dim_bu: {orphan_count} orphan rows")
    results.append(orphan_count == 0)

# Check all dates in actuals and accruals exist in gold_dim_date
dim_dates = [r.Date for r in spark.table("gold_dim_date").select("Date").collect()]
for table, col in [
    ("gold_fact_net_sales", "Date"),
    ("gold_fact_accruals", "Date")
]:
    orphan_count = spark.table(table).filter(
        ~F.col(col).isin(dim_dates)
    ).count()
    status = "✓ PASS" if orphan_count == 0 else "✗ FAIL"
    print(f"  {status} | {table}[Date] → gold_dim_date: {orphan_count} orphan rows")
    results.append(orphan_count == 0)

print()
print("─" * 55)
print("CHECK 5 — Combined forecast total")
print("─" * 55)
actuals_total = df_actuals.agg(F.sum("Net_Sales")).collect()[0][0]
accruals_total = df_accruals.agg(F.sum("Accrued_Net_Sales_Forecast")).collect()[0][0]
pipeline_total = df_pipeline.agg(F.sum("Weighted_Value_This_Year")).collect()[0][0]
forecast_total = actuals_total + accruals_total + pipeline_total

print(f"  Actuals:  €{actuals_total:>15,.2f}")
print(f"  Accruals: €{accruals_total:>15,.2f}")
print(f"  Pipeline: €{pipeline_total:>15,.2f}")
print(f"  {'─' * 35}")
print(f"  Total:    €{forecast_total:>15,.2f}")

print()
print("=" * 55)
passed = sum(results)
total_checks = len(results)
overall = "ALL CHECKS PASSED" if passed == total_checks else f"{passed}/{total_checks} CHECKS PASSED"
print(f"OVERALL: {overall}")
print(f"Validation completed: {datetime.now()}")
print("=" * 55)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
