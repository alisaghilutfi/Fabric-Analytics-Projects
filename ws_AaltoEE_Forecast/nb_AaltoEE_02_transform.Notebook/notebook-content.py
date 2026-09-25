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

# nb_AaltoEE_02_transform
# Purpose: Read Silver Delta tables, apply star schema transformation,
# write Gold fact and dimension tables to Lakehouse
# ----------------------------------------------------------------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

print(f"Transform session started: {datetime.now()}")

# Verify Silver tables are available
for table in ["silver_net_sales", "silver_accruals", "silver_pipeline"]:
    count = spark.table(table).count()
    print(f"  {table}: {count} rows")

print("Silver tables verified — proceeding to Gold transformation")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD: Dim BU ─────────────────────────────────────────────────
print("Building gold_dim_bu...")

dim_bu_data = [
    (1, "ExEd Programs BU"),
    (2, "Qualification Programs BU"),
    (3, "University Programs BU"),
    (4, "Unassigned")
]

df_dim_bu = spark.createDataFrame(
    dim_bu_data,
    ["BU_Sort", "BU_Name"]
)

(
    df_dim_bu
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_dim_bu")
)

count = spark.table("gold_dim_bu").count()
print(f"gold_dim_bu written: {count} rows")
spark.table("gold_dim_bu").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD: Dim Date (extended 2024–2028) ─────────────────────────
print("Building gold_dim_date (2024–2028)...")

df_dim_date = spark.sql("""
    SELECT
        date_trunc('month', date_add(DATE('2024-01-01'), pos)) AS Date,
        year(date_trunc('month', date_add(DATE('2024-01-01'), pos))) AS Year,
        month(date_trunc('month', date_add(DATE('2024-01-01'), pos))) AS Month_Number,
        date_format(date_trunc('month', date_add(DATE('2024-01-01'), pos)), 'MMMM') AS Month_Name,
        date_format(date_trunc('month', date_add(DATE('2024-01-01'), pos)), 'MMM') AS Month_Short,
        CASE 
            WHEN month(date_trunc('month', date_add(DATE('2024-01-01'), pos))) <= 3 THEN 'Q1'
            WHEN month(date_trunc('month', date_add(DATE('2024-01-01'), pos))) <= 6 THEN 'Q2'
            WHEN month(date_trunc('month', date_add(DATE('2024-01-01'), pos))) <= 9 THEN 'Q3'
            ELSE 'Q4'
        END AS Quarter,
        date_format(date_trunc('month', date_add(DATE('2024-01-01'), pos)), 'yyyy-MM') AS Year_Month,
        year(date_trunc('month', date_add(DATE('2024-01-01'), pos))) * 100 +
        month(date_trunc('month', date_add(DATE('2024-01-01'), pos))) AS Month_Sort,
        CASE
            WHEN year(date_trunc('month', date_add(DATE('2024-01-01'), pos))) < 2026
            THEN 'Historical'
            WHEN month(date_trunc('month', date_add(DATE('2024-01-01'), pos))) <= 8
            AND year(date_trunc('month', date_add(DATE('2024-01-01'), pos))) = 2026
            THEN 'Actuals'
            WHEN year(date_trunc('month', date_add(DATE('2024-01-01'), pos))) = 2026
            THEN 'Forecast'
            ELSE 'Future'
        END AS Period_Type
    FROM (
        SELECT explode(sequence(0, 1826)) AS pos
    )
    WHERE day(date_add(DATE('2024-01-01'), pos)) = 1
""")

(
    df_dim_date
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_dim_date")
)

count = spark.table("gold_dim_date").count()
print(f"gold_dim_date written: {count} rows")
spark.table("gold_dim_date").orderBy("Month_Sort").show(70, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD: Fact Net Sales ─────────────────────────────────────────
print("Building gold_fact_net_sales...")

df_fact_net_sales = spark.sql("""
    SELECT
        s.BU,
        s.Cost_Center,
        s.Project,
        s.Year_Month,
        s.Date,
        s.Net_Sales,
        s.Validation_Flag,
        d.Quarter,
        d.Month_Name,
        d.Period_Type,
        s.Ingestion_Timestamp,
        s.Source_File
    FROM silver_net_sales s
    LEFT JOIN gold_dim_date d
        ON s.Date = d.Date
    LEFT JOIN gold_dim_bu b
        ON s.BU = b.BU_Name
""")

(
    df_fact_net_sales
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_fact_net_sales")
)

count = spark.table("gold_fact_net_sales").count()
total = spark.table("gold_fact_net_sales").agg(
    F.sum("Net_Sales").alias("total")
).collect()[0]["total"]

print(f"gold_fact_net_sales written: {count} rows")
print(f"Total Net Sales: €{total:,.2f}")
print(f"\nBy BU:")
spark.table("gold_fact_net_sales").groupBy("BU").agg(
    F.sum("Net_Sales").alias("Net_Sales"),
    F.count("*").alias("Rows")
).orderBy("Net_Sales", ascending=False).show(truncate=False)
print(f"\nBy Period_Type:")
spark.table("gold_fact_net_sales").groupBy("Period_Type").agg(
    F.sum("Net_Sales").alias("Net_Sales")
).show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD: Fact Accruals ──────────────────────────────────────────
print("Building gold_fact_accruals...")

df_fact_accruals = spark.sql("""
    SELECT
        s.BU,
        s.Project_Number,
        s.Accrual_Basis,
        s.Accrual_Type,
        s.Net_Sales_Year,
        s.Net_Sales_Month,
        s.Date,
        s.Accrued_Net_Sales_Forecast,
        s.Validation_Flag,
        d.Quarter,
        d.Month_Name,
        d.Period_Type,
        s.Ingestion_Timestamp,
        s.Source_File
    FROM silver_accruals s
    LEFT JOIN gold_dim_date d
        ON s.Date = d.Date
    LEFT JOIN gold_dim_bu b
        ON s.BU = b.BU_Name
""")

(
    df_fact_accruals
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_fact_accruals")
)

count = spark.table("gold_fact_accruals").count()
total = spark.table("gold_fact_accruals").agg(
    F.sum("Accrued_Net_Sales_Forecast").alias("total")
).collect()[0]["total"]

print(f"gold_fact_accruals written: {count} rows")
print(f"Total Accrued Forecast: €{total:,.2f}")
print(f"\nBy BU:")
spark.table("gold_fact_accruals").groupBy("BU").agg(
    F.sum("Accrued_Net_Sales_Forecast").alias("Forecast"),
    F.count("*").alias("Rows")
).orderBy("Forecast", ascending=False).show(truncate=False)
print(f"\nBy Accrual_Type:")
spark.table("gold_fact_accruals").groupBy("Accrual_Type").agg(
    F.sum("Accrued_Net_Sales_Forecast").alias("Forecast")
).orderBy("Forecast", ascending=False).show(truncate=False)
print(f"\nBy Month:")
spark.table("gold_fact_accruals").groupBy("Month_Name", "Net_Sales_Month").agg(
    F.sum("Accrued_Net_Sales_Forecast").alias("Forecast")
).orderBy("Net_Sales_Month").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD: Fact Pipeline ──────────────────────────────────────────
print("Building gold_fact_pipeline...")

df_fact_pipeline = spark.sql("""
    SELECT
        s.BU,
        s.Opportunity_Name,
        s.Est_Closing_Date,
        s.Closing_Date,
        s.Program_Start_Date,
        s.This_Year_Pct,
        s.Next_Year_Pct,
        s.Probability,
        s.Probability_Band,
        s.Value_EUR,
        s.Est_Value,
        s.Weighted_Value,
        s.Weighted_Value_This_Year,
        s.Validation_Flag,
        YEAR(s.Est_Closing_Date) AS Closing_Year,
        MONTH(s.Est_Closing_Date) AS Closing_Month,
        b.BU_Sort,
        s.Ingestion_Timestamp,
        s.Source_File
    FROM silver_pipeline s
    LEFT JOIN gold_dim_bu b
        ON s.BU = b.BU_Name
""")

(
    df_fact_pipeline
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_fact_pipeline")
)

count = spark.table("gold_fact_pipeline").count()
total_weighted = spark.table("gold_fact_pipeline").agg(
    F.sum("Weighted_Value_This_Year").alias("total")
).collect()[0]["total"]
total_value = spark.table("gold_fact_pipeline").agg(
    F.sum("Value_EUR").alias("total")
).collect()[0]["total"]

print(f"gold_fact_pipeline written: {count} rows")
print(f"Total Value EUR: €{total_value:,.2f}")
print(f"Total Weighted Value This Year: €{total_weighted:,.2f}")
print(f"\nBy BU:")
spark.table("gold_fact_pipeline").groupBy("BU").agg(
    F.sum("Value_EUR").alias("Total_Value"),
    F.sum("Weighted_Value_This_Year").alias("Weighted_This_Year"),
    F.count("*").alias("Opportunities")
).orderBy("Total_Value", ascending=False).show(truncate=False)
print(f"\nBy Probability Band:")
spark.table("gold_fact_pipeline").groupBy("Probability_Band").agg(
    F.sum("Value_EUR").alias("Total_Value"),
    F.count("*").alias("Opportunities")
).orderBy("Total_Value", ascending=False).show(truncate=False)
print(f"\nValidation flags:")
spark.table("gold_fact_pipeline").groupBy("Validation_Flag").count().show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD VALIDATION SUMMARY ──────────────────────────────────────
print("=" * 60)
print("nb_AaltoEE_02_transform — COMPLETED")
print("=" * 60)
print(f"Run timestamp: {datetime.now()}")
print()

# Dimension tables
print("DIMENSION TABLES:")
for table, expected in [("gold_dim_bu", 4), ("gold_dim_date", 12)]:
    count = spark.table(table).count()
    status = "✓ PASS" if count == expected else "✗ FAIL"
    print(f"  {status} | {table}: {count} rows (expected {expected})")

print()

# Fact tables
print("FACT TABLES:")
fact_checks = {
    "gold_fact_net_sales": {
        "expected_rows": 795,
        "expected_total": 14635067.81,
        "amount_col": "Net_Sales"
    },
    "gold_fact_accruals": {
        "expected_rows": 313,
        "expected_total": 6616025.78,
        "amount_col": "Accrued_Net_Sales_Forecast"
    },
    "gold_fact_pipeline": {
        "expected_rows": 126,
        "expected_total": 828900.00,
        "amount_col": "Weighted_Value_This_Year"
    }
}

all_passed = True
for table, config in fact_checks.items():
    count = spark.table(table).count()
    total = spark.table(table).agg(
        F.sum(config["amount_col"]).alias("total")
    ).collect()[0]["total"]
    row_ok = count == config["expected_rows"]
    total_ok = abs(total - config["expected_total"]) < 1.0
    status = "✓ PASS" if (row_ok and total_ok) else "✗ FAIL"
    if not (row_ok and total_ok):
        all_passed = False
    print(f"  {status} | {table}")
    print(f"         Rows:  {count} (expected {config['expected_rows']})")
    print(f"         Total: €{total:,.2f} (expected €{config['expected_total']:,.2f})")

print()
print("=" * 60)
print(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print("=" * 60)

# Combined forecast summary
print("\nCOMBINED FORECAST SUMMARY:")
actuals = spark.table("gold_fact_net_sales").agg(
    F.sum("Net_Sales")).collect()[0][0]
accruals = spark.table("gold_fact_accruals").agg(
    F.sum("Accrued_Net_Sales_Forecast")).collect()[0][0]
pipeline = spark.table("gold_fact_pipeline").agg(
    F.sum("Weighted_Value_This_Year")).collect()[0][0]
total_forecast = actuals + accruals + pipeline

print(f"  Actual Net Sales (Jan-Aug):     €{actuals:>15,.2f}")
print(f"  Accrued Forecast (Sep-Dec):     €{accruals:>15,.2f}")
print(f"  Weighted Pipeline (this year):  €{pipeline:>15,.2f}")
print(f"  {'─' * 40}")
print(f"  Total Forecast EUR:             €{total_forecast:>15,.2f}")
print()

# All Gold tables
print("GOLD TABLES IN LAKEHOUSE:")
gold_tables = [
    "gold_dim_bu", "gold_dim_date",
    "gold_fact_net_sales", "gold_fact_accruals", "gold_fact_pipeline"
]
for t in gold_tables:
    count = spark.table(t).count()
    print(f"  {t}: {count} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── GOLD: Forecast Snapshot ──────────────────────────────────────
print("Building gold_forecast_snapshot...")

from pyspark.sql import functions as F
from datetime import datetime

# Snapshot timestamp — identifies this pipeline run
snapshot_ts = datetime.now()
snapshot_date = snapshot_ts.strftime("%Y-%m-%d")
snapshot_version = snapshot_ts.strftime("%Y%m%d_%H%M%S")

# Build snapshot from Gold tables
df_actuals_snap = (
    spark.table("gold_fact_net_sales")
    .groupBy("BU", "Date", "Month_Name", "Quarter", "Period_Type")
    .agg(F.sum("Net_Sales").alias("Amount"))
    .withColumn("Component", F.lit("Actuals"))
)

df_accruals_snap = (
    spark.table("gold_fact_accruals")
    .groupBy("BU", "Date", "Month_Name", "Quarter", "Period_Type")
    .agg(F.sum("Accrued_Net_Sales_Forecast").alias("Amount"))
    .withColumn("Component", F.lit("Accruals"))
)

df_pipeline_snap = (
    spark.table("gold_fact_pipeline")
    .groupBy("BU")
    .agg(F.sum("Weighted_Value_This_Year").alias("Amount"))
    .withColumn("Date", F.lit(None).cast("timestamp"))
    .withColumn("Month_Name", F.lit("Full Year"))
    .withColumn("Quarter", F.lit("Full Year"))
    .withColumn("Period_Type", F.lit("Pipeline"))
    .withColumn("Component", F.lit("Pipeline"))
)

# Union all three components
df_snapshot = (
    df_actuals_snap
    .unionByName(df_accruals_snap)
    .unionByName(df_pipeline_snap)
    .withColumn("Snapshot_Timestamp", F.lit(snapshot_ts))
    .withColumn("Snapshot_Date", F.lit(snapshot_date))
    .withColumn("Snapshot_Version", F.lit(snapshot_version))
)

# Append to snapshot table — never overwrite
(
    df_snapshot
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable("gold_forecast_snapshot")
)

# Verify
count = spark.table("gold_forecast_snapshot").count()
versions = spark.table("gold_forecast_snapshot").select(
    "Snapshot_Version"
).distinct().count()

print(f"gold_forecast_snapshot: {count} total rows")
print(f"Snapshot versions stored: {versions}")
print(f"This snapshot version: {snapshot_version}")

# Show latest snapshot summary
spark.table("gold_forecast_snapshot").filter(
    F.col("Snapshot_Version") == snapshot_version
).groupBy("Component").agg(
    F.sum("Amount").alias("Total_Amount")
).orderBy("Component").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
