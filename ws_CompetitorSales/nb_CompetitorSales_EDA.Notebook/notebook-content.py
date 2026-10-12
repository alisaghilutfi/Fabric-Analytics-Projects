# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "46efa811-c9ab-4952-99e3-9c5ee33f09c5",
# META       "default_lakehouse_name": "lh_CompetitorSales",
# META       "default_lakehouse_workspace_id": "bdf49d9a-d2e4-444b-b666-735d2460b5cb",
# META       "known_lakehouses": [
# META         {
# META           "id": "46efa811-c9ab-4952-99e3-9c5ee33f09c5"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "46efa811-c9ab-4952-99e3-9c5ee33f09c5",
# META       "default_lakehouse_name": "lh_CompetitorSales",
# META       "default_lakehouse_workspace_id": "bdf49d9a-d2e4-444b-b666-735d2460b5cb",
# META       "known_lakehouses": [
# META         {
# META           "id": "46efa811-c9ab-4952-99e3-9c5ee33f09c5"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# ws_CompetitorSales — Gold Layer EDA
# Run after every pipeline execution to validate Gold data quality.
# Known issue: 1,530 US Zip codes unmatched in Geography (7.38% of US rows).

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, count, when, sum as spark_sum

# ── Cell 2: Row counts and NULL check ────────────────────────────────────────
gold_tables = [
    "gold_dim_geography",
    "gold_dim_product",
    "gold_dim_manufacturer",
    "gold_fact_sales"
]

for table_name in gold_tables:
    df = spark.read.table(table_name)
    total = df.count()
    print("=" * 50)
    print(f"Table: {table_name}  |  Total rows: {total:,}")
    print("NULL counts per column:")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Cell 3: Primary key uniqueness ───────────────────────────────────────────
pk_checks = [
    ("gold_dim_geography",    "ZipCountry"),
    ("gold_dim_product",      "ProductID"),
    ("gold_dim_manufacturer", "ManufacturerID"),
]

print("=== Primary Key Uniqueness ===\n")
for table_name, pk_col in pk_checks:
    df = spark.read.table(table_name)
    total = df.count()
    distinct = df.select(pk_col).distinct().count()
    if total == distinct:
        print(f"✅ {table_name}.{pk_col} — No duplicates ({total:,} rows)")
    else:
        dupes = total - distinct
        print(f"❌ {table_name}.{pk_col} — {dupes:,} duplicate rows found")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Cell 4: Referential integrity ────────────────────────────────────────────
KNOWN_GEO_ORPHANS = 55594

print("=== Referential Integrity ===\n")

df_sales   = spark.read.table("gold_fact_sales")
df_product = spark.read.table("gold_dim_product")
df_geo     = spark.read.table("gold_dim_geography")

# Sales → Product
orphans_product = df_sales.join(df_product, on="ProductID", how="left_anti").count()
if orphans_product == 0:
    print(f"✅ Sales→Product: 0 orphans")
else:
    print(f"❌ FAIL Sales→Product: {orphans_product:,} orphans (should be 0)")

# Sales → Geography
orphans_geo = df_sales.join(df_geo, on="ZipCountry", how="left_anti").count()
if orphans_geo <= KNOWN_GEO_ORPHANS:
    print(f"✅ Sales→Geography: {orphans_geo:,} orphans (within known baseline of {KNOWN_GEO_ORPHANS:,})")
else:
    regression = orphans_geo - KNOWN_GEO_ORPHANS
    print(f"⚠️  Sales→Geography: {orphans_geo:,} orphans — regression of {regression:,} above baseline")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Cell 5: Value distributions ──────────────────────────────────────────────
from pyspark.sql.functions import min as spark_min, max as spark_max, avg, round as spark_round

print("=== Value Distributions ===\n")

df_sales = spark.read.table("gold_fact_sales")

# Date range
date_stats = df_sales.agg(
    spark_min("Date").alias("earliest"),
    spark_max("Date").alias("latest")
).collect()[0]
print(f"Date range: {date_stats['earliest']}  →  {date_stats['latest']}")

# Revenue and Units
rev_stats = df_sales.agg(
    spark_min("Revenue").alias("min_rev"),
    spark_max("Revenue").alias("max_rev"),
    spark_round(avg("Revenue"), 2).alias("avg_rev"),
    spark_min("Units").alias("min_units"),
    spark_max("Units").alias("max_units")
).collect()[0]
print(f"Revenue — min: ${rev_stats['min_rev']:.2f}  max: ${rev_stats['max_rev']:,.2f}  avg: ${rev_stats['avg_rev']:,.2f}")
print(f"Units   — min: {rev_stats['min_units']}  max: {rev_stats['max_units']}")

# Country distribution
print("\nCountry distribution:")
df_sales.groupBy("Country").count().orderBy("count", ascending=False).show(truncate=False)

# Year distribution
from pyspark.sql.functions import year
print("Year distribution:")
df_sales.withColumn("Year", year("Date")).groupBy("Year").count().orderBy("Year").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Cell 6: Revenue sanity check ─────────────────────────────────────────────
REVENUE_MIN = 600_000_000
REVENUE_MAX = 800_000_000

print("=== Revenue Sanity Check ===\n")

df_sales   = spark.read.table("gold_fact_sales")
df_product = spark.read.table("gold_dim_product")
df_mfr     = spark.read.table("gold_dim_manufacturer")

total_rev = df_sales.agg(spark_sum("Revenue")).collect()[0][0]
print(f"Total revenue: ${total_rev:,.0f}")
if REVENUE_MIN <= total_rev <= REVENUE_MAX:
    print(f"✅ Within expected range (${REVENUE_MIN/1e6:.0f}M – ${REVENUE_MAX/1e6:.0f}M)")
else:
    print(f"❌ OUTSIDE expected range (${REVENUE_MIN/1e6:.0f}M – ${REVENUE_MAX/1e6:.0f}M)")

# Revenue by country
print("\nRevenue by country:")
df_sales.groupBy("Country") \
    .agg(spark_round(spark_sum("Revenue"), 2).alias("TotalRevenue")) \
    .orderBy("TotalRevenue", ascending=False) \
    .show(truncate=False)

# Revenue by top 5 manufacturers (snowflake join: Sales → Product → Manufacturer)
top5 = df_sales \
    .join(df_product, on="ProductID") \
    .join(df_mfr, on="ManufacturerID") \
    .groupBy("Manufacturer") \
    .agg(spark_round(spark_sum("Revenue"), 2).alias("TotalRevenue")) \
    .orderBy("TotalRevenue", ascending=False) \
    .limit(5)

print("Revenue by top 5 manufacturers:")
top5.show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
