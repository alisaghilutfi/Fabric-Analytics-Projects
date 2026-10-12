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

from pyspark.sql.functions import col, concat, lit, to_date, coalesce, when
from pyspark.sql.functions import lit as lit_fn
from pyspark.sql.types import DateType

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Gold: dim_product ────────────────────────────────────────────────────────

df_product = spark.read.table("silver_product") \
    .withColumn("Category", coalesce(col("Category"), lit("Uncategorised")))
df_product.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("gold_dim_product")
print(f"✅ gold_dim_product: {df_product.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Gold: dim_manufacturer ───────────────────────────────────────────────────

df_manufacturer = spark.read.table("silver_manufacturer")
df_manufacturer.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("gold_dim_manufacturer")
print(f"✅ gold_dim_manufacturer: {df_manufacturer.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Gold: dim_geography ──────────────────────────────────────────────────────

df_geography = spark.read.table("silver_geography") \
    .withColumn("Country", when(col("Country") == "USA", "United States").otherwise(col("Country"))) \
    .withColumn("ZipCountry", concat(col("Zip"), lit("_"), col("Country")))
df_geography.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("gold_dim_geography")
print(f"✅ gold_dim_geography: {df_geography.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Gold: fact_sales ─────────────────────────────────────────────────────────
# Validate before writing: check for orphaned ProductIDs (AC-10)

df_sales = spark.read.table("silver_sales") \
    .withColumn("Date", to_date(col("Date"), "MM/dd/yyyy")) \
    .withColumn("ZipCountry", concat(col("Zip"), lit_fn("_"), col("Country")))

df_product = spark.read.table("silver_product")

orphaned = df_sales.join(df_product, on="ProductID", how="left_anti")
orphan_count = orphaned.count()
print(f"Orphaned fact rows (unmatched ProductID): {orphan_count}")
assert orphan_count == 0, f"AC-10 FAILED: {orphan_count} orphaned rows"

df_sales.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("gold_fact_sales")
print(f"✅ gold_fact_sales: {df_sales.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Gold: OPTIMIZE fact_sales (V-Order for DirectLake) ───────────────────────

spark.sql("OPTIMIZE gold_fact_sales")
print("✅ OPTIMIZE complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
