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

from pyspark.sql.functions import col, lit, regexp_replace, trim
from pyspark.sql.types import DoubleType

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Silver: Sales UNION ALL ──────────────────────────────────────────────────
# USA file has no Country column — add it explicitly before union.
# All international files already carry a Country column.

df_usa = spark.read.table("bronze_sales_usa").withColumn("Country", lit("United States"))

intl_tables = [
    "bronze_sales_canada",
    "bronze_sales_germany",
    "bronze_sales_japan",
    "bronze_sales_mexico",
    "bronze_sales_nigeria",
]

df_intl = [spark.read.table(t) for t in intl_tables]

from functools import reduce
from pyspark.sql import DataFrame

df_all = [df_usa] + df_intl
df_sales = reduce(DataFrame.unionByName, df_all)

df_sales.write.mode("overwrite").saveAsTable("silver_sales")
print(f"✅ silver_sales: {df_sales.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Silver: Product ──────────────────────────────────────────────────────────
# Rename PerformanceID → ProductID (same key, different column name in source)
# Parse Price: strip currency symbols and whitespace, cast to double

df_product = spark.read.table("bronze_product") \
    .withColumnRenamed("PerformanceID", "ProductID") \
    .withColumn("Price", regexp_replace(col("Price"), r"[^0-9.]", "")) \
    .withColumn("Price", trim(col("Price")).cast(DoubleType()))

df_product.write.mode("overwrite").saveAsTable("silver_product")
print(f"✅ silver_product: {df_product.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Silver: Manufacturer (unpivot) ───────────────────────────────────────────
# Source is wide format: one column per manufacturer attribute.
# Target: ManufacturerID, Manufacturer, Logo — one row per manufacturer.
# Use pandas for the unpivot since the source is small (15 manufacturers).

import pandas as pd

df_mfr_pd = spark.read.table("bronze_manufacturer").toPandas()

# Inspect columns at runtime — wide format means column names encode the data.
# Standard unpivot: melt all non-ID columns.
# Adjust id_vars if the source has a different structure after Bronze inspection.
df_mfr_melted = df_mfr_pd.melt(var_name="Manufacturer", value_name="Logo")
df_mfr_melted.insert(0, "ManufacturerID", range(1, len(df_mfr_melted) + 1))

df_manufacturer = spark.createDataFrame(df_mfr_melted)
df_manufacturer.write.mode("overwrite").saveAsTable("silver_manufacturer")
print(f"✅ silver_manufacturer: {df_manufacturer.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Silver: Geography (pass-through) ─────────────────────────────────────────
# Already clean after Bronze header-row skip. Pass through as-is.

df_geography = spark.read.table("bronze_geography")
df_geography.write.mode("overwrite").saveAsTable("silver_geography")
print(f"✅ silver_geography: {df_geography.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
