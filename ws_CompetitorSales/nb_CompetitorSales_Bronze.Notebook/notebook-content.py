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

# ── Bronze: Sales CSVs ──────────────────────────────────────────────────────
# Read all 6 regional sales CSVs and write as individual Bronze Delta tables.
# Schema is inferred — no casting at Bronze; that happens in Silver.

from pyspark.sql.functions import lit

sales_files = {
    "bronze_sales_usa":     "Files/Bronze/Sales/Sales.csv",
    "bronze_sales_canada":  "Files/Bronze/Sales/Canada.csv",
    "bronze_sales_germany": "Files/Bronze/Sales/Germany.csv",
    "bronze_sales_japan":   "Files/Bronze/Sales/Japan.csv",
    "bronze_sales_mexico":  "Files/Bronze/Sales/Mexico.csv",
    "bronze_sales_nigeria": "Files/Bronze/Sales/Nigeria.csv",
}

for table_name, file_path in sales_files.items():
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    df.write.mode("overwrite").saveAsTable(table_name)
    print(f"✅ {table_name}: {df.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── Bronze: Dimensions (bi_dimensions.xlsx) ─────────────────────────────────
# Product Details sheet — standard header, no skip needed
# Manufacturer sheet — wide format (15 cols); written as-is, unpivot in Silver
# Geography sheet — 3 metadata header rows skipped via skiprows

import pandas as pd
from pyspark.sql import SparkSession

xlsx_path = "/lakehouse/default/Files/Bronze/Dimensions/bi_dimensions.xlsx"

# Product Details
df_product = pd.read_excel(xlsx_path, sheet_name="Product Details")
spark.createDataFrame(df_product).write.mode("overwrite").saveAsTable("bronze_product")
print(f"✅ bronze_product: {len(df_product)} rows")

# Manufacturer (wide — write as-is)
df_manufacturer = pd.read_excel(xlsx_path, sheet_name="Manufacturer")
spark.createDataFrame(df_manufacturer).write.mode("overwrite").saveAsTable("bronze_manufacturer")
print(f"✅ bronze_manufacturer: {len(df_manufacturer)} rows")

# Geography (skip 3 metadata header rows)
df_geography = pd.read_excel(xlsx_path, sheet_name="Geography", skiprows=3)
spark.createDataFrame(df_geography).write.mode("overwrite").saveAsTable("bronze_geography")
print(f"✅ bronze_geography: {len(df_geography)} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
