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
#
# Sheet structure (discovered at runtime):
# - Product Details: row 0 = sheet title, row 1 = real header → use header=1
# - Manufacturer: 4 rows × 15 cols wide format
#     row 0 = Column1..Column15 (junk)
#     row 1 = ManufacturerID values (1–14)
#     row 2 = Manufacturer names
#     row 3 = Logo URLs
#   → manual transpose into 3-column, 15-row table
# - Geography: 3 metadata header rows → skiprows=3, header=0

import pandas as pd

xlsx_path = "/lakehouse/default/Files/Bronze/Dimensions/bi_dimensions.xlsx"

# Product Details
df_product = pd.read_excel(xlsx_path, sheet_name="Product Details", header=1)
df_product.columns = df_product.columns.str.strip()
spark.createDataFrame(df_product).write.mode("overwrite").saveAsTable("bronze_product")
print(f"✅ bronze_product: {len(df_product)} rows")

# Manufacturer — manual unpivot from wide format
df_mfr_raw = pd.read_excel(xlsx_path, sheet_name="Manufacturer", header=None)
df_mfr = pd.DataFrame({
    "ManufacturerID": pd.to_numeric(df_mfr_raw.iloc[1, :].values, errors="coerce"),
    "Manufacturer":   df_mfr_raw.iloc[2, :].values,
    "Logo":           df_mfr_raw.iloc[3, :].values,
})
df_mfr = df_mfr.dropna(subset=["ManufacturerID"])
df_mfr["ManufacturerID"] = df_mfr["ManufacturerID"].astype(int)
spark.createDataFrame(df_mfr).write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("bronze_manufacturer")
print(f"✅ bronze_manufacturer: {len(df_mfr)} rows")

# Geography
df_geography = pd.read_excel(xlsx_path, sheet_name="Geography", skiprows=3, header=0)
df_geography.columns = df_geography.columns.str.strip()
spark.createDataFrame(df_geography).write.mode("overwrite").saveAsTable("bronze_geography")
print(f"✅ bronze_geography: {len(df_geography)} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import requests
import notebookutils

# Get token
token = notebookutils.credentials.getToken("pbi")

# Trigger refresh via REST API
workspace_id = "bdf49d9a-d2e4-444b-b666-735d2460b5cb"
dataset_id = "2d55bd9f-5512-40de-803b-6d5be8fc4eb0"

url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets/{dataset_id}/refreshes"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json={})
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
