# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "0e4ac3b2-b113-465b-8587-844ffa45271e",
# META       "default_lakehouse_name": "lh_Finance_Bronze",
# META       "default_lakehouse_workspace_id": "61549e76-c4d4-4b27-9018-ab9b04eab5dc",
# META       "known_lakehouses": [
# META         {
# META           "id": "0e4ac3b2-b113-465b-8587-844ffa45271e"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # nb_Finance_Bronze
# Reads raw CSVs from GitHub and writes them as Delta tables to lh_Finance_Bronze.
# No transformations — Bronze is raw as-landed.
# Source repo: https://github.com/alisaghilutfi/Fabric-Analytics-Projects
# CSV path:    ws_Finance_Analysis/data/

# CELL ********************

# GitHub raw base URL — repo must be public when this notebook runs
GITHUB_RAW = "https://raw.githubusercontent.com/alisaghilutfi/Fabric-Analytics-Projects/main/ws_Finance_Analysis/data"

SOURCES = {
    "raw_customers":     f"{GITHUB_RAW}/customers.csv",
    "raw_transactions":  f"{GITHUB_RAW}/finance_transactions.csv",
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp

spark = SparkSession.builder.getOrCreate()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

for table_name, url in SOURCES.items():
    print(f"Loading {table_name} from {url}...")
    
    # Read via pandas (handles HTTPS URLs reliably)
    pdf = pd.read_csv(url)
    df = spark.createDataFrame(pdf)
    df = df.withColumn("_bronze_loaded_at", current_timestamp())
    
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    count = spark.table(table_name).count()
    print(f"  ✓ {table_name}: {count:,} rows written")

print("\nBronze load complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
