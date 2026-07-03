# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "fda3c40e-3b18-4230-b5a8-18334d845083",
# META       "default_lakehouse_name": "lh_USGS_Earthquake",
# META       "default_lakehouse_workspace_id": "39caa3ab-a964-45c2-bd5b-7d46ad66c985",
# META       "known_lakehouses": [
# META         {
# META           "id": "fda3c40e-3b18-4230-b5a8-18334d845083"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)

# CELL ********************

url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2014-01-01&endtime=2014-01-02"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import requests
import json

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

response = requests.get(url)
response

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

response_json = response.json()
response_json

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

response_json_features = response.json()["features"]
response_json_features

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("""
    SELECT 
        MIN(Date) as min_date,
        MAX(Date) as max_date,
        COUNT(*) as row_count
    FROM lh_USGS_Earthquake.dbo.DateDimension
""").show()

spark.sql("""
    SELECT 
        MIN(time) as min_event,
        MAX(time) as max_event
    FROM lh_USGS_Earthquake.dbo.earthquake_events_gold
""").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Full diagnostic — Direct Lake framing health check
import json

# 1. Confirm Delta log is readable and column mapping is intact
print("=== GOLD TABLE DETAIL ===")
detail = spark.sql("DESCRIBE DETAIL lh_USGS_Earthquake.dbo.earthquake_events_gold").collect()[0]
props = detail["properties"]
print(f"columnMapping.mode : {props.get('delta.columnMapping.mode', 'NOT SET')}")
print(f"minReaderVersion   : {detail['minReaderVersion']}")
print(f"minWriterVersion   : {detail['minWriterVersion']}")
print(f"numFiles           : {detail['numFiles']}")
print(f"location           : {detail['location']}")

print("\n=== DATE DIMENSION DETAIL ===")
detail_date = spark.sql("DESCRIBE DETAIL lh_USGS_Earthquake.dbo.DateDimension").collect()[0]
print(f"numFiles           : {detail_date['numFiles']}")
print(f"location           : {detail_date['location']}")

# 2. Confirm sig_class is in the physical schema (not sig_calss)
print("\n=== GOLD SCHEMA ===")
spark.sql("DESCRIBE lh_USGS_Earthquake.dbo.earthquake_events_gold").show(20, truncate=False)

# 3. Check Delta log transaction history for recent operations
print("\n=== GOLD TRANSACTION HISTORY (last 5) ===")
spark.sql("""
    DESCRIBE HISTORY lh_USGS_Earthquake.dbo.earthquake_events_gold
""").select("version", "timestamp", "operation", "operationParameters").show(5, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("""
    SELECT *
    FROM information_schema.columns
    WHERE table_name = 'earthquake_events_gold'
    AND column_name = 'sig_class'
""").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from delta.tables import DeltaTable

# Read current clean data
df = spark.table("earthquake_events_gold")
print(f"Row count before: {df.count():,}")
print("Schema:")
df.printSchema()

# Write to a temp table with column mapping enabled from scratch
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.columnMapping.mode", "name")
    .option("delta.minReaderVersion", "2")
    .option("delta.minWriterVersion", "5")
    .saveAsTable("earthquake_events_gold_clean")
)

# Verify
df_clean = spark.table("earthquake_events_gold_clean")
print(f"Row count after: {df_clean.count():,}")
df_clean.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Drop the broken table and rename the clean one
spark.sql("DROP TABLE earthquake_events_gold")
spark.sql("ALTER TABLE earthquake_events_gold_clean RENAME TO earthquake_events_gold")

# Verify Delta log is clean
spark.sql("""
    DESCRIBE HISTORY earthquake_events_gold
""").select("version", "timestamp", "operation").show(5, truncate=False)

# Confirm column mapping
detail = spark.sql("DESCRIBE DETAIL earthquake_events_gold").collect()[0]
print(f"columnMapping.mode: {detail['properties'].get('delta.columnMapping.mode', 'NOT SET')}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

detail = spark.sql("DESCRIBE DETAIL earthquake_events_gold").collect()[0]
print(detail['location'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from delta.tables import DeltaTable

GOLD_PATH = "abfss://39caa3ab-a964-45c2-bd5b-7d46ad66c985@onelake.dfs.fabric.microsoft.com/fda3c40e-3b18-4230-b5a8-18334d845083/Tables/dbo/earthquake_events_gold"

# Read current data via path (bypasses any catalog name confusion)
df = spark.read.format("delta").load(GOLD_PATH)
print(f"Row count: {df.count():,}")
print(f"sig_class distinct: {df.select('sig_class').distinct().collect()}")

# Overwrite directly to the ABFSS path — column mapping baked in at write time
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.columnMapping.mode", "name")
    .option("delta.minReaderVersion", "2")
    .option("delta.minWriterVersion", "5")
    .save(GOLD_PATH)
)

# Verify via path
df_verify = spark.read.format("delta").load(GOLD_PATH)
print(f"Row count after: {df_verify.count():,}")

detail = DeltaTable.forPath(spark, GOLD_PATH).detail().collect()[0]
print(f"columnMapping.mode: {detail['properties'].get('delta.columnMapping.mode', 'NOT SET')}")
print(f"numFiles: {detail['numFiles']}")
print(f"location: {detail['location']}")

# Confirm sig_class values
spark.read.format("delta").load(GOLD_PATH).groupBy("sig_class").count().orderBy("sig_class").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
