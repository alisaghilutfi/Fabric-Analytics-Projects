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

# CELL ********************

# from pyspark.sql.functions import *
# from pyspark.sql.types import TimestampType


# # Crate a spark dataframe containing JSON data
# df = spark.read.option("multiline", "true").json(f"Files/{start_date}_earthquake_data.json")


# # Reshape earthquake data by extracting and renaming key attributes for further analysis
# df = \
# df.\
#     select(
#         'id',
#         col("geometry.coordinates").getItem(0).alias("longitude"),
#         col("geometry.coordinates").getItem(1).alias("latitude"),
#         col("geometry.coordinates").getItem(2).alias("elevation"),
#         col('properties.title').alias('title'),
#         col('properties.place').alias('place_description'),
#         col('properties.sig').alias('sig'),
#         col('properties.mag').alias('mag'),
#         col('properties.magType').alias('magType'),
#         col('properties.time').alias('time'),
#         col('properties.updated').alias('updated'),
#     )


#     # Convert 'time' and 'updated' columns from milliseconds to timestamp format for clearer datetime representation
# df = df.\
#     withColumn("time", col("time")/1000).\
#     withColumn("updated", col("updated")/1000).\
#     withColumn("time", col("time").cast(TimestampType())).\
#     withColumn("updated", col("updated").cast(TimestampType()))


#     # Appending the data to the silver table
# df.write.mode("append").saveAsTable("earthquake_events_silver")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Fabric notebook source — nb_USGS_Earthquake_Silver (FIXED)
# Fix: replaced mode("append") with Delta MERGE on id to prevent duplicate rows
#      on pipeline re-runs within the same date window.
# Fix: explicit StructType schema — no more runtime schema inference drift.

# METADATA ********************
# META {
# META   "kernel_info": { "name": "synapse_pyspark" },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "fda3c40e-3b18-4230-b5a8-18334d845083",
# META       "default_lakehouse_name": "lh_USGS_Earthquake",
# META       "default_lakehouse_workspace_id": "39caa3ab-a964-45c2-bd5b-7d46ad66c985"
# META     }
# META   }
# META }

# CELL ********************
# Parameters injected by Data Factory pipeline
# start_date: str  e.g. "2024-12-01"  (used only to locate the Bronze file)

from pyspark.sql.functions import col, from_unixtime
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, LongType, ArrayType
)
from delta.tables import DeltaTable

# ── 1. Explicit schema — never infer from runtime JSON ───────────────────────
# Mirrors the GeoJSON feature structure written by the Bronze notebook.
# If USGS adds a field, this schema is intentionally narrow — extra columns
# are silently dropped, which is safer than letting schema evolution
# corrupt the Silver table.

FEATURE_SCHEMA = StructType([
    StructField("id", StringType(), nullable=False),
    StructField("geometry", StructType([
        StructField("coordinates", ArrayType(DoubleType()), nullable=True)
    ]), nullable=True),
    StructField("properties", StructType([
        StructField("title",   StringType(), nullable=True),
        StructField("place",   StringType(), nullable=True),
        StructField("sig",     LongType(),   nullable=True),
        StructField("mag",     DoubleType(), nullable=True),
        StructField("magType", StringType(), nullable=True),
        StructField("time",    LongType(),   nullable=True),   # epoch ms
        StructField("updated", LongType(),   nullable=True),   # epoch ms
    ]), nullable=True),
])

# ── 2. Read Bronze file ───────────────────────────────────────────────────────
bronze_path = f"Files/{start_date}_earthquake_data.json"

df_raw = (
    spark.read
    .option("multiline", "true")
    .schema(FEATURE_SCHEMA)       # FIXED: explicit schema, no inference
    .json(bronze_path)
)

# Guard: USGS can return 200 with zero features (quiet seismic period)
row_count = df_raw.count()
if row_count == 0:
    raise ValueError(
        f"[Silver] Bronze file '{bronze_path}' contains zero records. "
        "Verify the USGS API returned data for this date window."
    )
print(f"[Silver] Read {row_count:,} raw features from {bronze_path}")

# ── 3. Reshape + cast ─────────────────────────────────────────────────────────
df_silver = (
    df_raw.select(
        "id",
        col("geometry.coordinates").getItem(0).alias("longitude"),
        col("geometry.coordinates").getItem(1).alias("latitude"),
        col("geometry.coordinates").getItem(2).alias("elevation"),
        col("properties.title").alias("title"),
        col("properties.place").alias("place_description"),
        col("properties.sig").alias("sig"),
        col("properties.mag").alias("mag"),
        col("properties.magType").alias("magType"),
        # FIXED: use from_unixtime (seconds) instead of cast chain
        # epoch ms → seconds → TimestampType via from_unixtime
        from_unixtime(col("properties.time")    / 1000).cast("timestamp").alias("time"),
        from_unixtime(col("properties.updated") / 1000).cast("timestamp").alias("updated"),
    )
    # Drop rows with no primary key — defensive, shouldn't happen with USGS data
    .filter(col("id").isNotNull())
)

# ── 4. Upsert into Silver (MERGE — idempotent) ────────────────────────────────
# ANTI-PATTERN REMOVED: mode("append") with no dedup guard.
# A pipeline re-run on the same date window would silently duplicate every row.
#
# Strategy: INSERT-only merge on id.
# We never UPDATE existing Silver rows — Bronze data for a given event id
# is immutable once written. If USGS revises an event, it gets a new id.
# If you need to honour USGS revisions, swap the WHEN MATCHED clause to
# UPDATE SET * (or update only the `updated` + `mag` columns).

SILVER_TABLE = "earthquake_events_silver"

if spark.catalog.tableExists(SILVER_TABLE):
    silver_delta = DeltaTable.forName(spark, SILVER_TABLE)

    (
        silver_delta.alias("target")
        .merge(
            df_silver.alias("source"),
            condition="target.id = source.id"
        )
        .whenNotMatchedInsertAll()   # only insert truly new events
        .execute()
    )
    print(f"[Silver] MERGE complete → {SILVER_TABLE}")
else:
    # First run — table doesn't exist yet, write normally
    (
        df_silver.write
        .format("delta")
        .mode("overwrite")          # safe on first write
        .option("overwriteSchema", "true")
        .saveAsTable(SILVER_TABLE)
    )
    print(f"[Silver] Created table {SILVER_TABLE} with {df_silver.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
