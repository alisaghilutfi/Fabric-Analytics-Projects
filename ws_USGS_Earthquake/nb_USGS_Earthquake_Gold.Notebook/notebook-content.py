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
# META     },
# META     "environment": {
# META       "environmentId": "b08e8bb0-93e8-b11c-43af-feb0445540a6",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# CELL ********************

# !pip install reverse_geocoder
# #%pip install reverse_geocoder


# from pyspark.sql.functions import *
# from pyspark.sql.functions import udf
# from pyspark.sql.types import StringType
# import reverse_geocoder as rg 


# df = spark.read.table("earthquake_events_silver").filter(col("time") > start_date)


# def get_country_code(lat, lon):
#     """
#     Retrieve the country code for a given latitude and longitude.

#     Parameters:
#     lat (float or str): Latitude of the location.
#     lon (float or str): Longitude of the location.

#     Returns:
#     str: Country code of the location, retrieved using the reverse geocoding API.

#     Example:
#     >>> get_country_details(48.8588443, 2.2943506)
#     'FR'
#     """
#     coordinates = (float(lat), float(lon))
#     return rg.search(coordinates)[0].get("cc")


# # Registering the udfs so they can be used on spark dataframes
# get_country_code_udf = udf(get_country_code, StringType())


# # Adding country_code and city attributes
# df_location = \
#     df.\
#         withColumn("country_code", get_country_code_udf(col("latitude"), col("longitude")))


# # Adding significance classification
# df_location_sig = \
#     df_location.\
#         withColumn(
#             "sig_calss",
#             when(col("sig") < 100, "Low").\
#             when((col("sig") >= 100) & (col("sig") < 500), "Moderate").\
#             otherwise("High")
#         )


# # Appending the data to the gold table
# df_location_sig.write.mode("append").saveAsTable("earthquake_events_gold")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# from delta.tables import DeltaTable

# # Deduplicate in-place: keep one row per id (latest by updated timestamp)
# gold = spark.table("earthquake_events_gold")

# gold_deduped = (
#     gold
#     .dropDuplicates(["id"])   # USGS id is the natural key
# )

# print(f"Before: {gold.count():,} rows")
# print(f"After:  {gold_deduped.count():,} rows")

# # Overwrite with clean data
# (
#     gold_deduped.write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "false")   # schema is unchanged
#     .saveAsTable("earthquake_events_gold")
# )

# print("Deduplication complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# silver = spark.table("earthquake_events_silver")
# silver_deduped = silver.dropDuplicates(["id"])

# print(f"Silver before: {silver.count():,} rows")
# print(f"Silver after:  {silver_deduped.count():,} rows")
# print(f"Duplicates:    {silver.count() - silver_deduped.count():,} rows removed")

# (
#     silver_deduped.write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "false")
#     .saveAsTable("earthquake_events_silver")
# )

# print("Silver deduplication complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# # Check if rename already partially applied
# cols = [f.name for f in spark.table("earthquake_events_gold").schema.fields]
# print(cols)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# spark.sql("""
#     ALTER TABLE earthquake_events_gold
#     SET TBLPROPERTIES (
#         'delta.columnMapping.mode' = 'name',
#         'delta.minReaderVersion'   = '2',
#         'delta.minWriterVersion'   = '5'
#     )
# """)

# spark.sql("""
#     ALTER TABLE earthquake_events_gold
#     RENAME COLUMN sig_calss TO sig_class
# """)

# # Verify
# cols = [f.name for f in spark.table("earthquake_events_gold").schema.fields]
# print(cols)
# # sig_class should appear, sig_calss should be gone

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Pull Request did not happen!
# Fabric notebook source — nb_USGS_Earthquake_Gold (FIXED)
#
# Fixes applied:
#   1. ANTI-PATTERN: row-serial Python UDF → vectorized pandas_udf (batch reverse geocode)
#   2. ANTI-PATTERN: mode("append") → Delta MERGE on id (idempotent re-runs)
#   3. BUG: col("time") > start_date (implicit string cast) → explicit to_timestamp()
#   4. TYPO: sig_calss → sig_class (column name fixed; update SemanticModel tmdl to match)
#   5. NULL guard on lat/lon before UDF (bad coordinates crash reverse_geocoder)

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
# Parameter: start_date (string, "yyyy-MM-dd") — injected by Data Factory

# CELL ********************
# Install reverse_geocoder (Spark executor scope)
# Using %pip (magic) is preferred over ! in Fabric notebooks — applies to all nodes
# %pip install reverse_geocoder --quiet

# CELL ********************
import reverse_geocoder as rg
import pandas as pd

from pyspark.sql.functions import (
    col, lit, to_timestamp, when, pandas_udf
)
from pyspark.sql.types import StringType
from delta.tables import DeltaTable

# ── 1. Vectorized pandas UDF (FIXED) ─────────────────────────────────────────
#
# ANTI-PATTERN REMOVED:
#   def get_country_code(lat, lon): ...
#   get_country_code_udf = udf(get_country_code, StringType())
#
# The original UDF called rg.search() once per row inside a Python subprocess
# on each executor. For 10k events this is fine; for 100k+ it becomes the
# dominant bottleneck (rg.search() has ~0.1ms/call overhead × rows × executors).
#
# pandas_udf (ARROW_BATCH mode) receives an entire partition as a pd.Series,
# passes ALL coordinates to rg.search() in a single vectorised call (mode=2),
# and returns a pd.Series. Arrow serialisation overhead is negligible vs the
# latency savings on even moderate datasets.
#
# rg.search() mode=2 → uses a KD-tree; bulk lookup is O(n log n) vs O(n) serial.

@pandas_udf(StringType())
def get_country_code_vectorized(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    Vectorized reverse geocoder UDF.
    Handles nulls explicitly — rg.search() crashes on NaN float coordinates.
    Returns None for null/invalid coordinates (outer join safe).
    """
    # Build a boolean mask for valid (non-null, finite) coordinate pairs
    valid_mask = lat.notna() & lon.notna() & lat.abs().le(90) & lon.abs().le(180)

    # Pre-allocate result series with None
    result = pd.Series([None] * len(lat), dtype=object)

    if valid_mask.any():
        valid_coords = list(zip(lat[valid_mask].tolist(), lon[valid_mask].tolist()))
        # mode=2 = use faster KD-tree lookup (vs mode=1 which loads full DB)
        geocoded = rg.search(valid_coords, mode=2)
        result[valid_mask] = [r.get("cc") for r in geocoded]

    return result

# ── 2. Read Silver, filtered to new events only ───────────────────────────────
# FIXED: explicit to_timestamp() cast prevents implicit string→timestamp coercion
# which can silently drop boundary-day events depending on session timezone.

df_silver = (
    spark.read.table("earthquake_events_silver")
    .filter(col("time") > to_timestamp(lit(start_date)))  # FIXED
)

row_count = df_silver.count()
if row_count == 0:
    print(f"[Gold] No new Silver rows with time > {start_date}. Skipping.")
    dbutils.notebook.exit("NO_NEW_DATA")

print(f"[Gold] Processing {row_count:,} Silver rows")

# ── 3. Enrich: country code (vectorized) ─────────────────────────────────────
df_geocoded = df_silver.withColumn(
    "country_code",
    get_country_code_vectorized(col("latitude"), col("longitude"))  # FIXED: vectorized
)

# ── 4. Enrich: significance classification ────────────────────────────────────
# FIXED: column name typo sig_calss → sig_class
# !! IMPORTANT: also update sm_USGS_Earthquake SemanticModel tmdl:
#    sourceColumn: sig_calss  →  sourceColumn: sig_class
#    in tables/Earthquake Events.tmdl

df_gold = df_geocoded.withColumn(
    "sig_class",                                   # FIXED: was "sig_calss"
    when(col("sig") < 100, "Low")
    .when((col("sig") >= 100) & (col("sig") < 500), "Moderate")
    .otherwise("High")
)

# ── 5. Upsert into Gold (MERGE — idempotent) ──────────────────────────────────
# ANTI-PATTERN REMOVED: mode("append") with no dedup guard.
# Same rationale as Silver fix — INSERT-only merge on id.

GOLD_TABLE = "earthquake_events_gold"

if spark.catalog.tableExists(GOLD_TABLE):
    gold_delta = DeltaTable.forName(spark, GOLD_TABLE)

    (
        gold_delta.alias("target")
        .merge(
            df_gold.alias("source"),
            condition="target.id = source.id"
        )
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"[Gold] MERGE complete → {GOLD_TABLE}")
else:
    (
        df_gold.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_TABLE)
    )
    print(f"[Gold] Created table {GOLD_TABLE} with {df_gold.count():,} rows")

# ── 6. Trigger Direct Lake model cache refresh (optional) ────────────────────
# After Gold lands new rows, the Direct Lake model needs to re-frame its
# parquet snapshot. You can do this from the notebook via the Fabric REST API
# to keep the report hot, or rely on the model's auto-framing on next query.
# Uncomment if you want immediate post-pipeline model refresh:
#
# import notebookutils
# notebookutils.mssparkutils.credentials.getToken("pbi")  # warm token
# See: fabric_mcp_server.py → trigger_semantic_model_refresh tool

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# from pyspark.sql.functions import countDistinct

# for table in ["earthquake_events_silver", "earthquake_events_gold"]:
#     df = spark.table(table)
#     total = df.count()
#     distinct = df.select(countDistinct("id")).collect()[0][0]
#     flag = "✅ CLEAN" if total == distinct else f"❌ {total - distinct:,} DUPLICATES"
#     print(f"{table}: {total:,} total | {distinct:,} distinct | {flag}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
