# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "4478b3d6-2e64-43ee-b120-3caa722abf48",
# META       "default_lakehouse_name": "lh_Governance",
# META       "default_lakehouse_workspace_id": "7ea6403c-9364-4cd5-94c3-3ba6fcfce918",
# META       "known_lakehouses": [
# META         {
# META           "id": "4478b3d6-2e64-43ee-b120-3caa722abf48"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Cell 1 - Imports
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType, IntegerType
from datetime import datetime, timezone
print("Imports ready ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 2 - gold_workspaces
# Source: silver_workspaces
# Enrich with admin_count proxy (distinct users who acted in that workspace)
# and git_connected flag (not in scanner output yet -- defaults to NULL).

silver_ws = spark.table("silver_workspaces")
silver_events = spark.table("silver_activity_events")

# Admin count proxy: distinct user_upn per workspace in activity events
admin_counts = (
    silver_events
    .filter(F.col("workspace_id").isNotNull())
    .groupBy("workspace_id")
    .agg(F.countDistinct("user_upn").alias("admin_count"))
)

scanned_at = datetime.now(timezone.utc).isoformat()

gold_ws = (
    silver_ws
    .join(admin_counts, on="workspace_id", how="left")
    .select(
        F.col("workspace_id"),
        F.col("name"),
        F.col("type"),
        F.col("state"),
        F.col("is_on_dedicated_capacity"),
        F.col("capacity_id"),
        F.col("description"),
        F.coalesce(F.col("admin_count"), F.lit(0)).cast(IntegerType()).alias("admin_count"),
        F.lit(None).cast(BooleanType()).alias("git_connected"),
        F.lit(scanned_at).alias("scanned_at")
    )
)

gold_ws = gold_ws.dropDuplicates(["workspace_id"])

gold_ws.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_workspaces")

print(f"gold_workspaces: {gold_ws.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 3 - gold_items
# Source: silver_items LEFT JOIN silver_activity_events on object_id -> item_id
# Enrich with last_activity_date per item and workspace_name

silver_items = spark.table("silver_items")
silver_events = spark.table("silver_activity_events")
silver_ws = spark.table("silver_workspaces")

last_activity = (
    silver_events
    .filter(F.col("object_id").isNotNull())
    .groupBy("object_id")
    .agg(F.max("event_date").alias("last_activity_date"))
)

ws_names = silver_ws.select(
    F.col("workspace_id"),
    F.col("name").alias("workspace_name")
)

gold_items = (
    silver_items
    .join(last_activity, silver_items["item_id"] == last_activity["object_id"], how="left")
    .join(ws_names, on="workspace_id", how="left")
    .select(
        F.col("workspace_id"),
        F.col("workspace_name"),
        F.col("item_id"),
        F.col("item_type"),
        silver_items["name"].alias("name"),
        F.col("configured_by"),
        F.col("created_date"),
        F.col("target_storage_mode"),
        F.col("content_provider_type"),
        F.col("last_activity_date"),
        F.lit(scanned_at).alias("scanned_at")
    )
)

gold_items.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_items")

print(f"gold_items: {gold_items.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 4 - gold_activity_summary
# Source: silver_activity_events
# Aggregate by object_id -- last viewed date, view counts, unused flag

from datetime import date, timedelta

today_str = date.today().isoformat()
cutoff_30d = (date.today() - timedelta(days=30)).isoformat()
cutoff_90d = (date.today() - timedelta(days=90)).isoformat()

silver_events = spark.table("silver_activity_events")
silver_ws = spark.table("silver_workspaces")

ws_names = silver_ws.select(
    F.col("workspace_id"),
    F.col("name").alias("workspace_name")
)

gold_activity = (
    silver_events
    .filter(F.col("object_id").isNotNull())
    .groupBy(
        "workspace_id",
        "object_id",
        "object_type",
        "object_display_name"
    )
    .agg(
        F.max("event_date").alias("last_viewed_date"),
        F.sum(
            F.when(F.col("event_date") >= cutoff_30d, 1).otherwise(0)
        ).alias("view_count_30d"),
        F.sum(
            F.when(F.col("event_date") >= cutoff_90d, 1).otherwise(0)
        ).alias("view_count_90d")
    )
    .withColumn(
        "is_unused_90d",
        F.col("last_viewed_date") < cutoff_90d
    )
    .join(ws_names, on="workspace_id", how="left")
    .select(
        "workspace_id",
        "workspace_name",
        "object_id",
        "object_type",
        "object_display_name",
        "last_viewed_date",
        "view_count_30d",
        "view_count_90d",
        "is_unused_90d"
    )
)

gold_activity.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_activity_summary")

print(f"gold_activity_summary: {gold_activity.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 5 - Validation
print("=== Gold layer row counts ===")
for t in ["gold_workspaces", "gold_items", "gold_activity_summary"]:
    try:
        n = spark.table(t).count()
        print(f"  {t}: {n} rows")
    except Exception as e:
        print(f"  {t}: ERROR -- {e}")

# Quick sanity: unused reports
try:
    unused = spark.table("gold_activity_summary") \
        .filter(F.col("is_unused_90d") == True) \
        .filter(F.col("object_type").isin(["Report", "PowerBIReport"])) \
        .count()
    print(f"  Unused reports (90d): {unused}")
except Exception as e:
    print(f"  Unused reports check: ERROR -- {e}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
