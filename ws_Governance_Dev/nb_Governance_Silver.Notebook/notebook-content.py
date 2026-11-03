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
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType,
    TimestampType, ArrayType
)
print("Imports ready ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 2 - Silver: workspaces
# Parse bronze_scan_workspaces raw_json -> silver_workspaces
# One row per workspace per scan, deduplicated on workspace_id + scanned_at

ws_schema = StructType([
    StructField("id", StringType()),
    StructField("name", StringType()),
    StructField("type", StringType()),
    StructField("state", StringType()),
    StructField("isOnDedicatedCapacity", BooleanType()),
    StructField("capacityId", StringType()),
    StructField("description", StringType()),
])

bronze_ws = spark.table("bronze_scan_workspaces")

silver_ws = (
    bronze_ws
    .withColumn("ws", F.from_json(F.col("raw_json"), ws_schema))
    .select(
        F.col("ws.id").alias("workspace_id"),
        F.col("ws.name").alias("name"),
        F.col("ws.type").alias("type"),
        F.col("ws.state").alias("state"),
        F.col("ws.isOnDedicatedCapacity").alias("is_on_dedicated_capacity"),
        F.col("ws.capacityId").alias("capacity_id"),
        F.col("ws.description").alias("description"),
        F.col("scanned_at")
    )
    .dropDuplicates(["workspace_id", "scanned_at"])
    .filter(F.col("workspace_id").isNotNull())
)

silver_ws.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver_workspaces")

print(f"silver_workspaces: {silver_ws.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 3 - Silver: items
# Extract datasets and reports arrays from each workspace scan result -> silver_items
# One row per item per scan

item_schema = StructType([
    StructField("id", StringType()),
    StructField("name", StringType()),
    StructField("configuredBy", StringType()),
    StructField("configuredById", StringType()),
    StructField("createdDate", StringType()),
    StructField("targetStorageMode", StringType()),
    StructField("contentProviderType", StringType()),
    StructField("isEffectiveIdentityRequired", BooleanType()),
    StructField("isEffectiveIdentityRolesRequired", BooleanType()),
])

ws_with_items_schema = StructType([
    StructField("id", StringType()),
    StructField("name", StringType()),
    StructField("datasets", ArrayType(item_schema)),
    StructField("reports", ArrayType(StructType([
        StructField("id", StringType()),
        StructField("name", StringType()),
        StructField("datasetId", StringType()),
        StructField("createdDateTime", StringType()),
        StructField("modifiedDateTime", StringType()),
        StructField("modifiedBy", StringType()),
        StructField("modifiedById", StringType()),
    ]))),
])

bronze_ws = spark.table("bronze_scan_workspaces")

parsed = bronze_ws.withColumn("ws", F.from_json(F.col("raw_json"), ws_with_items_schema))

# Datasets -> items
datasets = (
    parsed
    .select(
        F.col("ws.id").alias("workspace_id"),
        F.col("scanned_at"),
        F.explode_outer(F.col("ws.datasets")).alias("item")
    )
    .select(
        F.col("workspace_id"),
        F.col("item.id").alias("item_id"),
        F.lit("SemanticModel").alias("item_type"),
        F.col("item.name").alias("name"),
        F.col("item.configuredBy").alias("configured_by"),
        F.col("item.configuredById").alias("configured_by_id"),
        F.col("item.createdDate").alias("created_date"),
        F.col("item.targetStorageMode").alias("target_storage_mode"),
        F.col("item.contentProviderType").alias("content_provider_type"),
        F.col("scanned_at")
    )
)

# Reports -> items
reports = (
    parsed
    .select(
        F.col("ws.id").alias("workspace_id"),
        F.col("scanned_at"),
        F.explode_outer(F.col("ws.reports")).alias("item")
    )
    .select(
        F.col("workspace_id"),
        F.col("item.id").alias("item_id"),
        F.lit("Report").alias("item_type"),
        F.col("item.name").alias("name"),
        F.col("item.modifiedBy").alias("configured_by"),
        F.col("item.modifiedById").alias("configured_by_id"),
        F.col("item.createdDateTime").alias("created_date"),
        F.lit(None).cast(StringType()).alias("target_storage_mode"),
        F.lit(None).cast(StringType()).alias("content_provider_type"),
        F.col("scanned_at")
    )
)

silver_items = (
    datasets.union(reports)
    .dropDuplicates(["item_id", "scanned_at"])
    .filter(F.col("item_id").isNotNull())
)

silver_items.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver_items")

print(f"silver_items: {silver_items.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 4 - Silver: activity events
# Parse bronze_activity_events raw_json -> silver_activity_events

event_schema = StructType([
    StructField("Id", StringType()),
    StructField("RecordType", StringType()),
    StructField("CreationTime", StringType()),
    StructField("Operation", StringType()),
    StructField("Activity", StringType()),
    StructField("UserId", StringType()),
    StructField("UserType", StringType()),
    StructField("WorkspaceId", StringType()),
    StructField("WorkSpaceName", StringType()),
    StructField("ObjectId", StringType()),
    StructField("ObjectType", StringType()),
    StructField("ObjectDisplayName", StringType()),
    StructField("Workload", StringType()),
    StructField("ResultStatus", StringType()),
])

bronze_events = spark.table("bronze_activity_events")

silver_events = (
    bronze_events
    .withColumn("evt", F.from_json(F.col("raw_json"), event_schema))
    .select(
        F.col("event_date"),
        F.col("evt.Id").alias("event_id"),
        F.col("evt.CreationTime").alias("creation_time"),
        F.col("evt.Operation").alias("operation"),
        F.col("evt.Activity").alias("activity"),
        F.col("evt.UserId").alias("user_upn"),
        F.col("evt.UserType").alias("user_type"),
        F.col("evt.WorkspaceId").alias("workspace_id"),
        F.col("evt.WorkSpaceName").alias("workspace_name"),
        F.col("evt.ObjectId").alias("object_id"),
        F.col("evt.ObjectType").alias("object_type"),
        F.col("evt.ObjectDisplayName").alias("object_display_name"),
        F.col("evt.Workload").alias("workload"),
        F.col("evt.ResultStatus").alias("result_status"),
    )
    .dropDuplicates(["event_id", "event_date"])
    .filter(F.col("event_id").isNotNull())
)

silver_events.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("silver_activity_events")

print(f"silver_activity_events: {silver_events.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 5 - Validation
print("=== Silver layer row counts ===")
for t in ["silver_workspaces", "silver_items", "silver_activity_events"]:
    try:
        n = spark.table(t).count()
        print(f"  {t}: {n} rows")
    except Exception as e:
        print(f"  {t}: ERROR -- {e}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F

df = spark.table("silver_activity_events")

null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

null_counts.show(vertical=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
