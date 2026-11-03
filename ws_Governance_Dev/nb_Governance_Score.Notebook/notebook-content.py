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

# Cell 1 - Imports and rule registry
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, IntegerType
)
from datetime import datetime, timezone

scanned_at = datetime.now(timezone.utc).isoformat()

# Rule registry -- id, severity, description
RULES = {
    "UNUSED_REPORT":        {"severity": "Medium", "description": "Report not viewed in 90 days"},
    "SINGLE_ADMIN":         {"severity": "High",   "description": "Workspace has only one distinct active user (admin proxy)"},
    "NO_GIT_CONNECTION":    {"severity": "Medium", "description": "Workspace not connected to Git (data unavailable at current scan depth)"},
    "NO_SENSITIVITY_LABEL": {"severity": "High",   "description": "Item has no sensitivity label (data unavailable at current scan depth)"},
    "REFRESH_FAILING":      {"severity": "High",   "description": "Semantic model failed refresh 3+ times in 7 days (no refresh history available for DirectLake tenant)"},
}

SEVERITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

print("Rule registry loaded ✅")
print(f"Rules: {list(RULES.keys())}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 2 - Load Gold tables
gold_ws       = spark.table("gold_workspaces")
gold_items    = spark.table("gold_items")
gold_activity = spark.table("gold_activity_summary")

print(f"gold_workspaces:       {gold_ws.count()} rows")
print(f"gold_items:            {gold_items.count()} rows")
print(f"gold_activity_summary: {gold_activity.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 3 - Evaluate rules
from pyspark.sql import Row

rule_results = []

# Rule: UNUSED_REPORT
# Reports with no activity in last 90 days
# Note: fires only after 90 days of accumulated activity data
unused_reports = (
    gold_activity
    .filter(F.col("is_unused_90d") == True)
    .filter(F.col("object_type").isin(["Report", "PowerBIReport"]))
)

for row in unused_reports.collect():
    rule_results.append({
        "workspace_id": row["workspace_id"],
        "item_id":      row["object_id"],
        "item_name":    row["object_display_name"],
        "item_type":    "Report",
        "rule_id":      "UNUSED_REPORT",
        "passed":       False,
        "detail":       f"Last viewed: {row['last_viewed_date']} -- {row['view_count_90d']} views in 90d",
        "severity":     RULES["UNUSED_REPORT"]["severity"],
        "scanned_at":   scanned_at,
    })

print(f"UNUSED_REPORT violations: {unused_reports.count()}")

# Rule: SINGLE_ADMIN
# Workspaces with admin_count <= 1 (proxy via distinct active users)
single_admin_ws = (
    gold_ws
    .filter(F.col("type") == "Workspace")
    .filter(F.col("admin_count") <= 1)
)

for row in single_admin_ws.collect():
    rule_results.append({
        "workspace_id": row["workspace_id"],
        "item_id":      row["workspace_id"],
        "item_name":    row["name"],
        "item_type":    "Workspace",
        "rule_id":      "SINGLE_ADMIN",
        "passed":       False,
        "detail":       f"admin_count proxy = {row['admin_count']} (based on distinct active users in activity events)",
        "severity":     RULES["SINGLE_ADMIN"]["severity"],
        "scanned_at":   scanned_at,
    })

print(f"SINGLE_ADMIN violations: {single_admin_ws.count()}")

# Rule: NO_GIT_CONNECTION
# git_connected is NULL for all workspaces at current scanner depth
# Flag all non-personal workspaces as data-unavailable
no_git_ws = (
    gold_ws
    .filter(F.col("type") == "Workspace")
    .filter(F.col("git_connected").isNull())
)

for row in no_git_ws.collect():
    rule_results.append({
        "workspace_id": row["workspace_id"],
        "item_id":      row["workspace_id"],
        "item_name":    row["name"],
        "item_type":    "Workspace",
        "rule_id":      "NO_GIT_CONNECTION",
        "passed":       False,
        "detail":       "Git connection status unavailable at current scanner depth -- manual verification required",
        "severity":     RULES["NO_GIT_CONNECTION"]["severity"],
        "scanned_at":   scanned_at,
    })

print(f"NO_GIT_CONNECTION violations: {no_git_ws.count()}")

# Rule: NO_SENSITIVITY_LABEL
# sensitivity_label not returned by scanner at current depth
# Flag all SemanticModel and Report items as data-unavailable
no_label_items = (
    gold_items
    .filter(F.col("item_type").isin(["SemanticModel", "Report"]))
)

for row in no_label_items.collect():
    rule_results.append({
        "workspace_id": row["workspace_id"],
        "item_id":      row["item_id"],
        "item_name":    row["name"],
        "item_type":    row["item_type"],
        "rule_id":      "NO_SENSITIVITY_LABEL",
        "passed":       False,
        "detail":       "Sensitivity label data unavailable at current scanner depth -- enable DAX/mashup expressions in tenant settings",
        "severity":     RULES["NO_SENSITIVITY_LABEL"]["severity"],
        "scanned_at":   scanned_at,
    })

print(f"NO_SENSITIVITY_LABEL violations: {no_label_items.count()}")

# Rule: REFRESH_FAILING
# No refresh history for DirectLake tenant -- skip
print("REFRESH_FAILING: skipped -- DirectLake tenant, no refresh history available")

print(f"\nTotal rule result rows: {len(rule_results)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 4 - Write gold_health_rule_results
if rule_results:
    schema = StructType([
        StructField("workspace_id", StringType()),
        StructField("item_id",      StringType()),
        StructField("item_name",    StringType()),
        StructField("item_type",    StringType()),
        StructField("rule_id",      StringType()),
        StructField("passed",       BooleanType()),
        StructField("detail",       StringType()),
        StructField("severity",     StringType()),
        StructField("scanned_at",   StringType()),
    ])

    df_results = spark.createDataFrame(
        [Row(**r) for r in rule_results],
        schema=schema
    )

    df_results.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("gold_health_rule_results")

    print(f"gold_health_rule_results: {df_results.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 5 - Compute gold_health_scores
# Score = 100 - weighted penalty per violation
# Weights: High=5pts, Medium=3pts, Low=1pt; floor at 0

df_results = spark.table("gold_health_rule_results")
gold_ws     = spark.table("gold_workspaces")

violations = (
    df_results
    .filter(F.col("passed") == False)
    .groupBy("workspace_id")
    .agg(
        F.sum(F.when(F.col("severity") == "High",   1).otherwise(0)).alias("critical_count"),
        F.sum(F.when(F.col("severity") == "Medium", 1).otherwise(0)).alias("medium_count"),
        F.sum(F.when(F.col("severity") == "Low",    1).otherwise(0)).alias("low_count"),
        F.count("*").alias("total_violations")
    )
)

rules_checked = (
    df_results
    .groupBy("workspace_id")
    .agg(F.count("*").alias("total_rules_checked"))
)

gold_scores = (
    gold_ws
    .join(violations,    on="workspace_id", how="left")
    .join(rules_checked, on="workspace_id", how="left")
    .withColumn("critical_count",      F.coalesce(F.col("critical_count"),      F.lit(0)))
    .withColumn("medium_count",        F.coalesce(F.col("medium_count"),        F.lit(0)))
    .withColumn("low_count",           F.coalesce(F.col("low_count"),           F.lit(0)))
    .withColumn("total_rules_checked", F.coalesce(F.col("total_rules_checked"), F.lit(0)))
    .withColumn("raw_penalty",
        F.col("critical_count") * 5 +
        F.col("medium_count")   * 3 +
        F.col("low_count")      * 1
    )
    .withColumn("score",
        F.greatest(F.lit(0), F.lit(100) - F.col("raw_penalty")).cast(IntegerType())
    )
    .select(
        "workspace_id",
        "name",
        "score",
        "critical_count",
        "medium_count",
        "low_count",
        "total_rules_checked",
        F.lit(scanned_at).alias("scanned_at")
    )
)

gold_scores.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_health_scores")

print(f"gold_health_scores: {gold_scores.count()} rows ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 6 - Validation
print("=== Health scoring results ===")

print("\n-- Score distribution --")
spark.table("gold_health_scores") \
    .select("name", "score", "critical_count", "medium_count", "low_count") \
    .orderBy("score") \
    .show(30, truncate=False)

print("\n-- Rule violations by rule --")
spark.table("gold_health_rule_results") \
    .groupBy("rule_id", "severity") \
    .agg(F.count("*").alias("violation_count")) \
    .orderBy("severity", "rule_id") \
    .show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
