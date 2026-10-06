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

from pyspark.sql.functions import (
    col, year, month, quarter, dayofmonth, date_format,
    dayofweek, when, concat, lit
)

# ── Configuration ────────────────────────────────────────────────────────────
start_date  = "2016-01-01"
end_date    = "2022-12-31"
today_year  = 2026  # static reference for YearOffset

# ── Generate date spine ───────────────────────────────────────────────────────
dates_df = spark.sql(f"""
    SELECT explode(sequence(
        to_date('{start_date}'),
        to_date('{end_date}'),
        interval 1 day
    )) AS Date
""")

# ── Add calendar columns ──────────────────────────────────────────────────────
dim_date = dates_df \
    .withColumn("Year",          year("Date")) \
    .withColumn("Quarter",       concat(lit("Q"), quarter("Date"))) \
    .withColumn("MonthNumber",   month("Date")) \
    .withColumn("MonthName",     date_format("Date", "MMMM")) \
    .withColumn("MonthYear",     date_format("Date", "MMM yyyy")) \
    .withColumn("MonthYearCode", (year("Date") * 100 + month("Date")).cast("int")) \
    .withColumn("Day",           dayofmonth("Date")) \
    .withColumn("DayOfWeek",     date_format("Date", "EEEE")) \
    .withColumn("IsWeekend",     when(dayofweek("Date").isin(1, 7), True).otherwise(False)) \
    .withColumn("YearOffset",    (year("Date") - today_year).cast("int"))

# ── Write to Lakehouse DateDimension table ────────────────────────────────────
dim_date.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("DateDimension")

print(f"✅ DateDimension: {dim_date.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
