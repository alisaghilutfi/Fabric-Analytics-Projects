# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "a68f2b2d-bf20-4553-ac41-4210294771dd",
# META       "default_lakehouse_name": "lh_AdventureWorks",
# META       "default_lakehouse_workspace_id": "772cb78b-6a90-493c-b6ff-ae836fa78ec1",
# META       "known_lakehouses": [
# META         {
# META           "id": "a68f2b2d-bf20-4553-ac41-4210294771dd"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

from delta.tables import DeltaTable
from pyspark.sql import functions as F

min_date, max_date = spark.table("gold_sales_fact").select(
    F.min("OrderDate"), F.max("OrderDate")
).first()

calendar_df = spark.range(1).select(
    F.explode(
        F.sequence(F.lit(min_date), F.lit(max_date), F.expr("interval 1 day"))
    ).alias("Date")
)

dim_date_df = (
    calendar_df
    .withColumn("Year", F.year("Date"))
    .withColumn("Quarter", F.quarter("Date"))
    .withColumn("MonthNumber", F.month("Date"))
    .withColumn("MonthName", F.date_format("Date", "MMMM"))
    .withColumn("Day", F.dayofmonth("Date"))
    .withColumn("DayOfWeekNumber", F.dayofweek("Date"))
    .withColumn("DayName", F.date_format("Date", "EEEE"))
    .withColumn("IsWeekend", F.dayofweek("Date").isin(1, 7))
    .withColumn("YearMonth", F.date_format("Date", "yyyyMM"))
)

target_table = "gold_dim_date"

if not spark.catalog.tableExists(target_table):
    (dim_date_df.write
        .format("delta")
        .mode("overwrite")
        .option("delta.columnMapping.mode", "name")
        .option("delta.minReaderVersion", "2")
        .option("delta.minWriterVersion", "5")
        .saveAsTable(target_table))
else:
    delta_target = DeltaTable.forName(spark, target_table)
    (delta_target.alias("target")
        .merge(dim_date_df.alias("source"), "target.Date = source.Date")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute())

print(f"{target_table}: {dim_date_df.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
