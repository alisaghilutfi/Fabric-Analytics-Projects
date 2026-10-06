# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

from pyspark.sql.functions import (
    col, year, month, quarter, dayofmonth, date_format,
    dayofweek, when, concat, lit, expr
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

start_date = "2016-01-01"
end_date   = "2022-12-31"

dates_df = spark.sql(f"""
    SELECT explode(sequence(to_date('{start_date}'), to_date('{end_date}'),
    interval 1 day)) AS Date
""")

from pyspark.sql.functions import trunc, last_day, months_between, floor

today_year = 2026  # static reference for Year Offset

dim_date = dates_df \
    .withColumn("Year",          year("Date")) \
    .withColumn("Quarter",       concat(lit("Q"), quarter("Date"))) \
    .withColumn("Month Number",  month("Date")) \
    .withColumn("Month Name",    date_format("Date", "MMMM")) \
    .withColumn("Month Year",    date_format("Date", "MMM yyyy")) \
    .withColumn("Month Year Code", (year("Date") * 100 + month("Date")).cast("int")) \
    .withColumn("Day",           dayofmonth("Date")) \
    .withColumn("Day of Week",   date_format("Date", "EEEE")) \
    .withColumn("Is Weekend",    when(dayofweek("Date").isin(1, 7), True).otherwise(False)) \
    .withColumn("Year Offset",   (year("Date") - today_year).cast("int"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dim_date.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("abfss://bdf49d9a-d2e4-444b-b666-735d2460b5cb@onelake.dfs.fabric.microsoft.com/46efa811-c9ab-4952-99e3-9c5ee33f09c5/Tables/dbo/DateDimension")

print(f"✅ DateDimension: {dim_date.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
