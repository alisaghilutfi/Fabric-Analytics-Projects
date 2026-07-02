# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

from pyspark.sql.functions import *
from pyspark.sql.functions import expr, year, quarter, month, dayofmonth, date_format, dayofweek, when

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

start_date = "2024-01-01"
end_date = "2027-12-31"

dates_df = spark.sql(f"""
  SELECT explode(sequence(to_date('{start_date}'), to_date('{end_date}'), interval 1 day)) AS Date
""")

date_dim = dates_df.withColumn("Year", year("Date")) \
    .withColumn("Quarter", concat(lit("Q"), quarter("Date"))) \
    .withColumn("Month Number", month("Date")) \
    .withColumn("Month Name", date_format("Date", "MMMM")) \
    .withColumn("Month Year", date_format("Date", "yyyy-MM")) \
    .withColumn("Day", dayofmonth("Date")) \
    .withColumn("Day of Week", date_format("Date", "EEEE")) \
    .withColumn("Weekday", dayofweek("Date")) \
    .withColumn("Is Weekend", when(dayofweek("Date").isin(1, 7), True).otherwise(False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

date_dim = date_dim \
    .withColumnRenamed("Month Number", "MonthNumber") \
    .withColumnRenamed("Month Name", "MonthName") \
    .withColumnRenamed("Month Year", "MonthYear") \
    .withColumnRenamed("Day of Week", "DayOfWeek") \
    .withColumnRenamed("Is Weekend", "IsWeekend")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

date_dim.show(5)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

date_dim.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("abfss://ws_USGS_Earthquake@onelake.dfs.fabric.microsoft.com/lh_USGS_Earthquake.Lakehouse/Tables/dbo/DateDimension")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
