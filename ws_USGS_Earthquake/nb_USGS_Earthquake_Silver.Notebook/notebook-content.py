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

from pyspark.sql.functions import *
from pyspark.sql.types import TimestampType

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Crate a spark dataframe containing JSON data
df = spark.read.option("multiline", "true").json(f"Files/{start_date}_earthquake_data.json")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Reshape earthquake data by extracting and renaming key attributes for further analysis
df = \
df.\
    select(
        'id',
        col("geometry.coordinates").getItem(0).alias("longitude"),
        col("geometry.coordinates").getItem(1).alias("latitude"),
        col("geometry.coordinates").getItem(2).alias("elevation"),
        col('properties.title').alias('title'),
        col('properties.place').alias('place_description'),
        col('properties.sig').alias('sig'),
        col('properties.mag').alias('mag'),
        col('properties.magType').alias('magType'),
        col('properties.time').alias('time'),
        col('properties.updated').alias('updated'),
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Convert 'time' and 'updated' columns from milliseconds to timestamp format for clearer datetime representation
df = df.\
    withColumn("time", col("time")/1000).\
    withColumn("updated", col("updated")/1000).\
    withColumn("time", col("time").cast(TimestampType())).\
    withColumn("updated", col("updated").cast(TimestampType()))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Appending the data to the silver table
df.write.mode("append").saveAsTable("earthquake_events_silver")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
