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

!pip install reverse_geocoder
#%pip install reverse_geocoder

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import reverse_geocoder as rg 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.read.table("earthquake_events_silver").filter(col("time") > start_date)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def get_country_code(lat, lon):
    """
    Retrieve the country code for a given latitude and longitude.

    Parameters:
    lat (float or str): Latitude of the location.
    lon (float or str): Longitude of the location.

    Returns:
    str: Country code of the location, retrieved using the reverse geocoding API.

    Example:
    >>> get_country_details(48.8588443, 2.2943506)
    'FR'
    """
    coordinates = (float(lat), float(lon))
    return rg.search(coordinates)[0].get("cc")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Registering the udfs so they can be used on spark dataframes
get_country_code_udf = udf(get_country_code, StringType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Adding country_code and city attributes
df_location = \
    df.\
        withColumn("country_code", get_country_code_udf(col("latitude"), col("longitude")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Adding significance classification
df_location_sig = \
    df_location.\
        withColumn(
            "sig_calss",
            when(col("sig") < 100, "Low").\
            when((col("sig") >= 100) & (col("sig") < 500), "Moderate").\
            otherwise("High")
        )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Appending the data to the gold table
df_location_sig.write.mode("append").saveAsTable("earthquake_events_gold")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
