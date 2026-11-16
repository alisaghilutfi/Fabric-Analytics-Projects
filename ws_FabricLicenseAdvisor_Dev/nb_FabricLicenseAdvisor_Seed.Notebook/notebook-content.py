# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse_name": "lh_pricing",
# META       "default_lakehouse_workspace_id": "",
# META       "known_lakehouses": []
# META     }
# META   }
# META }

# CELL ********************

# MARKDOWN
# # nb_FabricLicenseAdvisor_Seed
# Loads pricing reference data into `lh_pricing` Delta tables.
# Source: RADACAD Nov 2026 + MS Learn /fabric/enterprise/licenses
# Region: US list prices. Re-run this notebook whenever prices change.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import datetime

spark = SparkSession.builder.getOrCreate()

LAKEHOUSE = "lh_pricing"
LOAD_DATE = datetime.date.today().isoformat()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MARKDOWN ## Table 1: pricing_sku

sku_schema = StructType([
    StructField("sku",                  StringType(),  False),
    StructField("cus",                  IntegerType(), False),
    StructField("vcores",               FloatType(),   True),
    StructField("payg_per_month_usd",   FloatType(),   False),
    StructField("reserved_1yr_usd",     FloatType(),   True),
    StructField("free_viewer_allowed",  BooleanType(), False),
    StructField("copilot_enabled",      BooleanType(), False),
    StructField("pbi_equivalent",       StringType(),  True),
    StructField("load_date",            StringType(),  False),
])

sku_data = [
    ("F2",      2,    0.25,     263.0,    156.0,  False, True,  None,      LOAD_DATE),
    ("F4",      4,    0.5,      526.0,    313.0,  False, True,  None,      LOAD_DATE),
    ("F8",      8,    1.0,     1052.0,    626.0,  False, True,  None,      LOAD_DATE),
    ("F16",    16,    2.0,     2104.0,   1252.0,  False, True,  None,      LOAD_DATE),
    ("F32",    32,    4.0,     4208.0,   2504.0,  False, True,  None,      LOAD_DATE),
    ("F64",    64,    8.0,     8410.0,   5003.0,  True,  True,  "P1/A4",   LOAD_DATE),
    ("F128",  128,   16.0,    16820.0,  10008.0,  True,  True,  "P2/A5",   LOAD_DATE),
    ("F256",  256,   32.0,    33640.0,  20016.0,  True,  True,  "P3/A6",   LOAD_DATE),
    ("F512",  512,   64.0,    67280.0,  40032.0,  True,  True,  "P4",      LOAD_DATE),
    ("F1024", 1024, 128.0,   134560.0,  80063.0,  True,  True,  "P5",      LOAD_DATE),
    ("F2048", 2048, 256.0,   269120.0, 160127.0,  True,  True,  None,      LOAD_DATE),
    ("F4096", 4096, 512.0,   538240.0, 320253.0,  True,  True,  None,      LOAD_DATE),
    ("F8192", 8192, 1024.0, 1076480.0, 640506.0,  True,  True,  None,      LOAD_DATE),
]

df_sku = spark.createDataFrame(sku_data, schema=sku_schema)
df_sku.write.format("delta").mode("overwrite").saveAsTable(f"{LAKEHOUSE}.pricing_sku")
print(f"pricing_sku: {df_sku.count()} rows loaded")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MARKDOWN ## Table 2: pricing_user_licenses

ul_schema = StructType([
    StructField("license_type",               StringType(),  False),
    StructField("price_per_user_usd",         FloatType(),   False),
    StructField("can_create_content",         BooleanType(), False),
    StructField("can_share_pbi_content",      BooleanType(), False),
    StructField("can_view_pbi_on_ppu_ws",     BooleanType(), False),
    StructField("can_view_pbi_on_premium_ws", BooleanType(), False),
    StructField("fabric_items_access",        BooleanType(), False),
    StructField("load_date",                  StringType(),  False),
])

ul_data = [
    ("Free",  0.0,  False, False, False, True,  False, LOAD_DATE),
    ("Pro",  14.0,  True,  True,  True,  True,  False, LOAD_DATE),
    ("PPU",  24.0,  True,  True,  True,  True,  True,  LOAD_DATE),
]

df_ul = spark.createDataFrame(ul_data, schema=ul_schema)
df_ul.write.format("delta").mode("overwrite").saveAsTable(f"{LAKEHOUSE}.pricing_user_licenses")
print(f"pricing_user_licenses: {df_ul.count()} rows loaded")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MARKDOWN ## Table 3: pricing_storage

st_schema = StructType([
    StructField("storage_type",        StringType(), False),
    StructField("price_per_gb_usd",    FloatType(),  True),
    StructField("free_tier_gb_per_cu", FloatType(),  True),
    StructField("notes",               StringType(), True),
    StructField("load_date",           StringType(), False),
])

st_data = [
    ("OneLake",          0.023, None,   "Standard OneLake storage, all regions approx.", LOAD_DATE),
    ("Mirrored_Storage", 0.0,   1024.0, "1 TB free per purchased CU; e.g. F2 = 2 TB free mirrored storage", LOAD_DATE),
]

df_st = spark.createDataFrame(st_data, schema=st_schema)
df_st.write.format("delta").mode("overwrite").saveAsTable(f"{LAKEHOUSE}.pricing_storage")
print(f"pricing_storage: {df_st.count()} rows loaded")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("All pricing tables loaded successfully.")
print(f"  pricing_sku:            {spark.table(f'{LAKEHOUSE}.pricing_sku').count()} rows")
print(f"  pricing_user_licenses:  {spark.table(f'{LAKEHOUSE}.pricing_user_licenses').count()} rows")
print(f"  pricing_storage:        {spark.table(f'{LAKEHOUSE}.pricing_storage').count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
