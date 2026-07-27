# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "b168228f-1e92-437d-bdfe-e60038e35cfc",
# META       "default_lakehouse_name": "lh_Ecommerce_Olist",
# META       "default_lakehouse_workspace_id": "225d8970-d000-435e-82f1-39db7b2eb063",
# META       "known_lakehouses": [
# META         {
# META           "id": "b168228f-1e92-437d-bdfe-e60038e35cfc"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# [Olist: Brazilian Ecommerce](https://github.com/Judithokon/olist-ecommerce-sales-data-analysis-using-python/blob/main/Olist%20Data%20Analysis.ipynb)

# CELL ********************

# Loop through all files and save them as Delta tables
file_names = ["olist_customers_dataset", "olist_geolocation_dataset", "olist_order_items_dataset", 
              "olist_order_payments_dataset", "olist_order_reviews_dataset", "olist_orders_dataset", 
              "olist_products_dataset", "olist_sellers_dataset", "product_category_name_translation"]

for file in file_names:
    df = spark.read.format("csv").option("header", "true").load(f"Files/Bronze/{file}.csv")
    df.write.format("delta").mode("overwrite").saveAsTable(file)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Fast EDA process for all delta tables
from pyspark.sql import functions as F 

def fast_eda(table_name):
    df = spark.read.table(table_name)
    print(f"---EDA for {table_name} ---")
    print(f"Record count: {df.count()}")
    print(f"Column count: {len(df.columns)}")

    # Check for nulls in each column
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    null_counts.show()

    # Show first 5 rows for visual shape check
    df.show(5)

# Run EDA for the customers table
for table_name in file_names:
    fast_eda(table_name)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
