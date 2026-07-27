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

# CELL ********************

# Checking Spark session!
spark

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Importing libraries

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.types import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Configurations

# CELL ********************

# Configuration based on paths of Delta tables
base_path = "abfss://ws_Ecommerce_Olist@onelake.dfs.fabric.microsoft.com/lh_Ecommerce_Olist.Lakehouse/Tables/dbo/"
tables = [
    "olist_customers_dataset", "olist_geolocation_dataset", "olist_order_items_dataset", 
    "olist_order_payments_dataset", "olist_order_reviews_dataset", "olist_orders_dataset", 
    "olist_products_dataset", "olist_sellers_dataset", "product_category_name_translation"
]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## EDA

# CELL ********************

# Automated EDA for Bronze Tables
def run_detailed_eda(table_name):
    print(f"{'='*20} ANALYZING TABLE: {table_name} {'='*20}")
    
    # Load the Delta table
    df = spark.read.format("delta").load(f"{base_path}{table_name}")
    
    # 1. Basic Stats (Count & Schema)
    row_count = df.count()
    col_count = len(df.columns)
    print(f"Shape: {row_count} rows | {col_count} columns")
    df.printSchema()
    
    # 2. Check for Null Values per column
    print("Null Value Counts:")
    null_counts = df.select([F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c) for c in df.columns])
    null_counts.show()
    
    # 3. Check for Duplicate Rows
    duplicate_count = row_count - df.dropDuplicates().count()
    print(f"Duplicate Rows Found: {duplicate_count}")
    
    # 4. Statistical Summary (Numerical columns only)
    print("Numerical Summary:")
    df.describe().show()
    
    # 5. Visual Sample
    print("First 5 records:")
    display(df.limit(5))
    print("\n" * 2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Execute loop for all tables
for table in tables:
    run_detailed_eda(table)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Basic Structural Inspection
# display(df)
# df.columns
# df.show(5, truncate=True)
# df.printSchema()

## Statistical Summaries
# df.describe().show()
# df.summary().show()

## Data Volume and Quality Checks
# df.count()
# df.distinct().count()
# df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

## Categorical Data Profiling
# Find the most frequent categories for some tables such as products and customers
# df.groupBy("customer_state").count().sort("count", ascending=False).show()

## Time-Series Inspection
# For the orders dataset, we need to ensure the date strings are interpreted correctly
# df.selectExpr("min(order_purchase_timestamp)", "max(order_purchase_timestamp)").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Cleaning

# CELL ********************

# Define specific cleaning rules for each table
cleaning_map = {
     "olist_geolocation_dataset": {
        "date_cols": [],
        "int_cols": [],
        "float_cols": ["geolocation_lat", "geolocation_lng"]
    },
    "olist_order_items_dataset": {
        "date_cols": ["shipping_limit_date"],
        "int_cols": [],
        "float_cols": ["price", "freight_value"]
    },
    "olist_order_payments_dataset": {
        "date_cols": [],
        "int_cols": ["payment_sequential", "payment_installments"],
        "float_cols": ["payment_value"]
    },
    "olist_order_reviews_dataset": {
        "date_cols": ["review_creation_date", "review_answer_timestamp"],
        "int_cols": [],
        "float_cols": []
    },
    "olist_orders_dataset": {
        "date_cols": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"],
        "int_cols": [],
        "float_cols": []
    },
    "olist_products_dataset": {
        "date_cols": [],
        "int_cols": ["product_name_lenght", "product_description_lenght", "product_photos_qty"],
        "float_cols": ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    }  
}

def transform_to_silver(table_name, rules):
    print(f"Transforming {table_name} to Silver...")
    
    # Load Bronze Table
    df = spark.read.table(f"dbo.{table_name}")
    
    # A. Remove Exact Duplicates
    df = df.dropDuplicates()
    
    # B. Apply Type Conversions based on the cleaning map
    if rules:
        for col_name in rules.get("date_cols", []):
            df = df.withColumn(col_name, F.to_timestamp(F.col(col_name)))
            
        for col_name in rules.get("int_cols", []):
            df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
            
        for col_name in rules.get("float_cols", []):
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    # C. Handle NULLs (Optional: Here we drop rows where primary keys are null)
    primary_keys = {"olist_orders_dataset": "order_id", "olist_customers_dataset": "customer_id"}
    pk = primary_keys.get(table_name)
    if pk:
        df = df.filter(F.col(pk).isNotNull())

    # D. Save to Silver Layer (prefixed with 'silver_')
    silver_table_name = f"silver_{table_name}"
    df.write.format("delta").mode("overwrite").saveAsTable(silver_table_name)
    print(f"Successfully saved {silver_table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Execute for tables defined in the map
for table, rules in cleaning_map.items():
    transform_to_silver(table, rules)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# For tables NOT in the map (those that might just need duplicate removal), run with empty rules
remaining_tables = ["olist_customers_dataset", "olist_sellers_dataset", "product_category_name_translation"]
for table in remaining_tables:
    transform_to_silver(table, {})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
