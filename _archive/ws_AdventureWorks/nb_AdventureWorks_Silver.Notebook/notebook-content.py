# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "05e9ba33-cc56-405e-b963-edce852987d8",
# META       "default_lakehouse_name": "lh_AdventureWorks",
# META       "default_lakehouse_workspace_id": "772cb78b-6a90-493c-b6ff-ae836fa78ec1",
# META       "known_lakehouses": [
# META         {
# META           "id": "a68f2b2d-bf20-4553-ac41-4210294771dd"
# META         },
# META         {
# META           "id": "05e9ba33-cc56-405e-b963-edce852987d8"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

from delta.tables import DeltaTable

# Primary key columns per table (composite for junction tables)
PRIMARY_KEYS = {
    "Address": ["AddressID"],
    "Customer": ["CustomerID"],
    "CustomerAddress": ["CustomerID", "AddressID"],
    "Product": ["ProductID"],
    "ProductCategory": ["ProductCategoryID"],
    "ProductDescription": ["ProductDescriptionID"],
    "ProductModel": ["ProductModelID"],
    "ProductModelProductDescription": ["ProductModelID", "ProductDescriptionID", "Culture"],
    "SalesOrderDetail": ["SalesOrderDetailID"],
    "SalesOrderHeader": ["SalesOrderID"],
}

def upsert_silver_table(table_name):
    pk_columns = PRIMARY_KEYS[table_name]
    bronze_table = f"bronze_{table_name.lower()}"
    silver_table = f"silver_{table_name.lower()}"

    df = spark.table(bronze_table)
    before_count = df.count()

    df = df.dropDuplicates()
    dedup_count = df.count()

    df = df.na.drop(subset=pk_columns)
    after_count = df.count()

    print(f"{silver_table}: before={before_count}, after_dedup={dedup_count}, after_pk_filter={after_count}")

    if not spark.catalog.tableExists(silver_table):
        (df.write
            .format("delta")
            .mode("overwrite")
            .option("delta.columnMapping.mode", "name")
            .option("delta.minReaderVersion", "2")
            .option("delta.minWriterVersion", "5")
            .saveAsTable(silver_table))
    else:
        delta_target = DeltaTable.forName(spark, silver_table)
        merge_condition = " AND ".join([f"target.{c} = source.{c}" for c in pk_columns])
        (delta_target.alias("target")
            .merge(df.alias("source"), merge_condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("Address")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("Customer")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("CustomerAddress")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("Product")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("ProductCategory")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("ProductDescription")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("ProductModel")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("ProductModelProductDescription")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("SalesOrderDetail")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

upsert_silver_table("SalesOrderHeader")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
