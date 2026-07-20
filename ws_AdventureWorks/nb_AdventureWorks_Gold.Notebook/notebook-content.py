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
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

GOLD_PRIMARY_KEYS = {
    "sales_fact": ["SalesOrderDetailID"],
    "dim_customer": ["CustomerID"],
    "dim_product": ["ProductID"],
    "dim_address": ["AddressID"],
    "dim_productcategory": ["ProductCategoryID"],
}

def write_gold_table(df, table_name, pk_columns):
    target_table = f"gold_{table_name.lower()}"

    if not spark.catalog.tableExists(target_table):
        (df.write
            .format("delta")
            .mode("overwrite")
            .option("delta.columnMapping.mode", "name")
            .option("delta.minReaderVersion", "2")
            .option("delta.minWriterVersion", "5")
            .saveAsTable(target_table))
    else:
        delta_target = DeltaTable.forName(spark, target_table)
        merge_condition = " AND ".join([f"target.{c} = source.{c}" for c in pk_columns])
        (delta_target.alias("target")
            .merge(df.alias("source"), merge_condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute())

    print(f"{target_table}: {df.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

header = spark.table("silver_salesorderheader")
detail = spark.table("silver_salesorderdetail")

fact_df = (
    header.join(detail, on="SalesOrderID", how="inner")
    .select(
        header["SalesOrderID"],
        detail["SalesOrderDetailID"],
        header["CustomerID"],
        detail["ProductID"],
        header["OrderDate"].cast(DateType()).alias("OrderDate"),
        header["ShipDate"],
        detail["OrderQty"],
        detail["UnitPrice"],
        detail["UnitPriceDiscount"],
        detail["LineTotal"],
        header["SubTotal"],
        header["TaxAmt"],
        header["Freight"],
        header["TotalDue"],
        F.datediff(header["ShipDate"], header["OrderDate"]).alias("DaysToShip"),
    )
)

write_gold_table(fact_df, "sales_fact", GOLD_PRIMARY_KEYS["sales_fact"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dim_customer_df = spark.table("silver_customer").select(
    "CustomerID", "FirstName", "LastName", "CompanyName", "EmailAddress", "Phone"
)

write_gold_table(dim_customer_df, "dim_customer", GOLD_PRIMARY_KEYS["dim_customer"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

product = spark.table("silver_product")
category = spark.table("silver_productcategory")

dim_product_df = (
    product.join(category, on="ProductCategoryID", how="left")
    .select(
        product["ProductID"],
        product["Name"].alias("ProductName"),
        product["ProductNumber"],
        product["Color"],
        product["ListPrice"],
        product["ProductCategoryID"],
        category["Name"].alias("CategoryName"),
    )
)

write_gold_table(dim_product_df, "dim_product", GOLD_PRIMARY_KEYS["dim_product"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dim_address_df = spark.table("silver_address").select(
    "AddressID", "City", "StateProvince", "CountryRegion", "PostalCode"
)

write_gold_table(dim_address_df, "dim_address", GOLD_PRIMARY_KEYS["dim_address"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dim_productcategory_df = spark.table("silver_productcategory").select(
    "ProductCategoryID",
    "ParentProductCategoryID",
    F.col("Name").alias("CategoryName"),
)

write_gold_table(dim_productcategory_df, "dim_productcategory", GOLD_PRIMARY_KEYS["dim_productcategory"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
