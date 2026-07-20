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

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DecimalType, DoubleType, TimestampType

# Column names for each headerless SalesLT CSV export (Files/data/{table}.csv)
TABLE_SCHEMAS = {
    "Address": ["AddressID", "AddressLine1", "AddressLine2", "City", "StateProvince", "CountryRegion", "PostalCode", "rowguid", "ModifiedDate"],
    "Customer": ["CustomerID", "NameStyle", "Title", "FirstName", "MiddleName", "LastName", "Suffix", "CompanyName", "SalesPerson", "EmailAddress", "Phone", "PasswordHash", "PasswordSalt", "rowguid", "ModifiedDate"],
    "CustomerAddress": ["CustomerID", "AddressID", "AddressType", "rowguid", "ModifiedDate"],
    "Product": ["ProductID", "Name", "ProductNumber", "Color", "StandardCost", "ListPrice", "Size", "Weight", "ProductCategoryID", "ProductModelID", "SellStartDate", "SellEndDate", "DiscontinuedDate", "ThumbNailPhoto", "ThumbnailPhotoFileName", "rowguid", "ModifiedDate"],
    "ProductCategory": ["ProductCategoryID", "ParentProductCategoryID", "Name", "rowguid", "ModifiedDate"],
    "ProductDescription": ["ProductDescriptionID", "Description", "rowguid", "ModifiedDate"],
    "ProductModel": ["ProductModelID", "Name", "CatalogDescription", "rowguid", "ModifiedDate"],
    "ProductModelProductDescription": ["ProductModelID", "ProductDescriptionID", "Culture", "rowguid", "ModifiedDate"],
    "SalesOrderDetail": ["SalesOrderID", "SalesOrderDetailID", "OrderQty", "ProductID", "UnitPrice", "UnitPriceDiscount", "LineTotal", "rowguid", "ModifiedDate"],
    "SalesOrderHeader": ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status", "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber", "CustomerID", "ShipToAddressID", "BillToAddressID", "ShipMethod", "CreditCardApprovalCode", "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "rowguid", "ModifiedDate"],
}

DECIMAL_COLUMNS = {"StandardCost", "ListPrice", "UnitPrice", "UnitPriceDiscount", "LineTotal", "SubTotal", "TaxAmt", "Freight", "TotalDue"}
DOUBLE_COLUMNS = {"Weight"}
TIMESTAMP_COLUMNS = {"OrderDate", "DueDate", "ShipDate", "ModifiedDate", "SellStartDate", "SellEndDate", "DiscontinuedDate"}
BOOLEAN_COLUMNS = {"NameStyle", "OnlineOrderFlag"}

def cast_column(df, column_name):
    if column_name.endswith("ID"):
        return df.withColumn(column_name, F.col(column_name).cast(IntegerType()))
    if column_name in DECIMAL_COLUMNS:
        return df.withColumn(column_name, F.col(column_name).cast(DecimalType(18, 4)))
    if column_name in DOUBLE_COLUMNS:
        return df.withColumn(column_name, F.col(column_name).cast(DoubleType()))
    if column_name in TIMESTAMP_COLUMNS:
        return df.withColumn(column_name, F.col(column_name).cast(TimestampType()))
    if column_name in BOOLEAN_COLUMNS:
        return df.withColumn(column_name, F.col(column_name) == "1")
    return df

def load_bronze_table(table_name):
    columns = TABLE_SCHEMAS[table_name]
    df = spark.read.csv(f"Files/data/{table_name}.csv", header=False, inferSchema=False)
    df = df.toDF(*columns)
    for column_name in columns:
        df = cast_column(df, column_name)
    return df

def write_bronze_table(table_name):
    df = load_bronze_table(table_name)
    target_table = f"bronze_{table_name.lower()}"
    df.write.format("delta").mode("overwrite").saveAsTable(target_table)
    print(f"{target_table}: {df.count()} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("Address")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("Customer")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("CustomerAddress")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("Product")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("ProductCategory")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("ProductDescription")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("ProductModel")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("ProductModelProductDescription")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("SalesOrderDetail")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

write_bronze_table("SalesOrderHeader")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
