# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "d726b863-4ac2-4206-85ac-409c87da9f22",
# META       "default_lakehouse_name": "lh_Finance_Silver",
# META       "default_lakehouse_workspace_id": "61549e76-c4d4-4b27-9018-ab9b04eab5dc",
# META       "known_lakehouses": [
# META         {
# META           "id": "d726b863-4ac2-4206-85ac-409c87da9f22"
# META         },
# META         {
# META           "id": "0e4ac3b2-b113-465b-8587-844ffa45271e"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # nb_Finance_Silver
# Reads from lh_Finance_Bronze, applies all data quality remediations,
# and writes the star schema tables to lh_Finance_Silver.
# Star schema:
#   fact_transactions  ←→  dim_customer   (customer_id)
#   fact_transactions  ←→  dim_date       (transaction_date)
#   fact_transactions  ←→  dim_channel    (channel_key)
#   fact_transactions  ←→  dim_merchant   (merchant_key)
# Data quality remediations applied:
#   1. Exact duplicate rows dropped (dropDuplicates on transaction_id + all columns)
#   2. channel: trim whitespace + normalize "M@bile App" → "Mobile App"
#   3. currency: upper() → all INR
#   4. fee_amount: fillna(0.0)
#   5. negative amount: flag is_reversal = True
#   6. Dates: parse dd-MM-yyyy strings → DateType

# CELL ********************

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, BooleanType
from pyspark.sql.functions import (
    col, to_date, trim, upper, regexp_replace, when, lit,
    monotonically_increasing_id, coalesce, year, quarter,
    month, dayofmonth, date_format, dayofweek, concat,
    current_timestamp, expr, sequence, explode
)

spark = SparkSession.builder.getOrCreate()

# Bronze lakehouse path — update workspace/lakehouse IDs after first Fabric sync
BRONZE_LAKEHOUSE = "lh_Finance_Bronze"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1. Load Bronze tables

# CELL ********************

raw_tx   = spark.table(f"{BRONZE_LAKEHOUSE}.dbo.raw_transactions")
raw_cust = spark.table(f"{BRONZE_LAKEHOUSE}.dbo.raw_customers")

print(f"raw_transactions: {raw_tx.count():,} rows")
print(f"raw_customers:    {raw_cust.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2. Clean transactions

# CELL ********************

tx = raw_tx

# ── Step 1: Drop exact full-row duplicates ───────────────────────────────────
# 68 exact duplicate rows identified in profiling
tx = tx.dropDuplicates()
print(f"After dropDuplicates: {tx.count():,} rows")

# ── Step 2: Normalize channel ────────────────────────────────────────────────
# Values found: "M@bile App" (typo), leading/trailing spaces in several values
tx = tx.withColumn(
    "channel",
    when(trim(col("channel")) == "M@bile App", lit("Mobile App"))
    .otherwise(trim(col("channel")))
)

# ── Step 3: Normalize currency ───────────────────────────────────────────────
# Values: "INR", "inr", "inR" → all uppercase
tx = tx.withColumn("currency", upper(trim(col("currency"))))

# ── Step 4: Fill fee_amount nulls ────────────────────────────────────────────
tx = tx.withColumn("fee_amount", coalesce(col("fee_amount").cast(DoubleType()), lit(0.0)))

# ── Step 5: Flag negative amounts as reversals ───────────────────────────────
tx = tx.withColumn("is_reversal", when(col("amount") < 0, lit(True)).otherwise(lit(False)))

# ── Step 6: Cast is_fraud to boolean ─────────────────────────────────────────
tx = tx.withColumn("is_fraud_bool", when(upper(col("is_fraud")) == "YES", lit(True)).otherwise(lit(False)))

# ── Step 7: Parse dates (dd-MM-yyyy → DateType) ──────────────────────────────
tx = tx.withColumn("transaction_date", to_date(col("transaction_date"), "dd-MM-yyyy"))

# ── Step 8: Cast numeric columns ─────────────────────────────────────────────
tx = (tx
    .withColumn("amount",      col("amount").cast(DoubleType()))
    .withColumn("tax_amount",  col("tax_amount").cast(DoubleType()))
    .withColumn("risk_score",  col("risk_score").cast(IntegerType()))
)

# ── Step 9: Add total_amount (amount + fee + tax) for convenience ─────────────
tx = tx.withColumn("total_amount", col("amount") + col("fee_amount") + col("tax_amount"))

print("Transaction cleaning complete.")
tx.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3. Clean customers

# CELL ********************

cust = raw_cust

# Fix column name typo: "fisrt_name" → "first_name"
cust = cust.withColumnRenamed("fisrt_name", "first_name")

# Parse dates
cust = (cust
    .withColumn("date_of_birth", to_date(col("date_of_birth"), "dd-MM-yyyy"))
    .withColumn("join_date",     to_date(col("join_date"),     "dd-MM-yyyy"))
)

# Trim string columns
for c in ["gender", "city", "state", "occupation", "customer_segment"]:
    cust = cust.withColumn(c, trim(col(c)))

# Cast annual_income
cust = cust.withColumn("annual_income", col("annual_income").cast(DoubleType()))

print("Customer cleaning complete.")
cust.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4. Build dim_date (2023-01-01 to 2026-12-31)

# CELL ********************

date_range = spark.sql("""
    SELECT explode(sequence(to_date('2023-01-01'), to_date('2026-12-31'), interval 1 day)) AS Date
""")

dim_date = (date_range
    .withColumn("Year",         year("Date"))
    .withColumn("Quarter",      concat(lit("Q"), quarter("Date")))
    .withColumn("MonthNumber",  month("Date"))
    .withColumn("MonthName",    date_format("Date", "MMMM"))
    .withColumn("MonthYear",    date_format("Date", "MMM yyyy"))
    .withColumn("MonthYearSort",date_format("Date", "yyyy-MM"))
    .withColumn("Day",          dayofmonth("Date"))
    .withColumn("DayName",      date_format("Date", "EEEE"))
    .withColumn("DayOfWeek",    dayofweek("Date"))
    .withColumn("IsWeekend",    when(dayofweek("Date").isin(1, 7), lit(True)).otherwise(lit(False)))
    .withColumn("YearHalf",     when(month("Date") <= 6, lit("H1")).otherwise(lit("H2")))
)

print(f"dim_date: {dim_date.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 5. Build dim_channel

# CELL ********************

dim_channel = (tx
    .select("channel")
    .distinct()
    .orderBy("channel")
    .withColumn("channel_key", monotonically_increasing_id().cast(IntegerType()))
    .select("channel_key", "channel")
)

print("dim_channel:")
dim_channel.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 6. Build dim_merchant

# CELL ********************

dim_merchant = (tx
    .select("merchant_category")
    .distinct()
    .orderBy("merchant_category")
    .withColumn("merchant_key", monotonically_increasing_id().cast(IntegerType()))
    .select("merchant_key", "merchant_category")
)

print("dim_merchant:")
dim_merchant.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 7. Build dim_customer (from cleaned customers table)

# CELL ********************

dim_customer = cust.select(
    "customer_id",
    "first_name",
    "second_name",
    "gender",
    "date_of_birth",
    "city",
    "state",
    "occupation",
    "customer_segment",
    "annual_income",
    "join_date"
)

print(f"dim_customer: {dim_customer.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 8. Build fact_transactions (join surrogates)

# CELL ********************

# Join channel_key and merchant_key onto transactions
fact = (tx
    .join(dim_channel,  on="channel",           how="left")
    .join(dim_merchant, on="merchant_category",  how="left")
    .select(
        "transaction_id",
        "transaction_date",
        "account_id",
        "customer_id",
        "channel_key",
        "merchant_key",
        "transaction_type",
        "transaction_status",
        "amount",
        "fee_amount",
        "tax_amount",
        "total_amount",
        "currency",
        "is_fraud_bool",
        "is_reversal",
        "risk_score",
        "reference_no",
    )
)

print(f"fact_transactions: {fact.count():,} rows")
fact.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 9. Write Silver tables

# CELL ********************

WRITE_OPTIONS = {
    "delta.columnMapping.mode": "name"
}

silver_tables = {
    "dim_date":          dim_date,
    "dim_customer":      dim_customer,
    "dim_channel":       dim_channel,
    "dim_merchant":      dim_merchant,
    "fact_transactions": fact,
}

for table_name, df in silver_tables.items():
    print(f"Writing {table_name}...")
    writer = (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
    )
    for k, v in WRITE_OPTIONS.items():
        writer = writer.option(k, v)
    writer.saveAsTable(table_name)
    print(f"  ✓ {table_name}: {spark.table(table_name).count():,} rows")

print("\nSilver build complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
