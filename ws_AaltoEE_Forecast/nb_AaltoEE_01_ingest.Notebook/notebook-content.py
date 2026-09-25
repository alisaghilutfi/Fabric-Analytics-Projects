# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "f40e0030-a5ab-40aa-ba4b-90edd23ba8d0",
# META       "default_lakehouse_name": "lh_AaltoEE",
# META       "default_lakehouse_workspace_id": "167ca0e1-7e56-40f5-8a99-ccf78ec0ac81",
# META       "known_lakehouses": [
# META         {
# META           "id": "f40e0030-a5ab-40aa-ba4b-90edd23ba8d0"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# nb_AaltoEE_01_ingest
# Purpose: Read raw Excel files from Bronze folder, apply Silver 
# transformations, write Delta tables to Lakehouse Tables section
# ----------------------------------------------------------------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    DoubleType, IntegerType, DateType
)
from datetime import datetime

# Configuration
BRONZE_PATH = "Files/Bronze"
LAKEHOUSE   = "lh_AaltoEE"

print(f"Ingest session started: {datetime.now()}")
print(f"Bronze path: {BRONZE_PATH}")
print("Libraries loaded successfully")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os

# Check what paths are available
print("=== Lakehouse mounts ===")
for p in ["/lakehouse", "/lakehouse/default", "/lakehouse/default/Files"]:
    exists = os.path.exists(p)
    print(f"{p}: {'EXISTS' if exists else 'NOT FOUND'}")

# List Bronze folder if it exists
bronze = "/lakehouse/default/Files/Bronze"
if os.path.exists(bronze):
    print(f"\n=== Files in Bronze ===")
    for f in os.listdir(bronze):
        print(f)
else:
    print(f"\nBronze folder not found at {bronze}")
    print("\n=== Searching for files ===")
    for root, dirs, files in os.walk("/lakehouse"):
        for f in files:
            if f.endswith(".xlsx"):
                print(os.path.join(root, f))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # **Net Sales Actuals**

# CELL ********************

# ── BRONZE: Net Sales Actuals ────────────────────────────────────
import pandas as pd

print("Reading Net Sales Actuals from Bronze...")
pdf_actuals = pd.read_excel(
    "/lakehouse/default/Files/Bronze/Net Sales actuals DATA.xlsx",
    sheet_name=0,
    header=0,
    dtype=str
)

print(f"Raw rows read: {len(pdf_actuals)}")
print(f"Columns: {list(pdf_actuals.columns)}")
print(pdf_actuals.head(3).to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── SILVER: Net Sales Actuals ────────────────────────────────────
print("Applying Silver transformations to Net Sales Actuals...")

# BU name mapping — controlled exact match
bu_mapping = {
    "ExEd": "ExEd Programs BU",
    "Qualification Programs": "Qualification Programs BU",
    "University Programs": "University Programs BU"
}

# Apply transformations
pdf_actuals_silver = pdf_actuals.copy()

# 1. Rename columns to standard names
pdf_actuals_silver.columns = [
    "BU", "Cost Center", "Project", "Year Month", "Net Sales"
]

# 2. Harmonise BU names using exact mapping
pdf_actuals_silver["BU"] = (
    pdf_actuals_silver["BU"]
    .str.strip()
    .map(bu_mapping)
    .fillna("Unassigned")
)

# 3. Convert Year Month to integer then to date
pdf_actuals_silver["Year Month"] = (
    pdf_actuals_silver["Year Month"]
    .astype(str)
    .str.strip()
    .astype(int)
)

pdf_actuals_silver["Date"] = pd.to_datetime(
    pdf_actuals_silver["Year Month"].astype(str).str[:4] + "-" +
    pdf_actuals_silver["Year Month"].astype(str).str[4:] + "-01"
)

# 4. Convert Net Sales to float
pdf_actuals_silver["Net Sales"] = (
    pdf_actuals_silver["Net Sales"]
    .astype(str)
    .str.strip()
    .astype(float)
)

# 5. Add ingestion metadata
pdf_actuals_silver["Ingestion Timestamp"] = datetime.now()
pdf_actuals_silver["Source File"] = "Net Sales actuals DATA.xlsx"

# 6. Validation flags
pdf_actuals_silver["Validation Flag"] = "OK"
pdf_actuals_silver.loc[
    pdf_actuals_silver["Net Sales"] < 0, "Validation Flag"
] = "NEGATIVE_NET_SALES"
pdf_actuals_silver.loc[
    pdf_actuals_silver["Net Sales"] == 0, "Validation Flag"
] = "ZERO_NET_SALES"

# 7. Report
print(f"Rows after Silver: {len(pdf_actuals_silver)}")
print(f"BU distribution:\n{pdf_actuals_silver['BU'].value_counts()}")
print(f"\nValidation flags:\n{pdf_actuals_silver['Validation Flag'].value_counts()}")
print(f"\nDate range: {pdf_actuals_silver['Date'].min()} to {pdf_actuals_silver['Date'].max()}")
print(f"\nSample:")
print(pdf_actuals_silver.head(3).to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── WRITE: silver_net_sales ──────────────────────────────────────
print("Writing silver_net_sales Delta table...")

# Rename columns to remove spaces (Delta requirement)
pdf_actuals_silver = pdf_actuals_silver.rename(columns={
    "BU": "BU",
    "Cost Center": "Cost_Center",
    "Project": "Project",
    "Year Month": "Year_Month",
    "Net Sales": "Net_Sales",
    "Date": "Date",
    "Ingestion Timestamp": "Ingestion_Timestamp",
    "Source File": "Source_File",
    "Validation Flag": "Validation_Flag"
})

# Convert to Spark DataFrame
df_actuals_silver = spark.createDataFrame(pdf_actuals_silver)

# Write as Delta table
(
    df_actuals_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("silver_net_sales")
)

# Verify
count = spark.table("silver_net_sales").count()
print(f"silver_net_sales written: {count} rows")
spark.table("silver_net_sales").printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # **Accruals**

# CELL ********************

# ── BRONZE: Accruals ─────────────────────────────────────────────
print("Reading Accruals from Bronze...")

pdf_accruals = pd.read_excel(
    "/lakehouse/default/Files/Bronze/Accruals DATA.xlsx",
    sheet_name=0,
    header=0,
    dtype=str
)

print(f"Raw rows read: {len(pdf_accruals)}")
print(f"Columns: {list(pdf_accruals.columns)}")
print(pdf_accruals.head(3).to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── INSPECT: Accruals raw data ───────────────────────────────────
print("Inspecting Accruals raw data...")

# Check BU values
print(f"BU value counts:")
print(pdf_accruals["BU"].value_counts(dropna=False))

# Check for Total row
total_rows = pdf_accruals[pdf_accruals["BU"] == "Total"]
print(f"\nTotal rows found: {len(total_rows)}")

# Check for null BU
null_rows = pdf_accruals[pdf_accruals["BU"].isna()]
print(f"Null BU rows found: {len(null_rows)}")
if len(null_rows) > 0:
    print(null_rows.to_string())

# Check total forecast value
pdf_accruals["Accrued Net Sales Forecast"] = (
    pd.to_numeric(pdf_accruals["Accrued Net Sales Forecast"], errors="coerce")
)
print(f"\nTotal Accrued Net Sales Forecast (all rows): €{pdf_accruals['Accrued Net Sales Forecast'].sum():,.2f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── SILVER: Accruals ─────────────────────────────────────────────
print("Applying Silver transformations to Accruals...")

pdf_accruals_silver = pdf_accruals.copy()

# 1. Remove Total row only — keep null BU rows
pdf_accruals_silver = pdf_accruals_silver[
    pdf_accruals_silver["BU"] != "Total"
].copy()
print(f"Rows after removing Total: {len(pdf_accruals_silver)}")

# 2. Replace null BU with Unassigned
pdf_accruals_silver["BU"] = (
    pdf_accruals_silver["BU"]
    .fillna("Unassigned")
    .str.strip()
)

# 3. Rename columns
pdf_accruals_silver = pdf_accruals_silver.rename(columns={
    "BU": "BU",
    "Accrual Basis": "Accrual_Basis",
    "PNo.": "Project_Number",
    "Net Sales Year": "Net_Sales_Year",
    "Net Sales Month": "Net_Sales_Month",
    "Accrued Net Sales Forecast": "Accrued_Net_Sales_Forecast"
})

# 4. Convert numeric columns
pdf_accruals_silver["Net_Sales_Year"] = (
    pd.to_numeric(pdf_accruals_silver["Net_Sales_Year"], errors="coerce")
    .astype("Int64")
)
pdf_accruals_silver["Net_Sales_Month"] = (
    pd.to_numeric(pdf_accruals_silver["Net_Sales_Month"], errors="coerce")
    .astype("Int64")
)
pdf_accruals_silver["Accrued_Net_Sales_Forecast"] = (
    pd.to_numeric(
        pdf_accruals_silver["Accrued_Net_Sales_Forecast"], 
        errors="coerce"
    )
)

# 5. Create Date column
pdf_accruals_silver["Date"] = pd.to_datetime(
    pdf_accruals_silver["Net_Sales_Year"].astype(str) + "-" +
    pdf_accruals_silver["Net_Sales_Month"].astype(str).str.zfill(2) + "-01"
)

# 6. Extract Accrual Type from Accrual Basis
def get_accrual_type(basis):
    if pd.isna(basis):
        return "Unknown"
    if "Project Accrual Plan" in str(basis):
        return "Project Accrual Plan"
    if "Monthly Auto-Accrual" in str(basis):
        return "Monthly Auto-Accrual"
    if "Actual Close Date" in str(basis):
        return "Actual Close Date"
    if "Module" in str(basis):
        return "Module Accrual"
    if "Program Start Month" in str(basis):
        return "Program Start Month"
    return str(basis)

pdf_accruals_silver["Accrual_Type"] = (
    pdf_accruals_silver["Accrual_Basis"].apply(get_accrual_type)
)

# 7. Validation flags
pdf_accruals_silver["Validation_Flag"] = "OK"
pdf_accruals_silver.loc[
    pdf_accruals_silver["BU"] == "Unassigned", "Validation_Flag"
] = "UNASSIGNED_BU"
pdf_accruals_silver.loc[
    pdf_accruals_silver["Accrued_Net_Sales_Forecast"] == 0,
    "Validation_Flag"
] = "ZERO_FORECAST"

# 8. Add ingestion metadata
pdf_accruals_silver["Ingestion_Timestamp"] = datetime.now()
pdf_accruals_silver["Source_File"] = "Accruals DATA.xlsx"

# 9. Report
total = pdf_accruals_silver["Accrued_Net_Sales_Forecast"].sum()
print(f"Rows after Silver: {len(pdf_accruals_silver)}")
print(f"BU distribution:\n{pdf_accruals_silver['BU'].value_counts()}")
print(f"\nAccrual Type distribution:\n{pdf_accruals_silver['Accrual_Type'].value_counts()}")
print(f"\nValidation flags:\n{pdf_accruals_silver['Validation_Flag'].value_counts()}")
print(f"\nTotal Accrued Forecast: €{total:,.2f}")
print(f"Date range: {pdf_accruals_silver['Date'].min()} to {pdf_accruals_silver['Date'].max()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── WRITE: silver_accruals ───────────────────────────────────────
print("Writing silver_accruals Delta table...")

# Convert Int64 to regular int for Spark compatibility
pdf_accruals_silver["Net_Sales_Year"] = (
    pdf_accruals_silver["Net_Sales_Year"].astype(float).astype(int)
)
pdf_accruals_silver["Net_Sales_Month"] = (
    pdf_accruals_silver["Net_Sales_Month"].astype(float).astype(int)
)

# Convert to Spark DataFrame
df_accruals_silver = spark.createDataFrame(pdf_accruals_silver)

# Write as Delta table
(
    df_accruals_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("silver_accruals")
)

# Verify
count = spark.table("silver_accruals").count()
total = spark.table("silver_accruals").agg(
    F.sum("Accrued_Net_Sales_Forecast").alias("total")
).collect()[0]["total"]

print(f"silver_accruals written: {count} rows")
print(f"Total Accrued Forecast in table: €{total:,.2f}")
spark.table("silver_accruals").printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # **Opportunities**

# CELL ********************

# ── BRONZE: Open Custom Opportunities ───────────────────────────
print("Reading Open Custom Opportunities from Bronze...")

pdf_pipeline = pd.read_excel(
    "/lakehouse/default/Files/Bronze/Open custom opportunities DATA.xlsx",
    sheet_name=0,
    header=0,
    dtype=str
)

print(f"Raw rows read: {len(pdf_pipeline)}")
print(f"Columns: {list(pdf_pipeline.columns)}")
print(pdf_pipeline.head(3).to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── SILVER: Open Custom Opportunities ───────────────────────────
print("Applying Silver transformations to Opportunities...")

pdf_pipeline_silver = pdf_pipeline.copy()

# 1. Rename columns to clean names
pdf_pipeline_silver = pdf_pipeline_silver.rename(columns={
    "Status": "Status",
    "Est. Closing Date": "Est_Closing_Date",
    "Closing Date": "Closing_Date",
    "Program start (simulated)": "Program_Start_Date",
    "Interest / Opportunity": "Opportunity_Name",
    "Expected Accrued Net Sales for This Year %": "This_Year_Pct",
    "Expected Accrued Net Sales for Next Year %": "Next_Year_Pct",
    "Probability": "Probability_Text",
    "Value in €": "Value_EUR",
    "Est. Value": "Est_Value",
    "Weighted Est. Value_": "Weighted_Value",
    "Weighted Est. Value this year": "Weighted_Value_This_Year",
    "Business Area": "Business_Area",
    "P&LS BU": "BU"
})

# 2. Parse dates
for col in ["Est_Closing_Date", "Closing_Date", "Program_Start_Date"]:
    pdf_pipeline_silver[col] = pd.to_datetime(
        pdf_pipeline_silver[col], errors="coerce"
    )

# 3. Parse Probability — remove % sign and convert to decimal
pdf_pipeline_silver["Probability"] = (
    pdf_pipeline_silver["Probability_Text"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
    .pipe(pd.to_numeric, errors="coerce")
    .div(100)
)

# 4. Convert numeric columns
for col in ["Value_EUR", "Est_Value", "Weighted_Value", 
            "Weighted_Value_This_Year"]:
    pdf_pipeline_silver[col] = pd.to_numeric(
        pdf_pipeline_silver[col], errors="coerce"
    ).fillna(0.0)

# 5. Convert percentage columns — null to 0
for col in ["This_Year_Pct", "Next_Year_Pct"]:
    pdf_pipeline_silver[col] = pd.to_numeric(
        pdf_pipeline_silver[col], errors="coerce"
    ).fillna(0.0)

# 6. Add Probability Band
def prob_band(p):
    if pd.isna(p):
        return "Unknown"
    if p <= 0.30:
        return "Low (<=30%)"
    if p <= 0.60:
        return "Medium (31-60%)"
    return "High (>60%)"

pdf_pipeline_silver["Probability_Band"] = (
    pdf_pipeline_silver["Probability"].apply(prob_band)
)

# 7. Trim BU
pdf_pipeline_silver["BU"] = (
    pdf_pipeline_silver["BU"].str.strip()
)

# 8. Validation flags
pdf_pipeline_silver["Validation_Flag"] = "OK"

# Probability missing
pdf_pipeline_silver.loc[
    pdf_pipeline_silver["Probability"].isna(),
    "Validation_Flag"
] = "MISSING_PROBABILITY"

# Year allocation exceeds 100%
pct_sum = (
    pdf_pipeline_silver["This_Year_Pct"] + 
    pdf_pipeline_silver["Next_Year_Pct"]
)
pdf_pipeline_silver.loc[
    pct_sum > 100, "Validation_Flag"
] = "ALLOCATION_EXCEEDS_100PCT"

# Weighted value mismatch
expected_weighted = (
    pdf_pipeline_silver["Value_EUR"] * 
    pdf_pipeline_silver["Probability"]
)
mismatch = (
    (expected_weighted - pdf_pipeline_silver["Weighted_Value"]).abs() > 1
) & (pdf_pipeline_silver["Weighted_Value"] > 0)
pdf_pipeline_silver.loc[mismatch, "Validation_Flag"] = "WEIGHTED_VALUE_MISMATCH"

# 9. Drop columns not needed downstream
pdf_pipeline_silver = pdf_pipeline_silver.drop(
    columns=["Probability_Text", "Business_Area", "Status"]
)

# 10. Add ingestion metadata
pdf_pipeline_silver["Ingestion_Timestamp"] = datetime.now()
pdf_pipeline_silver["Source_File"] = "Open custom opportunities DATA.xlsx"

# 11. Report
print(f"Rows after Silver: {len(pdf_pipeline_silver)}")
print(f"BU distribution:\n{pdf_pipeline_silver['BU'].value_counts()}")
print(f"\nProbability Band:\n{pdf_pipeline_silver['Probability_Band'].value_counts()}")
print(f"\nValidation flags:\n{pdf_pipeline_silver['Validation_Flag'].value_counts()}")
print(f"\nTotal Weighted Value This Year: €{pdf_pipeline_silver['Weighted_Value_This_Year'].sum():,.2f}")
print(f"\nDate range (Est Closing): {pdf_pipeline_silver['Est_Closing_Date'].min()} to {pdf_pipeline_silver['Est_Closing_Date'].max()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── WRITE: silver_pipeline ───────────────────────────────────────
print("Writing silver_pipeline Delta table...")

# Convert to Spark DataFrame
df_pipeline_silver = spark.createDataFrame(pdf_pipeline_silver)

# Write as Delta table
(
    df_pipeline_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("silver_pipeline")
)

# Verify
count = spark.table("silver_pipeline").count()
total_weighted = spark.table("silver_pipeline").agg(
    F.sum("Weighted_Value_This_Year").alias("total")
).collect()[0]["total"]

print(f"silver_pipeline written: {count} rows")
print(f"Total Weighted Value This Year: €{total_weighted:,.2f}")
spark.table("silver_pipeline").printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # **Summary and Validation**

# CELL ********************

# ── INGESTION SUMMARY ────────────────────────────────────────────
print("=" * 60)
print("nb_AaltoEE_01_ingest — COMPLETED")
print("=" * 60)
print(f"Run timestamp: {datetime.now()}")
print()

tables = {
    "silver_net_sales": {
        "expected_rows": 795,
        "expected_total": 14635067.81,
        "amount_col": "Net_Sales"
    },
    "silver_accruals": {
        "expected_rows": 313,
        "expected_total": 6616025.78,
        "amount_col": "Accrued_Net_Sales_Forecast"
    },
    "silver_pipeline": {
        "expected_rows": 126,
        "expected_total": 828900.00,
        "amount_col": "Weighted_Value_This_Year"
    }
}

all_passed = True

for table, config in tables.items():
    df = spark.table(table)
    actual_rows = df.count()
    actual_total = df.agg(
        F.sum(config["amount_col"]).alias("total")
    ).collect()[0]["total"]

    row_ok = actual_rows == config["expected_rows"]
    total_ok = abs(actual_total - config["expected_total"]) < 1.0

    status = "✓ PASS" if (row_ok and total_ok) else "✗ FAIL"
    if not (row_ok and total_ok):
        all_passed = False

    print(f"{status} | {table}")
    print(f"       Rows:  {actual_rows} (expected {config['expected_rows']})")
    print(f"       Total: €{actual_total:,.2f} (expected €{config['expected_total']:,.2f})")
    print()

print("=" * 60)
print(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print("=" * 60)

# Validation flag summary across all tables
print("\nValidation flags summary:")
for table in tables:
    df = spark.table(table)
    if "Validation_Flag" in df.columns:
        flags = df.groupBy("Validation_Flag").count().orderBy("count", ascending=False)
        print(f"\n{table}:")
        flags.show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
