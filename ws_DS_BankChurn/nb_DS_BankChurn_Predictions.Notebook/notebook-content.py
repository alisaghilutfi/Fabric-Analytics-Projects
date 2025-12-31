# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "3d14c19d-21db-49af-873f-1621ee27460c",
# META       "default_lakehouse_name": "lh_DS_BankChurn",
# META       "default_lakehouse_workspace_id": "e82dfb36-dba0-483b-8860-67b2a08d0487",
# META       "known_lakehouses": [
# META         {
# META           "id": "3d14c19d-21db-49af-873f-1621ee27460c"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import numpy as np
import pandas as pd

from synapse.ml.predict import MLFlowTransformer
from pyspark.ml.feature import SQLTransformer
from pyspark.sql.functions import col, pandas_udf, udf, lit

import mlflow
from mlflow.models.signature import infer_signature

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load the test data
df_test = spark.read.format("delta").load("Tables/dbo/churn_test")
display(df_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_test.toPandas().info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **PREDICT with the Transformer API**

# CELL ********************

print(type(df_test))
print(type(df_test.toPandas()))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_test.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

model = MLFlowTransformer(
    inputCols=list(df_test.columns),
    outputCol='predictions',
    modelName='lgbm_sm',
    modelVersion=1
)

predictions = model.transform(df_test)
display(predictions)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# # Initialize with the dataframe, then update it iteratively
# df_test_bool = df_test
# bool_cols = [c for c, t in df_test.dtypes if t=='boolean']
# for c in bool_cols:
#     df_test_bool = df_test_bool.withColumn(c, col(c).cast("int"))
# # Check again - all should be int32/int64 now
# df_test_bool.toPandas().info()


# model = MLFlowTransformer(
#     inputCols=list(df_test_bool.columns),
#     outputCol='predictions',
#     modelName='lgbm_sm',
#     modelVersion=2
# )

# predictions = model.transform(df_test_bool)
# display(predictions)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **PREDICT with the Spark SQL API**

# CELL ********************

# #model_name = 'lgbm_sm'
# model_name = 'rfc1_sm'
# model_version = 1
# features = df_test_bool.columns

# sqlt = SQLTransformer().setStatement( 
#     f"SELECT PREDICT('{model_name}/{model_version}', {','.join(features)}) as predictions FROM __THIS__")

# # Substitute "X_test" below with your own test dataset
# display(sqlt.transform(df_test_bool))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **PREDICT with a user-defined function (UDF)**

# CELL ********************

# my_udf = model.to_udf()
# features = df_test_bool.columns

# display(df_test_bool.withColumn("predictions", my_udf(*[col(f) for f in features])))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Write model prediction results to the lakehouse**

# CELL ********************

# Save predictions to lakehouse to be used for generating a Power BI report

table_name = "customer_churn_test_predictions"
predictions.write.format('delta').mode("overwrite").save(f"Tables/dbo/{table_name}")
print(f"Spark DataFrame saved to delta table: {table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
