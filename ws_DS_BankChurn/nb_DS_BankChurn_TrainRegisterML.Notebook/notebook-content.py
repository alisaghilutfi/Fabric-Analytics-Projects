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

# Install imblearn for SMOTE using pip
# imblearn is a library for Synthetic Minority Oversampling Technique (SMOTE) which is used when dealing with imbalanced datasets

%pip install imblearn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# %pip install --upgrade scikit-learn imbalanced-learn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import numpy as np
import pandas as pd

import seaborn as sns
sns.set_theme(style="whitegrid", palette="tab10", rc = {'figure.figsize':(9,6)})
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc, rcParams
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score, roc_auc_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import mlflow
from mlflow.models.signature import infer_signature
from collections import Counter

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import imblearn
from imblearn.over_sampling import SMOTE

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Setting up the prerequisites**

# CELL ********************

# Load the data
SEED = 12345
churn_clean = spark.read.format("delta").load("Tables/dbo/churn_clean").toPandas()
churn_clean["Exited"] = churn_clean["Exited"].astype("int")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

churn_clean.info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Generate experiment for tracking and logging the model using MLflow

EXPERIMENT_NAME = "bank-churn-experiment"  # MLflow experiment name
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(exclusive=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Prepare training, validation and test datasets

y = churn_clean["Exited"]
X = churn_clean.drop("Exited", axis=1)

# Split the dataset to 60%, 20%, 20% for training, validation, and test datasets
# Train-Test Separation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
# Train-Validation Separation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Save test data to a delta table

table_name = "churn_test"

# Create PySpark DataFrame from Pandas
df_test = spark.createDataFrame(X_test)
df_test.write.mode("overwrite").format("delta").save(f"Tables/dbo/{table_name}")
print(f"Spark test DataFrame saved to delta table: {table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Apply SMOTE to the training data to synthesize new samples for the minority class**

# CELL ********************

sm = SMOTE(random_state=SEED)
X_res, y_res = sm.fit_resample(X_train, y_train)
new_train = pd.concat([X_res, y_res], axis=1)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

sm

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

X_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

new_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Model training**

# CELL ********************

# Train the model using Random Forest with maximum depth of 4 and 4 features

mlflow.sklearn.autolog(registered_model_name='rfc1_sm') # Register the trained model with autologging
rfc1_sm = RandomForestClassifier(max_depth=4, max_features=4, min_samples_split=3, random_state=1) # Pass hyperparameters
with mlflow.start_run(run_name="rfc1_sm") as run:
    rfc1_sm_run_id = run.info.run_id # Capture run_id for model prediction later
    print("run_id: {}; status: {}".format(rfc1_sm_run_id, run.info.status))
    rfc1_sm.fit(X_res, y_res.ravel()) # Balanced training data
    rfc1_sm.score(X_val, y_val)
    y_pred = rfc1_sm.predict(X_val)
    cr_rfc1_sm = classification_report(y_val, y_pred)
    cm_rfc1_sm = confusion_matrix(y_val, y_pred)
    val_proba = rfc1_sm.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, val_proba)
    mlflow.log_metric("val_roc_auc", val_roc_auc)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Train the model using Random Forest with maximum depth of 8 and 6 features

mlflow.sklearn.autolog(registered_model_name='rfc2_sm') # Register the trained model with autologging
rfc2_sm = RandomForestClassifier(max_depth=8, max_features=6, min_samples_split=3, random_state=1) # Pass hyperparameters
with mlflow.start_run(run_name="rfc2_sm") as run:
    rfc2_sm_run_id = run.info.run_id # Capture run_id for model prediction later
    print("run_id: {}; status: {}".format(rfc2_sm_run_id, run.info.status))
    rfc2_sm.fit(X_res, y_res.ravel()) # Balanced training data
    rfc2_sm.score(X_val, y_val)
    y_pred = rfc2_sm.predict(X_val)
    cr_rfc2_sm = classification_report(y_val, y_pred)
    cm_rfc2_sm = confusion_matrix(y_val, y_pred)
    val_proba = rfc2_sm.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, val_proba)
    mlflow.log_metric("val_roc_auc", val_roc_auc)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Train the model using LightGBM

# lgbm_model
mlflow.lightgbm.autolog(registered_model_name='lgbm_sm') # Register the trained model with autologging

lgbm_sm_model = LGBMClassifier(learning_rate = 0.07, 
                        max_delta_step = 2, 
                        n_estimators = 100,
                        max_depth = 10, 
                        eval_metric = "logloss", 
                        objective='binary', 
                        random_state=42)

with mlflow.start_run(run_name="lgbm_sm") as run:
    lgbm1_sm_run_id = run.info.run_id # Capture run_id for model prediction later
    lgbm_sm_model.fit(X_res, y_res.ravel()) # Balanced training data
    y_pred = lgbm_sm_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    cr_lgbm_sm = classification_report(y_val, y_pred)
    cm_lgbm_sm = confusion_matrix(y_val, y_pred)
    val_proba = lgbm_sm_model.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, val_proba)
    mlflow.log_metric("val_roc_auc", val_roc_auc)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Champion model selection**

# CELL ********************

best_run = mlflow.search_runs(
    experiment_names=["bank-churn-experiment"],
    order_by=["metrics.val_roc_auc DESC"],
    max_results=1
)
best_run_id = best_run.iloc[0]["run_id"]
champion_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(champion_uri, "champion_BankChurn")
print(f"Champion registered: run {best_run_id}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Assess the performances of the trained models on the validation dataset**

# CELL ********************

ypred_rfc1_sm_v = rfc1_sm.predict(X_val) # Random Forest with max depth of 4 and 4 features
ypred_rfc2_sm_v = rfc2_sm.predict(X_val) # Random Forest with max depth of 8 and 6 features
ypred_lgbm1_sm_v = lgbm_sm_model.predict(X_val) # LightGBM version 1

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Show True/False Positives/Negatives using the Confusion Matrix**

# CELL ********************

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)
    plt.figure(figsize=(4,4))
    plt.rcParams.update({'font.size': 10})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, color="blue")
    plt.yticks(tick_marks, classes, color="blue")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Confusion Matrix for Random Forest Classifier with maximum depth of 4 and 4 features

cfm = confusion_matrix(y_val, y_pred=ypred_rfc1_sm_v)
plot_confusion_matrix(cfm, classes=['Non Churn','Churn'],
                      title='Random Forest with max depth of 4')
tn, fp, fn, tp = cfm.ravel()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Confusion Matrix for Random Forest Classifier with maximum depth of 8 and 6 features

cfm = confusion_matrix(y_val, y_pred=ypred_rfc2_sm_v)
plot_confusion_matrix(cfm, classes=['Non Churn','Churn'],
                      title='Random Forest with max depth of 8')
tn, fp, fn, tp = cfm.ravel()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Confusion Matrix for LightGBM version 1

cfm = confusion_matrix(y_val, y_pred=ypred_lgbm1_sm_v)
plot_confusion_matrix(cfm, classes=['Non Churn','Churn'],
                      title='LightGBM')
tn, fp, fn, tp = cfm.ravel()

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
