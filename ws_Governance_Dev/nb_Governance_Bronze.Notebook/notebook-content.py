# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "4478b3d6-2e64-43ee-b120-3caa722abf48",
# META       "default_lakehouse_name": "lh_Governance",
# META       "default_lakehouse_workspace_id": "7ea6403c-9364-4cd5-94c3-3ba6fcfce918",
# META       "known_lakehouses": [
# META         {
# META           "id": "4478b3d6-2e64-43ee-b120-3caa722abf48"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Cell 1 - Parameters & auth
from datetime import datetime, timedelta, timezone
import requests
import json
from pyspark.sql.types import StringType, StructType, StructField

TENANT_ID     = "c8cb697b-07d1-4c61-a5d9-bbe5ad099fe8"
CLIENT_ID     = "4c046cb3-0862-4862-b228-5fee8381deca"
CLIENT_SECRET = "<paste-secret-here>"  # governance-scanner-key-1 -- fill in Fabric UI only

PBI_SCOPE  = "https://analysis.windows.net/powerbi/api/.default"
PBI_BASE   = "https://api.powerbi.com/v1.0/myorg"
ADMIN_BASE = f"{PBI_BASE}/admin"

def get_token():
    resp = requests.post(
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": PBI_SCOPE
        }
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

TOKEN = get_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
print("Token acquired")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 2 - Bronze: activity events (yesterday)
import time

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
start_dt  = f"{yesterday}T00:00:00.000Z"
end_dt    = f"{yesterday}T23:59:59.999Z"

all_events = []
url = f"{ADMIN_BASE}/activityevents?startDateTime='{start_dt}'&endDateTime='{end_dt}'"

while url:
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    all_events.extend(data.get("activityEventEntities", []))
    url = data.get("continuationUri")
    if url:
        time.sleep(0.5)

print(f"Activity events fetched: {len(all_events)}")

if all_events:
    df = spark.createDataFrame(
        [(yesterday, json.dumps(e)) for e in all_events],
        schema=StructType([
            StructField("event_date", StringType()),
            StructField("raw_json", StringType())
        ])
    )
    df.write.format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable("bronze_activity_events")
    print(f"bronze_activity_events: {len(all_events)} rows written")
else:
    print("No activity events for yesterday")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 3 - get modified workspace IDs (incremental)
try:
    last_run_df = spark.sql("SELECT MAX(scanned_at) as last_run FROM bronze_scan_workspaces")
    last_run = last_run_df.collect()[0]["last_run"]
except Exception:
    last_run = None

if last_run:
    modified_url = f"{ADMIN_BASE}/workspaces/modified?modifiedSince={last_run}"
else:
    modified_url = f"{ADMIN_BASE}/workspaces/modified"

resp = requests.get(modified_url, headers=HEADERS)
resp.raise_for_status()
workspace_ids = [w["id"] for w in resp.json()]
print(f"Workspaces to scan: {len(workspace_ids)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 4 - Scanner API: trigger scan, poll, fetch result
BATCH_SIZE = 100

def trigger_scan(ws_ids):
    resp = requests.post(
        f"{ADMIN_BASE}/workspaces/getInfo"
        "?lineage=false&datasourceDetails=true"
        "&datasetSchema=false&datasetExpressions=false",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"workspaces": ws_ids}
    )
    resp.raise_for_status()
    return resp.json()["id"]

def poll_scan(scan_id, max_wait=120):
    for _ in range(max_wait // 5):
        resp = requests.get(
            f"{ADMIN_BASE}/workspaces/scanStatus/{scan_id}",
            headers=HEADERS
        )
        resp.raise_for_status()
        status = resp.json()["status"]
        if status == "Succeeded":
            return True
        elif status == "Failed":
            raise RuntimeError(f"Scan {scan_id} failed")
        time.sleep(5)
    raise TimeoutError(f"Scan {scan_id} timed out")

def fetch_result(scan_id):
    resp = requests.get(
        f"{ADMIN_BASE}/workspaces/scanResult/{scan_id}",
        headers=HEADERS
    )
    resp.raise_for_status()
    return resp.json()

scan_results = []
scanned_at = datetime.now(timezone.utc).isoformat()

for i in range(0, len(workspace_ids), BATCH_SIZE):
    batch = workspace_ids[i:i + BATCH_SIZE]
    scan_id = trigger_scan(batch)
    print(f"Scan triggered: {scan_id} ({len(batch)} workspaces)")
    poll_scan(scan_id)
    result = fetch_result(scan_id)
    scan_results.extend(result.get("workspaces", []))
    print(f"Batch complete: {len(result.get('workspaces', []))} workspaces")

print(f"Total workspaces scanned: {len(scan_results)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 5 - write scan results to bronze_scan_workspaces
if scan_results:
    rows = [(scanned_at, json.dumps(ws)) for ws in scan_results]
    df = spark.createDataFrame(
        rows,
        schema=StructType([
            StructField("scanned_at", StringType()),
            StructField("raw_json", StringType())
        ])
    )
    df.write.format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable("bronze_scan_workspaces")
    print(f"bronze_scan_workspaces: {len(scan_results)} rows written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 6 - Bronze: refresh history per semantic model
semantic_models = []
for ws in scan_results:
    ws_id = ws.get("id")
    for ds in ws.get("datasets", []):
        semantic_models.append((ws_id, ds.get("id")))

print(f"Semantic models found: {len(semantic_models)}")

refresh_rows = []
for ws_id, sm_id in semantic_models:
    try:
        resp = requests.get(
            f"{PBI_BASE}/groups/{ws_id}/datasets/{sm_id}/refreshes?$top=30",
            headers=HEADERS
        )
        if resp.status_code == 200:
            for r in resp.json().get("value", []):
                refresh_rows.append((scanned_at, ws_id, sm_id, json.dumps(r)))
        time.sleep(0.2)
    except Exception as e:
        print(f"Skipped {sm_id}: {e}")

print(f"Refresh history records: {len(refresh_rows)}")

if refresh_rows:
    df = spark.createDataFrame(
        refresh_rows,
        schema=StructType([
            StructField("scanned_at", StringType()),
            StructField("workspace_id", StringType()),
            StructField("semantic_model_id", StringType()),
            StructField("raw_json", StringType())
        ])
    )
    df.write.format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable("bronze_refresh_history")
    print(f"bronze_refresh_history: {len(refresh_rows)} rows written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
