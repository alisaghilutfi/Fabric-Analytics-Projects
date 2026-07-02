# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "fda3c40e-3b18-4230-b5a8-18334d845083",
# META       "default_lakehouse_name": "lh_USGS_Earthquake",
# META       "default_lakehouse_workspace_id": "39caa3ab-a964-45c2-bd5b-7d46ad66c985",
# META       "known_lakehouses": [
# META         {
# META           "id": "fda3c40e-3b18-4230-b5a8-18334d845083"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import requests
import json

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# API URL with start and end dates provided by Data Factory, formatted for geojson output
url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_date}&endtime={end_date}"

# GET request to fetch data
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Get the JSON response
    data = response.json()
    data = data["features"]

    # Specify the file name (and path if needed)
    file_path = f"/lakehouse/default/Files/{start_date}_earthquake_data.json"

    # Open the file in write mode ('w') and save the JSON data
    with open(file_path, "w") as file:
        # `json.dump` method serializes `data` as a JSON formatted stream to `file`
        # `indent=4` makes the file human-readable by adding whitespace
        json.dump(data, file, indent=4)
    
    print(f"Data successfully saved to {file_path}")
else:
    print("Failed to fetch data. Status code:", response.status_code)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
