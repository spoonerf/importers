"""snippet for downloading UN SDG data in CSV format from the SDG API.
"""
import json
import requests
import pandas as pd
import os
import zipfile
from io import BytesIO


from un_sdg import INFILE, METAPATH

base_url = "https://unstats.un.org/sdgapi"

# retrieves all goal codes
url = f"{base_url}/v1/sdg/Goal/List"
res = requests.get(url)
assert res.ok

goals = json.loads(res.content)
goal_codes = [int(goal['code']) for goal in goals]
# retrieves all area codes
url = f"{base_url}/v1/sdg/GeoArea/List"
res = requests.get(url)
assert res.ok

areas = json.loads(res.content)
area_codes = [int(area['geoAreaCode']) for area in areas]
# retrieves csv with data for all codes and areas
url = f"{base_url}/v1/sdg/Goal/DataCSV"
res = requests.post(url, data={'goal': goal_codes, 'areaCodes': area_codes})
assert res.ok
df = pd.read_csv(BytesIO(res.content), low_memory = False)
df.to_csv(os.path.join(INFILE), index=False)


# Download and unzip metadata

zip_url = 'https://unstats.un.org/sdgs/metadata/files/SDG-indicator-metadata.zip'
r = requests.get(zip_url)  
with open(os.path.join(METAPATH, 'sdg-metadata.zip'), 'wb') as f:
    f.write(r.content)

with zipfile.ZipFile(os.path.join(METAPATH, 'sdg-metadata.zip'), 'r') as zip_ref:
    zip_ref.extractall(METAPATH)

files_in_directory = os.listdir(METAPATH)
filtered_files = [file for file in files_in_directory if not file.endswith(".pdf")]
for file in filtered_files:
	path_to_file = os.path.join(METAPATH, file)
	os.remove(path_to_file)