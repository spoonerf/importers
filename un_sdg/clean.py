
import pandas as pd
import os
import shutil
import glob

from typing import List, Tuple, Dict

#from db import connection
#from db_utils import DBUtils
from un_sdg import (
    INFILE,
    ENTFILE,
    DATASET_NAME,
    DATASET_AUTHORS,
    DATASET_VERSION,
    DATASET_LINK,
    DATASET_RETRIEVED_DATE,
    CONFIGPATH,
    INPATH,
    OUTPATH
)
# Get entities from the dataset 

df_entities = pd.read_csv(INFILE, low_memory = False)
assert not df_entities['GeoAreaName'].duplicated().any()
df_entities[['GeoAreaName']].drop_duplicates() \
                                       .dropna() \
                                        .rename(columns={'GeoAreaName': 'Country'}) \
                                        .to_csv(ENTFILE, index=False)
"""
Now use the country standardiser tool to standardise $ENTFILE
1. Open the OWID Country Standardizer Tool
   (https://owid.cloud/admin/standardize);
2. Change the "Input Format" field to "Non-Standard Country Name";
3. Change the "Output Format" field to "Our World In Data Name"; 
4. In the "Choose CSV file" field, upload {outfpath};
5. For any country codes that do NOT get matched, enter a custom name on
   the webpage (in the "Or enter a Custom Name" table column);
    * NOTE: For this dataset, you will most likely need to enter custom
      names for regions/continents (e.g. "Arab World", "Lower middle
      income");
6. Click the "Download csv" button;
7. Replace {outfpath} with the downloaded CSV;
8. Rename the "Country" column to "country_code".
"""

KEEP_PATHS = ['standardized_entity_names.csv']

# Max length of source name.
MAX_SOURCE_NAME_LEN = 256


### Start main() here

delete_output(KEEP_PATHS)

# Datasets.csv

df_datasets = clean_datasets()
assert df_datasets.shape[0] == 1, f"Only expected one dataset in {os.path.join(OUTPATH, 'datasets.csv')}."
df_datasets.to_csv(os.path.join(OUTPATH, 'datasets.csv'), index=False)


original_df = pd.read_csv(
    INFILE, 
    converters={'Value': str_to_float},
    low_memory=False
)
original_df = original_df[original_df['Value'].notnull()]






#### Helper functions: 

def clean_datasets():
    """Constructs a dataframe where each row represents a dataset to be upserted."""
    data = [
        {"id": 0, "name": f"{DATASET_NAME} - {DATASET_AUTHORS} ({DATASET_VERSION})"}
    ]
    df = pd.DataFrame(data)
    return df

## Not sure how well this works when the list is longer than one
def delete_output(keep_paths: List[str]) -> None:
    for path in keep_paths:
        if os.path.exists(os.path.join(OUTPATH, path)):
            for CleanUp in glob.glob(os.path.join(OUTPATH, '*.*')):
                if not CleanUp.endswith(path):    
                    os.remove(CleanUp)



#for path in KEEP_PATHS:
#    pd.Series(out_files).str.contains(path).tolist()

#out_files = pd.Series(glob.glob(os.path.join(OUTPATH, '*.*')))

#out_files = glob.glob(os.path.join(OUTPATH, '*.*'))
#for file in out_files:
#    file_st = pd.Series(file)
#    if ~file_st.str.contains('|'.join(KEEP_PATHS)):
#        os.remove(file)

#    files_to_remove = out_files[~out_files.str.contains('|'.join(KEEP_PATHS))]
#    files_to_remove
#    os.remove(files_to_remove)

def str_to_float(s):
    try:
        # Parse strings with thousands (,) separators
        return float(s.replace(',','')) if type(s) == str else s
    except ValueError:
        return None
