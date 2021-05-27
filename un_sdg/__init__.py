import os
import pandas as pd
import os
import glob
from datetime import datetime
import json
import itertools
import functools
import math
import numpy as np
import requests

from typing import List


#from standard_importer.import_dataset import USER_ID
# Dataset constants.
DATASET_NAME = "United Nations Sustainable Development Goals Indicators"
DATASET_AUTHORS = "United Nations"
DATASET_VERSION = "2021-03"
DATASET_LINK = "https://unstats.un.org/sdgs/indicators/database/" # Have to request dataset by email
DATASET_RETRIEVED_DATE = "10-May-2021"
DATASET_DIR = os.path.dirname(__file__) 
DATASET_NAMESPACE = f"{DATASET_DIR.split('/')[-1]}@{DATASET_VERSION}"
CONFIGPATH = os.path.join(DATASET_DIR, 'config')
INPATH = os.path.join(DATASET_DIR, 'input')
DATA_PATH = os.path.join(DATASET_DIR, 'output')
INFILE = os.path.join(INPATH, 'un-sdg-' + DATASET_VERSION + '.csv')
ENTFILE =  os.path.join(INPATH, 'entities-' + DATASET_VERSION + '.csv')
METAPATH = os.path.join(DATASET_DIR, 'metadata')
METADATA_LOC = 'https://unstats.un.org/sdgs/metadata/files/SDG-indicator-metadata.zip'
USER_ID = 54
# Should add commonly used vars in here e.g. UNITS = '[UNITS] ?

#### Helper functions: 

## Not sure how well this works when the list is longer than one
def delete_output(keep_paths: List[str]) -> None:
    for path in keep_paths:
        if os.path.exists(os.path.join(DATA_PATH, path)):
            for CleanUp in glob.glob(os.path.join(DATA_PATH, '*.*')):
                if not CleanUp.endswith(path):    
                    os.remove(CleanUp)

@functools.lru_cache(maxsize=256)
def get_series_with_relevant_dimensions(data_filtered,indicator, series, DIMENSIONS, NON_DIMENSIONS):
    """ For a given indicator and series, return a tuple:
    
      - data filtered to that indicator and series
      - names of relevant dimensions
      - unique values for each relevant dimension
    """
   # data_filtered = original_df[(original_df.Indicator == indicator) & (original_df.SeriesCode == series)]
    non_null_dimensions_columns = [col for col in DIMENSIONS if data_filtered.loc[:, col].notna().any()]
    dimension_names = []
    dimension_unique_values = []
    
    for c in non_null_dimensions_columns:
        print(non_null_dimensions_columns)
        uniques = data_filtered[c].unique()
        if len(uniques) > 1: # Means that columns where the value doesn't change aren't included e.g. Nature is typically consistent across a dimension whereas Age and Sex are less likely to be. 
            dimension_names.append(c)
            dimension_unique_values.append(list(uniques))
    return (data_filtered[data_filtered.columns.intersection(list(NON_DIMENSIONS)+ list(dimension_names))], dimension_names, dimension_unique_values)

@functools.lru_cache(maxsize=256)
def generate_tables_for_indicator_and_series(data_filtered, indicator, series, DIMENSIONS, NON_DIMENSIONS):
    tables_by_combination = {}
    data_filtered, dimensions, dimension_values = get_series_with_relevant_dimensions(data_filtered, indicator, series, DIMENSIONS, NON_DIMENSIONS)
    if len(dimensions) == 0:
        # no additional dimensions
        export = data_filtered
        return export
    else:
        for dimension_value_combination in itertools.product(*dimension_values):
            # build filter by reducing, start with a constant True boolean array
            filt = [True] * len(data_filtered)
            for dim_idx, dim_value in enumerate(dimension_value_combination):
                dimension_name = dimensions[dim_idx]
                value_is_nan = type(dim_value) == float and math.isnan(dim_value)
                filt = filt & (data_filtered[dimension_name].isnull() if value_is_nan else data_filtered[dimension_name] == dim_value)
            tables_by_combination[dimension_value_combination] = data_filtered[filt].drop(dimensions, axis=1)   
    return tables_by_combination

def extract_datapoints(df):
    return pd.DataFrame({
        'country': df['country'],
        'year': df['TimePeriod'],
        'value': df['Value']
    }).drop_duplicates(subset=['country', 'year']).dropna()


def get_distinct_entities() -> List[str]:
    """retrieves a list of all distinct entities that contain at least
    on non-null data point that was saved to disk from the
    `clean_and_create_datapoints()` method.
    Returns:
        entities: List[str]. List of distinct entity names.
    """
    fnames = [fname for fname in os.listdir(os.path.join(DATA_PATH, 'datapoints')) if fname.endswith('.csv')]
    entities = set({})
    for fname in fnames:
        df_temp = pd.read_csv(os.path.join(DATA_PATH, 'datapoints', fname))
        entities.update(df_temp['country'].unique().tolist())
    
    entities = list(entities)
    assert pd.notnull(entities).all(), (
        "All entities should be non-null. Something went wrong in "
        "`clean_and_create_datapoints()`."
    )
    return entities


def clean_datasets() -> pd.DataFrame:
    """Constructs a dataframe where each row represents a dataset to be
    upserted.
    Note: often, this dataframe will only consist of a single row.
    """
    data = [
        {"id": 0, "name": f"{DATASET_NAME} - {DATASET_AUTHORS} ({DATASET_VERSION})"}
    ]
    df = pd.DataFrame(data)
    return df


def dimensions_description() -> None:
    base_url = "https://unstats.un.org/sdgapi"
    # retrieves all goal codes
    url = f"{base_url}/v1/sdg/Goal/List"
    res = requests.get(url)
    assert res.ok
    goals = json.loads(res.content)
    goal_codes = [int(goal['code']) for goal in goals]
    # retrieves all area codes
    d = []
    for goal in goal_codes:
        url = f"{base_url}/v1/sdg/Goal/{goal}/Dimensions"
        res = requests.get(url)
        assert res.ok
        dims = json.loads(res.content)
        for dim in dims:
            for code in dim['codes']:
                d.append(
                    {
                        'code': code['code'],
                        'description': code['description'],
                    }
                )
    dim_dict = pd.DataFrame(d).drop_duplicates().set_index('code').squeeze().to_dict()
    return(dim_dict)


def attributes_description() -> None:
    base_url = "https://unstats.un.org/sdgapi"
    # retrieves all goal codes
    url = f"{base_url}/v1/sdg/Goal/List"
    res = requests.get(url)
    assert res.ok
    goals = json.loads(res.content)
    goal_codes = [int(goal['code']) for goal in goals]
    # retrieves all area codes
    a = []
    for goal in goal_codes:
        url = f"{base_url}/v1/sdg/Goal/{goal}/Attributes"
        res = requests.get(url)
        assert res.ok
        attr = json.loads(res.content)
        for att in attr:
            for code in att['codes']:
                a.append(
                    {
                        'code': code['code'],
                        'description': code['description'],
                    }
                )
    att_dict = pd.DataFrame(a).drop_duplicates().set_index('code').squeeze().to_dict()
    return(att_dict)
