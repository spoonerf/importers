import pandas as pd
import os
import shutil
import glob
from datetime import datetime
import json
import io
import itertools
import functools
import math
import pdfminer.high_level
import pdfminer.layout
import lxml.html


from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
from utils import str_to_float, extract_description, clean_datasets

#from db import connection
#from db_utils import DBUtils
from un_sdg import (
    DATAPATH,
    INFILE,
    ENTFILE,
    DATASET_NAME,
    DATASET_AUTHORS,
    DATASET_VERSION,
    DATASET_LINK,
    DATASET_RETRIEVED_DATE,
    CONFIGPATH,
    INPATH,
    OUTPATH,
    METAPATH
)
# Get entities from the dataset 

#df_entities = pd.read_csv(INFILE, low_memory = False)
#assert not df_entities['GeoAreaName'].duplicated().any()
#df_entities[['GeoAreaName']].drop_duplicates() \
#                                       .dropna() \
#                                        .rename(columns={'GeoAreaName': 'Country'}) \
#                                        .to_csv(ENTFILE, index=False)
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
#delete_output(KEEP_PATHS)

# Max length of source name.
MAX_SOURCE_NAME_LEN = 256

# Make the datapoints folder
Path(DATAPATH).mkdir(parents=True, exist_ok=True)

# Load and clean the data 
original_df = pd.read_csv(
    INFILE, 
    converters={'Value': str_to_float},
    low_memory=False
)
original_df = original_df[original_df['Value'].notnull()]

original_df[['GeoAreaName']].drop_duplicates() \
                                       .dropna() \
                                        .rename(columns={'GeoAreaName': 'Country'}) \
                                        .to_csv(ENTFILE, index=False)

DIMENSIONS = [c for c in original_df.columns if c[0] == '[' and c[-1] == ']']

### Start main() here

### Datasets

df_datasets = clean_datasets()
assert df_datasets.shape[0] == 1, f"Only expected one dataset in {os.path.join(OUTPATH, 'datasets.csv')}."
df_datasets.to_csv(os.path.join(OUTPATH, 'datasets.csv'), index=False)

### Sources

df_sources = pd.DataFrame(columns=['id', 'name', 'description', 'dataset_id'])

source_description_template = {
    'dataPublishedBy': "United Nations Statistics Division",
    'dataPublisherSource': None,
    'link': "https://unstats.un.org/sdgs/indicators/database/",
    'retrievedDate': datetime.now().strftime("%d-%B-%y"),
    'additionalInfo': None
}

all_series = original_df[['Indicator', 'SeriesCode', 'Source','SeriesDescription', '[Units]']]   .groupby(by=['Indicator', 'SeriesCode', 'Source','SeriesDescription', '[Units]'])   .count()   .reset_index()

all_series = original_df[['Indicator', 'SeriesCode', 'SeriesDescription', '[Units]']]   .groupby(by=['Indicator', 'SeriesCode', 'SeriesDescription', '[Units]'])   .count()   .reset_index()


df_sources = pd.DataFrame(columns=['id', 'name', 'description', 'dataset_id'])

source_description = source_description_template.copy()

for i, row in tqdm(all_series.iterrows(), total=len(all_series)):
   # print(row['Indicator'])   
    try:
        source_description['additionalInfo'] = extract_description(os.path.join(METAPATH,'Metadata-%s.pdf') % '-'.join([part.rjust(2, '0') for part in row['Indicator'].split('.')]))
        print(source_description['additionalInfo'])
    except:
        pass
    df_sources = df_sources.append({
        'id': i,
        #'name': "%s (UN SDG, 2021)" % row['Source'],
        'name': "%s (UN SDG, 2021)" % row['SeriesDescription'],
        'description': json.dumps(source_description),
        'dataset_id': df_datasets['id'] # this may need to be more flexible! 
    }, ignore_index=True)

df_sources.to_csv(os.path.join(OUTPATH, 'sources.csv'), index=False)

### Variables

variable_codes = original_df['SeriesCode'].drop_duplicates()

variable_codes = {
    'id': 0,
    'dataset_id': 0,
    'unit': original_df['[Units]']
}

entity2owid_name = pd.read_csv(os.path.join(OUTPATH, 'standardized_entity_names.csv')) \
                              .set_index('country_code') \
                              .squeeze() \
                              .to_dict()

original_df['country'] = original_df['GeoAreaName'].apply(lambda x: entity2owid_name[x])

variable_codes = original_df['SeriesCode'].drop_duplicates()

NON_DIMENSIONS = [c for c in original_df.columns if c not in set(DIMENSIONS)]# not sure if units should be in here

variable_idx = 0
variables = pd.DataFrame(columns=['id', 'name', 'unit', 'dataset_id'])

for i, row in tqdm(all_series.iterrows(), total=len(all_series)):
    _, dimensions, dimension_members = get_series_with_relevant_dimensions(row['Indicator'], row['SeriesCode'])
    if len(dimensions) == 0:
        # no additional dimensions
        table = generate_tables_for_indicator_and_series(row['Indicator'], row['SeriesCode'])
        variable = {
            'id': variable_idx,
            'dataset_id': i,
            'unit': row['[Units]'],
            'name': "%s - %s - %s" % (row['Indicator'], row['SeriesDescription'], row['SeriesCode'])
        }
        variables = variables.append(variable, ignore_index=True)
        extract_datapoints(table).to_csv(os.path.join(DATAPATH,'datapoints_%d.csv' % variable_idx), index=False)
        variable_idx += 1
    else:
        # has additional dimensions
        for member_combination, table in generate_tables_for_indicator_and_series(row['Indicator'], row['SeriesCode']).items():
            variable = {
                'id': variable_idx,
                'dataset_id': i,
                'unit': row['[Units]'],
                'name': "%s - %s - %s - %s" % (
                    row['Indicator'], 
                    row['SeriesDescription'], 
                    row['SeriesCode'],
                    ' - '.join(map(str, member_combination)))
                
            }
            variables = variables.append(variable, ignore_index=True)
            extract_datapoints(table).to_csv(os.path.join(DATAPATH,'datapoints_%d.csv' % variable_idx), index=False)
            variable_idx += 1

variables.to_csv(os.path.join(OUTPATH,'variables.csv'), index=False)



df_distinct_entities = pd.DataFrame(get_distinct_entities(), columns=['name']) # Goes through each datapoints to get the distinct entities

df_distinct_entities.to_csv(os.path.join(OUTPATH, 'distinct_countries_standardized.csv'), index=False)

#### Helper functions: 

## Not sure how well this works when the list is longer than one
def delete_output(keep_paths: List[str]) -> None:
    for path in keep_paths:
        if os.path.exists(os.path.join(OUTPATH, path)):
            for CleanUp in glob.glob(os.path.join(OUTPATH, '*.*')):
                if not CleanUp.endswith(path):    
                    os.remove(CleanUp)

@functools.lru_cache(maxsize=256)
def get_series_with_relevant_dimensions(indicator, series):
    """ For a given indicator and series, return a tuple:
    
      - data filtered to that indicator and series
      - names of relevant dimensions
      - unique values for each relevant dimension
    """
    data_filtered = original_df[(original_df.Indicator == indicator) & (original_df.SeriesCode == series)]
    non_null_dimensions_columns = [col for col in DIMENSIONS if data_filtered.loc[:, col].notna().any()]
    dimension_names = []
    dimension_unique_values = []
    
    for c in non_null_dimensions_columns:
        print(non_null_dimensions_columns)
        uniques = data_filtered[c].unique()
        if len(uniques) > 1: # Means that columns where the value doesn't change aren't included e.g. Nature is typically consistent across a dimension whereas Age and Sex are less likely to be. 
            dimension_names.append(c)
            dimension_unique_values.append(list(uniques))
    return (data_filtered[NON_DIMENSIONS + dimension_names], dimension_names, dimension_unique_values)

@functools.lru_cache(maxsize=256)
def generate_tables_for_indicator_and_series(indicator, series):
    tables_by_combination = {}
    data_filtered, dimensions, dimension_values = get_series_with_relevant_dimensions(indicator, series)
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
    fnames = [fname for fname in os.listdir(os.path.join(OUTPATH, 'datapoints')) if fname.endswith('.csv')]
    entities = set({})
    for fname in fnames:
        df_temp = pd.read_csv(os.path.join(OUTPATH, 'datapoints', fname))
        entities.update(df_temp['country'].unique().tolist())
    
    entities = list(entities)
    assert pd.notnull(entities).all(), (
        "All entities should be non-null. Something went wrong in "
        "`clean_and_create_datapoints()`."
    )
    return entities

if __name__ == '__main__':
    main()