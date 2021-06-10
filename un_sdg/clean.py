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
import numpy as np
import re

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict

#from db import connection
#from db_utils import DBUtils
from un_sdg import (
    INFILE,
    ENTFILE,
    DATA_PATH,
    DATASET_NAME,
    DATASET_AUTHORS,
    DATASET_VERSION
)

from un_sdg.core import (
    create_short_unit,
    extract_datapoints,
    get_distinct_entities,
    clean_datasets,
    dimensions_description,
    attributes_description,
    create_short_unit,
    get_series_with_relevant_dimensions,
    generate_tables_for_indicator_and_series,
    str_to_float, 
    extract_description
)

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

def load_and_clean():
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
    # Make the datapoints folder
    Path(DATA_PATH, 'datapoints').mkdir(parents=True, exist_ok=True)
    return original_df

### Datasets
def create_datasets():
    df_datasets = clean_datasets(DATASET_NAME, DATASET_AUTHORS, DATASET_VERSION)
    assert df_datasets.shape[0] == 1, f"Only expected one dataset in {os.path.join(DATA_PATH, 'datasets.csv')}."
    df_datasets.to_csv(os.path.join(DATA_PATH, 'datasets.csv'), index=False)
    return df_datasets

### Sources

def create_sources(original_df, df_datasets):
    df_sources = pd.DataFrame(columns=['id', 'name', 'description', 'dataset_id'])
    source_description_template = {
        'dataPublishedBy': "United Nations Statistics Division",
        'dataPublisherSource': None,
        'link': "https://unstats.un.org/sdgs/indicators/database/",
        'retrievedDate': datetime.now().strftime("%d-%B-%y"),
        'additionalInfo': None
    }
    #all_series = original_df[['SeriesCode', 'SeriesDescription', '[Units]']]   .groupby(by=['SeriesCode', 'SeriesDescription', '[Units]'])   .count()   .reset_index()
    all_series = original_df[['SeriesCode', 'SeriesDescription', '[Units]']]   .groupby(by=['SeriesCode', 'SeriesDescription', '[Units]'])   .count()   .reset_index()
    source_description = source_description_template.copy()
    for i, row in tqdm(all_series.iterrows(), total=len(all_series)):
        dp_source = original_df[original_df.SeriesCode == row['SeriesCode']].Source.drop_duplicates()
        if len(dp_source) <= 2:
            source_description['dataPublisherSource'] = dp_source.str.cat(sep='; ')
        else: 
            source_description['dataPublisherSource'] = 'Data from multiple sources compiled by UN Global SDG Database - https://unstats.un.org/sdgs/indicators/database/'    
        print(source_description['dataPublisherSource'])   
        try:
            source_description['additionalInfo'] = None
        except:
            pass
        df_sources = df_sources.append({
            'id': i,
            #'name': "%s (UN SDG, 2021)" % row['Source'],
            'name': "%s (UN SDG, 2021)" % row['SeriesDescription'],
            'description': json.dumps(source_description),
            'dataset_id': df_datasets.iloc[0]['id'], # this may need to be more flexible! 
            'series_code': row['SeriesCode']
        }, ignore_index=True)
    df_sources.to_csv(os.path.join(DATA_PATH, 'sources.csv'), index=False)
    
### Variables

def create_variables_datapoints(original_df):
    variable_idx = 0
    variables = pd.DataFrame(columns=['id', 'name', 'unit', 'dataset_id', 'source_id'])
    
    new_columns = [] 
    for k in original_df.columns:
        new_columns.append(re.sub(r"[\[\]]", '',k))

    original_df.columns = new_columns

    entity2owid_name = pd.read_csv(os.path.join(DATA_PATH, 'standardized_entity_names.csv')) \
                              .set_index('country_code') \
                              .squeeze() \
                              .to_dict()

    series2source_id = pd.read_csv(os.path.join(DATA_PATH, 'sources.csv'))\
                            .drop(['name','description', 'dataset_id'], 1)\
                            .set_index('series_code')\
                            .squeeze() \
                            .to_dict()
 
    unit_description = attributes_description()

    dim_description = dimensions_description()

    original_df['country'] = original_df['GeoAreaName'].apply(lambda x: entity2owid_name[x])
    original_df['Units_long'] = original_df['Units'].apply(lambda x: unit_description[x])

    DIMENSIONS = tuple(dim_description.id.unique())
    NON_DIMENSIONS = tuple([c for c in original_df.columns if c not in set(DIMENSIONS)])# not sure if units should be in here
    
    all_series = original_df[['Indicator', 'SeriesCode', 'SeriesDescription', 'Units_long']]   .groupby(by=['Indicator', 'SeriesCode', 'SeriesDescription', 'Units_long'])   .count()   .reset_index()
    all_series = create_short_unit(all_series)

    for i, row in tqdm(all_series.iterrows(), total=len(all_series)): 
        data_filtered =  pd.DataFrame(original_df[(original_df.Indicator == row['Indicator']) & (original_df.SeriesCode == row['SeriesCode'])])
        _, dimensions, dimension_members = get_series_with_relevant_dimensions(data_filtered, DIMENSIONS, NON_DIMENSIONS)
        print(i)
        if len(dimensions) == 0|(data_filtered[dimensions].isna().sum().sum() > 0):
            # no additional dimensions
            table = generate_tables_for_indicator_and_series(data_filtered, DIMENSIONS, NON_DIMENSIONS)
            variable = {
                'dataset_id': 0,
                'source_id': series2source_id[row['SeriesCode']],
                'id': variable_idx,
                'name': "%s - %s - %s" % (row['Indicator'], row['SeriesDescription'], row['SeriesCode']),
                'description': None,
                'code': row['SeriesCode'],
                'unit': row['Units_long'],
                'short_unit': row['short_unit'],
                'timespan': "%s - %s" % (int(np.min(data_filtered['TimePeriod'])), int(np.max(data_filtered['TimePeriod']))),
                'coverage': None,
                'display': None,
                'original_metadata': None
            }
            variables = variables.append(variable, ignore_index=True)
            extract_datapoints(table).to_csv(os.path.join(DATA_PATH,'datapoints','datapoints_%d.csv' % variable_idx), index=False)
            variable_idx += 1
        else:
        # has additional dimensions
            for member_combination, table in generate_tables_for_indicator_and_series(data_filtered, DIMENSIONS, NON_DIMENSIONS).items():
                variable = {
                    'dataset_id': 0,
                    'source_id': series2source_id[row['SeriesCode']],
                    'id': variable_idx,
                    'name': "%s - %s - %s - %s" % (
                        row['Indicator'], 
                        row['SeriesDescription'], 
                        row['SeriesCode'],
                        ' - '.join(map(str, member_combination))),
                    'description': None,
                    'code': row['SeriesCode'],
                    'unit': row['Units_long'],
                    'short_unit': row['short_unit'],
                    'timespan': "%s - %s" % (int(np.min(data_filtered['TimePeriod'])), int(np.max(data_filtered['TimePeriod']))),
                    'coverage': None,
                    'display': None,
                    'original_metadata': None  
                }
                print(member_combination)
                variables = variables.append(variable, ignore_index=True)
                extract_datapoints(table).to_csv(os.path.join(DATA_PATH,'datapoints','datapoints_%d.csv' % variable_idx), index=False)
                variable_idx += 1
                print(table)
    variables.to_csv(os.path.join(DATA_PATH,'variables.csv'), index=False)

def create_distinct_entities(): 
    df_distinct_entities = pd.DataFrame(get_distinct_entities(), columns=['name']) # Goes through each datapoints to get the distinct entities
    df_distinct_entities.to_csv(os.path.join(DATA_PATH, 'distinct_countries_standardized.csv'), index=False)




# Max length of source name.
MAX_SOURCE_NAME_LEN = 256


def main():
    original_df = load_and_clean() 
    df_datasets = create_datasets()
    create_sources(original_df, df_datasets)
    create_variables_datapoints(original_df) #numexpr can't be installed for this function to work - need to formalise this somehow
    create_distinct_entities()

if __name__ == '__main__':
    main()