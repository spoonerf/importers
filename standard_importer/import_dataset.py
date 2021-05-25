"""Imports a dataset and associated data sources, variables, and data points
into the SQL database.

Usage:

    python -m standard_importer.import_dataset
"""

import re
import json
from glob import glob
import sys
import os

from tqdm import tqdm
import pandas as pd

sys.path.append("/mnt/importers/scripts/importers")
from db import connection
from db_utils import DBUtils
from utils import import_from

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#DATASET_DIR = "who_gho"
#DATASET_VERSION = import_from(DATASET_DIR, 'DATASET_VERSION')

#USER_ID = 46

#CURRENT_DIR = os.path.dirname(__file__)
# CURRENT_DIR = os.path.join(os.getcwd(), 'standard_importer')
#DATA_PATH = os.path.join(CURRENT_DIR, f"../{DATASET_DIR}/output/")


def main(DATASET_DIR, DATA_PATH, DATASET_VERSION, USER_ID):

    with connection.cursor() as cursor:
        db = DBUtils(cursor)


        # Upsert entities
        print("---\nUpserting entities...")
        entities = pd.read_csv(os.path.join(DATA_PATH, "distinct_countries_standardized.csv"))
        for entity_name in tqdm(entities.name):
            db_entity_id = db.get_or_create_entity(entity_name)
            entities.loc[entities.name == entity_name, "db_entity_id"] = db_entity_id
        print(f"Upserted {len(entities)} entities.")


        # Upsert datasets
        print("---\nUpserting datasets...")
        datasets = pd.read_csv(os.path.join(DATA_PATH, "datasets.csv"))
        for i, dataset_row in tqdm(datasets.iterrows()):
            db_dataset_id = db.upsert_dataset(
                name=dataset_row["name"],
                namespace=f"{DATASET_DIR.split('/')[-1]}@{DATASET_VERSION}",
                user_id=USER_ID
            )
            datasets.at[i, "db_dataset_id"] = db_dataset_id
        print(f"Upserted {len(datasets)} datasets.")


        # Upsert sources
        print("---\nUpserting sources...")
        sources = pd.read_csv(os.path.join(DATA_PATH, "sources.csv"))
        sources = pd.merge(sources, datasets, left_on="dataset_id", right_on="id", suffixes=['__source', '__dataset'])
        for i, source_row in tqdm(sources.iterrows()):
            db_source_id = db.upsert_source(
                name=source_row.name__source,
                description=source_row.description,
                dataset_id=source_row.db_dataset_id
            )
            sources.at[i, "db_source_id"] = db_source_id
        print(f"Upserted {len(sources)} sources.")


        # Upsert variables
        print("---\nUpserting variables...")
        variables = pd.read_csv(os.path.join(DATA_PATH, "variables.csv"))
        variables = variables.fillna("")
        if 'notes' in variables:
            logger.warning(
                'The "notes" column in `variables.csv` is '
                'deprecated, and should be named "description" instead.'
            )
            variables.rename(columns={'notes': 'description'}, inplace=True)
        if 'source_id' in variables:
            on = 'source_id'
        else:
            on = 'dataset_id'
        variables = pd.merge(
            variables, sources, left_on=on, right_on='id__source', how='left', 
            validate='m:1', suffixes=['__variable', '__source']
        )
        for i, variable_row in tqdm(variables.iterrows()):
            db_variable_id = db.upsert_variable(
                name=variable_row["name"],
                source_id=variable_row["db_source_id"],
                dataset_id=variable_row["db_dataset_id"],
                description=variable_row["description__variable"],
                code=variable_row["code"] if "code" in variable_row else "",
                unit=variable_row["unit"] if "unit" in variable_row else None,
                short_unit=variable_row["short_unit"] if "short_unit" in variable_row else None,
                timespan=variable_row["timespan"] if "timespan" in variable_row else "",
                coverage=variable_row["coverage"] if "coverage" in variable_row else "",
                display=variable_row["display"] if "display" in variable_row else None,
                original_metadata=variable_row["original_metadata"] if "original_metadata" in variable_row else None
            )
            variables.at[i, "db_variable_id"] = db_variable_id
        print(f"Upserted {len(variables)} variables.")


        # Upserting datapoints
        print("---\nUpserting datapoints...")
        datapoint_files = glob(os.path.join(DATA_PATH, "datapoints/datapoints_*.csv"))
        for datapoint_file in tqdm(datapoint_files):
            variable_id = int(re.search("\\d+", datapoint_file)[0])
            db_variable_id = variables[variables['id'] == variable_id]["db_variable_id"]
            data = pd.read_csv(datapoint_file)
            data = pd.merge(
                data, entities, left_on="country", right_on="name", how='left',
                validate='m:1'
            )
            data_tuples = zip(
                data["value"],
                data["year"].astype(int),
                data["db_entity_id"].astype(int),
                [int(db_variable_id)] * len(data)
            )
            query = f"""
                INSERT INTO data_values
                    (value, year, entityId, variableId)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    value = VALUES(value),
                    year = VALUES(year),
                    entityId = VALUES(entityId),
                    variableId = VALUES(variableId)
            """
            db.upsert_many(query, data_tuples)
        print(f"Upserted {len(datapoint_files)} datapoint files.")


if __name__ == "__main__":
    main()
