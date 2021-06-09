import os
import pandas as pd
import os
from datetime import datetime
import numpy as np

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