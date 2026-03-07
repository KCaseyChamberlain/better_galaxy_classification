import os
import pandas as pd
import pandasql as ps
import numpy as np
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from tqdm import tqdm

#################
# set constants #
#################

PROCESSED_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
FLOAT_COLS = [
            'Pdisk',
            'Pedgeon',
            'PbarGZ2',
            'PbarNair10',
            'Pmerg',
            'Pbulge',
            'Pcigar',
            'TType',
            'PS0',
            'ra',
            'dec',
            'u',
            'g',
            'r',
            'i',
            'z',
            'petroMag_r',
            'petroR50_r'
        ]

###############
# import data #
###############

data = pd.read_parquet( os.path.join(PROCESSED_DIR, PROCESSED_NAME) )

#################################
# perform some general cleaning #
#################################

# rename objID column to something more reasonable
data = data.rename(columns={'objID_morph': 'objID'})

# cast some columns as more reasonable types
data['objID'] = data['objID'].astype(int)
for col in FLOAT_COLS:
    data[col] = data[col].astype(float)

########################
# derive some features #
########################



print(data)
