import os
import pandas as pd

#################
# set constants #
#################

MORPH_LOC = 'dominguez_raw.tsv'
SDSS_LOC = 'sdss_raw.parquet'
DATA_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
MORPH_COLS = [
        'objID',
        'TType'
    ]

#################################
# import the raw morphology CSV #
#################################

print("Loading morphology data from file...")

hrow = 47
morph_raw = pd.read_csv(
        os.path.join(DATA_DIR, MORPH_LOC), 
        sep='|', 
        header=hrow-2, 
        skiprows=[hrow,hrow+1], 
        usecols=MORPH_COLS,
        engine='python'
    )
morph_raw['objID'] = morph_raw['objID'].astype(int)
morph_raw.set_index('objID')

############################
# import the raw SDSS data #
############################

sdss_raw = pd.read_parquet( os.path.join(DATA_DIR, SDSS_LOC) )
sdss_raw.set_index('objID')

##################################
# match morphology & SDSS tables #
##################################

print("Joining morphology and SDSS tables...")

joined = morph_raw.join(sdss_raw, how='left', lsuffix='_morph', rsuffix='_sdss')

########################
# save the joined data #
########################

print("Saving joined data...")

os.makedirs(DATA_DIR, exist_ok=True)
joined.to_parquet( os.path.join(DATA_DIR, PROCESSED_NAME) )
