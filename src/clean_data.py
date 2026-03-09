import os
import pandas as pd
import numpy as np

#################
# set constants #
#################

DATA_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
PROCESSED_CLEANED_NAME = 'data_full_cleaned.parquet'
FLOAT_COLS = [
            'TType',
            'ra',
            'dec',
            'u',
            'g',
            'r',
            'i',
            'z',
            'petroMag_r',
            'petroR50_r',
            'petroR90_r'
        ]
COLS_TO_DROP = [  # these columns are not physical, so they obviously will not relate to morphology
            'petroR50_r',
            'petroR90_r',
            'petroMag_r',
            'u',
            'g',
            'r',
            'i',
            'z'
        ]

###############
# import data #
###############

data = pd.read_parquet( os.path.join(DATA_DIR, PROCESSED_NAME) )

#################################
# perform some general cleaning #
#################################

print("Cleaning some features...")

# rename objID column to something more reasonable
data = data.rename(columns={'objID_morph': 'objID'})

# drop any and all NaN
data.dropna(inplace=True)

# cast some columns as more reasonable types
data['objID'] = data['objID'].astype(int)
for col in FLOAT_COLS:
    data[col] = data[col].astype(float)

########################
# derive some features #
########################

print("Deriving physical features...")

# calculate surface brightness
petroMag_r_half = data['petroMag_r'] + 0.7562
half_area_r = np.pi * data['petroR50_r']**2
data['sb50_r'] = petroMag_r_half + 2.512*np.log10(half_area_r)

# calculate surface brightness concentration
data['sb_conc_r'] = data['petroR90_r']/data['petroR50_r']

# calculate colors
data['ug_color'] = data['u'] - data['g']
data['ur_color'] = data['u'] - data['r']
data['ui_color'] = data['u'] - data['i']
data['uz_color'] = data['u'] - data['z']
data['gr_color'] = data['g'] - data['r']
data['gi_color'] = data['g'] - data['i']
data['gz_color'] = data['g'] - data['z']
data['ri_color'] = data['r'] - data['i']
data['rz_color'] = data['r'] - data['z']
data['iz_color'] = data['i'] - data['z']

#############################################
# remove features that are no longer needed #
#############################################

data = data.drop(columns=COLS_TO_DROP)

######################
# save cleaned table #
######################

data.to_parquet( os.path.join(DATA_DIR, PROCESSED_CLEANED_NAME) )
