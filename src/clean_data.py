import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

#################
# set constants #
#################

DATA_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
PROCESSED_CLEANED_NAME = 'data_full_cleaned_noimpute.parquet'
PROCESSED_CLEANED_NAME_IMPUTED = 'data_full_cleaned_impute.parquet'
FLOAT_COLS = [
            'T',
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
            '_RAJ2000',
            '_DEJ2000',
            'sidx',
            'petroR50_r',
            'petroR90_r',
            'petroMag_r',
            'u',
            'g',
            'r',
            'i',
            'z'
        ]
OUT_COLS = ['T', 'ug_color', 'ur_color', 'ui_color', 'uz_color', 'gr_color', 'gi_color', 'gz_color', 'ri_color', 'rz_color', 'iz_color', 'sb50_r', 'sb_conc_r']

###############
# import data #
###############

data = pd.read_parquet( os.path.join(DATA_DIR, PROCESSED_NAME) )

##############################
# cast some columns as float #
##############################

print("Cleaning the data...")

for col in FLOAT_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")

###################
# data imputation #
###################

COLS_TO_IMPUTE = ['T', 'u', 'g', 'r', 'i', 'z', 'petroMag_r', 'petroR50_r', 'petroR90_r']

data['imputed_brightness'] = data['z'].isna().astype(int)
imputer = KNNImputer(n_neighbors=50, weights='uniform')
data[COLS_TO_IMPUTE] = imputer.fit_transform(data[COLS_TO_IMPUTE])

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

# calculate type
spel = np.array(['spiral']*len(data['T']))
spel[ data['T'] < 0. ] = 'elliptical'
data['morphology'] = spel

###################
# remove outliers #
###################

for cc in OUT_COLS:
    nonan = np.array(data[cc])[~np.isnan(np.array(data[cc]))]
    fq = np.quantile(nonan, 0.25)
    tq = np.quantile(nonan, 0.75)
    iqr = tq - fq
    ubound = tq + 1.5*iqr
    lbound = fq - 1.5*iqr
    data = data[ data[cc]<ubound ]
    data = data[ data[cc]>lbound ]

#############################################
# remove features that are no longer needed #
#############################################

data = data.drop(columns=COLS_TO_DROP)

##############################
# save cleaned imputed table #
##############################

data.to_parquet( os.path.join(DATA_DIR, PROCESSED_CLEANED_NAME_IMPUTED) )

######################
# save cleaned table #
######################

data = data[ data['imputed_brightness']==0 ]
data.to_parquet( os.path.join(DATA_DIR, PROCESSED_CLEANED_NAME) )
