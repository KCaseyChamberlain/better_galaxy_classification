import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sqlalchemy import create_engine

#################
# set constants #
#################

# data directory
DATA_DIR = '../data/'
# where is the processed/uncleaned data from the other script stored?
PROCESSED_NAME = 'data_full.parquet'
# where to store our cleaned data from this script?
PROCESSED_CLEANED_NAME = 'data_full_cleaned.parquet'
# columns that should be of type float
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
            'petroR90_r',
            'logdc',
            'bri25',
            'mabs'
        ]
# which columns to drop when we are done?
COLS_TO_DROP = [  # these columns are not physical, so they obviously will not relate to morphology
            'objID',
            'ra_leda',
            'dec_leda',
            'ra_devac',
            'dec_devac',
            'leda_idx',
            'devac_idx',
            'objtype',
            'petroR50_r',
            'petroR90_r',
            'petroMag_r',
            'ub_color',
            'bv_color',
            'vmaxg',
            'vmaxs',
            'vdis',
            'u',
            'g',
            'i',
            'z'
        ]
# which columns to exclude outliers?
OUT_COLS = ['T', 'ug_color', 'ur_color', 'ui_color', 'uz_color', 'gr_color', 'gi_color', 'gz_color', 'ri_color', 'rz_color', 'iz_color', 'sb50_r', 'sb_conc_r']
# which columns to impute the data?
COLS_TO_IMPUTE = ['logdc','bri25','mabs']
# table name for PostgreSQL
POSTGRES_TABLE = 'galaxy_data_cleaned'

# default local Postgres connection used when DATABASE_URL is not set
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
)

def write_df_to_postgres(df, table_name):
    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    if db_url == DEFAULT_DATABASE_URL:
        print("DATABASE_URL not set; using default local PostgreSQL connection.")

    try:
        print(f"Writing {len(df)} rows to Postgres table '{table_name}'...")
        engine = create_engine(db_url)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Finished writing table '{table_name}'.")
    except Exception as exc:
        print("PostgreSQL export failed.")
        print("Parquet output was still saved successfully.")
        print(f"Reason: {exc}")

###############
# import data #
###############

data = pd.read_parquet( os.path.join(DATA_DIR, PROCESSED_NAME) )

####################################
# combine morphology label columns #
####################################

data['T_devac'] = pd.to_numeric(data['T_devac'], errors="coerce")
data['T'] = data[['T_leda','T_devac']].mean(axis=1)
data.drop(['T_devac','T_leda'],axis=1,inplace=True)
data.dropna(subset=['T'], inplace=True)

##############################
# cast some columns as float #
##############################

print("Cleaning the data...")

for col in FLOAT_COLS:
    data[col] = pd.to_numeric(data[col], errors="coerce")

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

print("Removing outliers...")

for cc in OUT_COLS:
    fq = np.nanquantile(np.array(data[cc]), 0.25)
    tq = np.nanquantile(np.array(data[cc]), 0.75)
    iqr = tq - fq
    ubound = tq + 1.5*iqr
    lbound = fq - 1.5*iqr
    data = data[ data[cc]<ubound ]
    data = data[ data[cc]>lbound ]

#############################################
# remove features that are no longer needed #
#############################################

data = data.drop(columns=COLS_TO_DROP)

########################
# impute some features #
########################

print("Imputing some of the features...")

# mark which columns have imputed values
for cc in COLS_TO_IMPUTE:
    data['imputed_'+cc] = data[cc].isna().astype(int)

# impute
imputer = KNNImputer(n_neighbors=50, weights='uniform')
data[COLS_TO_IMPUTE] = imputer.fit_transform(data[COLS_TO_IMPUTE])

######################
# save cleaned table #
######################

print("Saving final cleaned data...")

data.to_parquet(os.path.join(DATA_DIR, PROCESSED_CLEANED_NAME))
write_df_to_postgres(data, POSTGRES_TABLE)
