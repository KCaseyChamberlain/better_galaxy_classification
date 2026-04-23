import os
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from tqdm import tqdm

#################
# set constants #
#################

# file locations
DEVAC_LOC = 'devac_raw.tsv'
LEDA_LOC = 'leda_raw.tsv'
SDSS_LOC = 'sdss_raw.parquet'
# data directory
DATA_DIR = '../data/'
# where to put the processed data from this script?
PROCESSED_NAME = 'data_full.parquet'
# columns to include from the de Vaucouleurs dataset
DEVAC_COLS = [
        '_RAJ2000',
        '_DEJ2000',
        'T'
    ]
# columns/indices to include from the HyperLEDA dataset
LEDA_COL_IDX = [0,2,5,6,17,40,41,42,44,46,58,63,80]
LEDA_COL_NAMES = [  # it's hard to read these from the data file so we define manually
        'objname',
        'objtype',
        'ra_leda',
        'dec_leda',
        'T_leda',
        'ub_color',
        'bv_color',
        'vmaxg',
        'vmaxs',
        'vdis',
        'logdc',
        'bri25',
        'mabs'
    ]

#################################
# import the raw morphology CSV #
#################################

print("Loading morphology data from files...")

# de Vaucouleurs
hrow = 94
devac_raw = pd.read_csv(
        os.path.join(DATA_DIR, DEVAC_LOC),
        sep='|',
        header=hrow-2,
        skiprows=[hrow,hrow+1],
        usecols=DEVAC_COLS,
        skipfooter=13,
        engine='python'
    )
## rename some columns to avoid conflict
devac_raw = devac_raw.rename(columns={
    '_RAJ2000' : 'ra_devac',
    '_DEJ2000': 'dec_devac',
    'T': 'T_devac'
})

# LEDA
try:
    leda_raw = pd.read_csv(
            os.path.join(DATA_DIR, LEDA_LOC),
            sep='\t',
            usecols=LEDA_COL_IDX,
            names=LEDA_COL_NAMES,
            skiprows=89
        )
except FileNotFoundError:
    print("LEDA data not found. Please download this as outlined in the README.")
## perform some initial rudimentary cleaning on LEDA data
### normalize objtype
leda_raw['objtype'] = leda_raw['objtype'].map(lambda ss: str(ss))
leda_raw['objtype'] = leda_raw['objtype'].map(lambda ss: ss.replace(" ",""))
leda_raw['objtype'] = leda_raw['objtype'].str.lower()
### select for only galaxies
leda_raw = leda_raw[leda_raw['objtype']=='g']
### drop any that don't have coordinates
leda_raw.dropna(subset=['ra_leda','dec_leda'], inplace=True)
### convert coordinates from hour angle to degree (for consistency)
leda_raw['ra_leda'] = leda_raw['ra_leda']*15.0

############################
# import the raw SDSS data #
############################

try:
    sdss_raw = pd.read_parquet( os.path.join(DATA_DIR, SDSS_LOC) )
except FileNotFoundError:
    print("SDSS data not found. Please run the appropriate script to create this file.")

##################################
# match morphology & SDSS tables #
##################################

# convert coordinates to skycoord objects for easy matching 
## de Vaucouleurs
print("Converting de Vaucouleurs coordinates for matching...")
devac_coords = SkyCoord(ra=tqdm(devac_raw['ra_devac'],desc="ra"), dec=tqdm(devac_raw['dec_devac'],desc="dec"), unit='deg')
## LEDA
print("Converting LEDA coordinates for matching...")
leda_coords = SkyCoord(ra=tqdm(leda_raw['ra_leda'],desc="ra"), dec=tqdm(leda_raw['dec_leda'],desc="dec"), unit='deg')
## SDSS
print("Converting SDSS coordinates for matching...")
sdss_coords = SkyCoord(ra=tqdm(sdss_raw['ra'],desc="ra"), dec=tqdm(sdss_raw['dec'],desc="dec"), unit='deg')

print("Matching SDSS data to de Vaucouleurs data...")

# get LEDA & de Vaucouleurs indices that match to SDSS objects
leda_idx, leda_dist, _ = sdss_coords.match_to_catalog_sky(leda_coords)
devac_idx, devac_dist, _ = sdss_coords.match_to_catalog_sky(devac_coords)

# filter for only close matches
for ii in range(len(leda_idx)):
    if leda_dist[ii] > Angle(5., unit=u.arcsec):
        leda_idx[ii] = int(9e15)
for ii in range(len(devac_idx)):
    if devac_dist[ii] > Angle(5., unit=u.arcsec):
        devac_idx[ii] = int(9e15)

# prepare tables for joining
sdss_raw['leda_idx'] = leda_idx
sdss_raw['devac_idx'] = devac_idx

# join tables
joined = sdss_raw.join(leda_raw, on='leda_idx', how='left', lsuffix='1', rsuffix='2')
joined = joined.join(devac_raw, on='devac_idx', how='left', lsuffix='3', rsuffix='4')
# reject rows that don't have a match in either LEDA or de Vaucouleurs
# (this effectively makes it so the previous left joins become an inner join from SDSS to de Vaucouleurs/LEDA at the same time)
joined = joined[~(joined['ra_devac'].isna() & joined['ra_leda'].isna())]

########################
# save the joined data #
########################

print("Saving joined data...")

os.makedirs(DATA_DIR, exist_ok=True)
joined.to_parquet( os.path.join(DATA_DIR, PROCESSED_NAME) )
