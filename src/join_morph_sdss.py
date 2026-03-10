import os
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from tqdm import tqdm

#################
# set constants #
#################

MORPH_LOC = 'devac_raw.tsv'
SDSS_LOC = 'sdss_raw.parquet'
DATA_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
MORPH_COLS = [
        '_RAJ2000',
        '_DEJ2000',
        'T'
    ]

#################################
# import the raw morphology CSV #
#################################

print("Loading morphology data from file...")

hrow = 94
morph_raw = pd.read_csv(
        os.path.join(DATA_DIR, MORPH_LOC),
        sep='|',
        header=hrow-2,
        skiprows=[hrow,hrow+1],
        usecols=MORPH_COLS,
        skipfooter=13,
        engine='python'
    )

############################
# import the raw SDSS data #
############################

sdss_raw = pd.read_parquet( os.path.join(DATA_DIR, SDSS_LOC) )

##################################
# match morphology & SDSS tables #
##################################

# convert coordinates to skycoord objects for easy matching 
## de Vaucouleurs
print("Converting de Vaucouleurs coordinates for matching")
devac_coords = SkyCoord(ra=tqdm(morph_raw['_RAJ2000']), dec=morph_raw['_DEJ2000'], unit='deg')
## SDSS
print("Converting SDSS coordinates for matching")
sdss_coords = SkyCoord(ra=tqdm(sdss_raw['ra']), dec=tqdm(sdss_raw['dec']), unit='deg')

print("Matching SDSS data to de Vaucouleurs data...")

# get SDSS indices that match to de Vaucouleurs objects
sidx, dist, _ = devac_coords.match_to_catalog_sky(sdss_coords)
for ii in range(len(sidx)):
    if dist[ii] > Angle(5., unit=u.arcsec):
        sidx[ii] = -1
morph_raw['sidx'] = sidx

# join tables
joined = morph_raw.join(sdss_raw, on='sidx', how='left')

########################
# save the joined data #
########################

print("Saving joined data...")

os.makedirs(DATA_DIR, exist_ok=True)
joined.to_parquet( os.path.join(DATA_DIR, PROCESSED_NAME) )
