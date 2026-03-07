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

DEVAC_LOC = '../data/devac_raw.tsv'
PROCESSED_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
DEVAC_COLS = [
        '_RAJ2000',
        '_DEJ2000',
        'name',
        'type',
        'T',
        'e_T',
        'D25',
        'e_D25',
        'R25',
        'e_R25',
        'BT',
        'e_BT',
        'cz',
        'e_cz'
    ]

#####################################
# import the raw de Vaucouleurs CSV #
#####################################

print("Loading de Vaucouleurs data from file...")

hrow = 94
devac_raw = pd.read_csv(
        DEVAC_LOC, 
        sep='|', 
        header=hrow-2, 
        skiprows=[hrow,hrow+1], 
        usecols=DEVAC_COLS,
        skipfooter=13,
        engine='python'
    )

##################
# clean the data #
##################

print("Loading de Vaucouleurs data...")

# make a new dataframe for cleaned data
devac_cleaned = devac_raw.copy()

################################
# query SDSS for matching data #
################################
# (or load preexisting)

print("Attempting to locate saved SDSS data...")

if not os.path.exists( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') ):
    print("No saved SDSS data; loading SDSS data from servers...")
    qres = []
    rastep = 2
    for ra in tqdm(np.arange(0.,360.,rastep)):
        rdf = SDSS.query_sql(f"""
            SELECT objid, ra, dec, u, g, r, i, z, petroMag_r, petroR50_r
            FROM PhotoObj
            WHERE
                type = 3
                AND r < 10.
                AND ra > {ra}
                AND ra < {ra+rastep}
        """).to_pandas()
        qres.append(rdf)
    sdss_raw = pd.concat(qres, ignore_index=True)
    sdss_raw.to_parquet( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') )
else:
    print("Found saved SDSS data; loading from file...")
    sdss_raw = pd.read_parquet( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') )

######################################
# match de Vaucouleurs & SDSS tables #
######################################

# convert coordinates to skycoord objects for easy matching 
## de Vaucouleurs
print("Converting de Vaucouleurs coordinates for matching")
devac_coords = SkyCoord(ra=tqdm(devac_cleaned['_RAJ2000']), dec=devac_cleaned['_DEJ2000'], unit='deg')
## SDSS
print("Converting SDSS coordinates for matching")
sdss_cleaned = sdss_raw.copy()
sdss_coords = SkyCoord(ra=tqdm(sdss_cleaned['ra']), dec=tqdm(sdss_cleaned['dec']), unit='deg')

print("Matching SDSS data to de Vaucouleurs data...")

# get SDSS indices that match to de Vaucouleurs objects
sidx, dist, _ = devac_coords.match_to_catalog_sky(sdss_coords)
for ii in range(len(sidx)):
    if dist[ii] > Angle(5., unit=u.arcsec):
        sidx[ii] = -1
devac_cleaned['sidx'] = sidx
print(devac_cleaned['sidx'].value_counts())

# join tables
joined_cleaned = devac_cleaned.join(sdss_cleaned, on='sidx', how='left')
print(joined_cleaned)

##################################
# save the cleaned, matched data #
##################################

print("Saving cleaned, matched data...")

os.makedirs(PROCESSED_DIR, exist_ok=True)
joined_cleaned.to_parquet( os.path.join(PROCESSED_DIR, PROCESSED_NAME) )
