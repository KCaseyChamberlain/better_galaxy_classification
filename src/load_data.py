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

LEDA_LOC = '../data/leda_raw.tsv'
PROCESSED_DIR = '../data/'
PROCESSED_NAME = 'data_full.parquet'
LEDA_COLS = [
        'PGC',
        '_RAJ2000',
        '_DEJ2000',
        'MType',
        'logD25',
        'e_logD25',
        'logR25',
        'e_logR25'
    ]

################################
# import the raw HyperLEDA CSV #
################################

print("Loading HyperLEDA data from file...")

hrow = 48
leda_raw = pd.read_csv(
        LEDA_LOC, 
        sep='|', 
        header=hrow-2, 
        skiprows=[hrow,hrow+1], 
        usecols=LEDA_COLS,
        skipfooter=13,
        engine='python'
    )

##################
# clean the data #
##################

print("Loading HyperLEDA data...")

# make a new dataframe for cleaned data
leda_cleaned = leda_raw.copy()

# keep only rows with a non-null MType (morphology classification)
blank_mval = (leda_cleaned['MType'].unique())[0]
leda_cleaned.replace(blank_mval, np.nan, inplace=True)
leda_cleaned.dropna(subset=['MType'], inplace=True)

# # keep only rows where target can be observed with SDSS
# leda_cleaned = ps.sqldf("""
#         SELECT *
#         FROM leda_cleaned
#         WHERE _DEJ2000 > 0
# """)

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
            SELECT objid, ra, dec, u, g, r, i, z
            FROM PhotoObj
            WHERE
                type = 3
                AND r < 19.
                AND ra > {ra}
                AND ra < {ra+rastep}
        """).to_pandas()
        qres.append(rdf)
    sdss_raw = pd.concat(qres, ignore_index=True)
    sdss_raw.to_parquet( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') )
else:
    print("Found saved SDSS data; loading from file...")
    sdss_raw = pd.read_parquet( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') )

#################################
# match HyperLEDA & SDSS tables #
#################################

# convert coordinates to skycoord objects for easy matching 
## HyperLEDA
print("Converting HyperLEDA coordinates for matching")
leda_coords = SkyCoord(ra=tqdm(leda_cleaned['_RAJ2000']), dec=leda_cleaned['_DEJ2000'], unit='deg')
## SDSS
print("Converting SDSS coordinates for matching")
sdss_cleaned = sdss_raw.copy()
sdss_coords = SkyCoord(ra=tqdm(sdss_cleaned['ra']), dec=tqdm(sdss_cleaned['dec']), unit='deg')

print("Matching SDSS data to HyperLEDA data...")

# get SDSS indices that match to LEDA objects
sidx, dist, _ = leda_coords.match_to_catalog_sky(sdss_coords)
for ii in range(len(sidx)):
    if dist[ii] > Angle(5., unit=u.arcsec):
        sidx[ii] = -1
leda_cleaned['sidx'] = sidx
print(leda_cleaned['sidx'].value_counts())

# join tables
joined_cleaned = leda_cleaned.join(sdss_cleaned, on='sidx', how='left')
print(joined_cleaned)

##################################
# save the cleaned, matched data #
##################################

print("Saving cleaned, matched data...")

os.makedirs(PROCESSED_DIR, exist_ok=True)
joined_cleaned.to_parquet( os.path.join(PROCESSED_DIR, PROCESSED_NAME) )
