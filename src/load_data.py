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
PROCESSED_NAME = 'cleaned_data_full.parquet'
LEDA_COLS = [
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

# log progress
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

# log progress
print("Cleaning HyperLEDA data...")

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

# convert coordinates to astropy objects
leda_coords = SkyCoord(ra=leda_cleaned['_RAJ2000'], dec=leda_cleaned['_DEJ2000'], unit='deg')

################################
# query SDSS for matching data #
################################

# log progress
print("Loading SDSS data...")

if not os.path.exists( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') ):
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
    sdss_raw = pd.read_parquet( os.path.join(PROCESSED_DIR, 'sdss_raw.parquet') )

# convert coordinates to astropy objects
sdss_cleaned = sdss_raw.copy()
sdss_coords = SkyCoord(ra=tqdm(sdss_cleaned['ra']), dec=sdss_cleaned['dec'], unit='deg')

#################################
# match HyperLEDA & SDSS tables #
#################################

# log progress
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

##################################
# save the cleaned, matched data #
##################################

# log progress
print("Saving cleaned, matched data...")

# TODO: remove skycoord object after matching

os.makedirs(PROCESSED_DIR, exist_ok=True)
joined_cleaned.to_parquet( os.path.join(PROCESSED_DIR, PROCESSED_NAME) )
