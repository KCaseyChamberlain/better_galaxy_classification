import os
import pandas as pd
import pandasql as ps
from astroquery.sdss import SDSS
from tqdm import tqdm

#################
# set constants #
#################

DATA_DIR = '../data/'
OUT_NAME = 'sdss_raw.parquet'

######################
# query SDSS servers #
######################

# for all galaxies brighter than rmag 19, query in ra bins to not upset servers

if not os.path.exists( os.path.join(DATA_DIR, OUT_NAME) ):
    print("Querying SDSS servers...")
    qres = []
    rastep = 2
    for ra in tqdm(np.arange(0.,360.,rastep)):
        rdf = SDSS.query_sql(f"""
            SELECT objid, ra, dec, u, g, r, i, z, petroMag_r, petroR50_r
            FROM PhotoObj
            WHERE
                type = 3
                AND r < 19.
                AND ra > {ra}
                AND ra < {ra+rastep}
        """).to_pandas()
        qres.append(rdf)
    sdss_raw = pd.concat(qres, ignore_index=True)
    sdss_raw = sdss_raw.rename(columns={'objid': 'objID'})
    sdss_raw.to_parquet( os.path.join(DATA_DIR, OUT_NAME) )
else:
    print("SDSS data has already been downloaded...")

