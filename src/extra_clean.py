import pandas as pd
import os
from sqlalchemy import create_engine

#################
# set constants #
#################

# data directory
DATA_DIR = '../data/'
# where is the data?
DATA_NAME = 'data_full_cleaned.parquet'
# where should we put the extra cleaned data from this script?
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

data = pd.read_parquet(os.path.join(DATA_DIR, DATA_NAME))

###############################
# perform some cleaning steps #
###############################

# make magnitude cut
data = data[data['r']<17.]

#############
# save data #
#############

data.to_parquet(os.path.join(DATA_DIR, DATA_NAME))
write_df_to_postgres(data, POSTGRES_TABLE)
