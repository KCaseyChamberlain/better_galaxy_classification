# Better Galaxy Classification Using ML



## Getting started
If you've just downloaded this codebase, the first step is to gather the data (it is not fully located in the repository since it's too big). Please see the following steps to collect the data:

1. Download the raw HyperLEDA data from [this](https://drive.google.com/file/d/1AefO001XVp6XCR9nvx0bXuXvkaR-xSpI/view?usp=sharing) drive link and put it (do not alter the filename) in the `data/` directory in this repository.
2. Run `clean_and_load_data.sh`. This will take up to an hour depending on how responsive the SDSS servers are. It runs a few python scripts to gather raw data from the SDSS servers, match this to the HyperLEDA and de Vaucouleurs data, then clean the data.

These steps will put a file called `data_full_cleaned_noimpute.parquet` (or `galaxy_data_cleaned_noimpute` in Postgres) into a parquet file in the `data/` folder or into the Postgres database (see below). This is the cleaned dataset to be used. (There are some other intermediate files also dumped into the `data/` directory.)

### Optional PostgreSQL export

By default, this project writes cleaned outputs to parquet files in the `data/` directory.

If you have PostgreSQL installed locally, `clean_data.py` can also write the cleaned datasets to PostgreSQL as tables. This is optional and is only used when a database connection is provided.

Before running `clean_data.py`, set a `DATABASE_URL` environment variable with your local PostgreSQL connection string. For a default local PostgreSQL setup, it will usually look like this:

```bash
export DATABASE_URL='postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/postgres'
```

## Description of features in the dataset
- `objname`: name of the galaxy, if any (probably just a catalog name in most cases)
- `T`: number relating to galaxy morphology (more negative means more elliptical, more positive means more spiral). This is the thing we are trying to predict.
- `ra`, `dec`: galaxy sky coordinates (stands for right ascension and declination, just to know where the galaxy is located in the sky). Not useful for our model, but just in case we want to know where an object is on the sky.
- `sb50_r`: mean surface brightness in the central part of the galaxy. Surface brightness is basically a "brightness density." This could be a good feature for ML.
- `sb_conc_r`: surface brightness concentration. A higher number means the light is more evenly spread across the galaxy, while a lower number means it is more concentrated in the center. This could be a good feature for ML.
- `logdc`: (log) central surface brightness. This could be a good feature for ML.
- `bri25`: the surface brightness along the outer edge of the galaxy. This could be a good feature for ML.
- `mabs`: the intrinsic brightness (a.k.a., absolute magnitude) of the galaxy as a whole. This could be a good feature for ML. Note that this number is the inverse of how you would expect it: higher numbers are fainter (e.g., magnitude 20 is fainter than magnitude 15).
- `\*\_color`: in general, color is quantified by taking the difference in galaxy brightness between two filters. The SDSS filters are called "u", "g", "r", "i", and "z". So all these colors are just the difference between two filters. It is always the redder filter minus the bluer filter, so a more positive number means redder, and more negative number means bluer. These could be good features for ML, and some colors might have more predictive power than others.
- `imputed_\*\`: a 1 indicates that the column on that row was imputed. A 0 means it is actual data.
