# Better Galaxy Classification Using ML



## Getting started
To download, join, and clean the data all in one go, you can run clean_and_load_data.sh. It might take a few minutes.

I store the data in 'parquet' files which are just like self-containd databases that can be read by pandas. You can see how to read the data in the sample_vis.ipynb notebook.

## Optional PostgreSQL export

By default, this project writes cleaned outputs to parquet files in the `data/` directory.

If you have PostgreSQL installed locally, `clean_data.py` can also write the cleaned datasets to PostgreSQL as tables. This is optional and is only used when a database connection is provided.

Before running `clean_data.py`, set a `DATABASE_URL` environment variable with your local PostgreSQL connection string. For a default local PostgreSQL setup, it will usually look like this:

```bash
export DATABASE_URL='postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/postgres'
```

## Features
Here are what the features of the cleaned dataset mean:
- objID: SDSS identifier
- TType: number relating to galaxy morphology (more negative means more elliptical, more positive means more spiral)
- ra, dec: galaxy sky coordinates (stands for right ascension and declination, just to know where the galaxy is located in the sky)
- sb50_r: mean surface brightness in the central part of the galaxy. Surface brightness is basically a "brightness density."
- sb_conc_r: surface brightness concentration. A higher number means the light is more evenly spread across the galaxy, while a lower number means it is more concentrated in the center.
- \*\_color: in general, color is quantified by taking the difference in galaxy brightness between two filters. The SDSS filters are calle "u", "g", "r", "i", and "z". So all these colors are just the difference between two filters. It is always the redder filter minus the bluer filter, so a more positive number means redder, and more negative number means bluer.
