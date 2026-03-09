# Better Galaxy Classification Using ML



## Getting started
To download, join, and clean the data all in one go, you can run clean_and_load_data.sh. It might take a few minutes.

I store the data in 'parquet' files which are just like self-containd databases that can be read by pandas. You can see how to read the data in the sample_vis.ipynb notebook.

## Features
Here are what the features of the cleaned dataset mean:
- objID: SDSS identifier
- TType: number relating to galaxy morphology (more negative means more elliptical, more positive means more spiral)
- ra, dec: galaxy sky coordinates (stands for right ascension and declination, just to know where the galaxy is located in the sky)
- sb50_r: mean surface brightness in the central part of the galaxy. Surface brightness is basically a "brightness density."
- sb_conc_r: surface brightness concentration. A higher number means the light is more evenly spread across the galaxy, while a lower number means it is more concentrated in the center.
- \*\_color: in general, color is quantified by taking the difference in galaxy brightness between two filters. The SDSS filters are calle "u", "g", "r", "i", and "z". So all these colors are just the difference between two filters. It is always the redder filter minus the bluer filter, so a more positive number means redder, and more negative number means bluer.
