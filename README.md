# Better Galaxy Classification Using ML

This project uses modern galaxy measurements from the Sloan Digital Sky Survey (SDSS) and legacy morphology data from HyperLEDA / de Vaucouleurs catalogs to classify galaxy morphology. The pipeline downloads SDSS data, joins it with morphology catalogs by sky coordinate, cleans and engineers features, exports the cleaned dataset, and runs supervised and unsupervised model analysis.

## Setup and Run

### Prerequisites

Install the following before running the project:

- Anaconda or Miniconda
- PostgreSQL
- Docker, optional depending on local setup

### 1. Download Required Data

The raw HyperLEDA data is too large to store directly in this repository.

Download the raw HyperLEDA data from this Google Drive link:

```text
https://drive.google.com/file/d/1AefO001XVp6XCR9nvx0bXuXvkaR-xSpI/view?usp=sharing
```

Place the file in the `data/` directory.

Do not rename the file.

Expected location:

```text
data/leda_raw.tsv
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate cs_galaxies
```

If the environment already exists and you need to update it:

```bash
conda env update -f environment.yml --prune
conda activate cs_galaxies
```

### 3. Run the Full Pipeline

From the project root, run:

```bash
bash clean_and_load_data.sh
```

This runs the full project pipeline in order:

```text
get_sdss_data.py       -> downloads or reuses SDSS raw data
join_morph_sdss.py     -> joins SDSS with HyperLEDA / de Vaucouleurs data
clean_data.py          -> cleans data, engineers features, saves parquet, exports to PostgreSQL
evaluate_models.py     -> runs supervised model evaluation
cluster_analysis.py    -> runs unsupervised K-Means / PCA analysis
```

If you want to run the steps manually instead:

```bash
cd src
python get_sdss_data.py
python join_morph_sdss.py
python clean_data.py
python evaluate_models.py
python cluster_analysis.py
cd ..
```

### 4. Expected Data Outputs

The main cleaned dataset is saved as:

```text
data/data_full_cleaned.parquet
```

Intermediate data files may also be created in the `data/` directory.

## PostgreSQL Export

The project always writes the cleaned dataset to parquet.

If PostgreSQL is running locally, `clean_data.py` also attempts to export the cleaned dataset to PostgreSQL.

Default local PostgreSQL connection:

```text
postgresql+psycopg2://postgres:postgres@localhost:5432/postgres
```

Default PostgreSQL table:

```text
galaxy_data_cleaned
```

To use a different database connection, set `DATABASE_URL` before running `clean_data.py`.

Linux / macOS:

```bash
export DATABASE_URL='postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/postgres'
```

Windows Command Prompt / Anaconda Prompt:

```bat
set "DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/postgres"
```

If PostgreSQL export fails, the parquet output is still saved.

## Model Evaluation Outputs

Supervised model results from `evaluate_models.py` are saved in:

```text
results/
```

Expected supervised outputs:

```text
results/model_comparison_results.csv
results/logistic_regression_confusion_matrix.png
results/gradient_boosting_confusion_matrix.png
results/random_forest_confusion_matrix.png
results/model_metric_comparison.png
```

The supervised evaluation currently compares:

```text
Logistic Regression
Gradient Boosting
Random Forest
```

Metrics include:

```text
RMSE
accuracy
balanced accuracy
precision
recall
F1 score
confusion matrix
```

## Clustering Outputs

Unsupervised clustering results from `cluster_analysis.py` are saved in:

```text
results/
```

Expected clustering outputs:

```text
results/kmeans_elbow_plot.png
results/kmeans_pca_clusters.png
results/kmeans_cluster_morphology_counts.csv
results/kmeans_cluster_morphology_percentages.csv
```

The clustering analysis uses K-Means with `k=2` as the main comparison because the cleaned morphology labels are binary:

```text
spiral
elliptical
```

The script also runs an elbow analysis over multiple `k` values to check whether the feature space suggests more than two natural clusters.

## Optional Extra Cleaning

`extra_clean.py` is an optional experimental script for applying additional filtering to the already-cleaned dataset.

It is not part of the default pipeline.

Only run it if you intentionally want to overwrite `data/data_full_cleaned.parquet` with a more aggressively filtered version.

Default pipeline:

```bash
cd src
python get_sdss_data.py
python join_morph_sdss.py
python clean_data.py
python evaluate_models.py
python cluster_analysis.py
cd ..
```

Optional extra-cleaning flow:

```bash
cd src
python clean_data.py
python extra_clean.py
python evaluate_models.py
python cluster_analysis.py
cd ..
```

## Description of Features in the Dataset

- `objname`: name of the galaxy, if available.
- `T`: numeric morphology value. More negative values indicate more elliptical galaxies, and more positive values indicate more spiral galaxies.
- `morphology`: binary class label derived from `T`. Values are `spiral` or `ellipt`.
- `ra`, `dec`: sky coordinates, right ascension and declination.
- `sb50_r`: mean surface brightness in the central region of the galaxy.
- `sb_conc_r`: surface brightness concentration.
- `logdc`: log central surface brightness from HyperLEDA.
- `bri25`: surface brightness along the outer edge of the galaxy.
- `mabs`: absolute magnitude / intrinsic brightness. Higher values are fainter.
- `*_color`: color features computed as differences between SDSS filter magnitudes.
- `imputed_*`: indicator columns showing whether a value was imputed. A value of `1` means the original value was missing and imputed; `0` means the value was observed.
