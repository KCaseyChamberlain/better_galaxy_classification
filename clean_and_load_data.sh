#!/bin/bash

cd src
python get_sdss_data.py
python join_morph_sdss.py
python clean_data.py
python evaluate_models.py
python cluster_analysis.py
cd ..