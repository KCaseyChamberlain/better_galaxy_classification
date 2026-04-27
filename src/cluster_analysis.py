# -----------------------------------------------------------------------------
# Unsupervised clustering analysis pipeline.

# This file contains the unsupervised learning work used to explore whether
# the cleaned galaxy feature space naturally forms morphology-like groups
# without using the morphology labels during training.

# Models and analysis included in this file:
# - K-Means clustering
# - K-Means elbow analysis, if used
# - PCA projection for visualization, if used
# - Cluster-to-morphology comparison after clustering

# This file is separate from evaluate_models.py because K-Means is an
# unsupervised method. It does not train directly on the spiral/elliptical
# labels. Instead, the labels are used afterward only to interpret how well
# the discovered clusters align with known morphology classes.
# -----------------------------------------------------------------------------