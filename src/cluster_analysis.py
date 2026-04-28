import os

import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="threadpoolctl",
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Unsupervised clustering analysis pipeline.
#
# This file contains the unsupervised learning work used to explore whether
# the cleaned galaxy feature space naturally forms morphology-like groups
# without using the morphology labels during training.
#
# Models and analysis included in this file:
# - K-Means clustering
# - K-Means elbow analysis
# - PCA projection for visualization
# - Cluster-to-morphology comparison after clustering
#
# This file is separate from evaluate_models.py because K-Means is an
# unsupervised method. It does not train directly on the spiral/elliptical
# labels. Instead, the labels are used afterward only to interpret how well
# the discovered clusters align with known morphology classes.
# -----------------------------------------------------------------------------

DATA_DIR = "../data/"
DATA_FILE = "data_full_cleaned.parquet"
RESULTS_DIR = "../results/"
RANDOM_STATE = 42

FEATURE_COLS = [
    "logdc",
    "bri25",
    "mabs",
    "sb50_r",
    "sb_conc_r",
    "ug_color",
    "ur_color",
    "ui_color",
    "uz_color",
    "gr_color",
    "gi_color",
    "gz_color",
    "ri_color",
    "rz_color",
    "iz_color",
    "imputed_logdc",
    "imputed_bri25",
    "imputed_mabs",
]

TARGET_COL = "morphology"


def load_data():
    path = os.path.join(DATA_DIR, DATA_FILE)
    data = pd.read_parquet(path)

    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    return X, y


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def save_kmeans_elbow_plot(X_scaled):
    # We test k=2 through k=6.
    # The main comparison uses k=2 because the cleaned labels are binary:
    # spiral vs elliptical.
    k_values = range(2, 7)
    inertias = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("K-Means Elbow Analysis")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "kmeans_elbow_plot.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Saved K-Means elbow plot to {plot_path}")


def run_kmeans(X_scaled, n_clusters=2):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    clusters = model.fit_predict(X_scaled)

    return model, clusters


def compare_clusters_to_morphology(clusters, y):
    comparison_df = pd.DataFrame(
        {
            "cluster": clusters,
            "morphology": y.values,
        }
    )

    cluster_table = pd.crosstab(
        comparison_df["cluster"],
        comparison_df["morphology"],
    )

    cluster_percentages = pd.crosstab(
        comparison_df["cluster"],
        comparison_df["morphology"],
        normalize="index",
    )

    print("\n===== Cluster to Morphology Counts =====")
    print(cluster_table)

    print("\n===== Cluster to Morphology Percentages =====")
    print(cluster_percentages.round(4))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    counts_path = os.path.join(
        RESULTS_DIR,
        "kmeans_cluster_morphology_counts.csv",
    )
    percentages_path = os.path.join(
        RESULTS_DIR,
        "kmeans_cluster_morphology_percentages.csv",
    )

    cluster_table.to_csv(counts_path)
    cluster_percentages.to_csv(percentages_path)

    print(f"\nSaved cluster counts to {counts_path}")
    print(f"Saved cluster percentages to {percentages_path}")

    # Adjusted Rand Index compares cluster assignments to true labels.
    # 1 means perfect agreement.
    # 0 means roughly random agreement.
    ari = adjusted_rand_score(y, clusters)

    print("\n===== K-Means Label Alignment =====")
    print(f"Adjusted Rand Index: {ari:.4f}")

    return cluster_table, cluster_percentages, ari


def save_pca_cluster_plot(X_scaled, clusters, y):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "pc1": X_pca[:, 0],
            "pc2": X_pca[:, 1],
            "cluster": clusters,
            "morphology": y.values,
        }
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_df["pc1"],
        pca_df["pc2"],
        c=pca_df["cluster"],
        alpha=0.6,
        s=12,
    )

    plt.title("K-Means Clusters Projected with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "kmeans_pca_clusters.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA cluster plot to {plot_path}")


def main():
    X, y = load_data()
    X_scaled = scale_features(X)

    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("Class counts:")
    print(y.value_counts())

    save_kmeans_elbow_plot(X_scaled)

    # k=2 is the main clustering result because the cleaned morphology labels
    # are binary: spiral vs elliptical.
    kmeans_model, clusters = run_kmeans(X_scaled, n_clusters=2)

    print("\n===== K-Means Summary =====")
    print("Number of clusters:", kmeans_model.n_clusters)
    print("Inertia:", round(kmeans_model.inertia_, 4))

    compare_clusters_to_morphology(clusters, y)

    save_pca_cluster_plot(X_scaled, clusters, y)


if __name__ == "__main__":
    main()