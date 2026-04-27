import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# Supervised model evaluation pipeline.

# This file contains the supervised classification models used to predict
# galaxy morphology from the cleaned tabular feature set.

# Models included in this file:
# - Logistic Regression baseline
# - Gradient Boosting classifier
# - KNN elbow method for selecting k
# - Optimized KNN classifier
# - Weighted KNN classifier
# - Random Forest classifier

# This file also owns the shared evaluation logic for supervised models,
# including train/test splitting, feature scaling where needed, RMSE,
# accuracy, balanced accuracy, precision, recall, F1 score, confusion
# matrix output, and final best-model selection.
# -----------------------------------------------------------------------------

DATA_DIR = "../data/"
DATA_FILE = "data_full_cleaned.parquet"
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


def evaluate_model(model_name, model, X_test, y_test):
    predictions = model.predict(X_test)

    # encode labels for RMSE
    y_test_encoded = (y_test == "spiral").astype(int)
    predictions_encoded = (predictions == "spiral").astype(int)

    rmse = np.sqrt(mean_squared_error(y_test_encoded, predictions_encoded))

    results = {
        "model": model_name,
        "rmse": rmse,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spiral"),
        "recall": recall_score(y_test, predictions, pos_label="spiral"),
        "f1": f1_score(y_test, predictions, pos_label="spiral"),
    }

    print(f"\n===== {model_name} =====")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    os.makedirs("../results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax,
    )
    ax.set_title(f"{model_name} Confusion Matrix")

    safe_name = model_name.lower().replace(" ", "_")
    plot_path = f"../results/{safe_name}_confusion_matrix.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix plot to {plot_path}")

    return results

def save_metric_comparison_plot(results_df):
    metrics_to_plot = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    plot_df = results_df[["model"] + metrics_to_plot].set_index("model")

    os.makedirs("../results", exist_ok=True)

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Metric Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.tight_layout()

    plot_path = "../results/model_metric_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Saved metric comparison plot to {plot_path}")    

def main():
    X, y = load_data()

    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("Class counts:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        model_results = evaluate_model(model_name, model, X_test, y_test)
        results.append(model_results)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    print("\n===== Model Comparison =====")
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]

    print("\n===== Best Model =====")
    print(f"Best model by F1 score: {best_model['model']}")
    print(f"F1 score: {best_model['f1']:.4f}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Balanced accuracy: {best_model['balanced_accuracy']:.4f}")
    print(f"RMSE: {best_model['rmse']:.4f}")

    os.makedirs("../results", exist_ok=True)
    results_df.to_csv("../results/model_comparison_results.csv", index=False)
    print("\nSaved model comparison results to ../results/model_comparison_results.csv")

    save_metric_comparison_plot(results_df)


if __name__ == "__main__":
    main()