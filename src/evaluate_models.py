import os
import numpy as np
import pandas as pd

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

    return results


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
        )
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


if __name__ == "__main__":
    main()