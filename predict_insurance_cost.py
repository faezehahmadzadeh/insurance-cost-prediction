"""
Predict Insurance Cost — a portfolio project by Faezeh Ahmadzadeh

Goal: predict a person's annual medical insurance charges from basic
information (age, sex, BMI, number of children, smoker status, region).

Dataset: Medical Cost Personal Dataset (1,338 people)
Source : https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

What this script does, step by step:
  1. Downloads the data (and saves a local copy in data/insurance.csv)
  2. Prints a small exploratory data analysis (EDA) summary
  3. Prepares the data: one-hot encoding for text columns, scaling for numbers
  4. Trains and compares 5 models:
       - Linear Regression  (simple baseline)
       - Decision Tree      (single tree)
       - Random Forest      (tree ensemble)
       - Gradient Boosting  (strong tree ensemble)
       - K-Nearest Neighbors (distance-based; uses scaled features)
  5. Evaluates each model with RMSE, MAE and R2 on a held-out test set
  6. Saves metrics to outputs/metrics.json and charts to outputs/
"""

import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, we only save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "insurance.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)
RANDOM_STATE = 42

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]


# ---------------------------------------------------------------------------
# 1. Load data (download once, then reuse the local copy)
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        print(f"Downloading data from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {DATA_FILE.name}")
    return df


# ---------------------------------------------------------------------------
# 2. Quick exploratory data analysis
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    print("\n--- EDA summary ---")
    print(df.describe(include="all").round(2).to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nAverage charges by smoker status:")
    print(df.groupby("smoker")[TARGET].agg(["count", "mean", "median"]).round(2).to_string())
    print("\nAverage charges by region:")
    print(df.groupby("region")[TARGET].mean().round(2).to_string())


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
def save_charges_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET], bins=40, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Annual medical charges (USD)")
    ax.set_ylabel("Number of people")
    ax.set_title("Distribution of insurance charges")
    fig.tight_layout()
    path = OUTPUT_DIR / "charges_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_actual_vs_predicted(y_true, y_pred, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, color="#4C72B0")
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    ax.plot([low, high], [low, high], color="#C44E52", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual charges (USD)")
    ax.set_ylabel("Predicted charges (USD)")
    ax.set_title(f"Actual vs predicted — {model_name}")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / f"actual_vs_predicted_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_model_comparison(results: dict) -> Path:
    ordered = sorted(results.items(), key=lambda kv: kv[1]["rmse"])
    names = [name for name, _ in ordered]
    rmses = [metrics["rmse"] for _, metrics in ordered]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], rmses[::-1], color="#4C72B0")
    ax.set_xlabel("RMSE on test set (USD) — lower is better")
    ax.set_title("Model comparison — RMSE by algorithm")
    for i, value in enumerate(rmses[::-1]):
        ax.text(value, i, f" {value:,.0f}", va="center")
    fig.tight_layout()
    path = OUTPUT_DIR / "model_comparison_rmse.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_feature_importance(fitted_pipeline, feature_groups, model_name: str) -> Path | None:
    """Only tree models have feature_importances_."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance — {model_name}")
    fig.tight_layout()
    path = OUTPUT_DIR / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # also print a readable grouped view (one-hot columns grouped back)
    print(f"\nFeature importance ({model_name}, grouped):")
    grouped = {}
    for name, value in zip(feature_names, model.feature_importances_):
        clean = name.split("__", 1)[-1]  # e.g. "smoker_yes" or "bmi"
        if clean.startswith(("sex_", "smoker_", "region_")):
            clean = clean.rsplit("_", 1)[0]
        grouped[clean] = grouped.get(clean, 0.0) + float(value)
    for feature, value in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {feature:12s} {value:.3f}")
    print("\nTop individual features:")
    for feature, value in importance.tail(5).iloc[::-1].items():
        print(f"  {feature:35s} {value:.3f}")
    return path


# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "rmse": round(float(rmse), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    run_eda(df)

    chart_paths = [save_charges_distribution(df)]
    print(f"\nSaved chart: {chart_paths[-1].name}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)} rows | Test size: {len(X_test)} rows")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        # KNN is distance-based, so it relies on the StandardScaler in the
        # preprocessing pipeline to put features on a comparable scale.
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    best_name, best_r2, best_pipeline, best_pred = None, -np.inf, None, None
    print("\n--- Model results on the test set ---")
    print(f"{'Model':20s} {'RMSE ($)':>10s} {'MAE ($)':>10s} {'R2':>8s}")
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)
        results[name] = metrics
        print(f"{name:20s} {metrics['rmse']:10.2f} {metrics['mae']:10.2f} {metrics['r2']:8.4f}")
        if metrics["r2"] > best_r2:
            best_name, best_r2, best_pipeline, best_pred = name, metrics["r2"], pipeline, predictions

    print(f"\nBest model by R2: {best_name}")

    chart_paths.append(save_actual_vs_predicted(y_test, best_pred, best_name))
    print(f"Saved chart: {chart_paths[-1].name}")

    importance_chart = save_feature_importance(best_pipeline, None, best_name)
    if importance_chart is not None:
        chart_paths.append(importance_chart)
        print(f"Saved chart: {importance_chart.name}")

    chart_paths.append(save_model_comparison(results))
    print(f"Saved chart: {chart_paths[-1].name}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_rows": int(len(df)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "random_state": RANDOM_STATE,
                "best_model": best_name,
                "models": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
