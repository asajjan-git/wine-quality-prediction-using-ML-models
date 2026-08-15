"""
Wine Quality Prediction Using Machine Learning
------------------------------------------------
Predicts the sensory quality score of white "Vinho Verde" wine from 11
physicochemical measurements, comparing Multiple Linear Regression,
Random Forest, and XGBoost.

Dataset: UCI Wine Quality (white wine subset), Cortez et al. (2009).

Usage:
    python wine_quality_analysis.py
"""
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "winequality-white.csv")
FIG_DIR = os.path.join(HERE, "..", "figures")
RESULTS_DIR = os.path.join(HERE, "..", "results")


def load_data() -> pd.DataFrame:
    """Load the dataset, downloading it from UCI if not already present locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found locally, downloading from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=";")
    return df


def explore(df: pd.DataFrame):
    os.makedirs(FIG_DIR, exist_ok=True)

    print("\n=== Dataset shape ===")
    print(df.shape)

    print("\n=== Missing values ===")
    print(df.isna().sum().sum(), "missing values found")

    print("\n=== Descriptive statistics ===")
    print(df.describe().T.round(3))

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap="RdBu", center=0, square=True)
    plt.title("Correlation Heatmap of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"\nSaved correlation heatmap -> {FIG_DIR}/correlation_heatmap.png")

    print("\nCorrelation with quality (sorted):")
    print(corr["quality"].sort_values(ascending=False))


def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate(name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    return {"Model": name, "RMSE": rmse, "R-squared": r2, "MAE": mae}


def plot_feature_importance(name, model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title(f"{name} — Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    fname = f"feature_importance_{name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved {name} feature importance -> {FIG_DIR}/{fname}")


def plot_model_comparison(results_df: pd.DataFrame):
    metrics = ["RMSE", "R-squared", "MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = {"Multiple Linear Regression": "#E07A5F",
               "Random Forest": "#3D9970",
               "XGBoost": "#4C72B0"}

    for ax, metric in zip(axes, metrics):
        colors = [palette.get(m, "#888") for m in results_df["Model"]]
        ax.bar(results_df["Model"], results_df[metric], color=colors)
        ax.set_title(metric)
        ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")

    plt.suptitle("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"\nSaved model comparison chart -> {FIG_DIR}/model_comparison.png")


def main():
    df = load_data()
    explore(df)

    X_train, X_test, X_train_s, X_test_s, y_train, y_test = preprocess(df)

    results = []

    # --- Multiple Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    results.append(evaluate("Multiple Linear Regression", y_test, pred_lr))

    print("\nLinear Regression coefficients:")
    print(pd.Series(lr.coef_, index=X_train.columns).sort_values(ascending=False))

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=500, max_features=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # tree ensembles don't need scaling
    pred_rf = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, pred_rf))
    plot_feature_importance("Random Forest", rf, X_train.columns)

    # --- XGBoost ---
    xgb = XGBRegressor(
        max_depth=6, learning_rate=0.1, n_estimators=100,
        random_state=RANDOM_STATE, objective="reg:squarederror"
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    results.append(evaluate("XGBoost", y_test, pred_xgb))
    plot_feature_importance("XGBoost", xgb, X_train.columns)

    # --- Compare & save ---
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== Final Comparison ===")
    print(results_df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    print(f"\nSaved metrics -> {RESULTS_DIR}/metrics_summary.csv")

    plot_model_comparison(results_df)


if __name__ == "__main__":
    main()
