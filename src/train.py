"""
train.py
--------
Trains and compares models for two tasks:

    * Classification: will the trip be delayed (> 5 min late)?   target = "delayed"
    * Regression:     how many minutes late will it be?          target = "arrival_delay_min"

For each task we try 4 models, measure them on a held-out test set (20% of
trips the models never saw), and save the best one to models/.

Outputs:
    models/best_classifier.joblib, models/best_regressor.joblib
    models/app_metadata.json                (info the dashboard needs)
    reports/metrics_classification.csv, reports/metrics_regression.csv
    reports/test_predictions.csv            (actual vs predicted, test set)
    reports/figures/09..12_*.png            (model comparison charts)

Run:
    python src/train.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import (CATEGORICAL_FEATURES, CLEAN_DATA, FEATURES, FIGURES_DIR, MODELS_DIR,
                    NUMERIC_FEATURES, RANDOM_STATE, REPORTS_DIR, TARGET_CLASS, TARGET_REG,
                    TEST_SIZE, DELAY_THRESHOLD_MIN)

RS = RANDOM_STATE


def make_preprocessor():
    """Turn text columns into 0/1 columns and put numbers on a similar scale."""
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])


def make_pipeline(model):
    """Preprocessing + model in one object, so new data is treated the same way."""
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
}

REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RS),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
}


def train_classifiers(X_train, X_test, y_train, y_test):
    rows, fitted = [], {}
    for name, model in CLASSIFIERS.items():
        pipe = make_pipeline(model)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "cv_f1 (train, 5-fold)": cv_f1,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        fitted[name] = pipe
        print(f"  {name:20s} test F1 = {rows[-1]['f1']:.3f}")
    table = pd.DataFrame(rows).sort_values("f1", ascending=False).round(4)
    return table, fitted


def train_regressors(X_train, X_test, y_train, y_test):
    rows, fitted = [], {}
    for name, model in REGRESSORS.items():
        pipe = make_pipeline(model)
        cv_mae = -cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "cv_mae (train, 5-fold)": cv_mae,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "r2": r2_score(y_test, pred),
        })
        fitted[name] = pipe
        print(f"  {name:20s} test MAE = {rows[-1]['mae']:.3f} min")
    table = pd.DataFrame(rows).sort_values("mae").round(4)
    return table, fitted


def save_comparison_chart(table, metrics, filename, title):
    long = table.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax)
    ax.set(title=title, xlabel="", ylabel="Score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def main():
    df = pd.read_csv(CLEAN_DATA, keep_default_na=False, na_values=[""])
    X = df[FEATURES]

    # One split for both tasks. "stratify" keeps the same share of delayed trips in each part.
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, df.index, test_size=TEST_SIZE, random_state=RS, stratify=df[TARGET_CLASS])
    print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    # Baselines: what you get without any machine learning
    yc_test = df.loc[idx_test, TARGET_CLASS]
    yr_train, yr_test = df.loc[idx_train, TARGET_REG], df.loc[idx_test, TARGET_REG]
    majority = df.loc[idx_train, TARGET_CLASS].mode()[0]
    baseline = {
        "classification_majority_class_accuracy": round(float((yc_test == majority).mean()), 4),
        "regression_predict_mean_mae": round(float(mean_absolute_error(yr_test, [yr_train.mean()] * len(yr_test))), 4),
    }
    print("Baselines:", baseline)

    print("\nClassification (delayed vs on time)")
    clf_table, clfs = train_classifiers(X_train, X_test, df.loc[idx_train, TARGET_CLASS], yc_test)
    print("\nRegression (delay in minutes)")
    reg_table, regs = train_regressors(X_train, X_test, yr_train, yr_test)

    best_clf_name = clf_table.iloc[0]["model"]
    best_reg_name = reg_table.iloc[0]["model"]
    best_clf, best_reg = clfs[best_clf_name], regs[best_reg_name]
    print(f"\nBest classifier: {best_clf_name}   Best regressor: {best_reg_name}")

    # ---- Save tables ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clf_table.to_csv(REPORTS_DIR / "metrics_classification.csv", index=False)
    reg_table.to_csv(REPORTS_DIR / "metrics_regression.csv", index=False)

    # Actual vs predicted on the test set (used by the dashboard)
    test_pred = df.loc[idx_test, ["trip_id", "transport_type", "route_id", "weather_condition", "event_type"]].copy()
    test_pred["actual"] = yr_test.values
    test_pred["predicted"] = best_reg.predict(X_test).round(2)
    test_pred["actual_delayed"] = yc_test.values
    test_pred["predicted_delayed"] = best_clf.predict(X_test)
    test_pred.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # ---- Charts ----
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_comparison_chart(clf_table, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                          "09_classifier_comparison.png", "Classification models (test set, higher is better)")
    save_comparison_chart(reg_table, ["mae", "rmse"],
                          "10_regressor_comparison.png", "Regression models (test set, lower is better)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(yc_test, test_pred["predicted_delayed"],
                                            display_labels=["On time", "Delayed"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {best_clf_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # Permutation importance: how much worse does the model get if we shuffle one column?
    imp = permutation_importance(best_reg, X_test, yr_test, n_repeats=5, random_state=RS,
                                 scoring="neg_mean_absolute_error")
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=importance, y="feature", x="importance", color="#2a6f97", ax=ax)
    ax.set(title=f"Feature importance ({best_reg_name} regressor)",
           xlabel="Increase in MAE when the feature is shuffled (min)", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_feature_importance.png", dpi=120)
    plt.close(fig)

    # ---- Save models + info the dashboard needs for its input form ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODELS_DIR / "best_classifier.joblib")
    joblib.dump(best_reg, MODELS_DIR / "best_regressor.joblib")
    routes = (df.groupby("route_id")
                .agg(transport_type=("transport_type", "first"),
                     distance_km=("distance_km", "first"),
                     scheduled_duration_min=("scheduled_duration_min", "median"))
                .reset_index())
    metadata = {
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
        "delay_threshold_min": DELAY_THRESHOLD_MIN,
        "baselines": baseline,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "routes": routes.to_dict(orient="records"),
    }
    (MODELS_DIR / "app_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n", clf_table.to_string(index=False))
    print("\n", reg_table.to_string(index=False))
    print("\nSaved models, metrics and charts.")


if __name__ == "__main__":
    main()
