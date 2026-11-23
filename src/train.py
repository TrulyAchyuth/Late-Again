"""
Trains the two models with a time-based split.

train: Jan-Sep 2023, validation: Oct-Dec 2023 (to pick the model), test: 2024.
1. major-delay classifier: will the next bus be 15+ minutes late?
2. delay regressor: how many minutes late?
Both are compared with simple baselines, and an ablation retrains them without
some inputs to see what those inputs add.

Run: python src/train.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import (EVENT_FEATURES, INCIDENT_CATEGORICAL, WEATHER_FEATURES,
                    INCIDENT_FEATURES, INCIDENT_NUMERIC, INCIDENTS_FILE, MAJOR_DELAY_MIN,
                    METRICS_FILE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TRAIN_END, VALID_END)

RS = RANDOM_STATE


# 
# Helpers
# 
def time_split(df, time_col):
    """Split rows by date, never randomly: the model must predict the future from the past."""
    t = pd.to_datetime(df[time_col])
    train = df[t <= TRAIN_END]
    valid = df[(t > TRAIN_END) & (t <= VALID_END)]
    test = df[t > VALID_END]
    return train, valid, test


def make_pipeline(model, categorical, numeric):
    """One-hot encode text columns (rare values grouped as 'infrequent'), scale numbers."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                              sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", prep), ("model", clone(model))])   # clone = fresh, unfitted copy


def classification_scores(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba) if proba is not None else None}


def regression_scores(y, pred):
    return {"mae": mean_absolute_error(y, pred), "r2": r2_score(y, pred)}


def rounded(d):
    return {k: (round(float(v), 4) if v is not None else None) for k, v in d.items()}


# 
# Baselines (no machine learning)
# 
class IncidentTypeBaseline:
    """Predict the training-set median delay of the incident type (a sensible 'rule of thumb')."""

    def fit(self, X, y):
        self.medians = y.groupby(X["incident"]).median()
        self.overall = float(y.median())
        return self

    def predict(self, X):
        return X["incident"].map(self.medians).fillna(self.overall).to_numpy()


# 
# 1 + 2: incident models
# 
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=20, max_depth=18,
                                            n_jobs=-1, random_state=RS),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=RS),
}
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=50, random_state=RS),
    "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=20, max_depth=18,
                                           n_jobs=-1, random_state=RS),
    "Gradient Boosting": HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
                                                       learning_rate=0.08, random_state=RS),
}


def train_incident_models(df, log=print):
    train, valid, test = time_split(df, "timestamp")
    full_train = pd.concat([train, valid])
    log(f"Incidents  train {len(train):,} | valid {len(valid):,} | test {len(test):,}")
    X = {k: v[INCIDENT_FEATURES] for k, v in [("train", train), ("valid", valid), ("test", test),
                                                ("full", full_train)]}

    # classification
    yc = {k: v["major_delay"] for k, v in [("train", train), ("valid", valid), ("test", test),
                                            ("full", full_train)]}
    clf_rows = []
    for name, model in CLASSIFIERS.items():
        pipe = make_pipeline(model, INCIDENT_CATEGORICAL, INCIDENT_NUMERIC).fit(X["train"], yc["train"])
        proba = pipe.predict_proba(X["valid"])[:, 1]
        scores = classification_scores(yc["valid"], (proba >= 0.5).astype(int), proba)
        clf_rows.append({"model": name, **rounded(scores)})
        log(f"  [clf] {name:20s} valid ROC-AUC {scores['roc_auc']:.3f}  F1 {scores['f1']:.3f}")
    best_clf_name = max(clf_rows, key=lambda r: r["roc_auc"])["model"]

    # regression
    yr = {k: v["delay_min"] for k, v in [("train", train), ("valid", valid), ("test", test),
                                          ("full", full_train)]}
    reg_rows = []
    for name, model in REGRESSORS.items():
        pipe = make_pipeline(model, INCIDENT_CATEGORICAL, INCIDENT_NUMERIC).fit(X["train"], yr["train"])
        scores = regression_scores(yr["valid"], pipe.predict(X["valid"]))
        reg_rows.append({"model": name, **rounded(scores)})
        log(f"  [reg] {name:20s} valid MAE {scores['mae']:.2f} min")
    best_reg_name = min(reg_rows, key=lambda r: r["mae"])["model"]

    # refit the winners on all of 2023, evaluate ONCE on 2024
    clf = make_pipeline(CLASSIFIERS[best_clf_name], INCIDENT_CATEGORICAL, INCIDENT_NUMERIC).fit(X["full"], yc["full"])
    reg = make_pipeline(REGRESSORS[best_reg_name], INCIDENT_CATEGORICAL, INCIDENT_NUMERIC).fit(X["full"], yr["full"])

    proba = clf.predict_proba(X["test"])[:, 1]
    clf_test = classification_scores(yc["test"], (proba >= 0.5).astype(int), proba)
    reg_pred = reg.predict(X["test"])
    reg_test = regression_scores(yr["test"], reg_pred)

    # Baselines on the same 2024 test set
    majority = int(yc["full"].mode()[0])
    rule = IncidentTypeBaseline().fit(X["full"], yr["full"])
    rule_pred = rule.predict(X["test"])
    rule_major = (rule_pred >= MAJOR_DELAY_MIN).astype(int)
    baselines = {
        "clf_majority": rounded(classification_scores(yc["test"], np.full(len(yc["test"]), majority), None)),
        "clf_incident_rule": rounded(classification_scores(yc["test"], rule_major, rule_pred)),
        "reg_global_median": rounded(regression_scores(yr["test"], np.full(len(yr["test"]), float(yr["full"].median())))),
        "reg_incident_median": rounded(regression_scores(yr["test"], rule_pred)),
    }

    # Which inputs matter? (permutation importance on a 2024 sample)
    sample = X["test"].sample(min(8000, len(X["test"])), random_state=RS)
    imp = permutation_importance(reg, sample, yr["test"].loc[sample.index], n_repeats=3,
                                 random_state=RS, scoring="neg_mean_absolute_error")
    importance = (pd.DataFrame({"feature": INCIDENT_FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))

    # Test predictions for the dashboard's "actual vs predicted" chart
    preds = test[["timestamp", "route", "incident", "weather", "delay_min", "major_delay"]].copy()
    preds["predicted_delay_min"] = reg_pred.round(2)
    preds["major_delay_proba"] = proba.round(4)

    # Ablation: retrain the winners WITHOUT some inputs and see how much worse they get
    groups = {"weather inputs": WEATHER_FEATURES, "event inputs": EVENT_FEATURES,
              "route": ["route"], "incident type": ["incident"]}
    ablation = {}
    for label, drop in groups.items():
        cat = [c for c in INCIDENT_CATEGORICAL if c not in drop]
        num = [c for c in INCIDENT_NUMERIC if c not in drop]
        c_pipe = make_pipeline(CLASSIFIERS[best_clf_name], cat, num).fit(X["full"][cat + num], yc["full"])
        r_pipe = make_pipeline(REGRESSORS[best_reg_name], cat, num).fit(X["full"][cat + num], yr["full"])
        ablation[label] = rounded({
            "roc_auc": roc_auc_score(yc["test"], c_pipe.predict_proba(X["test"][cat + num])[:, 1]),
            "mae": mean_absolute_error(yr["test"], r_pipe.predict(X["test"][cat + num])),
        })
        log(f"  [ablation] without {label}: done")

    results = {
        "ablation": ablation,
        "classifier": {"best": best_clf_name, "validation": clf_rows, "test": rounded(clf_test)},
        "regressor": {"best": best_reg_name, "validation": reg_rows, "test": rounded(reg_test)},
        "baselines": baselines,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
    }
    return clf, reg, rule, results, importance, preds


# 
def main(log=print):
    incidents = pd.read_csv(INCIDENTS_FILE, parse_dates=["timestamp"])
    clf, reg, rule, inc_results, importance, preds = train_incident_models(incidents, log)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "major_delay_classifier.joblib", compress=3)
    joblib.dump(reg, MODELS_DIR / "delay_regressor.joblib", compress=3)
    joblib.dump(rule.medians.to_dict(), MODELS_DIR / "incident_medians.joblib")

    metrics = inc_results
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    preds.to_csv(REPORTS_DIR / "test_predictions.csv.gz", index=False)

    c, r, b, ab = metrics["classifier"], metrics["regressor"], metrics["baselines"], metrics["ablation"]
    log("\n==== 2024 test results ====")
    log(f"Major delay ({c['best']}): ROC-AUC {c['test']['roc_auc']:.3f}, F1 {c['test']['f1']:.3f} "
        f"| incident-type rule: ROC-AUC {b['clf_incident_rule']['roc_auc']:.3f}, F1 {b['clf_incident_rule']['f1']:.3f}")
    log(f"Delay minutes ({r['best']}): MAE {r['test']['mae']:.2f} | global median {b['reg_global_median']['mae']:.2f} "
        f"| incident-type median {b['reg_incident_median']['mae']:.2f}")
    for name, row in ab.items():
        log(f"Without {name:14s}: ROC-AUC {row['roc_auc']:.3f}, MAE {row['mae']:.2f}")


if __name__ == "__main__":
    main()
