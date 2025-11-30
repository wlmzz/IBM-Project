import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .data_ingestion import load_data, basic_clean
from .logging_utils import logger
from .config import config

def build_pipeline(model):
    categorical_features = ["country"]
    numeric_features = ["tenure", "monthly_charges", "num_streams"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return clf

def train_and_evaluate():
    df = basic_clean(load_data())
    X = df[["country", "tenure", "monthly_charges", "num_streams"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    baseline_model = build_pipeline(LogisticRegression(max_iter=1000))
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_pred)
    logger.info(f"Baseline LogisticRegression AUC: {baseline_auc:.3f}")

    rf_model = build_pipeline(
        RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    logger.info(f"RandomForest AUC: {rf_auc:.3f}")

    best_model, best_auc, label = (
        (rf_model, rf_auc, "RandomForest")
        if rf_auc >= baseline_auc
        else (baseline_model, baseline_auc, "LogisticRegression")
    )

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    joblib.dump(best_model, config.model_path)
    logger.info(f"Saved best model ({label}) with AUC={best_auc:.3f} to {config.model_path}")
    return {"baseline_auc": baseline_auc, "rf_auc": rf_auc, "best_auc": best_auc, "best_label": label}

if __name__ == "__main__":
    train_and_evaluate()
