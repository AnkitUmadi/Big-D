"""
Baseline Healthcare Readmission Risk Models
Trains clinical risk prediction models (LightGBM, XGBoost, Logistic Regression)
on the processed analytical cohort.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, f1_score
from lightgbm import LGBMClassifier


FEATURES = [
    "age",
    "length_of_stay",
    "num_prior_inpatient",
    "num_prior_emergency",
    "num_prior_ambulatory",
    "comorbidity_count",
    "has_diabetes",
    "has_hypertension",
    "has_heart_failure",
    "has_copd",
    "has_ckd",
    "has_cad",
    "has_depression",
    "total_medications",
    "total_procedures",
    "total_claim_cost"
]

SENSITIVE_ATTRS = ["age_group", "gender", "race", "ethnicity"]
TARGET = "readmission_30d"


def prepare_data(df, test_size=0.25, random_state=42):
    """Splits dataset into train and test sets preserving subgroup indices."""
    X = df[FEATURES].copy()
    y = df[TARGET].values
    sensitive_data = df[SENSITIVE_ATTRS].copy()

    # Create composite intersectional column
    sensitive_data["intersectional_group"] = (
        sensitive_data["age_group"].astype(str) + " | " + 
        sensitive_data["race"].astype(str) + " | " + 
        sensitive_data["gender"].astype(str)
    )

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_data, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "sens_train": sens_train,
        "sens_test": sens_test,
        "scaler": scaler
    }


def train_baseline_models(data_dict, model_save_dir="models/saved"):
    """
    Trains baseline models:
    - LightGBM (Non-linear high capacity gradient booster)
    - Logistic Regression (Standard interpretable clinical benchmark)
    - Random Forest
    """
    os.makedirs(model_save_dir, exist_ok=True)
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    print("[*] Training Baseline Healthcare Risk Models...")

    # 1. LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=24,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # 2. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # 3. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    models = {
        "LightGBM": lgb_model,
        "LogisticRegression": lr_model,
        "RandomForest": rf_model
    }

    # Evaluate overall metrics on test set
    overall_metrics = []
    predictions_dict = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        overall_metrics.append({
            "model": name,
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4)
        })

        # Save predictions for test set audit
        pred_df = data_dict["sens_test"].copy()
        pred_df["y_true"] = y_test
        pred_df["y_prob"] = y_prob
        pred_df["y_pred"] = y_pred
        predictions_dict[name] = pred_df

        # Save model artifact
        joblib.dump(model, os.path.join(model_save_dir, f"{name.lower()}_baseline.joblib"))

    df_metrics = pd.DataFrame(overall_metrics)
    print("\n[+] Overall Test Performance Summary:")
    print(df_metrics.to_string(index=False))

    return models, predictions_dict, df_metrics
