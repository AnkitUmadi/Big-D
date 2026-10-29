"""
Baseline Healthcare Readmission Risk Models
Trains clinical risk prediction models (LightGBM, Logistic Regression, Random Forest)
on patient-level partitioned and standardized features.

Enforces patient-level train/validation/test split independence (zero patient overlap)
and fits preprocessing strictly on training data.

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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, recall_score, precision_score
)
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


def prepare_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits cohort into Train, Validation, and Test sets by unique PATIENT IDENTIFIER.
    Ensures zero patient leakage across sets, stratifies patient-level readmission status,
    and fits feature scaling strictly on the training set.
    """
    # 1. Stratify unique patients by outcome
    patient_summary = df.groupby("patient_id").agg({
        TARGET: "max"
    }).reset_index()

    # Split off test patients
    train_val_pats, test_pats = train_test_split(
        patient_summary["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=patient_summary[TARGET]
    )

    # From train_val, split off validation patients
    val_ratio_in_dev = val_size / (1.0 - test_size)
    dev_summary = patient_summary[patient_summary["patient_id"].isin(train_val_pats)]
    train_pats, val_pats = train_test_split(
        dev_summary["patient_id"],
        test_size=val_ratio_in_dev,
        random_state=random_state,
        stratify=dev_summary[TARGET]
    )

    # 2. Slice encounters by patient sets
    train_mask = df["patient_id"].isin(train_pats)
    val_mask = df["patient_id"].isin(val_pats)
    test_mask = df["patient_id"].isin(test_pats)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    X_train_raw = df_train[FEATURES].copy()
    X_val_raw = df_val[FEATURES].copy()
    X_test_raw = df_test[FEATURES].copy()

    y_train = df_train[TARGET].values.astype(int)
    y_val = df_val[TARGET].values.astype(int)
    y_test = df_test[TARGET].values.astype(int)

    # 3. Fit scaler strictly on training set
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=FEATURES, index=df_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=FEATURES, index=df_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=FEATURES, index=df_test.index)

    # 4. Sensitive attributes with composite intersectional column
    def make_sens_df(data_slice):
        s_df = data_slice[SENSITIVE_ATTRS + ["patient_id", "encounter_id"]].copy()
        s_df["intersectional_group"] = (
            s_df["age_group"].astype(str) + " | " +
            s_df["race"].astype(str) + " | " +
            s_df["gender"].astype(str)
        )
        return s_df

    sens_train = make_sens_df(df_train)
    sens_val = make_sens_df(df_val)
    sens_test = make_sens_df(df_test)

    print(f"[*] Patient-Level Split Summary:")
    print(f"    - Training Set:   {len(train_pats):,} patients, {len(df_train):,} encounters (Readm Rate: {y_train.mean():.2%})")
    print(f"    - Validation Set: {len(val_pats):,} patients, {len(df_val):,} encounters (Readm Rate: {y_val.mean():.2%})")
    print(f"    - Test Set:       {len(test_pats):,} patients, {len(df_test):,} encounters (Readm Rate: {y_test.mean():.2%})")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "sens_train": sens_train,
        "sens_val": sens_val,
        "sens_test": sens_test,
        "scaler": scaler,
        "train_patients": set(train_pats),
        "val_patients": set(val_pats),
        "test_patients": set(test_pats)
    }


def train_baseline_models(data_dict, model_save_dir="models/saved"):
    """
    Trains baseline healthcare risk models on X_train:
    - LightGBM (Non-linear gradient boosted tree)
    - Logistic Regression (Interpretable clinical standard)
    - Random Forest (Bagged decision ensemble)

    Evaluates models on held-out test data.
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

    # Evaluate test performance
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
        tpr = recall_score(y_test, y_pred, zero_division=0)
        ppv = precision_score(y_test, y_pred, zero_division=0)

        overall_metrics.append({
            "model": name,
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "sensitivity_tpr": round(tpr, 4),
            "precision_ppv": round(ppv, 4)
        })

        pred_df = data_dict["sens_test"].copy()
        pred_df["y_true"] = y_test
        pred_df["y_prob"] = y_prob
        pred_df["y_pred"] = y_pred
        predictions_dict[name] = pred_df

        joblib.dump(model, os.path.join(model_save_dir, f"{name.lower()}_baseline.joblib"))

    df_metrics = pd.DataFrame(overall_metrics)
    print("\n[+] Overall Test Performance Summary (Patient-Split Held-Out):")
    print(df_metrics.to_string(index=False))

    return models, predictions_dict, df_metrics
