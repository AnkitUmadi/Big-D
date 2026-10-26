"""
Responsible AI Bias Mitigation Suite for Clinical Risk Models
Implements three principled mitigation strategies:
1. Pre-Processing: Demographic Reweighing
2. Calibration-Aware Post-Processing: Subgroup-Specific Isotonic Calibration
3. Threshold-Optimization Post-Processing: Fairlearn ThresholdOptimizer for Equal Opportunity

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from fairlearn.postprocessing import ThresholdOptimizer
from lightgbm import LGBMClassifier


def compute_sample_weights_reweighing(y_train, sensitive_series):
    """
    Computes sample weights using demographic reweighing:
    W(S = s, Y = y) = [ P(S = s) * P(Y = y) ] / P(S = s, Y = y )
    Eliminates statistical dependence between sensitive attribute S and label Y.
    """
    n_samples = len(y_train)
    weights = np.zeros(n_samples)

    groups = sensitive_series.unique()
    labels = np.unique(y_train)

    # Probabilities
    p_y = {y: np.mean(y_train == y) for y in labels}
    p_s = {s: np.mean(sensitive_series == s) for s in groups}

    for s in groups:
        for y in labels:
            idx = (sensitive_series == s) & (y_train == y)
            n_sy = np.sum(idx)
            if n_sy > 0:
                p_sy = n_sy / n_samples
                w = (p_s[s] * p_y[y]) / p_sy
                weights[idx] = w
            else:
                weights[idx] = 1.0

    return weights


class SubgroupCalibrator:
    """
    Fits separate Isotonic Regression calibrators per demographic subgroup.
    Ensures that calibrated predicted probabilities align with true empirical risk
    across Age, Sex, and Race groups.
    """
    def __init__(self, fallback_calibrator=True):
        self.calibrators = {}
        self.global_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fallback = fallback_calibrator

    def fit(self, y_prob, y_true, group_series):
        # Fit global calibrator first
        self.global_calibrator.fit(y_prob, y_true)

        for g in group_series.unique():
            idx = (group_series == g)
            if np.sum(idx) >= 30 and len(np.unique(y_true[idx])) > 1:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(y_prob[idx], y_true[idx])
                self.calibrators[g] = iso
            else:
                self.calibrators[g] = self.global_calibrator
        return self

    def predict_proba(self, y_prob, group_series):
        calibrated = np.zeros_like(y_prob)
        for g in group_series.unique():
            idx = (group_series == g)
            if np.sum(idx) > 0:
                calibrator = self.calibrators.get(g, self.global_calibrator)
                calibrated[idx] = calibrator.predict(y_prob[idx])
        return np.clip(calibrated, 0.001, 0.999)


def train_mitigated_models(data_dict, baseline_lgb, sensitive_col="race", model_save_dir="models/saved"):
    """
    Trains and evaluates mitigated versions of the LightGBM model:
    1. Reweighted LightGBM (Pre-processing)
    2. Subgroup-Calibrated LightGBM (Post-processing calibration)
    3. Fairlearn Threshold-Optimized LightGBM (Post-processing Equal Opportunity)
    """
    os.makedirs(model_save_dir, exist_ok=True)
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    sens_train = data_dict["sens_train"]
    sens_test = data_dict["sens_test"]

    print(f"\n[*] Applying Bias Mitigation targeting sensitive attribute: [{sensitive_col}]")

    # Strategy 1: Pre-processing Fair Reweighing
    print("    [1/3] Training Pre-processing Fair Reweighted LightGBM...")
    train_weights = compute_sample_weights_reweighing(y_train, sens_train[sensitive_col])
    
    lgb_reweighted = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=24,
        random_state=42,
        verbose=-1
    )
    lgb_reweighted.fit(X_train, y_train, sample_weight=train_weights)

    prob_reweighted_test = lgb_reweighted.predict_proba(X_test)[:, 1]
    pred_reweighted_test = (prob_reweighted_test >= 0.5).astype(int)

    # Strategy 2: Post-processing Subgroup Isotonic Calibration
    print("    [2/3] Fitting Post-processing Subgroup-Specific Calibrator...")
    prob_base_train = baseline_lgb.predict_proba(X_train)[:, 1]
    prob_base_test = baseline_lgb.predict_proba(X_test)[:, 1]

    subgroup_cal = SubgroupCalibrator()
    subgroup_cal.fit(prob_base_train, y_train, sens_train[sensitive_col])
    prob_calibrated_test = subgroup_cal.predict_proba(prob_base_test, sens_test[sensitive_col])
    pred_calibrated_test = (prob_calibrated_test >= 0.5).astype(int)

    # Strategy 3: Post-processing Fairlearn ThresholdOptimizer
    print("    [3/3] Optimizing Decision Thresholds for Equal Opportunity (Fairlearn)...")
    threshold_opt = ThresholdOptimizer(
        estimator=baseline_lgb,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba"
    )
    threshold_opt.fit(X_train, y_train, sensitive_features=sens_train[sensitive_col])
    pred_thresh_test = threshold_opt.predict(X_test, sensitive_features=sens_test[sensitive_col])

    # Assemble prediction dictionaries
    mitigated_predictions = {
        "Baseline": {
            "y_prob": prob_base_test,
            "y_pred": (prob_base_test >= 0.5).astype(int)
        },
        "Reweighted": {
            "y_prob": prob_reweighted_test,
            "y_pred": pred_reweighted_test
        },
        "SubgroupCalibrated": {
            "y_prob": prob_calibrated_test,
            "y_pred": pred_calibrated_test
        },
        "ThresholdOptimized": {
            "y_prob": prob_base_test,
            "y_pred": pred_thresh_test
        }
    }

    # Save artifacts
    joblib.dump(lgb_reweighted, os.path.join(model_save_dir, "lgb_reweighted.joblib"))
    joblib.dump(subgroup_cal, os.path.join(model_save_dir, "subgroup_calibrator.joblib"))
    joblib.dump(threshold_opt, os.path.join(model_save_dir, "threshold_optimizer.joblib"))

    print("[OK] Mitigation complete and artifacts saved.")
    return mitigated_predictions
