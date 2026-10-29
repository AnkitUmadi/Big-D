"""
Responsible AI Bias Mitigation Suite for Clinical Risk Models
Implements three principled algorithmic fairness strategies:
1. Pre-Processing: Demographic Sample Reweighing (fitted on Training data)
2. Post-Processing Calibration: Subgroup-Specific Isotonic Calibration (fitted on Validation data)
3. Post-Processing Thresholds: Fairlearn ThresholdOptimizer for Equal Opportunity / Equalized Odds (fitted on Validation data)

Evaluates all mitigated variants on a frozen Held-Out Test Set.
Robust to degenerate single-class subgroups and sparse sample sizes.

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
    Computes demographic reweighing sample weights on training data:
    W(S = s, Y = y) = [ P(S = s) * P(Y = y) ] / P(S = s, Y = y )

    Eliminates statistical dependence between sensitive attribute S and target Y.
    Validates against zero counts, NaNs, and infinite weights.
    """
    y_train = np.asarray(y_train)
    sensitive_series = pd.Series(sensitive_series).reset_index(drop=True)
    n_samples = len(y_train)

    if n_samples == 0:
        return np.ones(0)

    groups = sensitive_series.unique()
    labels = np.unique(y_train)

    p_y = {y: float(np.mean(y_train == y)) for y in labels}
    p_s = {s: float(np.mean(sensitive_series == s)) for s in groups}

    weights = np.ones(n_samples, dtype=float)

    for s in groups:
        for y in labels:
            idx = (sensitive_series == s) & (y_train == y)
            n_sy = int(np.sum(idx))
            if n_sy > 0:
                p_sy = n_sy / n_samples
                w = (p_s[s] * p_y[y]) / p_sy
                weights[idx] = w
            else:
                weights[idx] = 1.0

    # Sanity checks: prevent NaN, inf, or negative weights
    weights = np.nan_to_num(weights, nan=1.0, posinf=10.0, neginf=1.0)
    weights = np.clip(weights, 0.05, 20.0)
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return weights


class SubgroupCalibrator:
    """
    Fits separate Isotonic Regression calibrators per demographic subgroup.
    Fitted strictly on held-out validation predictions to avoid in-sample overfitting.
    Falls back to a global calibrator when subgroup sample size < 20 or single-class.
    """
    def __init__(self, min_samples=20):
        self.min_samples = min_samples
        self.calibrators = {}
        self.global_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_prob, y_true, group_series):
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)
        group_series = pd.Series(group_series).reset_index(drop=True)

        # Fit global calibrator on validation set
        self.global_calibrator.fit(y_prob, y_true)

        for g in group_series.unique():
            idx = (group_series == g).values
            n_g = np.sum(idx)
            # Ensure sufficient subgroup sample size and both classes present
            if n_g >= self.min_samples and len(np.unique(y_true[idx])) > 1:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(y_prob[idx], y_true[idx])
                self.calibrators[g] = iso
            else:
                self.calibrators[g] = self.global_calibrator
        return self

    def predict_proba(self, y_prob, group_series):
        y_prob = np.asarray(y_prob)
        group_series = pd.Series(group_series).reset_index(drop=True)
        calibrated = np.zeros_like(y_prob)

        for g in group_series.unique():
            idx = (group_series == g).values
            if np.sum(idx) > 0:
                calibrator = self.calibrators.get(g, self.global_calibrator)
                calibrated[idx] = calibrator.predict(y_prob[idx])
        return np.clip(calibrated, 0.001, 0.999)


def train_mitigated_models(data_dict, baseline_lgb, sensitive_col="race", model_save_dir="models/saved"):
    """
    Trains and evaluates mitigated versions of the LightGBM model:
    1. Reweighted LightGBM (Pre-processing on Training Set)
    2. Subgroup-Calibrated LightGBM (Post-processing Calibration on Validation Set)
    3. Fairlearn Threshold-Optimized LightGBM (Post-processing on Validation Set for Equal Opportunity)

    All evaluations occur on the frozen test set (X_test, y_test).
    Handles degenerate (single-class) or sparse subgroups gracefully.
    """
    os.makedirs(model_save_dir, exist_ok=True)
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    sens_train = data_dict["sens_train"]
    sens_val = data_dict["sens_val"]
    sens_test = data_dict["sens_test"]

    print(f"\n[*] Applying Bias Mitigation targeting sensitive attribute: [{sensitive_col}]")

    # -------------------------------------------------------------
    # Strategy 1: Pre-processing Fair Sample Reweighing (Train-Only)
    # -------------------------------------------------------------
    print("    [1/3] Training Pre-processing Fair Reweighted LightGBM (weights fit on train)...")
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

    # -------------------------------------------------------------
    # Strategy 2: Post-processing Subgroup Isotonic Calibration (Val-Fit)
    # -------------------------------------------------------------
    print("    [2/3] Fitting Post-processing Subgroup-Specific Calibrator on Validation set...")
    prob_base_val = baseline_lgb.predict_proba(X_val)[:, 1]
    prob_base_test = baseline_lgb.predict_proba(X_test)[:, 1]

    subgroup_cal = SubgroupCalibrator(min_samples=20)
    subgroup_cal.fit(prob_base_val, y_val, sens_val[sensitive_col])

    prob_calibrated_test = subgroup_cal.predict_proba(prob_base_test, sens_test[sensitive_col])
    pred_calibrated_test = (prob_calibrated_test >= 0.5).astype(int)

    # -------------------------------------------------------------
    # Strategy 3: Post-processing Fairlearn ThresholdOptimizer (Val-Fit)
    # -------------------------------------------------------------
    print("    [3/3] Optimizing Decision Thresholds for Equal Opportunity on Validation set...")
    # Fairlearn constraint: 'true_positive_rate_parity' (Equal Opportunity)
    # Handle degenerate subgroups (single-class in validation split) by falling back to reference group
    val_sens_series = sens_val[sensitive_col].reset_index(drop=True)
    test_sens_series = sens_test[sensitive_col].reset_index(drop=True)

    # Determine reference group (most frequent category)
    ref_group = val_sens_series.value_counts().index[0]

    # Detect groups with < 2 classes in validation split
    degenerate_groups = set()
    for g in val_sens_series.unique():
        idx = (val_sens_series == g)
        if len(np.unique(y_val[idx])) < 2 or int(np.sum(idx)) < 5:
            degenerate_groups.add(g)

    if degenerate_groups:
        print(f"    [!] Subgroups {degenerate_groups} have single-class or sparse distribution in validation set; falling back to reference group [{ref_group}] threshold.")

    # Map degenerate groups to reference group for threshold optimization
    sens_val_opt = val_sens_series.apply(lambda g: ref_group if g in degenerate_groups else g)
    sens_test_opt = test_sens_series.apply(lambda g: ref_group if g in degenerate_groups else g)

    threshold_opt = ThresholdOptimizer(
        estimator=baseline_lgb,
        constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba"
    )
    # Fit threshold optimizer strictly on Validation set
    threshold_opt.fit(X_val, y_val, sensitive_features=sens_val_opt)
    # Predict binary decisions on Test set
    pred_thresh_test = threshold_opt.predict(X_test, sensitive_features=sens_test_opt)

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
        "ThresholdOptimized_EO": {
            "y_prob": None,  # ThresholdOptimizer produces binary decisions, not calibrated probabilities
            "y_pred": pred_thresh_test.astype(int)
        }
    }

    # Save artifacts
    joblib.dump(lgb_reweighted, os.path.join(model_save_dir, "lgb_reweighted.joblib"))
    joblib.dump(subgroup_cal, os.path.join(model_save_dir, "subgroup_calibrator.joblib"))
    joblib.dump(threshold_opt, os.path.join(model_save_dir, "threshold_optimizer.joblib"))

    print("[OK] Mitigation complete and artifacts saved.")
    return mitigated_predictions
