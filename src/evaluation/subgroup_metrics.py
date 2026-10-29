"""
Subgroup Fairness & Clinical Performance Evaluation Engine
Computes granular classification, calibration, and algorithmic fairness metrics
across Age, Sex/Gender, Race, Ethnicity, and Intersectional subgroups.

Handles edge cases: empty groups, small sample sizes, single-class subgroups,
zero denominators, and provides confidence intervals and warnings.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, accuracy_score, f1_score
)


def compute_wilson_ci(k, n, confidence=0.95):
    """Computes Wilson score interval for a binomial proportion k / n."""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf((1 + confidence) / 2.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """
    Computes comprehensive clinical classification & triage metrics:
    - Sample count, base rate (prevalence), selection rate
    - Sensitivity / TPR, Specificity / TNR
    - False Positive Rate (FPR), False Negative Rate / Miss Rate (FNR)
    - Precision / PPV, Negative Predictive Value (NPV)
    - Accuracy, F1 Score
    - ROC-AUC, PR-AUC, Brier score (when probabilities provided)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n == 0:
        return {
            "n_samples": 0, "base_rate": np.nan, "selection_rate": np.nan,
            "tpr_sensitivity": np.nan, "tnr_specificity": np.nan, "fpr": np.nan,
            "fnr_miss_rate": np.nan, "precision_ppv": np.nan, "npv": np.nan,
            "accuracy": np.nan, "f1_score": np.nan, "roc_auc": np.nan,
            "pr_auc": np.nan, "brier_score": np.nan
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc = (tp + tn) / n
    f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv is not np.nan and tpr is not np.nan and (ppv + tpr) > 0) else 0.0

    auc = np.nan
    pr_auc = np.nan
    brier = np.nan
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
            except Exception:
                auc = np.nan
                pr_auc = np.nan
        brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "n_samples": n,
        "base_rate": float(np.mean(y_true)),
        "selection_rate": float(np.mean(y_pred)),
        "tpr_sensitivity": float(tpr) if not np.isnan(tpr) else np.nan,
        "tnr_specificity": float(tnr) if not np.isnan(tnr) else np.nan,
        "fpr": float(fpr) if not np.isnan(fpr) else np.nan,
        "fnr_miss_rate": float(fnr) if not np.isnan(fnr) else np.nan,
        "precision_ppv": float(ppv) if not np.isnan(ppv) else np.nan,
        "npv": float(npv) if not np.isnan(npv) else np.nan,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
        "brier_score": float(brier) if not np.isnan(brier) else np.nan
    }


def evaluate_subgroups(y_true, y_pred, y_prob, group_series, group_name="subgroup", reference_group=None):
    """
    Evaluates model performance across subgroups of a single sensitive attribute.
    Computes disparities relative to a reference group or range-based disparities.
    Adds sample size flags and confidence intervals for sensitivity.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob) if y_prob is not None else None
    group_series = pd.Series(group_series).reset_index(drop=True)

    unique_groups = sorted([g for g in group_series.unique() if pd.notna(g)])
    rows = []

    for g in unique_groups:
        idx = (group_series == g).values
        n = np.sum(idx)
        if n == 0:
            continue

        g_true = y_true[idx]
        g_pred = y_pred[idx]
        g_prob = y_prob[idx] if y_prob is not None else None

        metrics = compute_binary_metrics(g_true, g_pred, g_prob)
        metrics["attribute"] = group_name
        metrics["subgroup"] = str(g)
        metrics["small_sample_flag"] = (n < 30)

        # Wilson CI for TPR
        pos_count = int(np.sum(g_true == 1))
        tp_count = int(np.sum((g_true == 1) & (g_pred == 1)))
        ci_low, ci_high = compute_wilson_ci(tp_count, pos_count)
        metrics["tpr_ci_95_low"] = round(ci_low, 4)
        metrics["tpr_ci_95_high"] = round(ci_high, 4)

        rows.append(metrics)

    df_subgroups = pd.DataFrame(rows)
    if df_subgroups.empty:
        return df_subgroups

    # Disparities relative to reference group
    if reference_group and reference_group in df_subgroups["subgroup"].values:
        ref_row = df_subgroups[df_subgroups["subgroup"] == reference_group].iloc[0]
        ref_sel = ref_row["selection_rate"]
        ref_tpr = ref_row["tpr_sensitivity"]
        ref_fpr = ref_row["fpr"]

        df_subgroups["selection_rate_ratio"] = df_subgroups["selection_rate"] / (ref_sel + 1e-8)
        df_subgroups["tpr_difference"] = df_subgroups["tpr_sensitivity"] - ref_tpr
        df_subgroups["fpr_difference"] = df_subgroups["fpr"] - ref_fpr
    else:
        # Range disparity relative to mean or max
        max_sel = df_subgroups["selection_rate"].max()
        df_subgroups["selection_rate_ratio"] = df_subgroups["selection_rate"] / (max_sel + 1e-8)
        df_subgroups["tpr_difference"] = df_subgroups["tpr_sensitivity"] - df_subgroups["tpr_sensitivity"].mean()
        df_subgroups["fpr_difference"] = df_subgroups["fpr"] - df_subgroups["fpr"].mean()

    # Lead columns ordering
    lead_cols = [
        "attribute", "subgroup", "n_samples", "base_rate", "selection_rate",
        "roc_auc", "pr_auc", "tpr_sensitivity", "fnr_miss_rate", "fpr", "brier_score",
        "small_sample_flag"
    ]
    other_cols = [c for c in df_subgroups.columns if c not in lead_cols]
    return df_subgroups[lead_cols + other_cols]


def compute_fairness_summary(df_subgroups, min_samples=0):
    """
    Computes overarching Responsible AI fairness disparity metrics for a SINGLE attribute:
    - Demographic Parity Difference: max(selection_rate) - min(selection_rate)
    - Demographic Parity Ratio: min(selection_rate) / max(selection_rate)
    - Equal Opportunity Difference: max(TPR) - min(TPR)
    - False Positive Rate Difference: max(FPR) - min(FPR)
    - Equalized Odds Disparity: max(EO_diff, FPR_diff)

    If min_samples > 0, filters df_subgroups for cohorts with at least min_samples (e.g. min_samples=30 for robust subsets).
    """
    if df_subgroups.empty:
        return {
            "demographic_parity_difference": 0.0,
            "demographic_parity_ratio": 1.0,
            "equal_opportunity_difference": 0.0,
            "false_positive_rate_difference": 0.0,
            "equalized_odds_disparity": 0.0
        }

    df_eval = df_subgroups
    if min_samples > 0:
        filtered = df_subgroups[df_subgroups["n_samples"] >= min_samples]
        if len(filtered) >= 2:
            df_eval = filtered

    sel_rates = df_eval["selection_rate"].dropna().values
    tprs = df_eval["tpr_sensitivity"].dropna().values
    fprs = df_eval["fpr"].dropna().values

    dp_diff = float(np.max(sel_rates) - np.min(sel_rates)) if len(sel_rates) > 0 else 0.0
    dp_ratio = float(np.min(sel_rates) / (np.max(sel_rates) + 1e-8)) if len(sel_rates) > 0 else 1.0
    eo_diff = float(np.max(tprs) - np.min(tprs)) if len(tprs) > 0 else 0.0
    fpr_diff = float(np.max(fprs) - np.min(fprs)) if len(fprs) > 0 else 0.0
    eq_odds = max(eo_diff, fpr_diff)

    return {
        "demographic_parity_difference": round(dp_diff, 4),
        "demographic_parity_ratio": round(dp_ratio, 4),
        "equal_opportunity_difference": round(eo_diff, 4),
        "false_positive_rate_difference": round(fpr_diff, 4),
        "equalized_odds_disparity": round(eq_odds, 4)
    }
