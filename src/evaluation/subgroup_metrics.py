"""
Subgroup Fairness & Clinical Performance Evaluation Engine
Computes granular metrics across Age, Sex, and Race groups.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, accuracy_score, f1_score
)


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """Computes standard classification & clinical triage metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Missed High-Risk Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision / Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    acc = (tp + tn) / n if n > 0 else 0.0
    f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0.0

    auc = np.nan
    pr_auc = np.nan
    brier = np.nan
    if y_prob is not None:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

    return {
        "n_samples": n,
        "base_rate": np.mean(y_true),
        "selection_rate": np.mean(y_pred),
        "tpr_sensitivity": tpr,
        "tnr_specificity": tnr,
        "fpr": fpr,
        "fnr_miss_rate": fnr,
        "precision_ppv": ppv,
        "npv": npv,
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier
    }


def evaluate_subgroups(y_true, y_pred, y_prob, group_series, group_name="subgroup", reference_group=None):
    """
    Evaluates model performance and computes fairness disparities for each subgroup.
    """
    groups = sorted(group_series.unique())
    rows = []

    for g in groups:
        idx = (group_series == g).values
        if np.sum(idx) == 0:
            continue
        g_true = y_true[idx]
        g_pred = y_pred[idx]
        g_prob = y_prob[idx] if y_prob is not None else None

        metrics = compute_binary_metrics(g_true, g_pred, g_prob)
        metrics["attribute"] = group_name
        metrics["subgroup"] = str(g)
        rows.append(metrics)

    df_subgroups = pd.DataFrame(rows)

    # Compute disparities relative to reference group
    if reference_group and reference_group in df_subgroups["subgroup"].values:
        ref_row = df_subgroups[df_subgroups["subgroup"] == reference_group].iloc[0]
        df_subgroups["selection_rate_ratio"] = df_subgroups["selection_rate"] / (ref_row["selection_rate"] + 1e-8)
        df_subgroups["tpr_difference"] = df_subgroups["tpr_sensitivity"] - ref_row["tpr_sensitivity"]
        df_subgroups["fpr_difference"] = df_subgroups["fpr"] - ref_row["fpr"]
    else:
        # If no explicit reference, compute range disparity
        max_sel = df_subgroups["selection_rate"].max()
        df_subgroups["selection_rate_ratio"] = df_subgroups["selection_rate"] / (max_sel + 1e-8)
        df_subgroups["tpr_difference"] = df_subgroups["tpr_sensitivity"] - df_subgroups["tpr_sensitivity"].mean()
        df_subgroups["fpr_difference"] = df_subgroups["fpr"] - df_subgroups["fpr"].mean()

    # Reorder columns
    lead_cols = ["attribute", "subgroup", "n_samples", "base_rate", "selection_rate", "roc_auc", "pr_auc", "tpr_sensitivity", "fnr_miss_rate", "fpr", "brier_score"]
    other_cols = [c for c in df_subgroups.columns if c not in lead_cols]
    df_subgroups = df_subgroups[lead_cols + other_cols]

    return df_subgroups


def compute_fairness_summary(df_subgroups):
    """
    Computes overarching Responsible AI fairness metrics:
    - Demographic Parity Difference: max(selection_rate) - min(selection_rate)
    - Demographic Parity Ratio (Disparate Impact): min(selection_rate) / max(selection_rate)
    - Equal Opportunity Difference: max(TPR) - min(TPR)
    - False Positive Rate Difference: max(FPR) - min(FPR)
    """
    sel_rates = df_subgroups["selection_rate"].values
    tprs = df_subgroups["tpr_sensitivity"].values
    fprs = df_subgroups["fpr"].values

    dp_diff = np.max(sel_rates) - np.min(sel_rates)
    dp_ratio = np.min(sel_rates) / (np.max(sel_rates) + 1e-8)
    eo_diff = np.max(tprs) - np.min(tprs)
    fpr_diff = np.max(fprs) - np.min(fprs)

    return {
        "demographic_parity_difference": round(dp_diff, 4),
        "demographic_parity_ratio": round(dp_ratio, 4),
        "equal_opportunity_difference": round(eo_diff, 4),
        "false_positive_rate_difference": round(fpr_diff, 4)
    }
