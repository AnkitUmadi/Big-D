"""
Intersectional Subgroup Audit Engine
Audits clinical performance and calibration at the demographic intersections of Age, Race, and Sex/Gender.

Identifies compounding disparities and vulnerable demographic pockets where models exhibit elevated miss rates.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import numpy as np
import pandas as pd
from src.evaluation.subgroup_metrics import compute_binary_metrics
from src.evaluation.calibration_audit import compute_calibration_metrics


def audit_intersectional_subgroups(y_true, y_pred, y_prob, df_sens, min_cohort_size=15):
    """
    Evaluates compounded disparities across (age_group x race x gender).
    Identifies intersectional pockets with small sample sizes and elevated miss rates.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob) if y_prob is not None else None

    inter_series = (
        df_sens["age_group"].astype(str) + " | " +
        df_sens["race"].astype(str) + " | " +
        df_sens["gender"].astype(str)
    ).reset_index(drop=True)

    unique_inter = sorted(inter_series.unique())
    rows = []

    for group_key in unique_inter:
        idx = (inter_series == group_key).values
        n = int(np.sum(idx))
        if n < min_cohort_size:
            continue

        g_true = y_true[idx]
        g_pred = y_pred[idx]
        g_prob = y_prob[idx] if y_prob is not None else None

        bin_m = compute_binary_metrics(g_true, g_pred, g_prob)
        cal_m = compute_calibration_metrics(g_true, g_prob) if g_prob is not None else {"ece": np.nan, "mce": np.nan}

        parts = group_key.split(" | ")
        rows.append({
            "intersectional_group": group_key,
            "age_group": parts[0],
            "race": parts[1],
            "gender": parts[2],
            "n_samples": n,
            "small_sample_flag": (n < 30),
            "base_rate": round(float(bin_m["base_rate"]), 4),
            "selection_rate": round(float(bin_m["selection_rate"]), 4),
            "roc_auc": round(float(bin_m["roc_auc"]), 4) if not np.isnan(bin_m["roc_auc"]) else np.nan,
            "tpr_sensitivity": round(float(bin_m["tpr_sensitivity"]), 4),
            "fnr_miss_rate": round(float(bin_m["fnr_miss_rate"]), 4),
            "fpr": round(float(bin_m["fpr"]), 4),
            "ece": round(float(cal_m["ece"]), 4) if not np.isnan(cal_m["ece"]) else np.nan,
            "brier_score": round(float(bin_m["brier_score"]), 4) if not np.isnan(bin_m["brier_score"]) else np.nan
        })

    df_inter = pd.DataFrame(rows)
    if not df_inter.empty:
        df_inter = df_inter.sort_values(by="ece", ascending=False).reset_index(drop=True)
    return df_inter
