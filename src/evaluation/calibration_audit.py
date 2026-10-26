"""
Subgroup Calibration Audit Engine for Clinical Risk Models
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Brier Scores, and generates subgroup reliability diagrams.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import numpy as np
import pandas as pd


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and reliability diagram bin statistics using equal-width probability bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_samples = len(y_true)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = (bin_indices == i)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_conf = np.mean(y_prob[mask])
            bin_acc = np.mean(y_true[mask])
            abs_err = np.abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * abs_err
            mce = max(mce, abs_err)

            bin_stats.append({
                "bin_idx": i,
                "bin_lower": bins[i],
                "bin_upper": bins[i+1],
                "bin_center": (bins[i] + bins[i+1]) / 2.0,
                "bin_count": bin_count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "calibration_error": abs_err
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "bin_lower": bins[i],
                "bin_upper": bins[i+1],
                "bin_center": (bins[i] + bins[i+1]) / 2.0,
                "bin_count": 0,
                "confidence": (bins[i] + bins[i+1]) / 2.0,
                "accuracy": np.nan,
                "calibration_error": 0.0
            })

    brier = np.mean((y_prob - y_true) ** 2)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "bin_stats": pd.DataFrame(bin_stats)
    }


def audit_subgroup_calibration(y_true, y_prob, group_series, group_name="subgroup", n_bins=10):
    """
    Performs calibration audit across all subgroups of a given sensitive attribute.
    """
    groups = sorted(group_series.unique())
    subgroup_calibration_summary = []
    reliability_curves = {}

    for g in groups:
        idx = (group_series == g).values
        if np.sum(idx) < 15:
            continue
        g_true = y_true[idx]
        g_prob = y_prob[idx]

        cal_res = compute_calibration_metrics(g_true, g_prob, n_bins=n_bins)
        subgroup_calibration_summary.append({
            "attribute": group_name,
            "subgroup": str(g),
            "n_samples": len(g_true),
            "observed_base_rate": round(float(np.mean(g_true)), 4),
            "mean_predicted_risk": round(float(np.mean(g_prob)), 4),
            "ece": round(float(cal_res["ece"]), 4),
            "mce": round(float(cal_res["mce"]), 4),
            "brier_score": round(float(cal_res["brier_score"]), 4)
        })
        reliability_curves[str(g)] = cal_res["bin_stats"]

    df_summary = pd.DataFrame(subgroup_calibration_summary)
    return df_summary, reliability_curves
