"""
Subgroup Calibration Audit Engine for Clinical Risk Models
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Brier Scores, and generates subgroup reliability curve bin statistics.

Handles empty bins, probability validation, and small sample sizes.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import numpy as np
import pandas as pd


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    Brier Score, and reliability diagram bin statistics using equal-width probability bins.

    Validates inputs and ensures empty bins are not presented as zero error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_prob ({len(y_prob)})")
    if len(y_true) == 0:
        return {"ece": np.nan, "mce": np.nan, "brier_score": np.nan, "bin_stats": pd.DataFrame()}

    # Validate probabilities
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_samples = len(y_true)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = (bin_indices == i)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_conf = float(np.mean(y_prob[mask]))
            bin_acc = float(np.mean(y_true[mask]))
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * abs_err
            mce = max(mce, abs_err)

            bin_stats.append({
                "bin_idx": i,
                "bin_lower": round(bins[i], 2),
                "bin_upper": round(bins[i+1], 2),
                "bin_center": round((bins[i] + bins[i+1]) / 2.0, 2),
                "bin_count": bin_count,
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "calibration_error": round(abs_err, 4)
            })
        else:
            # Empty bins must not be presented as having 0 calibration error
            bin_stats.append({
                "bin_idx": i,
                "bin_lower": round(bins[i], 2),
                "bin_upper": round(bins[i+1], 2),
                "bin_center": round((bins[i] + bins[i+1]) / 2.0, 2),
                "bin_count": 0,
                "confidence": np.nan,
                "accuracy": np.nan,
                "calibration_error": np.nan
            })

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier), 4),
        "bin_stats": pd.DataFrame(bin_stats)
    }


def audit_subgroup_calibration(y_true, y_prob, group_series, group_name="subgroup", n_bins=10):
    """
    Performs calibration audit across all subgroups of a given sensitive attribute.
    Evaluates observed prevalence vs. mean predicted risk and computes subgroup ECE/MCE.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group_series = pd.Series(group_series).reset_index(drop=True)

    unique_groups = sorted([g for g in group_series.unique() if pd.notna(g)])
    subgroup_calibration_summary = []
    reliability_curves = {}

    for g in unique_groups:
        idx = (group_series == g).values
        n = np.sum(idx)
        if n == 0:
            continue

        g_true = y_true[idx]
        g_prob = y_prob[idx]

        cal_res = compute_calibration_metrics(g_true, g_prob, n_bins=n_bins)
        subgroup_calibration_summary.append({
            "attribute": group_name,
            "subgroup": str(g),
            "n_samples": int(len(g_true)),
            "observed_base_rate": round(float(np.mean(g_true)), 4),
            "mean_predicted_risk": round(float(np.mean(g_prob)), 4),
            "ece": cal_res["ece"],
            "mce": cal_res["mce"],
            "brier_score": cal_res["brier_score"],
            "small_sample_flag": (n < 30)
        })
        reliability_curves[str(g)] = cal_res["bin_stats"]

    df_summary = pd.DataFrame(subgroup_calibration_summary)
    return df_summary, reliability_curves
