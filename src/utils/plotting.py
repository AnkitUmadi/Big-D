"""
Visualization Suite for Healthcare Subgroup Fairness & Calibration Audits
Generates publication-quality figures for reports, presentations, and notebooks.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def plot_subgroup_disparities(df_subgroups, output_path="reports/figures/subgroup_disparities.png"):
    """
    Plots Sensitivity (TPR) and False Positive Rate (FPR) disparities across demographic subgroups.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # TPR plot
    sns.barplot(
        data=df_subgroups, x="subgroup", y="tpr_sensitivity", hue="attribute",
        ax=axes[0], palette="Blues_r", dodge=False
    )
    axes[0].set_title("Clinical Sensitivity (TPR) Disparity by Subgroup\n(Missed Readmissions = Denied Care)", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].set_xlabel("Subgroup")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].axhline(df_subgroups["tpr_sensitivity"].mean(), color="red", linestyle="--", label="Mean Sensitivity")
    axes[0].legend(loc="upper right")

    # FPR plot
    sns.barplot(
        data=df_subgroups, x="subgroup", y="fpr", hue="attribute",
        ax=axes[1], palette="Oranges_r", dodge=False
    )
    axes[1].set_title("False Positive Rate (FPR) Disparity by Subgroup\n(False Alarms = Resource Waste)", fontweight="bold")
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_xlabel("Subgroup")
    axes[1].set_ylim(0, 0.5)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].axhline(df_subgroups["fpr"].mean(), color="red", linestyle="--", label="Mean FPR")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Subgroup disparities chart saved: {output_path}")


def plot_calibration_comparison(baseline_curves, mitigated_curves, output_path="reports/figures/calibration_comparison.png"):
    """
    Plots Reliability Diagrams comparing Baseline vs Mitigated Subgroup-Calibrated models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    groups = [g for g in ["White", "Black", "Hispanic", "Geriatric (65+)", "Young Adult (18-44)"] if g in baseline_curves]
    
    n_cols = len(groups)
    if n_cols == 0:
        groups = list(baseline_curves.keys())[:4]
        n_cols = len(groups)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.2), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        # Ideal diagonal line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

        # Baseline curve
        if g in baseline_curves:
            b_df = baseline_curves[g].dropna(subset=["accuracy"])
            ax.plot(b_df["confidence"], b_df["accuracy"], marker="o", color="#d62728", label="Baseline (Unmitigated)", linewidth=2)

        # Mitigated curve
        if g in mitigated_curves:
            m_df = mitigated_curves[g].dropna(subset=["accuracy"])
            ax.plot(m_df["confidence"], m_df["accuracy"], marker="s", color="#2ca02c", label="Subgroup Calibrated", linewidth=2)

        ax.set_title(f"Subgroup: {g}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Mean Predicted Risk", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Observed Fraction Readmitted", fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper left", frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Calibration reliability curves saved: {output_path}")


def plot_fairness_utility_pareto(df_tradeoff, output_path="reports/figures/fairness_utility_tradeoff.png"):
    """
    Plots the Fairness-Utility Pareto frontier (ROC-AUC vs Equal Opportunity Disparity).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Baseline": "#d62728", "Reweighted": "#ff7f0e", "SubgroupCalibrated": "#1f77b4", "ThresholdOptimized": "#2ca02c"}
    
    for _, row in df_tradeoff.iterrows():
        name = row["model_variant"]
        c = colors.get(name, "#333333")
        ax.scatter(row["equal_opportunity_difference"], row["roc_auc"], color=c, s=160, zorder=5, label=name)
        ax.annotate(
            f"  {name}\n  (AUC: {row['roc_auc']:.3f}, EO Diff: {row['equal_opportunity_difference']:.3f})",
            (row["equal_opportunity_difference"], row["roc_auc"]),
            fontsize=9, fontweight="bold", verticalalignment="center"
        )

    ax.set_title("Responsible AI Trade-off: Overall Utility vs. Equal Opportunity Disparity", fontsize=12, fontweight="bold")
    ax.set_xlabel("Equal Opportunity Disparity [Max(TPR) - Min(TPR)] (Lower is Fairer)", fontsize=11)
    ax.set_ylabel("Global ROC-AUC (Higher is Better)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Shade ideal zone
    min_eo = df_tradeoff["equal_opportunity_difference"].min()
    max_auc = df_tradeoff["roc_auc"].max()
    ax.axvspan(0.0, 0.10, color="green", alpha=0.08, label="Acceptable Disparity Zone (<0.10)")

    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Fairness-Utility Pareto plot saved: {output_path}")


def plot_intersectional_heatmap(df_inter, output_path="reports/figures/intersectional_ece_heatmap.png"):
    """
    Plots intersectional heatmap of Expected Calibration Error (ECE) across Age Group and Race.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pivot = df_inter.pivot_table(index="age_group", columns="race", values="ece", aggfunc="mean")
    
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'Expected Calibration Error (ECE)'})
    plt.title("Intersectional Expected Calibration Error (Age x Race)\n(Darker = Higher Clinical Miscalibration)", fontweight="bold", fontsize=12)
    plt.ylabel("Age Group", fontsize=11)
    plt.xlabel("Race", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Intersectional ECE heatmap saved: {output_path}")
