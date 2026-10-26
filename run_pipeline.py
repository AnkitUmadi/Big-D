"""
Master End-to-End Orchestration Pipeline
Subgroup Performance in Healthcare Risk Models
Course: Big Data Systems & Responsible AI Engineering (BDSRAE) - PES University
Team: Ankith Umadi, Haneesh E, Vikas R

Executes:
1. Data Generation (Synthea EHR)
2. PySpark Big Data ETL & Partitioned Parquet Lakehouse
3. Baseline Model Training (LightGBM, Logistic Regression)
4. Granular Subgroup Fairness & Calibration Auditing
5. Responsible AI Mitigation (Reweighing, Subgroup Calibration, Threshold Optimizer)
6. Comparative Trade-Off Evaluation & Figure Generation
7. Big Data Scalability Benchmarks
"""

import os
import sys
import argparse
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from tabulate import tabulate

from data.synthea_generator import generate_synthea_cohort
from src.big_data.etl_pipeline import run_etl_pipeline, load_processed_data
from src.big_data.benchmark_scale import run_scaling_benchmark
from src.models.train import prepare_data, train_baseline_models
from src.models.mitigate import train_mitigated_models
from src.evaluation.subgroup_metrics import evaluate_subgroups, compute_fairness_summary
from src.evaluation.calibration_audit import audit_subgroup_calibration
from src.evaluation.intersectional_audit import audit_intersectional_subgroups
from src.utils.plotting import (
    plot_subgroup_disparities,
    plot_calibration_comparison,
    plot_fairness_utility_pareto,
    plot_intersectional_heatmap
)


def run_full_pipeline(cohort_size=30000, run_benchmarks=True):
    print("=" * 80)
    print("  BDSRAE TOPIC 19: SUBGROUP PERFORMANCE IN HEALTHCARE RISK MODELS")
    print("  PES University - Dept. of CSE (AI&ML)")
    print("  Team: Ankith Umadi (PES1UG24AM422), Haneesh E (PES1UG24AM471), Vikas R (PES1UG24AM477)")
    print("=" * 80)

    # ---------------------------------------------------------
    # STAGE 1: Synthetic Synthea EHR Generation
    # ---------------------------------------------------------
    print("\n>>> [STAGE 1] Synthetic Synthea EHR Cohort Generation")
    raw_dir = "data/raw"
    if not os.path.exists(os.path.join(raw_dir, "patients.csv")):
        generate_synthea_cohort(n_patients=cohort_size, output_dir=raw_dir)
    else:
        print(f"[*] Found existing raw Synthea files in {raw_dir}. Proceeding to ETL.")

    # ---------------------------------------------------------
    # STAGE 2: PySpark Distributed ETL & Lakehouse Creation
    # ---------------------------------------------------------
    print("\n>>> [STAGE 2] Apache PySpark Distributed ETL & Partitioned Parquet Lakehouse")
    proc_dir = "data/processed"
    parquet_path = run_etl_pipeline(raw_dir=raw_dir, processed_dir=proc_dir)

    # Load analytical cohort into memory for ML & audits
    df_cohort = load_processed_data(processed_dir=proc_dir)
    print(f"[*] Loaded processed analytical cohort: {len(df_cohort):,} index admissions.")

    # ---------------------------------------------------------
    # STAGE 3: Baseline Risk Models Training
    # ---------------------------------------------------------
    print("\n>>> [STAGE 3] Training Baseline Healthcare Readmission Models")
    data_dict = prepare_data(df_cohort)
    models, predictions, df_overall = train_baseline_models(data_dict)

    baseline_lgb = models["LightGBM"]
    baseline_preds = predictions["LightGBM"]

    # ---------------------------------------------------------
    # STAGE 4: Subgroup Performance & Calibration Audit (Baseline)
    # ---------------------------------------------------------
    print("\n>>> [STAGE 4] Baseline Subgroup Fairness & Calibration Audit")
    
    # Audit across Age Groups
    df_age_sub = evaluate_subgroups(
        baseline_preds["y_true"], baseline_preds["y_pred"], baseline_preds["y_prob"],
        baseline_preds["age_group"], group_name="Age Group", reference_group="Young Adult (18-44)"
    )
    # Audit across Race
    df_race_sub = evaluate_subgroups(
        baseline_preds["y_true"], baseline_preds["y_pred"], baseline_preds["y_prob"],
        baseline_preds["race"], group_name="Race", reference_group="White"
    )
    # Audit across Gender
    df_sex_sub = evaluate_subgroups(
        baseline_preds["y_true"], baseline_preds["y_pred"], baseline_preds["y_prob"],
        baseline_preds["gender"], group_name="Gender", reference_group="Male"
    )

    df_all_subgroups = pd.concat([df_age_sub, df_race_sub, df_sex_sub], ignore_index=True)
    
    print("\n[!] Baseline Subgroup Performance & Disparities:")
    print(tabulate(
        df_all_subgroups[["attribute", "subgroup", "n_samples", "base_rate", "roc_auc", "tpr_sensitivity", "fnr_miss_rate", "fpr", "brier_score"]],
        headers="keys", tablefmt="github", floatfmt=".4f"
    ))

    # Calibration Audit
    df_cal_race, race_curves_base = audit_subgroup_calibration(
        baseline_preds["y_true"], baseline_preds["y_prob"], baseline_preds["race"], group_name="Race"
    )
    df_cal_age, age_curves_base = audit_subgroup_calibration(
        baseline_preds["y_true"], baseline_preds["y_prob"], baseline_preds["age_group"], group_name="Age Group"
    )
    df_all_cal = pd.concat([df_cal_race, df_cal_age], ignore_index=True)
    print("\n[!] Subgroup Expected Calibration Error (ECE):")
    print(tabulate(df_all_cal, headers="keys", tablefmt="github", floatfmt=".4f"))

    # Intersectional Audit
    df_inter = audit_intersectional_subgroups(
        baseline_preds["y_true"], baseline_preds["y_pred"], baseline_preds["y_prob"], data_dict["sens_test"]
    )
    print("\n[!] Top 5 Most Vulnerable Intersectional Subgroups (Highest ECE / Miss Rate):")
    print(tabulate(df_inter.head(5)[["intersectional_group", "n_samples", "base_rate", "roc_auc", "tpr_sensitivity", "fnr_miss_rate", "ece"]],
                   headers="keys", tablefmt="github", floatfmt=".4f"))

    # ---------------------------------------------------------
    # STAGE 5: Responsible AI Bias Mitigation
    # ---------------------------------------------------------
    print("\n>>> [STAGE 5] Responsible AI Bias Mitigation")
    mitigated_preds = train_mitigated_models(data_dict, baseline_lgb, sensitive_col="race")

    # Audit Mitigated Models
    tradeoff_rows = []
    mitigated_curves = {}

    for name, m_dict in mitigated_preds.items():
        y_prob = m_dict["y_prob"]
        y_pred = m_dict["y_pred"]
        y_true = data_dict["y_test"]

        # Evaluate subgroup metrics on Race
        sub_eval = evaluate_subgroups(y_true, y_pred, y_prob, data_dict["sens_test"]["race"], group_name="Race")
        fair_summary = compute_fairness_summary(sub_eval)
        
        # Overall AUC and Calibration
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        acc_val = accuracy_score(y_true, y_pred)

        # Calibration on Race
        cal_eval, curves = audit_subgroup_calibration(y_true, y_prob, data_dict["sens_test"]["race"], group_name="Race")
        if name == "SubgroupCalibrated":
            mitigated_curves = curves

        mean_ece = cal_eval["ece"].mean()

        tradeoff_rows.append({
            "model_variant": name,
            "roc_auc": round(auc_val, 4),
            "accuracy": round(acc_val, 4),
            "equal_opportunity_difference": fair_summary["equal_opportunity_difference"],
            "demographic_parity_difference": fair_summary["demographic_parity_difference"],
            "mean_subgroup_ece": round(mean_ece, 4)
        })

    df_tradeoff = pd.DataFrame(tradeoff_rows)
    print("\n[+] Mitigation Comparison & Trade-Off Summary:")
    print(tabulate(df_tradeoff, headers="keys", tablefmt="github", floatfmt=".4f"))

    # ---------------------------------------------------------
    # STAGE 6: Visualizations & Artifacts Generation
    # ---------------------------------------------------------
    print("\n>>> [STAGE 6] Generating Publication-Quality Figures & Evaluation Artifacts")
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)

    plot_subgroup_disparities(df_all_subgroups, output_path=os.path.join(fig_dir, "subgroup_disparities.png"))
    plot_calibration_comparison(race_curves_base, mitigated_curves, output_path=os.path.join(fig_dir, "calibration_comparison.png"))
    plot_fairness_utility_pareto(df_tradeoff, output_path=os.path.join(fig_dir, "fairness_utility_tradeoff.png"))
    plot_intersectional_heatmap(df_inter, output_path=os.path.join(fig_dir, "intersectional_ece_heatmap.png"))

    # Save summary tables to CSV
    df_all_subgroups.to_csv(os.path.join(fig_dir, "baseline_subgroup_performance.csv"), index=False)
    df_all_cal.to_csv(os.path.join(fig_dir, "baseline_calibration_ece.csv"), index=False)
    df_inter.to_csv(os.path.join(fig_dir, "intersectional_audit.csv"), index=False)
    df_tradeoff.to_csv(os.path.join(fig_dir, "mitigation_tradeoff_summary.csv"), index=False)

    # ---------------------------------------------------------
    # STAGE 7: Big Data Scalability Benchmarks
    # ---------------------------------------------------------
    if run_benchmarks:
        print("\n>>> [STAGE 7] Big Data Systems Scalability Benchmarks")
        run_scaling_benchmark(scales=[5000, 10000, 20000], output_dir=fig_dir)

    print("\n" + "=" * 80)
    print("  [OK] ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts & Visualizations generated in: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Execution Pipeline")
    parser.add_argument("--cohort_size", type=int, default=30000, help="Initial patient cohort size")
    parser.add_argument("--skip_benchmarks", action="store_true", help="Skip scalability benchmark suite")
    args = parser.parse_args()

    run_full_pipeline(cohort_size=args.cohort_size, run_benchmarks=not args.skip_benchmarks)
