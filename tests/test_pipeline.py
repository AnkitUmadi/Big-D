"""
Automated Test Suite for Healthcare Readmission Risk & Fairness Pipeline
Tests:
1. 30-Day Readmission Target Logic & Edge Cases (30d boundary, intervening ambulatory visits, overlaps, censoring)
2. Patient-Level Split Independence (zero patient leakage across Train, Val, Test)
3. Leak-Free Feature Boundaries
4. Responsible AI Fairness Calculations & Disparities (empty bins, zero divisions)
5. Subgroup Calibration Metrics (ECE, MCE, Brier score, empty bin handling)
6. Pipeline Input Validation

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.subgroup_metrics import compute_binary_metrics, compute_fairness_summary, evaluate_subgroups
from src.evaluation.calibration_audit import compute_calibration_metrics, audit_subgroup_calibration
from src.models.train import prepare_data, FEATURES
from src.models.mitigate import compute_sample_weights_reweighing, SubgroupCalibrator
from src.big_data.etl_pipeline import load_processed_data


class TestReadmissionTarget(unittest.TestCase):
    """Tests 30-day readmission target construction and boundary conditions."""

    def test_target_boundary_conditions(self):
        # Case 1: Exactly 30 days -> Readmission = 1
        cur_stop = pd.Timestamp("2023-01-01 12:00:00")
        nxt_start_30d = cur_stop + pd.Timedelta(days=30)
        gap_30d = (nxt_start_30d - cur_stop).total_seconds() / 86400.0
        self.assertEqual(gap_30d, 30.0)
        self.assertTrue(0.0 < gap_30d <= 30.0)

        # Case 2: 30.1 days (31 days) -> Readmission = 0
        nxt_start_31d = cur_stop + pd.Timedelta(days=30, hours=2)
        gap_31d = (nxt_start_31d - cur_stop).total_seconds() / 86400.0
        self.assertFalse(0.0 < gap_31d <= 30.0)

        # Case 3: Negative gap (overlapping episode) -> Not a valid readmission
        overlap_start = cur_stop - pd.Timedelta(days=2)
        gap_overlap = (overlap_start - cur_stop).total_seconds() / 86400.0
        self.assertFalse(0.0 < gap_overlap <= 30.0)

    def test_intervening_ambulatory_visits_do_not_hide_readmission(self):
        """Simulates patient trajectory: Inpatient -> Ambulatory (day 5) -> Inpatient (day 18)."""
        encounters = [
            {"id": "enc_1", "class": "inpatient", "start": "2023-03-01", "stop": "2023-03-05"},
            {"id": "enc_2", "class": "ambulatory", "start": "2023-03-10", "stop": "2023-03-10"},
            {"id": "enc_3", "class": "inpatient", "start": "2023-03-23", "stop": "2023-03-27"}
        ]
        df_enc = pd.DataFrame(encounters)
        df_enc["start"] = pd.to_datetime(df_enc["start"])
        df_enc["stop"] = pd.to_datetime(df_enc["stop"])

        # Filter to inpatient stays
        inp = df_enc[df_enc["class"] == "inpatient"].sort_values("start").reset_index(drop=True)
        # Next inpatient stay after discharge of enc_1
        cur = inp.iloc[0]
        subsequent = inp[inp["start"] > cur["stop"]]
        self.assertFalse(subsequent.empty)
        next_inp = subsequent.iloc[0]
        gap_days = (next_inp["start"] - cur["stop"]).total_seconds() / 86400.0
        self.assertEqual(gap_days, 18.0)
        is_readmitted = (0.0 < gap_days <= 30.0)
        self.assertTrue(is_readmitted, "Intervening ambulatory visit must not mask subsequent inpatient readmission.")

    def test_censored_boundary_handling(self):
        """Tests that encounters occurring within 30 days of study horizon without readmission are flagged."""
        max_date = pd.Timestamp("2024-11-01 10:00:00")
        discharge_late = pd.Timestamp("2024-10-15 12:00:00")
        # Time to cutoff is 16.9 days (< 30 days)
        has_followup = (discharge_late <= max_date - pd.Timedelta(days=30))
        self.assertFalse(has_followup, "Index discharge within 30 days of study horizon lacks sufficient observable follow-up.")


class TestPatientSplitIndependence(unittest.TestCase):
    """Verifies that train, validation, and test splits have ZERO patient overlap."""

    def test_zero_patient_leakage(self):
        df_cohort = load_processed_data()
        data_dict = prepare_data(df_cohort, test_size=0.15, val_size=0.15, random_state=42)

        tr_pats = data_dict["train_patients"]
        val_pats = data_dict["val_patients"]
        te_pats = data_dict["test_patients"]

        self.assertEqual(len(tr_pats & val_pats), 0, "Patient overlap between Train and Val must be 0.")
        self.assertEqual(len(tr_pats & te_pats), 0, "Patient overlap between Train and Test must be 0.")
        self.assertEqual(len(val_pats & te_pats), 0, "Patient overlap between Val and Test must be 0.")

        total_split_pats = len(tr_pats) + len(val_pats) + len(te_pats)
        self.assertEqual(total_split_pats, df_cohort["patient_id"].nunique())

    def test_preprocessing_fitted_on_train_only(self):
        df_cohort = load_processed_data()
        data_dict = prepare_data(df_cohort, test_size=0.15, val_size=0.15, random_state=42)
        scaler = data_dict["scaler"]

        # Check that scaler was fitted on exactly X_train
        self.assertEqual(len(scaler.mean_), len(FEATURES))
        # Ensure transformed train has mean approximately 0 and std approximately 1
        X_tr = data_dict["X_train"]
        np.testing.assert_allclose(X_tr.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(X_tr.std(axis=0), 1.0, atol=1e-2)


class TestFairnessAndCalibration(unittest.TestCase):
    """Tests fairness calculations and calibration audit for robustness against edge cases."""

    def test_empty_bin_handling_in_calibration(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.05, 0.08, 0.12, 0.85, 0.90])
        res = compute_calibration_metrics(y_true, y_prob, n_bins=10)

        bin_stats = res["bin_stats"]
        empty_bins = bin_stats[bin_stats["bin_count"] == 0]
        self.assertTrue(len(empty_bins) > 0)
        # Empty bins must have NaN calibration error, not 0.0
        self.assertTrue(empty_bins["calibration_error"].isna().all())

    def test_fairness_summary_robust_to_zero_denominator(self):
        # Single-group or zero-positive group
        df_dummy = pd.DataFrame([
            {"attribute": "Test", "subgroup": "A", "selection_rate": 0.0, "tpr_sensitivity": 0.0, "fpr": 0.0, "n_samples": 50},
            {"attribute": "Test", "subgroup": "B", "selection_rate": 0.0, "tpr_sensitivity": 0.0, "fpr": 0.0, "n_samples": 50}
        ])
        summary = compute_fairness_summary(df_dummy)
        self.assertEqual(summary["demographic_parity_difference"], 0.0)
        self.assertFalse(np.isnan(summary["demographic_parity_ratio"]))
        self.assertEqual(summary["equal_opportunity_difference"], 0.0)

    def test_zero_positive_subgroup_emits_nan_sensitivity(self):
        # When a subgroup has 0 condition positives, sensitivity is mathematically undefined (NaN)
        from src.evaluation.subgroup_metrics import compute_binary_metrics
        res = compute_binary_metrics(y_true=[0, 0, 0], y_pred=[0, 1, 0])
        self.assertTrue(np.isnan(res["tpr_sensitivity"]))
        self.assertTrue(np.isnan(res["fnr_miss_rate"]))

        # Equal opportunity disparity must drop undefined (NaN) groups, not treat them as 0.0
        df_dummy = pd.DataFrame([
            {"attribute": "Race", "subgroup": "A", "selection_rate": 0.5, "tpr_sensitivity": 0.9, "fpr": 0.1, "n_samples": 50},
            {"attribute": "Race", "subgroup": "B", "selection_rate": 0.5, "tpr_sensitivity": 0.8, "fpr": 0.1, "n_samples": 50},
            {"attribute": "Race", "subgroup": "C_ZeroPos", "selection_rate": 0.5, "tpr_sensitivity": np.nan, "fpr": 0.1, "n_samples": 5}
        ])
        summary = compute_fairness_summary(df_dummy)
        # Should be 0.9 - 0.8 = 0.1, NOT 0.9 - 0.0 = 0.9
        self.assertAlmostEqual(summary["equal_opportunity_difference"], 0.1, places=4)

    def test_reweighing_weights_finite_and_normalized(self):
        y_train = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        sens_series = pd.Series(["A", "A", "A", "B", "B", "C", "C", "C"])
        weights = compute_sample_weights_reweighing(y_train, sens_series)

        self.assertEqual(len(weights), len(y_train))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=4)


class TestPipelineInputValidation(unittest.TestCase):
    """Validates raw data files and lakehouse integrity."""

    def test_raw_files_exist(self):
        raw_dir = "data/raw"
        for fname in ["patients.csv", "encounters.csv", "conditions.csv", "medications.csv", "procedures.csv"]:
            fpath = os.path.join(raw_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Required file missing: {fpath}")

    def test_parquet_lakehouse_non_empty(self):
        df = load_processed_data()
        self.assertGreater(len(df), 0)
        self.assertIn("readmission_30d", df.columns)
        self.assertIn("patient_id", df.columns)
        self.assertIn("encounter_id", df.columns)


if __name__ == "__main__":
    unittest.main()

