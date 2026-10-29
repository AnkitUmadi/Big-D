# Model Card: Hospital 30-Day Readmission Risk Predictor with Subgroup Fairness Auditing

Following the Mitchell et al. (2019) *Model Cards for Model Reporting* standard.

---

## 1. Model Details

- **Model Name:** ClinicalReadmission-SubgroupAudit-v1.0
- **Model Architectures:**
  - Gradient Boosted Decision Trees (LightGBM) — primary non-linear clinical model
  - Logistic Regression (L2 penalized) — standard interpretable clinical baseline
  - Random Forest Classifier (100 estimators, max depth 8)
- **Mitigation Suites Evaluated:**
  1. *Pre-processing:* Demographic Sample Reweighing (fitted on training data)
  2. *Post-processing Calibration:* Subgroup-Specific Isotonic Calibration (fitted on validation data)
  3. *Post-processing Thresholds:* Fairlearn ThresholdOptimizer for Equal Opportunity (fitted on validation data)
- **Model Date:** October 2026
- **Model Version:** 1.0.0
- **Developers:** Ankith Umadi (PES1UG24AM422), Haneesh E (PES1UG24AM471), Vikas R (PES1UG24AM477)
- **Institution:** Department of CSE (AI & ML), PES University, Bangalore, India
- **Course:** Big Data Systems & Responsible AI Engineering (BDSRAE)
- **Faculty Supervisor:** Dr. Manjula Devananda

---

## 2. Intended Use & Clinical Deployment Boundaries

### Primary Intended Use:
- **Transitional Care Resource Allocation Support:** Designed to assist multidisciplinary hospital discharge teams (inpatient case managers, transitional care nurses, clinical pharmacists) in prioritizing patients for post-discharge supportive interventions prior to discharge:
  - Post-discharge phone follow-up within 48–72 hours
  - In-home nurse visits and vitals check
  - Medication reconciliation and adherence counseling
  - Rapid outpatient primary care follow-up scheduling

### Out-of-Scope & Strictly Prohibited Uses:
- **Automated Discharge Denial:** Strictly prohibited from being used to deny, delay, or mandate hospital discharge.
- **Payer Underwriting or Claim Denials:** Must never be used by commercial payers, Medicare Advantage, or insurance adjusters to deny coverage, adjust capitation, or penalize providers.
- **Autonomous Clinical Decisions:** The model is a clinical decision-*support* tool. Attending clinical judgment must always supersede model predictions.
- **Pediatric Decision-Making:** Not validated for standalone pediatric use (<18 years); pediatric readmissions exhibit distinct pathophysiology and require dedicated pediatric scoring frameworks (e.g., Pediatric Risk of Readmission / PRR).

---

## 3. Factors, Features & Subgroups

### Demographic Attributes Audited:
- **Age Groups:** Pediatric (<18), Young Adult (18–44), Middle Age (45–64), Geriatric (65+)
- **Sex / Gender:** Female, Male
- **Race:** White, Black, Hispanic, Asian, Native American, Other
- **Ethnicity:** Non-Hispanic, Hispanic
- **Intersectional Strata:** Composite $(\text{Age Group} \times \text{Race} \times \text{Gender})$

### Clinical Predictors (16 Leak-Free Features Available at Discharge Cutoff):
- Patient age at admission
- Length of hospital stay (days)
- Prior 12-month inpatient admissions, emergency visits, and ambulatory visits (strictly prior to index encounter start)
- Chronic comorbidity indicators (diagnosed on or before index discharge): Hypertension, Type 2 Diabetes, Congestive Heart Failure, COPD, Chronic Kidney Disease, Coronary Artery Disease, Depression
- Cumulative comorbidity count
- Polypharmacy (distinct active prescribed medications at discharge)
- Cumulative diagnostic and surgical procedures performed at/before discharge
- Total claim expense of index encounter

---

## 4. Quantitative Evaluation & Experimental Results

### 4.1 Overall Performance on Frozen Held-Out Test Set (N = 550 Encounters across 343 Patients)
Evaluated on patient-independent held-out test data (zero patient overlap with train/val):

| Model | ROC-AUC | PR-AUC | Brier Score | Accuracy | F1-Score | Sensitivity (TPR) | Precision (PPV) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.6710 | 0.3057 | 0.1437 | 80.73% | 0.0536 | 2.91% | 33.33% |
| **Logistic Regression** | 0.6824 | 0.2866 | 0.1430 | 81.09% | 0.0000 | 0.00% | 0.00% |
| **Random Forest** | 0.6835 | 0.3078 | 0.1421 | 80.73% | 0.0000 | 0.00% | 0.00% |

*Key Clinical Insight:* Standard unadjusted 0.5 decision thresholds on imbalanced readmission data (18.73% base rate) cause baseline linear and forest models to predict 0 for all patients, yielding high accuracy (81%) but 0% sensitivity. This demonstrates why accuracy alone is fundamentally misleading in healthcare risk models.

### 4.2 Subgroup Fairness Disparities (Baseline LightGBM on Test Set)

| Demographic Subgroup | Sample Count (N) | Base Readmission Rate | Selection Rate (0.5 threshold) | Subgroup ROC-AUC | Sensitivity (TPR) | Miss Rate (FNR) | False Alarm Rate (FPR) | Expected Calibration Error (ECE) | Small Sample Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **White** | 317 | 17.67% | 1.58% | 0.6556 | 3.57% | 96.43% | 1.15% | 0.0375 | Robust ($N \ge 30$) |
| **Black** | 110 | 23.64% | 1.82% | 0.7422 | **0.00%** | **100.0%** | 2.38% | 0.0798 | Robust ($N \ge 30$) |
| **Hispanic** | 78 | 20.51% | 1.28% | 0.6643 | **0.00%** | **100.0%** | 1.61% | 0.0673 | Robust ($N \ge 30$) |
| **Asian** | 25 | 16.00% | 0.00% | 0.5000 | **0.00%** | **100.0%** | 0.00% | 0.0772 | Warning ($N<30$) |
| **Other** | 13 | 7.69% | 7.69% | 1.0000 | 100.0% | 0.00% | 0.00% | 0.2324 | High Variance ($N_{\text{pos}}=1$) |
| **Native American** | 7 | 0.00% | 0.00% | N/A | N/A | N/A | 0.00% | 0.2299 | High Risk ($N_{\text{pos}}=0$) |
| **Geriatric (65+)** | 208 | 25.96% | 2.40% | 0.6871 | 3.70% | 96.30% | 1.95% | 0.0692 | Robust ($N \ge 30$) |
| **Middle Age (45-64)** | 176 | 14.77% | 1.14% | 0.5977 | 3.85% | 96.15% | 0.67% | 0.0526 | Robust ($N \ge 30$) |
| **Young Adult (18-44)**| 118 | 13.56% | 1.69% | 0.7010 | 0.00% | 100.0% | 1.96% | 0.0232 | Robust ($N \ge 30$) |
| **Pediatric (<18)** | 48 | 14.58% | 0.00% | **0.4530** | 0.00% | 100.0% | 0.00% | **0.1621** | Robust ($N \ge 30$) |

### 4.3 Mitigation Comparison & Trade-Off Summary

All mitigations were tuned on the validation set and tested on the frozen held-out test set:

| Model Variant | Strategy | Overall ROC-AUC | Overall Accuracy | F1-Score | All-Groups $\Delta_{\text{EO}}$ | Robust ($N \ge 30$) $\Delta_{\text{EO}}$ | Mean Subgroup ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM** | Unmitigated (0.5 threshold) | **0.6710** | 80.73% | 0.0536 | 1.0000 | 0.0357 | 0.1207 |
| **Demographic Reweighed** | Pre-processing sample weights | 0.6605 | **81.09%** | 0.0545 | 1.0000 | 0.0357 | 0.1339 |
| **Subgroup Isotonic Calibrated** | Post-processing calibration | 0.6627 | 80.55% | 0.0360 | **0.1250** | 0.1250 | **0.1004** |
| **Fairlearn ThresholdOptimized** | Post-processing decision threshold | N/A | 40.91% | **0.3762** | 0.2500 | **0.0714** | N/A |

### Empirical Mitigation Findings:
1. **Subgroup Isotonic Calibration achieved the most balanced Pareto improvement:** It lowered the mean demographic calibration error from 0.1207 to 0.1004 (-16.8% reduction) and reduced Equal Opportunity Disparity from 1.0000 to 0.1250 (-87.5% reduction) with virtually no loss in global AUC (0.6627 vs 0.6710).
2. **Threshold Optimization shifts the recall-specificity trade-off:** Fairlearn's post-processing optimizer boosted clinical recall (F1 jumped from 0.0536 to 0.3762), but dropped overall accuracy to 40.91% due to increased false positives. In clinical triage, false alarms consume nursing follow-up resources.
3. **Sparse Subgroup Behavior:** Small groups (Native American $N=7$) have single-class representation in validation data, requiring explicit fallback to reference-group thresholds rather than unconstrained group-specific optimization.

---

## 5. Ethical & Clinical Safety Warnings

> [!WARNING]
> **Warning 1: Severe False Negative Risk in Minority Subgroups**
> Under default decision thresholds, Black, Hispanic, and Asian patients experienced a 100% false negative rate (miss rate), meaning zero high-risk minority patients were flagged for transitional care. Default 0.5 probability thresholds must never be used in production clinical risk stratification.

> [!WARNING]
> **Warning 2: Pediatric Miscalibration**
> Pediatric patients (<18) face an ECE of 0.1621 and an AUC of only 0.4530 (worse than chance). Automated readmission scoring must be completely disabled for pediatric admissions.

> [!WARNING]
> **Warning 3: Operational Constraints of Group-Specific Post-Processing**
> While subgroup-specific decision thresholds reduce measured disparities mathematically, applying different clinical risk cutoffs based on a patient's race or ethnicity carries legal, regulatory, and ethical challenges in healthcare delivery. Calibrated risk probabilities coupled with clinical judgment are strongly preferred over rigid group-based cutoffs.
