# Clinical Subgroup Performance & Disparity Report: 30-Day Hospital Readmission Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE)  
**Department:** Computer Science & Engineering (AI & ML), PES University  
**Faculty Evaluator:** Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)  
**Authors (Team Members):**  
- **Ankith Umadi** (PES1UG24AM422)  
- **Haneesh E** (PES1UG24AM471)  
- **Vikas R** (PES1UG24AM477)  
**Date:** October 2026  
**Artifact Version:** 1.0.0  

---

## 1. Executive Clinical Summary

Machine learning models deployed for inpatient 30-day hospital readmission risk prediction often exhibit deceptive aggregate performance: an algorithm may achieve >80% accuracy while harboring severe, systematic failure modes across vulnerable demographic subgroups. In clinical decision-support environments, these disparities manifest in two forms:

1. **Sensitivity / False Negative Disparities:** High-risk patients belonging to historically underserved or minority demographics are misclassified as low-risk, denying them post-discharge transitional care, medication reconciliation, and home nurse follow-up.
2. **Subgroup Miscalibration:** Predicted risk probabilities diverge from true empirical event frequencies within specific age, sex, and racial strata, causing erroneous clinical triage tiering and inefficient resource allocation.

This report documents an empirical fairness and calibration audit conducted across an inpatient cohort of synthetic Electronic Health Records (Synthea-style EHR) partitioned by patient identifier into train, validation, and held-out test sets.

---

## 2. Demographic Subgroups and Audit Methodology

The evaluation cohort consists of 3,630 inpatient index hospitalizations extracted via an Apache PySpark distributed Big Data pipeline. Subgroups are defined across four sensitive attributes and their intersectional combinations:
- **Age Stratification:** Pediatric (<18), Young Adult (18–44), Middle Age (45–64), Geriatric (65+).
- **Racial Stratification:** White (majority reference cohort), Black, Hispanic, Asian, Native American, Other.
- **Gender Stratification:** Female and Male cohorts.
- **Ethnicity Stratification:** Non-Hispanic and Hispanic cohorts.
- **Intersectional Strata:** Composite groupings across $\text{Age Group} \times \text{Race} \times \text{Gender}$.

### Key Clinical Metrics
- **Clinical Sensitivity / True Positive Rate ($\text{TPR}$):** Fraction of patients who actually readmit within 30 days that the model correctly flags for preventive intervention.
- **False Negative / Miss Rate ($\text{FNR}$):** $1 - \text{TPR}$ — The danger zone where preventable readmissions occur unmonitored.
- **False Positive Rate ($\text{FPR}$):** False alarms that allocate scarce nursing follow-up to patients who would not have readmitted.
- **Expected Calibration Error ($\text{ECE}$):** Weighted average deviation between predicted risk confidence and observed empirical readmission rate across $M = 10$ probability bins.
- **Equal Opportunity Disparity ($\Delta_{\text{EO}}$):** $\max_{s}(\text{TPR}_s) - \min_{s}(\text{TPR}_s)$.
- **Demographic Parity Disparity ($\Delta_{\text{DP}}$):** $\max_{s}(\text{selection rate}_s) - \min_{s}(\text{selection rate}_s)$.

---

## 3. Empirical Baseline Audit Findings (Held-Out Test Set, N = 550)

### 3.1 Subgroup Performance Disparities (Baseline LightGBM)

| Demographic Subgroup | Sample Count (N) | Base Readmission Rate | Subgroup ROC-AUC | Sensitivity (TPR) | Miss Rate (FNR) | False Alarm Rate (FPR) | Expected Calibration Error (ECE) | Small Sample Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **White** | 317 | 17.67% | 0.6556 | 3.57% | 96.43% | 1.15% | 0.0375 | Robust ($N \ge 30$) |
| **Black** | 110 | 23.64% | 0.7422 | **0.00%** | **100.0%** | 2.38% | 0.0798 | Robust ($N \ge 30$) |
| **Hispanic** | 78 | 20.51% | 0.6643 | **0.00%** | **100.0%** | 1.61% | 0.0673 | Robust ($N \ge 30$) |
| **Asian** | 25 | 16.00% | 0.5000 | **0.00%** | **100.0%** | 0.00% | 0.0772 | Warning ($N<30$) |
| **Other** | 13 | 7.69% | 1.0000 | 100.0% | 0.00% | 0.00% | 0.2324 | High Variance ($N_{\text{pos}}=1$) |
| **Native American** | 7 | 0.00% | N/A | N/A | N/A | 0.00% | 0.2299 | High Risk ($N_{\text{pos}}=0$) |
| **Geriatric (65+)** | 208 | 25.96% | 0.6871 | 3.70% | 96.30% | 1.95% | 0.0692 | Robust ($N \ge 30$) |
| **Middle Age (45-64)** | 176 | 14.77% | 0.5977 | 3.85% | 96.15% | 0.67% | 0.0526 | Robust ($N \ge 30$) |
| **Young Adult (18-44)**| 118 | 13.56% | 0.7010 | 0.00% | 100.0% | 1.96% | 0.0232 | Robust ($N \ge 30$) |
| **Pediatric (<18)** | 48 | 14.58% | **0.4530** | 0.00% | 100.0% | 0.00% | **0.1621** | Robust ($N \ge 30$) |

### Disparity Metric Definitions & Derivation:
1. **Disparity Across All Groups with Condition Positives ($\Delta_{\text{EO}}^{\text{all}} = 1.0000$):**
   Calculated as $\max(\text{TPR}) - \min(\text{TPR}) = 1.0000 - 0.0000 = 1.0000$. This upper bound is driven by the 'Other' category, where only 1 readmitted patient existed ($N_{\text{pos}}=1$) and was correctly identified by the baseline model ($\hat{p} = 0.577 \ge 0.5$, Wilson 95% CI: $[0.2065, 1.0000]$), while Black, Hispanic, and Asian groups had 0.0% sensitivity. Native American has $N_{\text{pos}}=0$ and is mathematically excluded (NaN) from sensitivity calculations.
2. **Disparity Across Statistically Robust Groups ($\Delta_{\text{EO}}^{\text{robust}} = 0.0357$):**
   Restricting the evaluation to sufficiently powered cohorts ($N \ge 30$: White, Black, Hispanic), baseline sensitivity is 3.57% for White patients and 0.00% for Black and Hispanic patients, yielding $\Delta_{\text{EO}}^{\text{robust}} = 3.57\% - 0.00\% = 0.0357$.
3. **Severe Sensitivity Collapse in Minorities:** Under baseline thresholding, Black ($N=110$) and Hispanic ($N=78$) patients experience a 100% false negative rate.
4. **Pediatric Performance Invalidation:** Pediatric patients (<18) display an ROC-AUC of 0.4530 (worse than chance) and an ECE of 0.1621.

---

## 4. Subgroup Calibration Failure Analysis

In clinical risk scoring, calibration is vital because medical interventions are triggered by estimated probability thresholds (e.g., predicted risk $\ge 15\%$).

| Attribute | Subgroup | Observed Base Rate | Mean Predicted Risk | ECE | Max Calibration Error (MCE) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Race** | White | 17.67% | 17.91% | **0.0375** | 0.1434 | 0.1388 |
| **Race** | Black | 23.64% | 18.23% | **0.0614** | 0.3800 | 0.1621 |
| **Race** | Hispanic | 20.51% | 19.01% | **0.0673** | 0.6757 | 0.1605 |
| **Race** | Native American | 0.00% | 22.99% | **0.2299** | 0.3344 | 0.0564 |
| **Race** | Other | 7.69% | 24.42% | **0.2324** | 0.4228 | 0.0658 |
| **Age Group** | Young Adult (18–44) | 13.56% | 15.16% | **0.0232** | 0.5406 | 0.1142 |
| **Age Group** | Middle Age (45–64) | 14.77% | 17.61% | **0.0526** | 0.2504 | 0.1253 |
| **Age Group** | Geriatric (65+) | 25.96% | 20.74% | **0.0692** | 0.6757 | 0.1777 |
| **Age Group** | Pediatric (<18) | 14.58% | 17.54% | **0.1621** | 0.5790 | 0.1359 |

---

## 5. Intersectional Audit Findings

Single-axis evaluations conceal compounding harms at demographic intersections:

| Intersectional Subgroup | $N$ | Base Rate | Sensitivity ($\text{TPR}$) | Miss Rate ($\text{FNR}$) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Geriatric (65+) \| Hispanic \| Female** | 21 | 28.57% | 0.00% | **100.0%** | **0.2514** |
| **Pediatric (<18) \| Black \| Female** | 11 | 27.27% | 0.00% | **100.0%** | **0.2312** |
| **Middle Age (45–64) \| Hispanic \| Female** | 10 | 30.00% | 0.00% | **100.0%** | **0.2014** |
| **Geriatric (65+) \| Black \| Male** | 21 | 38.10% | 0.00% | **100.0%** | **0.1908** |

---

## 6. Responsible AI Mitigation & Comparative Trade-Offs

We evaluated three mitigation interventions on the held-out test set:

| Model Configuration | Strategy | Global ROC-AUC | Overall Accuracy | F1-Score | All-Groups $\Delta_{\text{EO}}$ | Robust ($N \ge 30$) $\Delta_{\text{EO}}$ | Mean Subgroup ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM** | Unmitigated | **0.6710** | 80.73% | 0.0536 | 1.0000 | 0.0357 | 0.1207 |
| **Demographic Reweighed** | Pre-processing sample weights | 0.6605 | **81.09%** | 0.0545 | 1.0000 | 0.0357 | 0.1339 |
| **Subgroup Isotonic Calibrated**| Post-processing calibration | 0.6627 | 80.55% | 0.0360 | **0.1250** | 0.1250 | **0.1004** |
| **ThresholdOptimized_EO** | Post-processing decision threshold | N/A | 40.91% | **0.3762** | 0.2500 | **0.0714** | N/A |

### Key Takeaway for Hospital Deployments:
Subgroup Isotonic Calibration provides the most clinically viable intervention: it reduces calibration error by **16.8%** and cuts Equal Opportunity disparity from 1.0000 to **0.1250** without the massive false alarm surge associated with unconstrained threshold optimization.
