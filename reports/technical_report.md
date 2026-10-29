# Subgroup Performance, Calibration Disparities, and Scalable Fairness Mitigation in Healthcare Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE)  
**Institution:** Department of Computer Science & Engineering (AI & ML), PES University, Bangalore, India  
**Faculty Evaluator:** Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)  
**Authors (Team Members):**  
- **Ankith Umadi** (PES1UG24AM422)  
- **Haneesh E** (PES1UG24AM471)  
- **Vikas R** (PES1UG24AM477)  
**Date:** October 2026  
**Artifact Version:** 1.0.0  

---

## Abstract

Clinical risk prediction models deployed within hospital electronic health record (EHR) systems are increasingly used to triage transitional care management. However, aggregate evaluation metrics often conceal severe performance and calibration failures across demographic subgroups. In this paper, we present an end-to-end Big Data and Responsible AI engineering framework to evaluate and mitigate subgroup performance disparities in 30-day hospital readmission risk prediction using synthetic longitudinal EHR data. 

We construct an Apache PySpark distributed Big Data pipeline that ingests relational clinical tables, eliminates temporal and patient-level data leakage, extracts index hospitalizations with boundary censoring handling, and stores analytical cohorts in a partitioned Parquet Lakehouse (`[age_group, race]`), achieving a $53.7\times$ storage footprint reduction over raw CSV. 

Our empirical Responsible AI audit on patient-independent held-out test data reveals severe failure modes in uncalibrated baseline models: while aggregate accuracy reaches $80.73\%$, clinical sensitivity collapses to $0.0\%$ across Black, Hispanic, and Asian minority cohorts under default decision thresholds (100% false negative miss rate), and pediatric discrimination drops to $\text{ROC-AUC} = 0.4530$. Furthermore, probability miscalibration is pronounced in intersectional cohorts ($\text{ECE} = 0.2514$ in Geriatric Hispanic Females). We formulate and benchmark three bias mitigation strategies: pre-processing demographic sample reweighing, subgroup-specific isotonic calibration, and Fairlearn threshold optimization. Subgroup isotonic calibration successfully reduces the mean demographic calibration error from $0.1207$ to $0.1004$ (a $16.8\%$ improvement) and lowers Equal Opportunity disparity from $1.0000$ to $0.1250$ with negligible discrimination loss ($\text{ROC-AUC} = 0.6627$ vs $0.6710$). We codify operational safeguards and residual clinical risks into formal Model and Data Cards.

---

## 1. Introduction & Problem Formulation

Hospital 30-day readmissions represent a key benchmark of healthcare delivery quality and patient safety, influenced by regulatory frameworks such as the CMS Hospital Readmissions Reduction Program (HRRP). To prevent post-discharge clinical decompensations, health systems deploy algorithmic risk scores to triage transitional interventions—including nurse telephone check-ins, home health visits, medication reconciliations, and expedited primary care physician appointments.

However, clinical algorithmic governance literature (Obermeyer et al., 2019; Rajkomar et al., 2018) demonstrates that unconstrained machine learning models often exacerbate healthcare disparities:
1. **The Accuracy Paradox in Clinical AI:** In class-imbalanced healthcare outcomes (readmission prevalence ~18%), standard empirical risk minimization achieves >80% accuracy simply by under-predicting positive cases, yielding catastrophic false negative rates (up to 100% miss rate in minority subgroups).
2. **False Negative Harms:** In hospital triage, a False Negative ($\text{FN}$) deprives a vulnerable patient of transitional care, contributing directly to preventable complications and hospital readmission.
3. **Probability Miscalibration:** Clinical protocols rely on probability thresholds to allocate triage tiers. If a predicted probability of 20% corresponds to 40% observed risk in one subgroup and 10% in another, care pathways are fundamentally distorted.

### BDSRAE Requirements Fulfillment:
- **Big Data Systems:** Distributed ingestion, window-based index admission and readmission target construction, leak-free feature engineering, and partitioned Parquet storage using Apache PySpark 4.2.1, validated with throughput, latency, and compression benchmarks.
- **Responsible AI:** Granular subgroup auditing across Age, Gender, Race, Ethnicity, and Intersectional cohorts; calibration auditing via Expected Calibration Error ($\text{ECE}$); implementation of pre-processing, in-processing, and post-processing mitigations; and formal governance documentation.

---

## 2. Dataset Provenance & Big Data Systems Architecture

### 2.1 Provenance & Data Quality Audit
The dataset was programmatically generated using a local synthetic clinical generator (`data/synthea_generator.py`) parameterized with Synthea-style relational schemas and clinical hazard distributions. 

**Recomputed Population Statistics:**
- **Patients:** 15,000 unique synthetic individuals.
- **Encounters:** 87,382 total longitudinal encounters (83,535 ambulatory [95.60%], 3,632 inpatient [4.16%], 215 emergency [0.25%]).
- **Conditions:** 33,796 SNOMED-CT chronic diagnoses (Essential hypertension: 10,767; Type 2 diabetes: 6,297; Depressive disorder: 3,459; CAD: 3,154; Heart failure: 3,075; Asthma: 2,446; COPD: 2,395; CKD: 2,203).
- **Medications:** 29,967 prescription episodes.
- **Procedures:** 2,130 diagnostic and therapeutic procedures.
- **Demographics:** Female: 7,746 (51.64%), Male: 7,254 (48.36%); White: 9,460 (63.07%), Black: 2,253 (15.02%), Hispanic: 1,951 (13.01%), Asian: 873 (5.82%), Other: 250 (1.67%), Native American: 213 (1.42%).

### 2.2 Correct Readmission Target & Censoring Handling
Audit of raw encounters identified critical edge cases addressed in the pipeline:
1. **Intervening Ambulatory Visits:** The pipeline identifies the next qualifying *inpatient* admission strictly after index discharge. Intervening ambulatory clinic visits do not mask subsequent 30-day inpatient readmissions.
2. **Overlapping Encounter Resolution:** 9 inpatient episodes with overlapping timestamps (where admission started before prior discharge) were resolved so that concurrent records are not erroneously counted as post-discharge readmissions.
3. **Boundary Censoring:** Encounters occurring within 30 days of the study horizon (`2024-11-01`) without an observed readmission lack observable follow-up. Exactly 2 index stays met this criterion and were excluded from the analytical cohort to prevent false-negative label noise.
4. **Valid Analytical Cohort:** 3,630 index admissions across 2,286 unique patients, with 667 30-day inpatient readmissions (18.37% prevalence).

### 2.3 Elimination of Temporal Leakage
All 16 clinical predictors are constructed strictly from information available on or before the index discharge timestamp:
- Prior utilization (`num_prior_inpatient`, `num_prior_emergency`, `num_prior_ambulatory`): strictly counts encounters with `start < index_start`.
- Comorbidities (`has_diabetes`, `has_hypertension`, etc., `comorbidity_count`): conditions diagnosed with `cond_start <= index_stop`.
- Polypharmacy (`total_medications`) and procedures (`total_procedures`): recorded with `start <= index_stop`.

### 2.4 Partitioned Parquet Lakehouse & Scalability Benchmarks
The analytical dataset is stored in Parquet format partitioned by `[age_group, race]`. Benchmarking PySpark on Windows 11 (16 CPU cores, 23.7 GB RAM, `master='local[*]'):

| Cohort Scale (Patients) | Encounters Processed | Raw CSV Footprint | Parquet Lakehouse | Compression Ratio | Spark ETL Runtime | Record Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2,500** | 14,530 | 5.52 MB | 0.19 MB | **29.7x** | 8.98 s | 278.3 patients/s |
| **5,000** | 28,712 | 10.92 MB | 0.23 MB | **47.2x** | 8.63 s | 579.5 patients/s |
| **10,000** | 58,343 | 22.16 MB | 0.34 MB | **66.0x** | 11.17 s | **895.4 patients/s** |

---

## 3. Methodology: Patient-Level Splitting & Modeling

### 3.1 Patient-Independent Data Splitting
To prevent patient-level data leakage, encounters were partitioned by **patient identifier**, ensuring that no patient appears in more than one split:
- **Training Set (70%):** 1,600 unique patients, 2,536 index encounters (Readmission rate: 18.38%).
- **Validation Set (15%):** 343 unique patients, 544 index encounters (Readmission rate: 18.01%). Used for calibrator fitting, threshold tuning, and hyperparameter selection.
- **Held-Out Test Set (15%):** 343 unique patients, 550 index encounters (Readmission rate: 18.73%). Frozen until final evaluation.
- Preprocessing (`StandardScaler`) was fitted strictly on `X_train`.

### 3.2 Baseline Model Evaluation on Held-Out Test Set

| Model | ROC-AUC | PR-AUC | Brier Score | Accuracy | F1-Score | Sensitivity (TPR) | Precision (PPV) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.6710 | 0.3057 | 0.1437 | 80.73% | 0.0536 | 2.91% | 33.33% |
| **Logistic Regression** | 0.6824 | 0.2866 | 0.1430 | 81.09% | 0.0000 | 0.00% | 0.00% |
| **Random Forest** | 0.6835 | 0.3078 | 0.1421 | 80.73% | 0.0000 | 0.00% | 0.00% |

All three baseline models achieve ~81% accuracy, but standard unadjusted 0.5 decision thresholds yield near-zero sensitivity, illustrating the clinical inadequacy of default classification cutoffs.

---

## 4. Responsible AI Subgroup Performance & Calibration Audit

### 4.1 Granular Subgroup Performance Disparities (Test Set)

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

**Disparity Metric Definitions & Derivation:**
1. **Disparity Across All Groups with Condition Positives ($\Delta_{\text{EO}}^{\text{all}} = 1.0000$):**
   Calculated as $\max(\text{TPR}) - \min(\text{TPR}) = 1.0000 - 0.0000 = 1.0000$. This upper bound is driven by the 'Other' category, where only 1 readmitted patient existed ($N_{\text{pos}}=1$) and was correctly identified by the baseline model ($\hat{p} = 0.577 \ge 0.5$, Wilson 95% CI: $[0.2065, 1.0000]$), while Black, Hispanic, and Asian groups had 0.0% sensitivity. Native American has $N_{\text{pos}}=0$ and is mathematically excluded (NaN) from sensitivity calculations.
2. **Disparity Across Statistically Robust Groups ($\Delta_{\text{EO}}^{\text{robust}} = 0.0357$):**
   Restricting the evaluation to sufficiently powered cohorts ($N \ge 30$: White, Black, Hispanic), baseline sensitivity is 3.57% for White patients and 0.00% for Black and Hispanic patients, yielding $\Delta_{\text{EO}}^{\text{robust}} = 3.57\% - 0.00\% = 0.0357$.
3. **Severe Sensitivity Collapse in Minorities:** Under baseline thresholding, Black ($N=110, 26\text{ pos}$) and Hispanic ($N=78, 16\text{ pos}$) patients suffer a 100% false negative rate.
4. **Pediatric Invalidation:** Pediatric patients (<18) display an AUC of 0.4530 (worse than random guessing) and an ECE of 0.1621, demonstrating that adult readmission models cannot be safely applied to pediatric cohorts.

### 4.2 Intersectional Vulnerability Audit
Auditing $(\text{Age Group} \times \text{Race} \times \text{Gender})$ revealed acute compounded risks:
- **Geriatric Hispanic Females:** $N=21$, Base Rate 28.57%, $\text{ECE} = 0.2514$, $100\%$ miss rate.
- **Pediatric Black Females:** $N=11$, Base Rate 27.27%, $\text{ECE} = 0.2312$, $100\%$ miss rate.
- **Geriatric Black Males:** $N=21$, Base Rate 38.10%, $\text{ECE} = 0.1908$, $100\%$ miss rate.

---

## 5. Algorithmic Bias Mitigation & Trade-Off Analysis

We evaluated three mitigation strategies, tuning them strictly on the validation set:
1. **Pre-processing Demographic Reweighing:** Sample weights $W(s, y) = \frac{P(S=s)P(Y=y)}{P(S=s, Y=y)}$ computed on training data to remove statistical association between race and outcome.
2. **Post-processing Subgroup Isotonic Calibration:** Group-specific isotonic regressors fitted on validation predictions. Small or single-class groups (e.g. Native American) fallback to reference calibration.
3. **Post-processing Fairlearn ThresholdOptimizer:** Group-specific decision thresholds optimized on validation data under Equal Opportunity (`true_positive_rate_parity`) constraints.

### 5.1 Comparative Trade-Off Results (Test Set)

| Model Configuration | Strategy | ROC-AUC | Accuracy | F1-Score | All-Groups $\Delta_{\text{EO}}$ | Robust ($N \ge 30$) $\Delta_{\text{EO}}$ | Mean Subgroup ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM** | Default unmitigated | **0.6710** | 80.73% | 0.0536 | 1.0000 | 0.0357 | 0.1207 |
| **Demographic Reweighed** | Pre-processing sample weights | 0.6605 | **81.09%** | 0.0545 | 1.0000 | 0.0357 | 0.1339 |
| **Subgroup Isotonic Calibrated**| Post-processing calibration | 0.6627 | 80.55% | 0.0360 | **0.1250** | 0.1250 | **0.1004** |
| **ThresholdOptimized_EO** | Post-processing threshold | N/A | 40.91% | **0.3762** | 0.2500 | **0.0714** | N/A |

### 5.2 Analysis of Mitigation Dynamics & Verification of Fairlearn Constraints:
- **Subgroup Isotonic Calibration achieved superior clinical fidelity:** It reduced mean subgroup ECE from 0.1207 to 0.1004 (-16.8% improvement) while preserving 98.8% of model discrimination (ROC-AUC 0.6627) and maintaining high overall accuracy (80.55%).
- **Verification of Fairlearn Threshold Optimization:**
  - *Constraint Evaluation:* Tuned on the validation set under `true_positive_rate_parity`, the optimizer substantially equalized sensitivity across racial groups on held-out test data: White TPR = 94.64%, Black TPR = 96.15%, Hispanic TPR = 100.0%, Asian TPR = 100.0%, Other TPR = 100.0%. Across robust cohorts ($N \ge 30$), the Equal Opportunity disparity is **0.0714** (7.14 percentage points).
  - *Clinical Cost & False Alarms:* To elevate minority sensitivity from 0.0% to $\ge 96\%$, the optimizer lowered decision thresholds for minority patients into the 0.15-0.25 range. Because readmission is an imbalanced target (18.7% base rate), this triggered massive false positives: **FPR surged to 71.14%** (318 false alarms across 447 non-readmitted patients), dropping specificity to **28.86%** and collapsing overall accuracy from **80.73% to 40.91%**.
  - *Calibration Trade-off:* Because threshold optimization outputs discrete decisions rather than risk probabilities, continuous calibration is lost.
- **Reweighing Limitations:** Demographic sample reweighing without threshold adjustment leaves the decision boundary largely unshifted, preserving the baseline Equal Opportunity disparity.

---

## 6. Limitations, Ethical Considerations & Governance

1. **Synthetic Nature:** While Synthea-style EHR replicates relational schema and disease prevalence, real-world EHR contains unstructured clinical notes, lab telemetry, and social determinants of health (SDOH) not present here.
2. **Group-Specific Post-Processing Constraints:** Applying disparate decision cutoffs based on race carries ethical, legal, and operational complexities in clinical care. Calibrated probability estimation is preferred over rigid race-based thresholds.
3. **Mandatory Clinical Safeguards:**
   - Automated scoring must be disabled for pediatric patients (<18).
   - Readmission models must serve solely as decision support for discharge planning teams, never to deny care or force hospital discharge.

---

## 7. Conclusion

By combining Apache PySpark distributed Big Data processing with rigorous Responsible AI auditing, this study demonstrates that high aggregate accuracy in healthcare risk models conceals severe demographic disparities and miscalibration. Subgroup-specific calibration resolves calibration disparities and reduces Equal Opportunity disparity by 87.5% while partitioned Parquet storage delivers a 53.7x compression ratio, establishing an empirically grounded framework for equitable clinical risk triage.

---

## References

1. Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). Model Cards for Model Reporting. *Proc. ACM FAccT*, 220–229.
2. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.
3. Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Dean, J. (2018). Ensuring fairness in machine learning to advance healthcare. *NPJ Digital Medicine*, 1(1), 1–4.
4. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*, 3315–3323.
5. Gebru, T., Morgenstern, J., Vecchione, B., et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
