# Model Card: Hospital 30-Day Readmission Risk Predictor with Subgroup Fairness Mitigation

Following the Mitchell et al. (2019) *Model Cards for Model Reporting* standard.

---

## 1. Model Details
- **Model Name:** ReadmissionRisk-FairBoost-v1.0
- **Model Architecture:** Gradient Boosted Decision Trees (LightGBM) with Post-Processing Equalized Odds Threshold Optimization & Subgroup Isotonic Calibration.
- **Model Date:** October 2026
- **Model Version:** 1.0.0
- **License:** MIT License
- **Developers:** Ankith Umadi (PES1UG24AM422), Haneesh E (PES1UG24AM471), Vikas R (PES1UG24AM477)
- **Institution:** Department of CSE (AI & ML), PES University, Bangalore, India
- **Course:** Big Data Systems & Responsible AI Engineering (BDSRAE)
- **Faculty Advisor:** Dr. Manjula Devananda

---

## 2. Intended Use Statement

### Primary Intended Uses:
- **Clinical Decision Support for Discharge Planning:** Designed to assist multidisciplinary hospital discharge teams (inpatient case managers, transitional care nurses, clinical pharmacists) in identifying adult patients at elevated risk of all-cause 30-day hospital readmission prior to discharge.
- **Transitional Resource Triage:** Facilitates targeted allocation of evidence-based transitional care interventions, including:
  1. Post-discharge telephone follow-up within 48–72 hours.
  2. In-home nurse visits and vitals monitoring.
  3. Medication reconciliation and compliance counseling.
  4. Rapid-turnaround primary care physician (PCP) follow-up appointment scheduling.

### Out-of-Scope and Prohibited Uses:
- **Automated Discharge Denial:** Strictly prohibited from being used to deny inpatient hospital discharge, delay necessary clinical transfers, or withhold palliative or hospice care.
- **Insurance Underwriting or Denials:** Must never be used by commercial payers or Medicare Advantage organizations to deny claims, determine insurance premiums, or adjust reimbursement capitation.
- **Pediatric Standalone Use:** The model is not validated for standalone decision-making in pediatric patients (<18 years) due to severe under-detection rates (miss rate 78.57%).
- **Sole Arbiter of Care:** The model is a decision-*support* tool; clinical judgment by attending physicians and nurse navigators must always supersede algorithmic outputs.

---

## 3. Factors and Subgroups
- **Demographic Attributes Evaluated:**
  - Age groups: Pediatric (<18), Young Adult (18–44), Middle Age (45–64), Geriatric (65+).
  - Sex / Gender: Female, Male.
  - Race & Ethnicity: White, Black/African American, Hispanic/Latino, Asian, Native American, Other.
  - Intersectional strata: Composite $(\text{Age Group} \times \text{Race} \times \text{Gender})$.
- **Clinical Features Utilized:**
  - Length of Stay (days), prior 12-month inpatient admissions, prior emergency visits, ambulatory encounters.
  - Chronic comorbidity indicators: Hypertension, Type 2 Diabetes, Congestive Heart Failure (CHF), COPD, Chronic Kidney Disease (CKD), Coronary Artery Disease (CAD), Depression.
  - Polypharmacy (total active prescribed medication count), diagnostic procedure count, and total claim expense.

---

## 4. Quantitative Metrics & Performance

### 4.1 Overall Performance
- **Global ROC-AUC:** 0.9010
- **Precision-Recall AUC (PR-AUC):** 0.6679
- **Overall Accuracy:** 80.07% (Post-Mitigation) / 82.82% (Baseline)
- **Global Brier Score:** 0.0999

### 4.2 Subgroup Disparities (Baseline vs. Mitigated)

| Metric | Baseline LightGBM | Post-Processing Mitigated | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Equal Opportunity Disparity ($\Delta_{\text{EO}}$)** | 0.7500 | **0.3333** | **-55.6%** |
| **Demographic Parity Disparity ($\Delta_{\text{DP}}$)** | 0.1758 | **0.1561** | **-11.2%** |
| **Mean Subgroup Calibration Error ($\text{ECE}$)** | 0.1026 | **0.0972** (Calibrated) | **-5.3%** |
| **Native American Miss Rate** | 100.0% | **33.3%** | **-66.7%** |
| **Geriatric Sensitivity** | 62.96% | **68.52%** | **+5.56%** |

---

## 5. Ethical Considerations & Disparity Root Causes
1. **Historical Access Bias in Synthetic & Real EHR:**
   Minority groups and underserved populations historically experience reduced access to outpatient primary care, leading to higher rates of acute emergency readmissions for ambulatory care-sensitive conditions (ACSCs). Unadjusted models penalize these patients or miscalculate their risk trajectory.
2. **False Negative Risk in Healthcare:**
   In clinical risk modeling, false negatives are vastly more harmful than false positives. A false negative deprives a high-risk patient of life-saving post-discharge monitoring, whereas a false positive merely results in an extra nurse telephone check-in or medication review.
3. **Severe Representation Gaps:**
   Synthetic EHR generators and real hospital registries suffer from small cohort sizes for Native American and Asian populations in specific hospital geographies. Training without subgroup-aware regularization causes empirical risk minimization (ERM) to ignore low-prevalence demographics.

---

## 6. Documented Residual Risks & Clinical Safety Warnings

> [!WARNING]
> **Residual Risk 1: Under-Detection in Pediatric (<18) Cohorts**
> Despite overall high AUC, the model's sensitivity in patients under 18 years is only 21.43% (78.57% miss rate). **Clinical action:** Disable automated scoring for patients under 18. Pediatric readmissions must use specialized pediatric scoring systems (e.g., Pediatric Risk of Readmission / PRR).

> [!WARNING]
> **Residual Risk 2: Intersectional Geriatric Black Male Vulnerability**
> Intersectional audits show that Geriatric Black Males face an Expected Calibration Error of 0.1796 and a 57.14% miss rate under baseline scoring. **Clinical action:** Clinical discharge coordinators must apply an automated risk floor multiplier of $1.25\times$ for geriatric patients with multiple chronic conditions.

> [!WARNING]
> **Residual Risk 3: Data Drift & Hospital Setting Shifts**
> The model was trained and calibrated on synthetic Massachusetts EHR distributions (Synthea). Deploying this model in distinct geographical jurisdictions (e.g., rural community hospitals or academic medical centers with different case mixes) will induce calibration drift. **Clinical action:** Recalibration using local hospital data is mandatory prior to clinical production use.

---

## 7. Model Governance & Maintenance Plan
- **Audit Cadence:** Monthly automated subgroup fairness reviews via PySpark batch jobs.
- **Trigger for Retraining:** Any drift where $\Delta_{\text{EO}} > 0.40$ or mean subgroup $\text{ECE} > 0.12$ across any single demographic stratum.
- **Human Oversight:** All algorithmic high-risk flags must be confirmed by an attending case manager prior to scheduling home nurse deployment.
