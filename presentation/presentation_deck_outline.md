# 8 to 10-Minute Capstone Presentation Deck Outline: Subgroup Performance in Healthcare Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE)  
**Department:** Computer Science & Engineering (AI & ML), PES University  
**Topic 19:** Subgroup Performance in Healthcare Risk Models  
**Faculty Evaluator:** Dr. Manjula Devananda  
**Team Members:**  
- **Ankith Umadi** (PES1UG24AM422)  
- **Haneesh E** (PES1UG24AM471)  
- **Vikas R** (PES1UG24AM477)  

**Presentation Duration:** 8 to 10 Minutes (Target: 9 Minutes + 2 Mins Q&A)  
**Template Standard:** PES University Capstone Defense Template  

---

## Slide 1: Title & Team Credentials (0:00 – 0:45)
- **Slide Content:**
  - Project Title: *Subgroup Performance, Calibration Disparities, and Scalable Fairness Mitigation in Healthcare Risk Models*
  - Subtitle: *An End-to-End Big Data Systems and Responsible AI Framework*
  - PES University Crest & CSE (AI & ML) Department Banner
  - Team Details: Ankith Umadi (PES1UG24AM422), Haneesh E (PES1UG24AM471), Vikas R (PES1UG24AM477)
  - Faculty Guide: Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Introduce team, SRNs, and project topic.
  - State the central thesis: Aggregate metrics in clinical machine learning hide life-threatening subgroup disparities; combining Big Data distributed engineering with Responsible AI auditing is necessary for safe deployment.

---

## Slide 2: Clinical Problem Statement & Case Study (0:45 – 1:30)
- **Slide Content:**
  - Case Scenario: Hospital 30-Day Readmission Risk Scoring (CMS HRRP context).
  - Clinical Decision-Support Pipeline: Inpatient discharge -> Risk score -> Triage care (phone calls, home nurse visits, medication reconciliation).
  - The Core Dilemma: False Negatives ($\text{FN}$) kill—patients discharged unmonitored experience acute decompensation. Miscalibration distorts triage tier thresholds.
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Explain why 30-day readmissions are the gold standard quality and reimbursement metric.
  - Explain the asymmetric cost of errors in healthcare: false alarms cost a \$20 phone call; missed readmissions cost \$15,000+ in emergency care and risk patient mortality.

---

## Slide 3: Big Data Systems Architecture & Parquet Lakehouse (1:30 – 2:30)
- **Slide Content:**
  - Architecture Diagram: Synthea Relational CSVs -> PySpark Distributed ETL -> Partitioned Parquet Lakehouse (`age_group`, `race`) -> Feature Store.
  - Technical Details:
    - PySpark Window Functions: `Window.partitionBy("patient_id").orderBy("enc_start")` for longitudinal readmission target derivation.
    - Distributed Joins across encounters, conditions (SNOMED-CT), medications (RxNorm), and procedures.
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Address Requirement 1 (Big Data systems design).
  - Highlight the partitioned Lakehouse design: partitioning on demographic audit axes enables predicate pushdown, allowing instant isolation of subgroup cohorts without full table scans.

---

## Slide 4: Measured Big Data Scalability Evidence (2:30 – 3:30)
- **Slide Content:**
  - Benchmark Table & Plots (`reports/figures/big_data_scaling.png`).
  - Empirical Evidence Metrics:
    - Storage Footprint: 43.9 MB CSV -> 1.4 MB Parquet (**$30.5\times$ compression ratio**).
    - Throughput: Scaling from 1,528 rec/s at 5k patients to **5,631 rec/s at 20k patients**.
    - Linear scaling latency: Runtime remains nearly flat (3.27s vs 3.55s) due to distributed Spark partition amortization.
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Emphasize Dr. Manjula's mandate for *measured evidence rather than description alone*.
  - Explain how Snappy columnar compression and Spark partition execution provide industrial-grade scalability.

---

## Slide 5: Baseline Healthcare Risk Models (3:30 – 4:15)
- **Slide Content:**
  - Model Architectures Evaluated: LightGBM, Logistic Regression, Random Forest.
  - Overall Test Performance Table:
    - LightGBM: ROC-AUC = 0.9010 | PR-AUC = 0.6679 | Accuracy = 82.82% | Brier Score = 0.0999.
    - Logistic Regression: ROC-AUC = 0.8910 | Accuracy = 81.50%.
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - On the surface, the baseline gradient-boosted tree model appears state-of-the-art with an AUC exceeding 0.90.
  - Transition to: "Why this number is dangerously deceptive."

---

## Slide 6: Responsible AI Subgroup Performance Disparity Audit (4:15 – 5:30)
- **Slide Content:**
  - Subgroup Disparity Table & Bar Chart (`reports/figures/subgroup_disparities.png`).
  - Key Findings Highlighted:
    - Pediatric (<18): Sensitivity drops to **21.43%** (**78.57% miss rate**).
    - Middle Age (45–64): Sensitivity is **45.16%** (**54.84% miss rate**).
    - Native American: Sensitivity drops to **0.00%** (**100% of readmitted patients missed**).
    - Equal Opportunity Disparity: **$\Delta_{\text{EO}} = 0.7500$**.
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - Walk through the clinical meaning of these numbers: under empirical risk minimization, the model prioritizes majority loss reduction and completely ignores low-prevalence demographics.

---

## Slide 7: Subgroup Calibration Disparities (ECE) (5:30 – 6:30)
- **Slide Content:**
  - Subgroup Expected Calibration Error Table.
  - Reliability Diagrams (`reports/figures/calibration_comparison.png`).
  - Key Disparities:
    - White Patients: ECE = **0.0287** (well-calibrated).
    - Asian Patients: ECE = **0.1107** ($4\times$ worse; predicted risk 16.17% vs observed 8.51%).
    - Native American: ECE = **0.2319**, MCE = **0.7562**.
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - Explain why calibration matters more than classification accuracy in medicine: clinical triage relies on risk tiers. Over-estimating Asian risk wastes hospital beds; under-estimating minority risk leads to fatal discharges.

---

## Slide 8: Intersectional Compounded Harms (6:30 – 7:15)
- **Slide Content:**
  - Intersectional ECE Heatmap (`reports/figures/intersectional_ece_heatmap.png`).
  - Worst Pockets:
    - **Geriatric Black Males**: ECE = **0.1796**, Sensitivity = **42.86%** (Miss rate 57.14%).
    - **Middle-Age White Females**: Sensitivity = **34.78%** (Miss rate 65.22%).
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - Demonstrate that aggregate one-dimensional audits miss compounded intersectional vulnerabilities (Simpson's Paradox in algorithmic bias).

---

## Slide 9: Responsible AI Mitigation Suite (7:15 – 8:15)
- **Slide Content:**
  - Three Mitigation Paradigms:
    1. Pre-Processing: Demographic Reweighing.
    2. Post-Processing: Subgroup Isotonic Calibration.
    3. Post-Processing: Fairlearn ThresholdOptimizer (Equalized Odds).
  - Before/After Comparison Table:
    - $\Delta_{\text{EO}}$ drops from **0.7500 to 0.3333** (**55.6% reduction in disparity**).
    - ROC-AUC remains rock-solid at **0.9010**; overall accuracy dips by only 2.75%.
- **Speaker:** Vikas R
- **Key Talking Points:**
  - Detail how threshold optimization dynamically adjusts the decision threshold per demographic group to ensure equal opportunity in post-discharge intervention.

---

## Slide 10: Fairness-Utility Pareto Frontier & Model Card Governance (8:15 – 9:15)
- **Slide Content:**
  - Pareto Frontier Scatter Plot (`reports/figures/fairness_utility_tradeoff.png`).
  - Model Card & Clinical Governance Highlights:
    - Intended Use: Clinical discharge planning decision support.
    - Prohibited Use: Automated care denial or insurance claims adjudication.
    - Residual Risk Warnings: Disable automated scoring for pediatric cohorts (<18); require human-in-the-loop verification.
- **Speaker:** Vikas R
- **Key Talking Points:**
  - Demonstrate that Responsible AI does not require sacrificing clinical predictive power.
  - Explain how Model and Data Cards operationalize accountability and governance.

---

## Slide 11: Summary of Deliverables & Live System Architecture (9:15 – 9:45)
- **Slide Content:**
  - Deliverables Matrix:
    - GitHub Repository with reproducible PySpark scripts & Docker Compose.
    - Pre-executed Results Notebook (`subgroup_risk_analysis.ipynb`).
    - 3-5 Page Technical Report, Model Card, Data Card, and Demo Video Script.
  - Summary of Impact: Bridging Big Data high-throughput lakehouse storage with Responsible AI clinical fairness.
- **Speaker:** Vikas R

---

## Slide 12: Conclusion & Q&A Defense Preparation (9:45 – 10:00)
- **Slide Content:**
  - "Thank you. We are open for questions."
  - Team Contact: `ankith.umadi@pes.edu`, `haneesh.e@pes.edu`, `vikas.r@pes.edu`.
- **Potential Evaluator Questions & Prepared Answers:**
  - *Q: Why choose Equal Opportunity over Demographic Parity in healthcare?*  
    **A:** In disease readmission, baseline illness prevalence differs across age and chronic disease groups. Demographic Parity would force equal readmission rates across healthy young and multimorbid elderly patients, which is medically invalid. Equal Opportunity ensures that *given* a patient is going to readmit, their probability of being detected and protected is equal regardless of race or age.
  - *Q: How does PySpark help Responsible AI?*  
    **A:** Big Data lakehouse partitioning by demographic attributes enables sub-second partition pruning, allowing real-time subgroup audits over millions of patient records without scanning unneeded data.
